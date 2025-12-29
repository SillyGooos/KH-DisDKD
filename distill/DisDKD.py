import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params

class FeatureHooks:
    """Helper class to extract intermediate features from a network using forward hooks."""
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class FeatureRegressor(nn.Module):
    """1x1 conv regressor to project features to a common hidden dimension."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)
    
class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.tensor([0.0]))
    def forward(self, x): return x

class DisDKD(nn.Module):
    """
    Modified DisDKD: Now performs FitNet-style MSE + DKD.
    Includes dummy attributes to maintain compatibility with adversarial Trainer logic.
    """
    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        hidden_channels=256,
        alpha=1.0,
        beta=8.0,
        temperature=4.0,
        lambda_feat=1.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lambda_feat = lambda_feat

        # --- MANDATORY DUMMY FOR TRAINER ---
        # Trainer looks for 'model.discriminator'. We provide a dummy to prevent NoneType error.
        self.discriminator = DummyDiscriminator()
        self.training_mode = "student" 

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Feature hooks
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        # Regressors
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        self.mse_loss = nn.MSELoss()

    def set_training_mode(self, mode):
        """Sets the internal mode to handle the Trainer's two-step call."""
        self.training_mode = mode

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        
        tckd_loss = F.kl_div(torch.log(pred_student_tckd + 1e-7), pred_teacher_tckd, reduction="batchmean") * (self.temperature**2)

        pred_teacher_nckd = F.softmax(logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1)
        log_pred_student_nckd = F.log_softmax(logits_student / self.temperature - 1000.0 * gt_mask, dim=1)
        
        nckd_loss = F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction="batchmean") * (self.temperature**2)

        return self.alpha * tckd_loss + self.beta * nckd_loss

    def forward(self, x, targets):
        # 1. Handle Discriminator phase (Dummy)
        if self.training_mode == "discriminator":
            # Access the dummy parameter directly to provide a differentiable 0.0
            # This ensures the optimizer has a 'loss' to step on without changing weights.
            dummy_loss = self.discriminator.dummy_param * 0 
            
            return {
                "total_disc_loss": dummy_loss, 
                "discriminator_loss": 0.0,
                "discriminator_accuracy": 1.0 
            }

        # 2. Handle Student phase (Actual MSE + DKD)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)

        # Retrieve feature maps from the hooks
        teacher_feat = self.teacher_hooks.features.get(list(self.teacher_hooks.features.keys())[0])
        student_feat = self.student_hooks.features.get(list(self.student_hooks.features.keys())[0])

        # --- MSE FEATURE ALIGNMENT ---
        # 1x1 convolutions project student/teacher to the same hidden dimension
        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)
        
        # This is the "Hint" or "FitNet" loss
        feat_loss = self.mse_loss(student_hidden, teacher_hidden)

        # --- DKD LOGIT DISTILLATION ---
        dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

        # Clean up hooks
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            # We pass the MSE loss here as the 'adversarial' component
            "total_student_loss": feat_loss * self.lambda_feat, 
            "kd_loss": dkd_loss,
            "method_specific_loss": dkd_loss,
            "adversarial_loss": feat_loss.item(), # This will show up in logs as MSE
            "fool_rate": 0.0
        }