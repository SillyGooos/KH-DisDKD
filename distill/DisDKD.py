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


class DisDKD(nn.Module):
    """
    FitNet-style DKD:
    - DKD logit distillation (TCKD + NCKD)
    - Intermediate feature regression (MSE) from teacher -> student
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
        lambda_feat=1.0,  # Weight for feature regression loss
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lambda_feat = lambda_feat

        # Freeze teacher parameters
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

        # Feature loss
        self.mse_loss = nn.MSELoss()

        print(f"Teacher regressor params: {count_params(self.teacher_regressor)*1e-6:.3f}M")
        print(f"Student regressor params: {count_params(self.student_regressor)*1e-6:.3f}M")
        print(f"DKD params: alpha={alpha}, beta={beta}, temperature={temperature}")
        print(f"Feature regression weight: lambda_feat={lambda_feat}")

    def set_training_mode(self, mode):
        """
        Dummy method to maintain compatibility with Trainer.
        For FitNet/MSE, there's no discriminator phase.
        """
        # Only 'student' mode exists now
        if mode != "student":
            print(f"Ignoring training mode {mode}, using student mode only.")

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """Decoupled Knowledge Distillation loss (TCKD + NCKD)."""
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        # TCKD
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd)
        tckd_loss = (
            F.kl_div(log_pred_student_tckd, pred_teacher_tckd, reduction="batchmean")
            * (self.temperature**2)
        )

        # NCKD
        pred_teacher_nckd = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_nckd = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction="batchmean")
            * (self.temperature**2)
        )

        return self.alpha * tckd_loss + self.beta * nckd_loss

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

    def forward(self, x, targets):
        # Teacher and student logits
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Intermediate features
        teacher_feat = self.teacher_hooks.features.get(list(self.teacher_hooks.features.keys())[0])
        student_feat = self.student_hooks.features.get(list(self.student_hooks.features.keys())[0])

        if teacher_feat is None or student_feat is None:
            raise ValueError("Missing features from hooks!")

        # Project to hidden space
        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)

        # Feature regression loss
        feat_loss = self.mse_loss(student_hidden, teacher_hidden)

        # DKD loss
        dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

        total_loss = dkd_loss + self.lambda_feat * feat_loss

        # Clear hooks
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return {
            "teacher_logits": teacher_logits,
            "student_logits": student_logits,
            "dkd_loss": dkd_loss.item(),
            "feat_loss": feat_loss.item(),
            "total_loss": total_loss
        }

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        """Single optimizer for student + regressor."""
        params = list(self.student.parameters()) + list(self.student_regressor.parameters())
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
