import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    """

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
        """Clears the stored features."""
        self.features.clear()

    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FeatureRegressor(nn.Module):
    """
    Light 1x1 convolutional regressor to project features to a common hidden dimension.

    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels for common space
    """

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """
    Lightweight discriminator to distinguish between teacher and student features.

    Args:
        hidden_channels (int): Number of channels in the hidden feature space
    """

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Simple discriminator network
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features [batch_size, hidden_channels, H, W]

        Returns:
            Tensor: Discriminator output [batch_size, 1] (probability of being teacher)
        """
        # Global average pooling
        pooled = self.global_pool(x)  # [batch_size, hidden_channels, 1, 1]

        # Discriminate
        output = self.discriminator(pooled)  # [batch_size, 1]

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
# ... (Keep FeatureHooks, FeatureRegressor, FeatureDiscriminator as they are)

class DisDKD(nn.Module):
    """
    Discriminator-enhanced Logit Knowledge Distillation.
    
    Combines standard KL-Divergence logit matching with 
    discriminator-based feature alignment.
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
        alpha=1.0, # Weight for CE
        beta=1.0,  # Weight for KL Logit KD
        temperature=4.0,
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Hooks
        self.teacher_hooks = FeatureHooks([(teacher_layer, get_module(self.teacher.model, teacher_layer))])
        self.student_hooks = FeatureHooks([(student_layer, get_module(self.student.model, student_layer))])
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Regressors and Discriminator
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)
        self.discriminator = FeatureDiscriminator(hidden_channels)
        self.bce_loss = nn.BCELoss()
        self.training_mode = "student"

    def set_training_mode(self, mode):
        # ... (Keep same logic as DisDKD to freeze/unfreeze params)
        pass

    def compute_kd_loss(self, logits_student, logits_teacher):
        """Standard KL Divergence for Logit Matching."""
        T = self.temperature
        p_s = F.log_softmax(logits_student / T, dim=1)
        p_t = F.softmax(logits_teacher / T, dim=1)
        # We multiply by T^2 as per original KD paper to keep gradients consistent
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    def forward(self, x, targets):
        batch_size = x.size(0)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        # Feature projection
        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)
        if student_hidden.shape[2:] != teacher_hidden.shape[2:]:
            student_hidden = F.interpolate(student_hidden, size=teacher_hidden.shape[2:], mode="bilinear")

        result = {"teacher_logits": teacher_logits, "student_logits": student_logits}

        if self.training_mode == "discriminator":
            teacher_pred = self.discriminator(teacher_hidden)
            student_pred = self.discriminator(student_hidden.detach())
            
            real_labels = torch.ones(batch_size, 1, device=x.device)
            fake_labels = torch.zeros(batch_size, 1, device=x.device)

            disc_loss = (self.bce_loss(teacher_pred, real_labels) + 
                         self.bce_loss(student_pred, fake_labels)) / 2
            
            result["discriminator_loss"] = disc_loss.item()
            result["discriminator_accuracy"] = ((teacher_pred > 0.5).float().mean() + 
                                               (student_pred <= 0.5).float().mean()).item() / 2
            result["total_disc_loss"] = disc_loss

        else: # student mode
            # Standard KD Loss instead of DKD
            kd_loss = self.compute_kd_loss(student_logits, teacher_logits)
            
            student_pred = self.discriminator(student_hidden)
            real_labels = torch.ones(batch_size, 1, device=x.device)
            adversarial_loss = self.bce_loss(student_pred, real_labels)

            result["kd_loss"] = kd_loss.item()
            result["adversarial_loss"] = adversarial_loss.item()
            result["fool_rate"] = (student_pred > 0.5).float().mean().item()
            result["total_student_loss"] = adversarial_loss
            result["method_specific_loss"] = kd_loss # This goes to beta in Trainer

        self.teacher_hooks.clear()
        self.student_hooks.clear()
        return result