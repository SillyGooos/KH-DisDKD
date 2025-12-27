import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """Helper class to extract intermediate features using forward hooks."""
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
    """1x1 convolutional regressor to project features to a common dimension."""
    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """Discriminator to distinguish between teacher and student features."""
    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
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
        pooled = self.global_pool(x)
        return self.discriminator(pooled)


class DisDKD(nn.Module):
    """
    Discriminator-enhanced Knowledge Distillation (Logit Matching version).
    
    NOTE: Name is kept as DisDKD for compatibility, but logic uses 
    Standard KL Divergence for logits instead of Decoupled KD.
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
        beta=1.0, # Usually beta is lower for standard KD compared to DKD
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

        # Set up hooks
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Projectors and Discriminator
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)
        self.discriminator = FeatureDiscriminator(hidden_channels)
        
        self.bce_loss = nn.BCELoss()
        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")
        self.training_mode = "student"

    def set_training_mode(self, mode):
        assert mode in ["student", "discriminator"]
        self.training_mode = mode
        
        # Toggle gradients
        is_disc = (mode == "discriminator")
        for param in self.student.parameters(): param.requires_grad = not is_disc
        for param in self.student_regressor.parameters(): param.requires_grad = not is_disc
        for param in self.discriminator.parameters(): param.requires_grad = is_disc
        for param in self.teacher_regressor.parameters(): param.requires_grad = is_disc

    def compute_logit_kd_loss(self, logits_student, logits_teacher):
        """
        Standard Logit Matching using KL Divergence.
        Replaces the Decoupled KD (TCKD/NCKD) logic.
        """
        T = self.temperature
        soft_targets = F.softmax(logits_teacher / T, dim=1)
        log_probs = F.log_softmax(logits_student / T, dim=1)
        
        # Return KL Divergence scaled by T^2
        return self.kd_criterion(log_probs, soft_targets) * (T * T)

    def forward(self, x, targets):
        batch_size = x.size(0)

        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)

        result = {"teacher_logits": teacher_logits, "student_logits": student_logits}

        if self.training_mode == "discriminator":
            teacher_pred = self.discriminator(teacher_hidden.detach()) # Added detach to avoid regressor updates
            student_pred = self.discriminator(student_hidden.detach())

            real_labels = torch.ones(batch_size, 1, device=x.device)
            fake_labels = torch.zeros(batch_size, 1, device=x.device)

            loss_real = self.bce_loss(teacher_pred, real_labels)
            loss_fake = self.bce_loss(student_pred, fake_labels)
            disc_loss = (loss_real + loss_fake) / 2

            result["total_disc_loss"] = disc_loss
            result["discriminator_loss"] = disc_loss.item()
            result["discriminator_accuracy"] = ((teacher_pred > 0.5).float().mean() + (student_pred <= 0.5).float().mean()).item() / 2

        else:
            # PHASE 2: Student training
            kd_loss = self.compute_logit_kd_loss(student_logits, teacher_logits)
            
            student_pred = self.discriminator(student_hidden)
            real_labels = torch.ones(batch_size, 1, device=x.device)
            adv_loss = self.bce_loss(student_pred, real_labels)

            # Match these keys to training.py
            result["kd_loss"] = kd_loss # Return tensor for backward
            result["adversarial_loss"] = adv_loss.item()
            result["total_student_loss"] = adv_loss
            result["fool_rate"] = (student_pred > 0.5).float().mean().item()
            # This is the 'method_specific_loss' used in the total_loss calculation
            result["method_specific_loss"] = kd_loss 

        self.teacher_hooks.clear()
        self.student_hooks.clear()
        return result