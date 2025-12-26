import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module


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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for CRD.
    
    Args:
        n_data (int): Number of training samples
        temperature (float): Temperature parameter for contrastive loss
        momentum (float): Momentum for updating negative prototypes
    """
    def __init__(self, n_data, temperature=0.07, momentum=0.5):
        super(ContrastiveLoss, self).__init__()
        self.n_data = n_data
        self.temperature = temperature
        self.momentum = momentum
        
        # Initialize negative prototypes
        # NOTE: feat dim is hard-coded as 128 here to match original code.
        self.register_buffer('negatives', torch.randn(n_data, 128))
        self.negatives = F.normalize(self.negatives, dim=1)
        
    def forward(self, student_feat, teacher_feat, indices):
        """
        Compute contrastive loss between student and teacher features.
        
        Args:
            student_feat (Tensor): Student features [batch_size, feat_dim]
            teacher_feat (Tensor): Teacher features [batch_size, feat_dim]
            indices (Tensor): Sample indices for negative selection [batch_size]
            
        Returns:
            Tensor: Contrastive loss
        """
        batch_size = student_feat.size(0)
        
        # Normalize features
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
        
        # Compute positive similarity (student-teacher pairs)
        pos_sim = torch.sum(student_feat * teacher_feat, dim=1) / self.temperature
        
        # Sample negative examples
        neg_indices = torch.randint(0, self.n_data, (batch_size, 4096), device=student_feat.device)
        neg_feat = self.negatives[neg_indices]  # [batch_size, 4096, feat_dim]
        
        # Compute negative similarities
        neg_sim = torch.bmm(
            student_feat.unsqueeze(1),  # [batch_size, 1, feat_dim]
            neg_feat.transpose(1, 2)    # [batch_size, feat_dim, 4096]
        ).squeeze(1) / self.temperature  # [batch_size, 4096]
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + 4096]
        
        # Labels: positive pairs are at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=student_feat.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Update negatives with momentum
        # Guard the update so it only happens during training and avoid shape mismatches:
        # - If `indices` length doesn't match current batch size (e.g., leftover training indices used during validation),
        #   take a defensive min-length slice so shapes align.
        # - Skip updates during evaluation (self.training==False).
        if self.training and indices is not None:
            with torch.no_grad():
                # ensure indices is on CPU/gpu consistently and is long
                idx = indices.long().to(self.negatives.device)
                n_idx = idx.size(0)
                n_feat = teacher_feat.size(0)
                if n_idx != n_feat:
                    # Defensive fallback: align to smallest length to avoid RHS/LHS shape mismatch
                    n = min(n_idx, n_feat)
                    idx = idx[:n]
                    update_feat = teacher_feat.detach()[:n]
                else:
                    update_feat = teacher_feat.detach()

                # Final sanity: if there are zero indices, skip
                if idx.numel() > 0:
                    # momentum update for the memory bank slots corresponding to idx
                    updated = self.momentum * self.negatives[idx] + (1 - self.momentum) * update_feat
                    updated = F.normalize(updated, dim=1)
                    self.negatives[idx] = updated
        
        return loss


class CRD(nn.Module):
    """
    Contrastive Representation Distillation.
    
    This method uses contrastive learning to match student and teacher 
    representations at intermediate layers.
    
    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network.
        teacher_layer (str): Name of the teacher layer to extract features from.
        student_layer (str): Name of the student layer to extract features from.
        teacher_channels (int): Number of channels in teacher feature map.
        student_channels (int): Number of channels in student feature map.
        n_data (int): Number of training samples for contrastive loss.
        feat_dim (int): Dimension of the projected features.
        temperature (float): Temperature for contrastive loss.
        momentum (float): Momentum for updating negative prototypes.
    """
    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        n_data,
        feat_dim=128,
        temperature=0.07,
        momentum=0.5
    ):
        super(CRD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.n_data = n_data
        self.feat_dim = feat_dim
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Set up hooks for feature extraction
        self.teacher_hooks = FeatureHooks([
            (teacher_layer, get_module(self.teacher.model, teacher_layer))
        ])
        self.student_hooks = FeatureHooks([
            (student_layer, get_module(self.student.model, student_layer))
        ])
        
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        
        # Projection heads to map features to common dimension
        self.teacher_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(teacher_channels, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.student_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(student_channels, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(n_data, temperature, momentum)
        
        # Store sample indices for each batch (used for negative sampling)
        self.sample_indices = None
    
    def set_sample_indices(self, indices):
        """Set sample indices for the current batch."""
        self.sample_indices = indices
    
    def forward(self, x):
        """
        Forward pass that computes teacher/student outputs and contrastive loss.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            tuple: (teacher_logits, student_logits, contrastive_loss)
        """
        # Forward pass through teacher and student networks
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        
        # Extract intermediate features
        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)
        
        if teacher_feat is None or student_feat is None:
            raise ValueError(f"Missing features for layers: {self.teacher_layer} or {self.student_layer}")
        
        # Project features to common dimension
        teacher_proj = self.teacher_projector(teacher_feat)
        student_proj = self.student_projector(student_feat)
        
        # Compute contrastive loss
        if self.sample_indices is not None:
            crd_loss = self.contrastive_loss(student_proj, teacher_proj, self.sample_indices)
        else:
            # Fallback: use batch indices if sample_indices not set
            batch_size = x.size(0)
            indices = torch.arange(batch_size, device=x.device)
            crd_loss = self.contrastive_loss(student_proj, teacher_proj, indices)
        
        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()
        
        return teacher_logits, student_logits, crd_loss