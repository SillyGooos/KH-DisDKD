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
    Contrastive Representation Distillation (CRD) Loss.

    Based on the paper: "Contrastive Representation Distillation" (Tian et al., ICLR 2020)
    Uses Noise Contrastive Estimation (NCE) framework.

    Args:
        n_data (int): Number of training samples
        feat_dim (int): Dimension of the feature embeddings
        temperature (float): Temperature parameter for contrastive loss
        momentum (float): Momentum for updating negative prototypes
        n_negatives (int): Number of negative samples to use
    """

    def __init__(
        self, n_data, feat_dim=128, temperature=0.07, momentum=0.5, n_negatives=4096
    ):
        super(ContrastiveLoss, self).__init__()
        if n_data < 2:
            raise ValueError(f"CRD requires n_data >= 2, got {n_data}")
        
        self.n_data = n_data
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.momentum = momentum
        self.n_negatives = n_negatives

        # Initialize memory bank for negative samples
        # FIXED: Use feat_dim instead of hard-coded 128
        self.register_buffer("memory_bank", torch.randn(n_data, feat_dim))
        self.memory_bank = F.normalize(self.memory_bank, dim=1)

    def forward(self, student_feat, teacher_feat, indices):
        """
        Compute contrastive loss between student and teacher features.

        Args:
            student_feat (Tensor): Student features [batch_size, feat_dim]
            teacher_feat (Tensor): Teacher features [batch_size, feat_dim]
            indices (Tensor): Sample indices for the current batch [batch_size]

        Returns:
            Tensor: Contrastive loss
        """
        batch_size = student_feat.size(0)

        # Normalize features
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)

        # =================================================================
        # POSITIVE PAIRS: student-teacher correspondence
        # =================================================================
        # Compute similarity between student and teacher (positive pairs)
        pos_logits = (
            torch.sum(student_feat * teacher_feat, dim=1, keepdim=True)
            / self.temperature
        )

        # =================================================================
        # NEGATIVE SAMPLES: from memory bank
        # =================================================================
        # We must never sample the positive index for each anchor.
        # Optional: also avoid sampling any index that is in the current batch.

        # indices: [B]
        idx = indices.long().to(student_feat.device).view(-1)  # [B]

        if idx.numel() != batch_size:
            raise RuntimeError(f"indices length {idx.numel()} != batch_size {batch_size}")

        B = idx.size(0)

        # 1) Sample from [0, n_data-2] then shift to skip idx per row.
        # This guarantees neg != idx for each sample without any while-loop.
        r = torch.randint(
            low=0,
            high=self.n_data - 1,
            size=(B, self.n_negatives),
            device=student_feat.device,
        )

        # Shift values >= idx upward by 1 -> now in [0, n_data-1] but never equal to idx
        neg_indices = r + (r >= idx.view(B, 1)).long()

        # # 2) OPTIONAL: avoid sampling any index from the current batch (stronger)
        # # This is slower but correct. If you want it, keep it.
        # avoid_batch = True
        # if avoid_batch:
        #     forbidden = idx.unique()
        #     # Resample only the forbidden positions until clean (bounded loop)
        #     # Using the same "shift trick" but we need per-row skip again, so we do it per-mask.
        #     for _ in range(10):  # 10 is plenty in practice
        #         mask = torch.isin(neg_indices, forbidden)
        #         if not mask.any():
        #             break
        #         # resample masked positions with the same trick
        #         rr = torch.randint(
        #             low=0,
        #             high=self.n_data - 1,
        #             size=(int(mask.sum().item()),),
        #             device=student_feat.device,
        #         )
        #         # Need the row-wise idx for each masked element
        #         row_ids = mask.nonzero(as_tuple=False)[:, 0]  # [K]
        #         rr = rr + (rr >= idx[row_ids]).long()
        #         neg_indices[mask] = rr
        #     else:
        #         # If we somehow cannot cleanly sample (extremely small datasets),
        #         # fall back to only excluding the positive per-row.
        #         pass
            
        # Retrieve negative samples from memory bank
        neg_feat = self.memory_bank[neg_indices]  # [B, n_negatives, feat_dim]
        neg_feat = F.normalize(neg_feat, dim=2)   # robust

        # Compute similarity between student features and negative samples
        neg_logits = torch.bmm(neg_feat, student_feat.unsqueeze(2)).squeeze(2) / self.temperature
        # neg_logits: [B, n_negatives]


        # =================================================================
        # NCE LOSS COMPUTATION
        # =================================================================
        # Concatenate positive and negative logits
        # Shape: [batch_size, 1 + n_negatives]
        logits = torch.cat([pos_logits, neg_logits], dim=1)

        # Labels: positive pair is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=student_feat.device)

        # Compute InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # =================================================================
        # UPDATE MEMORY BANK
        # =================================================================
        # Update memory bank with current teacher features using momentum
        if self.training and indices is not None:
            with torch.no_grad():
                # Ensure indices are valid
                idx = indices.long().to(self.memory_bank.device)

                # Ensure we have matching batch sizes
                if idx.size(0) != teacher_feat.size(0):
                    raise RuntimeError(f"CRD memory update mismatch: idx {idx.size(0)} vs teacher_feat {teacher_feat.size(0)}")

                if idx.numel() > 0:
                    # Momentum update: new = momentum * old + (1 - momentum) * new
                    updated_feat = (
                        self.momentum * self.memory_bank[idx]
                        + (1 - self.momentum) * teacher_feat.detach()
                    )
                    # Re-normalize after momentum update
                    updated_feat = F.normalize(updated_feat, dim=1)
                    self.memory_bank[idx] = updated_feat

        return loss


class CRD(nn.Module):
    """
    Contrastive Representation Distillation.

    This method uses contrastive learning to match student and teacher
    representations at intermediate layers.

    Reference: "Contrastive Representation Distillation" (Tian et al., ICLR 2020)

    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network.
        teacher_layer (str): Name of the teacher layer to extract features from.
        student_layer (str): Name of the student layer to extract features from.
        teacher_channels (int): Number of channels in teacher feature map.
        student_channels (int): Number of channels in student feature map.
        n_data (int): Number of training samples for contrastive loss.
        feat_dim (int): Dimension of the projected features (default: 128).
        temperature (float): Temperature for contrastive loss (default: 0.07).
        momentum (float): Momentum for updating memory bank (default: 0.5).
        n_negatives (int): Number of negative samples (default: 4096).
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
        momentum=0.5,
        n_negatives=4096,
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
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Projection heads to map features to common dimension
        # Teacher projector
        self.teacher_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(teacher_channels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # Student projector
        self.student_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(student_channels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # Contrastive loss with memory bank
        self.contrastive_loss = ContrastiveLoss(
            n_data=n_data,
            feat_dim=feat_dim,
            temperature=temperature,
            momentum=momentum,
            n_negatives=n_negatives,
        )

        # Store sample indices for each batch (used for memory bank updates)
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
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Extract intermediate features
        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        if teacher_feat is None or student_feat is None:
            raise ValueError(
                f"Missing features for layers: {self.teacher_layer} or {self.student_layer}"
            )

        # Project features to common dimension
        teacher_proj = self.teacher_projector(teacher_feat)
        student_proj = self.student_projector(student_feat)

        if self.sample_indices is None:
            if self.training:
                raise RuntimeError("CRD: sample_indices not set. Train loader must return (x,y,idx).")
            # eval mode: do not compute CRD loss
            crd_loss = torch.zeros([], device=student_proj.device)
        else:
            crd_loss = self.contrastive_loss(student_proj, teacher_proj, self.sample_indices)

        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return teacher_logits, student_logits, crd_loss
