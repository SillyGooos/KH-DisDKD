import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.utils import accuracy, AverageMeter
from utils.model_factory import create_distillation_model, print_model_parameters
from utils.checkpoint import save_checkpoint

class Trainer:
    """Handles model training and validation."""

    def __init__(
        self, teacher, student, num_classes, criterion, loss_tracker, device, args
    ):
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.device = device
        self.args = args

        # Create distillation model
        self.model = create_distillation_model(args, teacher, student, num_classes).to(
            device
        )
        print_model_parameters(self.model, args.method)

        # Setup optimizers
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup optimizer(s) and scheduler(s)."""
        args = self.args

        # Methods with discriminator need separate optimizers
        if args.method in ["DisDKD"]:
            self.student_optimizer, self.student_scheduler = self._create_optimizer(
                self.model
            )

            discriminator_obj = getattr(self.model, "discriminator", None)
            if discriminator_obj is not None:
                self.discriminator_optimizer = optim.Adam(
                    discriminator_obj.parameters(),
                    lr=args.discriminator_lr * args.disc_lr_multiplier,
                    weight_decay=args.weight_decay,
                )
            else:
                self.discriminator_optimizer = None
                print(f"Warning: {args.method} selected but no discriminator found.")
        else:
            self.student_optimizer, self.student_scheduler = self._create_optimizer(
                self.model
            )
            self.discriminator_optimizer = None

    def _create_optimizer(self, model):
        """Create optimizer and scheduler for a model."""
        args = self.args
        optimizers = {"adam": optim.Adam, "sgd": optim.SGD, "adamw": optim.AdamW}

        optimizer_class = optimizers[args.optimizer.lower()]
        optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

        if args.optimizer.lower() == "sgd":
            optimizer_kwargs["momentum"] = args.momentum

        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

        return optimizer, scheduler

    def train(self, train_loader, val_loader):
        """Main training loop with FitNet 2-Stage Logic."""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        best_acc = 0.0

        # Store original weights to allow restoration during Stage 2
        original_alpha = self.args.alpha
        original_beta = self.args.beta
        original_gamma = self.args.gamma

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # --- FITNET STAGE SWITCHING LOGIC ---
            if self.args.method == 'FitNet' and self.args.fitnet_stage1_epochs > 0:
                if epoch < self.args.fitnet_stage1_epochs:
                    # STAGE 1: HINT ONLY (Train Regressor + Student backbone)
                    self.args.alpha = 0.0
                    self.args.beta = 0.0
                    self.args.gamma = original_gamma
                    stage_name = "Stage 1 (Hint Only)"
                else:
                    # STAGE 2: TASK + DISTILLATION (Standard training)
                    self.args.alpha = original_alpha
                    self.args.beta = original_beta
                    self.args.gamma = 0.0 # Typically 0.0 in FitNet stage 2, or keep original if desired
                    stage_name = "Stage 2 (Task)"
            else:
                stage_name = "Standard"

            # Train and validate
            if self.args.method in ["DisDKD"]:
                train_losses, train_acc = self._train_epoch_adversarial(train_loader, epoch)
            else:
                train_losses, train_acc = self._train_epoch_standard(train_loader, epoch)

            val_losses, val_acc = self._validate(val_loader, epoch)

            # Log metrics
            lr = self.student_optimizer.param_groups[0]["lr"]
            self.loss_tracker.log_epoch(epoch, "train", train_losses, train_acc, lr=lr)
            self.loss_tracker.log_epoch(epoch, "val", val_losses, val_acc, lr=lr)

            self.student_scheduler.step()

            # --- MERGED LOGGING ---
            elapsed = time.time() - start_time
            if self.args.method == "DisDKD":
                disc_acc = train_losses.get("disc_accuracy", 0) * 100
                fool_rate = train_losses.get("fool_rate", 0) * 100
                print(f"Epoch {epoch}: Train {train_acc:.2f}%, Val {val_acc:.2f}% | "
                      f"Disc_Acc: {disc_acc:.1f}%, Fool: {fool_rate:.1f}% | "
                      f"Time: {elapsed:.1f}s")
            else:
                print(f"Epoch {epoch} [{stage_name}]: Train {train_acc:.2f}%, Val {val_acc:.2f}%, Time {elapsed:.1f}s")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(self.model, self.student_optimizer, epoch, val_acc, self.args, is_best=True)

            print("-" * 80)

        return best_acc

    def _train_epoch_standard(self, train_loader, epoch):
        """Train for one epoch (standard methods)."""
        self.model.train()
        meters = self._init_meters()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            if self.args.method == "CRD":
                inputs, targets, indices = batch_data
                self.model.set_sample_indices(indices)
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.student_optimizer.zero_grad()

            # Forward pass
            if self.args.method in ["DKD"]:
                teacher_logits, student_logits, method_specific_loss = self.model(inputs, targets)
            else:
                teacher_logits, student_logits, method_specific_loss = self.model(inputs)

            # Compute losses
            total_loss, losses_dict = self._compute_losses(
                teacher_logits, student_logits, targets, method_specific_loss, inputs
            )

            total_loss.backward()
            self.student_optimizer.step()

            self._update_meters(meters, losses_dict, student_logits, targets, inputs.size(0))
            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                self._update_progress_bar(progress_bar, meters)

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _train_epoch_adversarial(self, train_loader, epoch):
        self.model.train()
        meters = self._init_meters(adversarial=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train ({self.args.method})", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Step 1: Train Discriminator
            self.model.set_training_mode("discriminator")
            self.discriminator_optimizer.zero_grad()
            disc_result = self.model(inputs, targets)
            disc_result["total_disc_loss"].backward()
            self.discriminator_optimizer.step()

            # Step 2: Train Student
            self.model.set_training_mode("student")
            self.student_optimizer.zero_grad()
            student_result = self.model(inputs, targets)
            
            student_logits = student_result["student_logits"]
            ce_loss = self.criterion(student_logits, targets)
            
            # Use the KD loss from the model forward pass
            dkd_kd_loss = student_result.get("kd_loss", torch.tensor(0.0, device=self.device))

            # Total Loss Assembly
            # alpha: CE, beta: standard KD (set to 0 if only using DisDKD), gamma: adversarial
            total_loss = (self.args.alpha * ce_loss + 
                         self.args.disdkd_adversarial_weight * student_result["total_student_loss"] + 
                         dkd_kd_loss) # This is the DisDKD logit matching loss

            total_loss.backward()
            self.student_optimizer.step()

            # Pass dkd_kd_loss into the meter update
            self._update_adversarial_meters(meters, disc_result, student_result, ce_loss, dkd_kd_loss, total_loss, student_logits, targets, inputs.size(0))

            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                self._update_adversarial_progress_bar(progress_bar, meters)

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _validate(self, val_loader, epoch=None):
        """Validate the model."""
        self.model.eval()
        if hasattr(self.model, "set_sample_indices"):
            self.model.set_sample_indices(None)

        meters = self._init_meters(adversarial=False)
        desc = f"Epoch {epoch} Val" if epoch is not None else "Validation"
        progress_bar = tqdm(val_loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                if len(batch) == 3:
                    inputs, targets, indices = batch
                    indices = indices.to(self.device)
                else:
                    inputs, targets = batch
                    indices = None

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.args.method == "CRD" and indices is not None:
                    self.model.set_sample_indices(indices)

                # Forward logic matching training
                if self.args.method in ["DisDKD"]:
                    self.model.set_training_mode("student")
                    result = self.model(inputs, targets)
                    student_logits = result["student_logits"]
                    teacher_logits = result.get("teacher_logits")
                    method_specific_loss = result.get("method_specific_loss")
                elif self.args.method in ["DKD"]:
                    teacher_logits, student_logits, method_specific_loss = self.model(inputs, targets)
                else:
                    teacher_logits, student_logits, method_specific_loss = self.model(inputs)

                ce_loss = self.criterion(student_logits, targets)
                kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
                
                total = self.args.alpha * ce_loss + self.args.beta * kd_loss
                losses_dict = {"ce": ce_loss.item(), "kd": kd_loss.item()}

                if method_specific_loss is not None:
                    total += self.args.gamma * method_specific_loss
                    if self.args.method == "CRD": losses_dict["contrastive"] = method_specific_loss.item()
                    elif self.args.method == "FitNet": losses_dict["hint"] = method_specific_loss.item()
                    elif self.args.method == "DKD": losses_dict["dkd"] = method_specific_loss.item()

                losses_dict["total"] = total.item()
                self._update_meters(meters, losses_dict, student_logits, targets, inputs.size(0))

                progress_bar.set_postfix({"val_loss": f"{meters['total'].avg:.4f}", "val_acc": f"{meters['accuracy'].avg:.2f}%"})

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _compute_kd_loss(self, teacher_logits, student_logits):
        if teacher_logits is None: return torch.tensor(0.0, device=student_logits.device)
        T = self.args.tau
        return nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_logits / T, dim=1),
            nn.functional.softmax(teacher_logits / T, dim=1),
        ) * (T * T)

    def _compute_losses(self, teacher_logits, student_logits, targets, method_specific_loss, inputs=None):
        ce_loss = self.criterion(student_logits, targets)
        kd_loss = torch.tensor(0.0, device=student_logits.device) if self.args.method in ["DKD"] else self._compute_kd_loss(teacher_logits, student_logits)

        total_loss = self.args.alpha * ce_loss + self.args.beta * kd_loss
        losses_dict = {"ce": ce_loss.item(), "kd": kd_loss.item()}

        if method_specific_loss is not None:
            if self.args.method == "DKD":
                total_loss += method_specific_loss # Already weighted in model forward
                tckd, nckd = self._compute_dkd_components(student_logits, teacher_logits, targets)
                losses_dict.update({"tckd": tckd, "nckd": nckd})
            else:
                total_loss += self.args.gamma * method_specific_loss
                key = {"FitNet": "hint", "CRD": "contrastive"}.get(self.args.method, "method_loss")
                losses_dict[key] = method_specific_loss.item()

        losses_dict["total"] = total_loss.item()
        return total_loss, losses_dict

    def _compute_dkd_components(self, student_logits, teacher_logits, targets):
        """Calculates TCKD and NCKD for logging purposes."""
        with torch.no_grad():
            gt_mask = self.model._get_gt_mask(student_logits, targets)
            other_mask = self.model._get_other_mask(student_logits, targets)
            pred_s = torch.softmax(student_logits / self.args.tau, dim=1)
            pred_t = torch.softmax(teacher_logits / self.args.tau, dim=1)

            # TCKD
            p_s = self.model._cat_mask(pred_s, gt_mask, other_mask)
            p_t = self.model._cat_mask(pred_t, gt_mask, other_mask)
            tckd = nn.functional.kl_div(torch.log(p_s), p_t, reduction="sum") * (self.args.tau**2) / targets.shape[0]

            # NCKD
            log_p_s_nckd = torch.log_softmax(student_logits / self.args.tau - 1000.0 * gt_mask, dim=1)
            p_t_nckd = torch.softmax(teacher_logits / self.args.tau - 1000.0 * gt_mask, dim=1)
            nckd = nn.functional.kl_div(log_p_s_nckd, p_t_nckd, reduction="sum") * (self.args.tau**2) / targets.shape[0]

        return tckd.item(), nckd.item()

    def _init_meters(self, adversarial=False):
        meters = {k: AverageMeter() for k in ["total", "ce", "kd", "accuracy"]}
        if adversarial:
            # Added 'dkd' to ensure the meter exists for DisDKD logging
            extra_keys = ["discriminator", "adversarial", "disc_accuracy", "fool_rate"]
            if self.args.method == "DisDKD":
                extra_keys.append("dkd")
            meters.update({k: AverageMeter() for k in extra_keys})
        else:
            m_map = {"FitNet": ["hint"], "CRD": ["contrastive"], "DKD": ["tckd", "nckd"]}
            for m in m_map.get(self.args.method, []): 
                meters[m] = AverageMeter()
        return meters

    def _update_meters(self, meters, losses_dict, student_logits, targets, batch_size):
        acc1 = accuracy(student_logits, targets, topk=(1,))[0]
        for key, value in losses_dict.items():
            if key in meters: meters[key].update(value, batch_size)
        meters["accuracy"].update(acc1.item(), batch_size)

    def _update_adversarial_meters(self, meters, disc_res, stud_res, ce, kd, total, logits, targets, b_size):
        acc1 = accuracy(logits, targets, topk=(1,))[0]
        
        # Helper to safely get scalar
        def get_val(v): return v.item() if isinstance(v, torch.Tensor) else v

        meters["total"].update(get_val(total), b_size)
        meters["ce"].update(get_val(ce), b_size)
        meters["kd"].update(get_val(kd), b_size)
        meters["accuracy"].update(acc1.item(), b_size)
        
        meters["discriminator"].update(disc_res.get("discriminator_loss", 0), b_size)
        meters["disc_accuracy"].update(disc_res.get("discriminator_accuracy", 0), b_size)
        meters["adversarial"].update(stud_res.get("adversarial_loss", 0), b_size)
        meters["fool_rate"].update(stud_res.get("fool_rate", 0), b_size)

        if self.args.method == "DisDKD" and "dkd" in meters:
            # Pulling the logit matching loss specifically
            dkd_val = stud_res.get("method_specific_loss", 0)
            meters["dkd"].update(get_val(dkd_val), b_size)

    def _update_progress_bar(self, progress_bar, meters):
        postfix = {"loss": f'{meters["total"].avg:.4f}', "acc": f'{meters["accuracy"].avg:.2f}%'}
        if self.args.method == "DKD" and "tckd" in meters:
            postfix.update({"tckd": f'{meters["tckd"].avg:.4f}', "nckd": f'{meters["nckd"].avg:.4f}'})
        progress_bar.set_postfix(postfix)

    def _update_adversarial_progress_bar(self, progress_bar, meters):
        postfix = {"loss": f'{meters["total"].avg:.4f}', "disc_acc": f'{meters["disc_accuracy"].avg:.2%}', "fool": f'{meters["fool_rate"].avg:.2%}'}
        if "dkd" in meters: postfix["dkd"] = f'{meters["dkd"].avg:.4f}'
        progress_bar.set_postfix(postfix)

    def _get_average_losses(self, meters):
        return {k: v.avg for k, v in meters.items() if k != "accuracy"}