# WH-DisDKD
# Knowledge Distillation Toolbox

Compact, easy-to-follow training repo for experimenting with multiple knowledge-distillation (KD) methods — logit-based, feature-based, contrastive, and a discriminator-guided variant.

---

## Supported methods
The code exposes the `--method` flag. Available choices:
- `Pretraining` — train student from data only (no KD).
- `LogitKD` — vanilla logit distillation (Hinton et al.).  
  https://arxiv.org/abs/1503.02531
- `DKD` — Decoupled Knowledge Distillation (Zhao et al.).  
  https://arxiv.org/abs/2203.08679
- `DisDKD` — our Discriminator-guided feature alignment + DKD at logits (custom).
- `FitNet` — FitNets feature distillation (Romero et al.).  
  https://arxiv.org/abs/1412.6550
- `CRD` — Contrastive Representation Distillation (Tian et al.).  
  https://arxiv.org/abs/1910.10699

> Note: the flag string must match exactly one of the choices above (see `config.py`).

---

## Quick start
1. Running LogitKD method:
```bash
python train.py \
  --method LogitKD \
  --teacher resnet50 \
  --student resnet18 \
  --dataset CIFAR100 \
  --batch_size 128 \
  --epochs 100 \
  --lr 0.01 \
  --tau 4.0 \
  --alpha 1.0 --beta 0.4 \
  --save_dir ./checkpoints/logitkd
  ```
  2. Running DisDKD (our custom method)
  ```bash
  python train.py \
  --method DisDKD \
  --teacher resnet50 \
  --student resnet18 \
  --dataset IMAGENETTE \
  --epochs 60 \
  --disdkd_adversarial_weight 0.01 \
  --discriminator_lr 1e-4 \
  --disc_lr_multiplier 1.0 \
  --dkd_alpha 1.0 --dkd_beta 8.0 \
  --save_dir ./checkpoints/disdkd
  ```
  3. Our script also supports OOD tests using domainbed. Simply specify the train domains (i.e., ones visible to the student during distillation) and the val-domains (these are only visible to the student during evaluation):
  ```bash
  python train.py \
  --dataset PACS \
  --method FitNet \
  --train_domains art_photo \
  --val_domains sketch \
  --classic_split False
```
---
## Important Flags
* ```bash --teacher```, ```bash --student``` — model architectures (e.g., resnet50, resnet18).

* ```bash --teacher_weights```, ```bash --student_weights``` — optional pretrained weights paths. This **must** be passed for the teacher for distillation.

* ```bash --pretrained``` — use ImageNet pretrained teacher weights. Only useful during pretraining. **Ignore this for now**__.

* ```bash --dataset```, ```bash --data_root```, ```bash --batch_size```, ```bash --num_workers```

* ```bash --method``` — KD method (see list above).

* Feature alignment: ```bash --teacher_layer```, ```bash --student_layer```, ```bash --adapter``` (who contains adapter i.e. teacher or student), ```bash --feat_dim```, ```bash--hidden_channels```.

* DisDKD specifics: ```bash --disdkd_adversarial_weight```, ```bash --discriminator_lr```, ```bash --disc_lr_multiplier```.

* DKD specifics: ```bash --dkd_alpha```, ```bash --dkd_beta```.

* Loss weights & KD temp: ```bash --alpha``` (CE), ```bash --beta``` (KD), ```bash --gamma``` (method-specific), ```bash --tau``` (temperature).

* Training: ```bash --epochs```, ```bash --lr```, ```bash --optimizer```, ```bash --momentum```, ```bash --weight_decay```, ```bash --step_size```, ```bash --lr_decay```.

* Logging/output: ```bash --save_dir```, ```bash --log_file```, ```bash --print_freq```, ```bash --device```.

Use python train.py --help (or the script entrypoint you have) for the full list generated from config.py.
  


