import torch.nn as nn
from pretraining import Pretraining
from distill.DisDKD import DisDKD
from distill.fitnet import FitNet
from distill.dkd import DKD
from distill.logits_kd import LogitKD
from distill.crd import CRD
from utils.utils import count_params


RESNET_CHANNELS = {
    'resnet18': {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
    'resnet34': {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512},
    'resnet50': {'conv1': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
}


def get_layer_channels(model_name: str, layer_name: str) -> int:
    """Get channel count for a layer."""
    layer_key = layer_name.split('.')[0] if layer_name else None
    return RESNET_CHANNELS.get(model_name, {}).get(layer_key)


def create_distillation_model(args, teacher, student, num_classes: int):
    """Create the appropriate distillation model."""
    teacher_channels = get_layer_channels(args.teacher, args.teacher_layer)
    student_channels = get_layer_channels(args.student, args.student_layer)

    models = {
        "Pretraining": lambda: Pretraining(teacher),
        "LogitKD": lambda: LogitKD(teacher, student),
        "DKD": lambda: DKD(teacher, student, args.dkd_alpha, args.dkd_beta, args.tau),
        "DisDKD": lambda: DisDKD(
            teacher,
            student,
            args.teacher_layer,
            args.student_layer,
            teacher_channels,
            student_channels,
            args.hidden_channels,
            args.dkd_alpha,
            args.dkd_beta,
            args.tau,
        ),
        "FitNet": lambda: FitNet(
            teacher,
            student,
            args.teacher_layer,
            args.student_layer,
            teacher_channels,
            student_channels,
            args.adapter,
        ),
        "CRD": lambda: CRD(
            teacher,
            student,
            args.teacher_layer,
            args.student_layer,
            teacher_channels,
            student_channels,
            len(args._train_dataset),
            feat_dim=args.feat_dim,
            temperature=args.crd_temperature,
            momentum=args.crd_momentum,
            n_data==args.crd_n_negatives,
        ),
    }

    return models[args.method]()


def print_model_parameters(model, method_name: str):
    """Print parameter counts for model components."""
    print(f"\n=== {method_name} Model Parameters ===")
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if method_name == 'Pretraining':
        print(f"Teacher Model: {count_params(model.teacher):,} parameters")
    else:
        if hasattr(model, 'student'):
            print(f"Student: {count_params(model.student):,} parameters")
        if hasattr(model, 'teacher'):
            print(f"Teacher: {count_params(model.teacher):,} parameters (frozen)")
    
    # Method-specific components
    method_params = {
        'FitNet': [('adapter', 'hint_criterion.adaptation', 'module')],
        'CRD': [('teacher_projector', 'teacher_projector', 'module'), 
                ('student_projector', 'student_projector', 'module')],
        'DisDKD': [('teacher_regressor', 'teacher_regressor', 'module'),
                   ('student_regressor', 'student_regressor', 'module'),
                   ('discriminator', 'discriminator', 'module')],
    }
    
    if method_name in method_params:
        for item in method_params[method_name]:
            if len(item) == 3:
                name, attr_path, obj_type = item
            else:
                name, attr_path = item
                obj_type = 'module'
            
            obj = model
            for attr in attr_path.split('.'):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            
            if obj is not None:
                if obj_type == 'parameter':
                    # For nn.Parameter objects
                    if isinstance(obj, nn.Parameter):
                        print(f"{name.capitalize()}: {obj.numel():,} parameters")
                    else:
                        print(f"{name.capitalize()}: Not a Parameter object")
                else:
                    # For nn.Module objects
                    print(f"{name.capitalize()}: {count_params(obj):,} parameters")
    
    print(f"Total trainable: {total_trainable:,} parameters")
    print("=" * 50)
