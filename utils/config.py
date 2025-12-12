import argparse
from utils.data import get_available_domains, validate_domain_config, DOMAINBED_DATASETS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    
    # Model configuration
    parser.add_argument('--seed', type=int, default=42, help='Seed param')
    parser.add_argument('--teacher', type=str, default='resnet50', help='Teacher architecture')
    parser.add_argument('--student', type=str, default='resnet18', help='Student architecture')
    parser.add_argument('--teacher_weights', type=str, help='Teacher pretrained weights path')
    parser.add_argument('--student_weights', type=str, help='Student pretrained weights path')
    parser.add_argument('--pretrained', action='store_true', 
                       help='Use ImageNet pretrained weights for teacher model')
    
    # Dataset configuration
    all_datasets = ['CIFAR100', 'CIFAR10', 'IMAGENETTE', 'FOOD101', 'PACS'] + list(DOMAINBED_DATASETS.keys())
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=all_datasets)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--jitter', action='store_true', help='Apply color jitter')
    
    # Domain-based dataset arguments
    parser.add_argument('--train_domains', type=str, nargs='+',
                       help='Domains for training')
    parser.add_argument('--val_domains', type=str, nargs='+',
                       help='Domains for validation')
    parser.add_argument('--classic_split', action='store_true',
                       help='Use classic ML setup: train/val split from same domains')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation fraction in classic split')
    parser.add_argument('--test_env', type=int, default=None, 
                       help='[DomainBed only] Test domain ID')
    
    # Method configuration
    parser.add_argument('--method', type=str, default='LogitKD',
                       choices=['Pretraining', 'LogitKD', 'DKD', 'DisDKD','FitNet','CRD'])
    parser.add_argument('--teacher_layer', type=str, default='layer3')
    parser.add_argument('--student_layer', type=str, default='layer2')
    parser.add_argument('--adapter', type=str, default='student', choices=['student', 'teacher'])
    parser.add_argument('--feat_dim', type=int, default=128, 
                       help='Feature dimension for projection')
    parser.add_argument('--hidden_channels', type=int, default=128, 
                       help='Hidden channels for feature transformation')
    
    # Method-specific hyperparameters
    parser.add_argument('--disdkd_adversarial_weight', type=float, default=0.01,
                       help='DisDKD adversarial weight (gamma)')
    parser.add_argument('--disc_lr_multiplier', type=float, default=1.0, 
                       help='Discriminator LR multiplier')
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, 
                       help='Discriminator learning rate')
    parser.add_argument('--dkd_alpha', type=float, default=1.0, help='DKD TCKD weight')
    parser.add_argument('--dkd_beta', type=float, default=8.0, help='DKD NCKD weight')
    parser.add_argument('--fitnet_stage1_epochs', type=int, default=0, 
                       help='Number of epochs for FitNet Stage 1 (Hint only)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--step_size', type=int, default=5, help='LR scheduler step size')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='LR decay factor after every step-size')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0, help='CE loss weight')
    parser.add_argument('--beta', type=float, default=0.4, help='KD loss weight')
    parser.add_argument('--gamma', type=float, default=1.0, help='Method-specific loss weight')
    parser.add_argument('--tau', type=float, default=4.0, help='Temperature for KD')
    
    # Logging and output
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_file', type=str, default='log.csv')
    
    return parser.parse_args()


def validate_and_setup_domains(args):
    """Validate and setup domain configuration for domain-based datasets."""
    dataset_name = args.dataset.upper()
    
    if dataset_name not in DOMAINBED_DATASETS and dataset_name not in ['PACS', 'PACS_DEEPLAKE']:
        if args.train_domains or args.val_domains or args.classic_split:
            print(f"Warning: Domain arguments ignored for non-domain-based dataset {args.dataset}")
        args.train_domains = None
        args.val_domains = None
        args.classic_split = False
        return
    
    available_domains = get_available_domains(args.dataset)
    
    if args.train_domains is None:
        if args.classic_split:
            args.train_domains = available_domains
            print(f"Classic split: using all domains {args.train_domains}")
        else:
            args.train_domains = available_domains[:-1]
            if args.val_domains is None:
                args.val_domains = [available_domains[-1]]
            print(f"OOD split - Training: {args.train_domains}, Validation: {args.val_domains}")
    
    if args.classic_split:
        if args.val_domains is not None:
            print("Warning: --val_domains ignored in classic split mode.")
        args.val_domains = None
        print(f"Classic ML setup with domains: {args.train_domains}")
        print(f"Train/val split: {int((1-args.test_size)*100)}%/{int(args.test_size*100)}%")
    else:
        if args.val_domains is None:
            args.val_domains = [d for d in available_domains if d not in args.train_domains]
            if not args.val_domains:
                raise ValueError(f"No available validation domains.")
            print(f"Using remaining domains for validation: {args.val_domains}")
    
    validate_domain_config(args.dataset, args.train_domains, args.val_domains, args.classic_split)


def print_training_config(args):
    """Print complete training configuration."""
    print(f"\n=== Training Configuration ===")
    print(f"Method: {args.method}")
    print(f"Teacher: {args.teacher} -> Student: {args.student}")
    print(f"Dataset: {args.dataset}")
    
    dataset_name = args.dataset.upper()
    if dataset_name in DOMAINBED_DATASETS or dataset_name in ['PACS', 'PACS_DEEPLAKE']:
        if args.classic_split:
            print(f"Setup: Classic ML (train/val split from same domains)")
            print(f"Domains: {args.train_domains}")
            print(f"Train/Val split: {int((1-args.test_size)*100)}%/{int(args.test_size*100)}%")
        else:
            print(f"Setup: Out-of-Distribution evaluation")
            print(f"Training domains: {args.train_domains}")
            print(f"Validation domains: {args.val_domains}")
    
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    print(f"Loss weights - α: {args.alpha}, β: {args.beta}, γ: {args.gamma}")
    
    if args.method == 'DKD':
        print(f"DKD weights - TCKD α: {args.dkd_alpha}, NCKD β: {args.dkd_beta}")
    elif args.method == 'DisDKD':
        print(f"DisDKD DKD weights - TCKD α: {args.dkd_alpha}, NCKD β: {args.dkd_beta}")
        print(f"DisDKD Adversarial weight (γ): {args.disdkd_adversarial_weight}")
        print(f"Discriminator LR: {args.discriminator_lr}, multiplier: {args.disc_lr_multiplier}")
        
    print(f"Temperature: {args.tau}")
    print("=" * 40)