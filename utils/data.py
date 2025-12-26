import os
import torch
import numpy as np
import tarfile
import requests
import torchvision
import torchvision.transforms as transforms
import deeplake
import random
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

g = torch.Generator()
g.manual_seed(42)

# DATA CARD
DOMAINBED_DATASETS = {
    'PACS': { 'name': 'PACS',
        'domains': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'num_classes': 7},
    'VLCS': { 'name': 'VLCS',
            'domains': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
            'num_classes': 5,},
    'OfficeHome': { 'name': 'OfficeHome',
            'domains': ['Art', 'Clipart', 'Product', 'RealWorld'],
            'num_classes': 65}
}

class JitterTransform:
    """Color jitter transform."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, img):
        return self.jitter(img)


def get_transforms(jitter=False, dataset_name=None):
    """Get data transforms."""
    # For domain/dataset style datasets (PACS, VLCS, OfficeHome) we disable jitter by default
    if dataset_name and dataset_name.upper() in ['PACS_DEEPLAKE', 'VLCS', 'OFFICEHOME']:
        jitter = False

    if jitter:
        pipeline = [
            transforms.Resize((224, 224)),
            JitterTransform(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ]
    else:
        pipeline = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ]
    return transforms.Compose(pipeline)


class PACSDataset(Dataset):
    """Dataset wrapper for PACS from DeepLake."""
    
    def __init__(self, deeplake_ds, transform=None):
        self.ds = deeplake_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        sample = self.ds[idx]
        
        # Handle different ways the image might be stored
        if hasattr(sample, 'images'):
            if hasattr(sample.images, 'pil'):
                image = sample.images.pil()
            elif hasattr(sample.images, 'numpy'):
                # Convert numpy array to PIL
                img_array = sample.images.numpy()
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                image = Image.fromarray(img_array)
            else:
                # Direct tensor access
                img_tensor = sample.images
                if isinstance(img_tensor, torch.Tensor):
                    if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:
                        img_array = img_tensor.permute(1, 2, 0).numpy()
                    else:
                        img_array = img_tensor.numpy()
                    
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
                else:
                    img_array = np.array(img_tensor)
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
        else:
            # Try other common field names
            for field_name in ['image', 'img', 'data']:
                if hasattr(sample, field_name):
                    image_data = getattr(sample, field_name)
                    if hasattr(image_data, 'pil'):
                        image = image_data.pil()
                    else:
                        img_array = np.array(image_data)
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        image = Image.fromarray(img_array)
                    break
            else:
                raise ValueError(f"Could not find image data in sample. Available keys: {list(sample.keys())}")
        
        # Handle labels
        if hasattr(sample, 'labels'):
            if hasattr(sample.labels, 'numpy'):
                label = sample.labels.numpy().item()
            else:
                label = int(sample.labels)
        elif hasattr(sample, 'label'):
            if hasattr(sample.label, 'numpy'):
                label = sample.label.numpy().item()
            else:
                label = int(sample.label)
        else:
            for field_name in ['target', 'class', 'y']:
                if hasattr(sample, field_name):
                    label_data = getattr(sample, field_name)
                    if hasattr(label_data, 'numpy'):
                        label = label_data.numpy().item()
                    else:
                        label = int(label_data)
                    break
            else:
                raise ValueError(f"Could not find label data in sample. Available keys: {list(sample.keys())}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_pacs_deeplake_dataset(train_domains: list, val_domains: list = None, 
                             classic_split: bool = False, test_size: float = 0.2,
                             jitter: bool = False):
    """
    Load PACS dataset using DeepLake with flexible domain configuration.
    """
    available_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    num_classes = 7
    
    # Validate domains
    for domain in train_domains:
        if domain not in available_domains:
            raise ValueError(f"Invalid training domain '{domain}' for PACS. Available domains: {available_domains}")
    if val_domains:
        for domain in val_domains:
            if domain not in available_domains:
                raise ValueError(f"Invalid validation domain '{domain}' for PACS. Available domains: {available_domains}")
    
    train_transform = get_transforms(jitter=False, dataset_name='PACS_DEEPLAKE')
    test_transform = get_transforms(jitter=False, dataset_name='PACS_DEEPLAKE')
    
    train_ds = deeplake.load("hub://activeloop/pacs-train")
    test_ds = deeplake.load("hub://activeloop/pacs-val")
    
    # Domain text fields
    train_domains_data = list(np.array(train_ds.domains.data()['text']).squeeze())
    test_domains_data = list(np.array(test_ds.domains.data()['text']).squeeze())
    
    if classic_split:
        # Classic ML: combine specified train domains and split into train/val
        print(f"Using classic ML setup for PACS: train/val split from domains {train_domains}")
        all_datasets = []
        train_indices = [i for i, domain in enumerate(train_domains_data) if domain in train_domains]
        if train_indices:
            train_subset = train_ds[train_indices]
            all_datasets.append(PACSDataset(train_subset, train_transform))
        test_indices = [i for i, domain in enumerate(test_domains_data) if domain in train_domains]
        if test_indices:
            test_subset = test_ds[test_indices]
            all_datasets.append(PACSDataset(test_subset, train_transform))
        combined_dataset = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
        
        total_size = len(combined_dataset)
        indices = list(range(total_size))
        all_labels = [combined_dataset[i][1] for i in range(total_size)]
        
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, stratify=all_labels, random_state=42
        )
        train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
        
        # Build test/val combined dataset similarly and subset for val
        all_datasets_test = []
        if train_indices:
            train_subset = train_ds[[i for i, domain in enumerate(train_domains_data) if domain in train_domains]]
            all_datasets_test.append(PACSDataset(train_subset, test_transform))
        if test_indices:
            test_subset = test_ds[[i for i, domain in enumerate(test_domains_data) if domain in train_domains]]
            all_datasets_test.append(PACSDataset(test_subset, test_transform))
        combined_dataset_test = ConcatDataset(all_datasets_test) if len(all_datasets_test) > 1 else all_datasets_test[0]
        val_dataset = torch.utils.data.Subset(combined_dataset_test, val_indices)
        
    else:
        # OOD setup: use distinct domains for train/val
        if val_domains is None:
            val_domains = [d for d in available_domains if d not in train_domains]
            if not val_domains:
                raise ValueError(f"No available validation domains. All domains are in train_domains: {train_domains}")
            print(f"Using remaining domains for validation: {val_domains}")
        
        overlap = set(train_domains) & set(val_domains)
        if overlap:
            print(f"Warning: Domains {overlap} appear in both training and validation.")
        
        train_datasets = []
        train_indices = [i for i, domain in enumerate(train_domains_data) if domain in train_domains]
        if train_indices:
            train_subset = train_ds[train_indices]
            train_datasets.append(PACSDataset(train_subset, train_transform))
        test_indices = [i for i, domain in enumerate(test_domains_data) if domain in train_domains]
        if test_indices:
            test_subset = test_ds[test_indices]
            train_datasets.append(PACSDataset(test_subset, train_transform))
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        
        val_datasets = []
        val_train_indices = [i for i, domain in enumerate(train_domains_data) if domain in val_domains]
        if val_train_indices:
            val_train_subset = train_ds[val_train_indices]
            val_datasets.append(PACSDataset(val_train_subset, test_transform))
        val_test_indices = [i for i, domain in enumerate(test_domains_data) if domain in val_domains]
        if val_test_indices:
            val_test_subset = test_ds[val_test_indices]
            val_datasets.append(PACSDataset(val_test_subset, test_transform))
        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    return train_dataset, val_dataset, num_classes


class ImagePathDataset(Dataset):
    """
    Simple dataset that takes lists of image file paths and labels and loads them using PIL.
    """
    def __init__(self, paths, labels, transform=None):
        assert len(paths) == len(labels)
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


def _find_local_dataset_root(root, candidates):
    """
    If root doesn't contain dataset, try candidate paths (for typical Kaggle / Colab layouts).
    candidates is a list of relative/absolute paths to try.
    """
    # Check root first
    if root and os.path.exists(root):
        return root
    for c in candidates:
        if os.path.exists(c):
            return c
    return root  # fallback


def get_vlcs_dataset(root="./data", train_domains=None, val_domains=None,
                     classic_split=False, test_size=0.2, jitter=False):
    """
    Load VLCS dataset from local folder structure.
    Expected structure:
      <root>/VLCS/Caltech101/...
      <root>/VLCS/LabelMe/...
      <root>/VLCS/SUN09/...
      <root>/VLCS/VOC2007/...
    If not found at root/VLCS, tries '../input/vlcs-dataset/VLCS' automatically.
    """
    dataset_name = 'VLCS'
    available_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    num_classes = 5  # as in original mapping
    
    root = _find_local_dataset_root(root, ['../input/vlcs-dataset/VLCS'])
    vlcs_root = os.path.join(root, 'VLCS') if os.path.basename(root) != 'VLCS' else root
    if not os.path.exists(vlcs_root):
        raise FileNotFoundError(f"VLCS root not found at {vlcs_root}. Place VLCS folder there with domain subfolders.")
    
    train_transform = get_transforms(jitter=False, dataset_name=dataset_name)
    test_transform = get_transforms(jitter=False, dataset_name=dataset_name)
    
    # Default domains
    if train_domains is None:
        train_domains = available_domains.copy()
    
    # Validate
    for d in train_domains:
        if d not in available_domains:
            raise ValueError(f"Invalid VLCS domain: {d}. Available: {available_domains}")
    if val_domains:
        for d in val_domains:
            if d not in available_domains:
                raise ValueError(f"Invalid VLCS val domain: {d}. Available: {available_domains}")
    
    if classic_split:
        # Combine specified domains and perform stratified split
        all_paths = []
        all_labels = []
        label_map = {}  # map class folder name to label index
        next_label = 0
        for domain in train_domains:
            domain_dir = os.path.join(vlcs_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            # Walk class subfolders
            for cls in sorted(os.listdir(domain_dir)):
                cls_dir = os.path.join(domain_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                if cls not in label_map:
                    label_map[cls] = next_label
                    next_label += 1
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        all_paths.append(os.path.join(cls_dir, fname))
                        all_labels.append(label_map[cls])
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No images found under {vlcs_root} for domains {train_domains}")
        
        indices = list(range(len(all_paths)))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, stratify=all_labels, random_state=42)
        
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        train_dataset = ImagePathDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = ImagePathDataset(val_paths, val_labels, transform=test_transform)
        
    else:
        # OOD: train_domains and val_domains are separate
        if val_domains is None:
            val_domains = [d for d in available_domains if d not in train_domains]
            if not val_domains:
                raise ValueError("No available VLCS validation domains; all domains are in train_domains.")
        # Build datasets by domain (ImageFolder per domain)
        train_datasets = []
        for domain in train_domains:
            domain_dir = os.path.join(vlcs_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            train_datasets.append(torchvision.datasets.ImageFolder(domain_dir, transform=train_transform))
        val_datasets = []
        for domain in val_domains:
            domain_dir = os.path.join(vlcs_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            val_datasets.append(torchvision.datasets.ImageFolder(domain_dir, transform=test_transform))
        if len(train_datasets) == 0 or len(val_datasets) == 0:
            raise FileNotFoundError(f"VLCS domains missing. Check path {vlcs_root}")
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    return train_dataset, val_dataset, num_classes


def get_officehome_dataset(root="./data", train_domains=None, val_domains=None,
                          classic_split=False, test_size=0.2, jitter=False):
    """
    Load OfficeHome dataset from local folder structure.
    Expected structure:
      <root>/OfficeHomeDataset_10072016/Art/...
      <root>/OfficeHomeDataset_10072016/Clipart/...
      <root>/OfficeHomeDataset_10072016/Product/...
      <root>/OfficeHomeDataset_10072016/Real World/...
    If not found at root, tries '../input/officehome/datasets/OfficeHomeDataset_10072016'.
    """
    dataset_name = 'OfficeHome'
    available_domains = ['Art', 'Clipart', 'Product', 'Real World']
    num_classes = 65  # as per your mapping
    
    root = _find_local_dataset_root(root, ['../input/officehome/datasets/OfficeHomeDataset_10072016'])
    # If the base root points directly to the OfficeHome folder
    if os.path.basename(root) == 'OfficeHomeDataset_10072016':
        oh_root = root
    else:
        oh_root = os.path.join(root, 'OfficeHomeDataset_10072016')
    if not os.path.exists(oh_root):
        raise FileNotFoundError(f"OfficeHome root not found at {oh_root}. Place OfficeHomeDataset_10072016 folder there.")
    
    train_transform = get_transforms(jitter=False, dataset_name=dataset_name)
    test_transform = get_transforms(jitter=False, dataset_name=dataset_name)
    
    if train_domains is None:
        train_domains = available_domains.copy()
    for d in train_domains:
        if d not in available_domains:
            raise ValueError(f"Invalid OfficeHome domain: {d}. Available: {available_domains}")
    if val_domains:
        for d in val_domains:
            if d not in available_domains:
                raise ValueError(f"Invalid OfficeHome val domain: {d}. Available: {available_domains}")
    
    if classic_split:
        # Combine specified domains and perform stratified split
        all_paths = []
        all_labels = []
        label_map = {}
        next_label = 0
        for domain in train_domains:
            domain_dir = os.path.join(oh_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            for cls in sorted(os.listdir(domain_dir)):
                cls_dir = os.path.join(domain_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                if cls not in label_map:
                    label_map[cls] = next_label
                    next_label += 1
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        all_paths.append(os.path.join(cls_dir, fname))
                        all_labels.append(label_map[cls])
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No images found under {oh_root} for domains {train_domains}")
        
        indices = list(range(len(all_paths)))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, stratify=all_labels, random_state=42)
        
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths = [all_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        train_dataset = ImagePathDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = ImagePathDataset(val_paths, val_labels, transform=test_transform)
    else:
        if val_domains is None:
            val_domains = [d for d in available_domains if d not in train_domains]
            if not val_domains:
                raise ValueError("No available OfficeHome validation domains; all domains are in train_domains.")
        train_datasets = []
        for domain in train_domains:
            domain_dir = os.path.join(oh_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            train_datasets.append(torchvision.datasets.ImageFolder(domain_dir, transform=train_transform))
        val_datasets = []
        for domain in val_domains:
            domain_dir = os.path.join(oh_root, domain)
            if not os.path.isdir(domain_dir):
                continue
            val_datasets.append(torchvision.datasets.ImageFolder(domain_dir, transform=test_transform))
        if len(train_datasets) == 0 or len(val_datasets) == 0:
            raise FileNotFoundError(f"OfficeHome domains missing. Check path {oh_root}")
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    return train_dataset, val_dataset, num_classes


def get_dataset(name="CIFAR100", root="./data", download=True, jitter=False, 
                train_domains=None, val_domains=None, classic_split=False, test_size=0.2):
    """
    Load dataset.
    """
    name = name.upper()
    
    # Force PACS to use DeepLake version
    if name == 'PACS':
        name = 'PACS_DEEPLAKE'
        print("Note: Using PACS from DeepLake instead of local DomainBed installation")
    
    # PACS DeepLake
    if name == 'PACS_DEEPLAKE':
        if train_domains is None:
            train_domains = ['art_painting', 'cartoon', 'photo']
        return get_pacs_deeplake_dataset(train_domains, val_domains, classic_split, test_size, jitter)
    
    # VLCS
    if name == 'VLCS':
        if train_domains is None:
            train_domains = ['Caltech101', 'LabelMe', 'SUN09']  # default (you can override)
        return get_vlcs_dataset(root=root, train_domains=train_domains, val_domains=val_domains,
                                classic_split=classic_split, test_size=test_size, jitter=jitter)
    
    # OfficeHome
    if name == 'OFFICEHOME' or name == 'OFFICE_HOME':
        if train_domains is None:
            train_domains = ['Art', 'Clipart', 'Product']  # default (you can override)
        return get_officehome_dataset(root=root, train_domains=train_domains, val_domains=val_domains,
                                      classic_split=classic_split, test_size=test_size, jitter=jitter)
    
    # Handle regular datasets (unchanged)
    train_transform = get_transforms(jitter=jitter)
    test_transform = get_transforms(jitter=False)

    if name == "CIFAR100":
        train = torchvision.datasets.CIFAR100(
            root, train=True, download=download, transform=train_transform
        )
        test = torchvision.datasets.CIFAR100(
            root, train=False, download=download, transform=test_transform
        )
        num_classes = 100
    elif name == "CIFAR10":
        train = torchvision.datasets.CIFAR10(
            root, train=True, download=download, transform=train_transform
        )
        test = torchvision.datasets.CIFAR10(
            root, train=False, download=download, transform=test_transform
        )
        num_classes = 10
    elif name == "IMAGENETTE":
        train_dir, val_dir = download_imagenette()  # function kept exactly as before
        train = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)
        num_classes = 10
    elif name == "FOOD101":
        train = torchvision.datasets.Food101(
            root=root, split="train", download=download, transform=train_transform
        )
        test = torchvision.datasets.Food101(
            root=root, split="test", download=download, transform=test_transform
        )
        num_classes = 101
    else:
        available_datasets = ["CIFAR100", "CIFAR10", "IMAGENETTE", "FOOD101", "PACS"] + ['VLCS', 'OFFICEHOME']
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available_datasets}")

    return train, test, num_classes


def get_dataloaders(dataset="CIFAR100", batch_size=64, root="./data", jitter=False, 
                   num_workers=4, train_domains=None, val_domains=None, 
                   classic_split=False, test_size=0.2):
    """
    Get data loaders.
    """
    train_set, test_set, num_classes = get_dataset(
        dataset, root, True, jitter, train_domains, val_domains, classic_split, test_size)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
    )

    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader, num_classes


class CRDTrainWrapper(Dataset):
    """Wraps ONLY the train dataset to also return a stable sample index for CRD."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i  # IMPORTANT: index in train dataset order


def seed_worker(worker_id: int):
    # Fully correct deterministic seeding for dataloader workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_crd_dataloaders(
    dataset,
    batch_size,
    data_root,
    jitter,
    num_workers,
    train_domains=None,
    val_domains=None,
    classic_split=False,
    test_size=0.2,
    seed=0,
):
    """
    Returns:
      train_loader: yields (x, y, idx)  <-- ONLY here
      val_loader: yields (x, y)
      num_classes
    """
    # 1) Build your underlying train_set / test_set the SAME way as get_dataloaders does
    train_set, val_set, num_classes = get_dataset(name=dataset,root=data_root,
    download=True,
    jitter=jitter,
    train_domains=train_domains,
    val_domains=val_domains,
    classic_split=classic_split,
    test_size=test_size,
    )


    # 2) Wrap ONLY train_set
    train_set = CRDTrainWrapper(train_set)

    # 3) Deterministic generator for shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # strongly recommended for CRD stability
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, num_classes


def print_dataset_info(dataset_name: str, train_domains: list = None, val_domains: list = None, 
                      classic_split: bool = False):
    """Print information about the dataset configuration."""
    dataset_name = dataset_name.upper()
    if dataset_name in ['VLCS', 'OFFICEHOME', 'PACS', 'PACS_DEEPLAKE']:
        if dataset_name == 'VLCS':
            info = {'num_classes': 5, 'domains': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']}
        elif dataset_name in ['PACS', 'PACS_DEEPLAKE']:
            info = {'num_classes': 7, 'domains': ['art_painting', 'cartoon', 'photo', 'sketch']}
        else:  # OfficeHome
            info = {'num_classes': 65, 'domains': ['Art', 'Clipart', 'Product', 'Real World']}
        
        print(f"\n=== {dataset_name} Dataset Configuration ===")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Available domains: {info['domains']}")
        
        if classic_split:
            print(f"Setup: Classic ML (train/val split from same domains)")
            print(f"Domains used: {train_domains}")
        else:
            print(f"Setup: Out-of-Distribution evaluation")
            print(f"Training domains: {train_domains}")
            print(f"Validation domains: {val_domains}")
        
        print(f"Note: Color jitter is disabled for domain-based datasets")
        print("=" * 50)
    else:
        print(f"\n=== {dataset_name} Dataset Configuration ===")
        print("Standard torchvision dataset")
        print("=" * 50)


def get_available_domains(dataset_name: str):
    """Get available domains for domain-based datasets."""
    dataset_name = dataset_name.upper()
    if dataset_name == 'VLCS':
        return ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset_name in ['PACS', 'PACS_DEEPLAKE']:
        return ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset_name == 'OFFICEHOME' or dataset_name == 'OFFICE_HOME':
        return ['Art', 'Clipart', 'Product', 'Real World']
    return None


def validate_domain_config(dataset_name: str, train_domains: list, val_domains: list = None, 
                          classic_split: bool = False):
    """Validate domain configuration for PACS, VLCS, OfficeHome."""
    dataset_name = dataset_name.upper()
    if dataset_name not in ['VLCS', 'PACS', 'PACS_DEEPLAKE', 'OFFICEHOME', 'OFFICE_HOME']:
        return  # Not a domain-based dataset
    
    available_domains = get_available_domains(dataset_name)
    if available_domains is None:
        return
    
    # Check training domains
    invalid_domains = [d for d in train_domains] if train_domains and any(d not in available_domains for d in train_domains) else []
    if invalid_domains:
        raise ValueError(f"Invalid training domains for {dataset_name}: {invalid_domains}. Available domains: {available_domains}")
    
    if val_domains:
        invalid_val_domains = [d for d in val_domains if d not in available_domains]
        if invalid_val_domains:
            raise ValueError(f"Invalid validation domains for {dataset_name}: {invalid_val_domains}. Available domains: {available_domains}")
        if not classic_split:
            overlap = set(train_domains) & set(val_domains)
            if overlap:
                print(f"Warning: Domains {overlap} appear in both training and validation.")
    
    # Inform about unused domains
    all_specified = set(train_domains or [])
    if val_domains:
        all_specified.update(val_domains)
    unused = [d for d in available_domains if d not in all_specified]
    if unused:
        print(f"Info: Unused domains for {dataset_name}: {unused}")


# Keep the imagenette downloader function as in your original file
def download_imagenette(dest_tgz="imagenette2-160.tgz", extract_dir="imagenette"):
    """Download and extract Imagenette dataset."""
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    if not os.path.exists(dest_tgz):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(dest_tgz, 'wb') as file, tqdm(
            desc=dest_tgz,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    if not os.path.exists(extract_dir):
        with tarfile.open(dest_tgz, "r:gz") as tar:
            tar.extractall(extract_dir)
    return os.path.join(extract_dir, "imagenette2-160", "train"), os.path.join(extract_dir, "imagenette2-160", "val")
