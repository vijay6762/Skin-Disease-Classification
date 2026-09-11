import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from PIL import Image

IMG_SIZE = (300, 300)

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Convert to RGB to ensure 3 channels
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def compute_class_weights(labels, num_classes):
    """Computes balanced class weights"""
    labels = np.array(labels)
    total = len(labels)
    class_weights = []
    
    for cls in range(num_classes):
        count = np.sum(labels == cls)
        if count == 0:
            weight = 1.0
        else:
            weight = total / (num_classes * count)
        class_weights.append(weight)
        
    return torch.FloatTensor(class_weights)

def load_directory_data(data_dir=r'os.environ.get('DATA_DIR', './data')',
                        batch_size=32, img_size=IMG_SIZE, test_size=0.2):
    
    print(f"Scanning directory: {data_dir}")
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist!")

    class_names = sorted([
        c for c in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, c))
    ])
    
    samples = []
    targets = []
    valid_classes = []
    class_to_idx = {}
    
    current_idx = 0
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        # Find all valid images
        images = []
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            images.extend([os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(ext)])
            
        if len(images) > 0:
            valid_classes.append(cls)
            class_to_idx[cls] = current_idx
            for img_path in images:
                samples.append(img_path)
                targets.append(current_idx)
            current_idx += 1

    num_classes = len(valid_classes)
    
    # Save indices map for API
    with open('class_indices.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print(f"Found {len(samples)} images across {num_classes} classes.")

    # Stratified split
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        list(zip(samples, targets)), targets, test_size=test_size, stratify=targets, random_state=42
    )

    class_weights = compute_class_weights(train_labels, num_classes)
    print(f"Class weights computed. Max weight: {class_weights.max():.2f}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Transforms
    # PyTorch transforms operate on PIL images, then convert to FloatTensors [0, 1]
    # EfficientNet_B3_Weights.IMAGENET1K_V1 expects ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    train_transform = transforms.Compose([
         transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
         transforms.ToTensor(),
         normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    train_ds = CustomDataset(train_samples, transform=train_transform)
    val_ds = CustomDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, num_classes, class_weights

if __name__ == "__main__":
    train_loader, val_loader, nc, cw = load_directory_data()
    for batch_x, batch_y in train_loader:
        print("Batch X:", batch_x.shape)
        print("Batch Y:", batch_y.shape)
        break