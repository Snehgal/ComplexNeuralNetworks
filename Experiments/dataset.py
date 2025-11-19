# 2️⃣ PyTorch Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import random

class SASDataset(Dataset):
    def __init__(self, images, masks, input_size=(992, 992), output_size=(992, 992), augment=False):
        self.images = torch.from_numpy(images).float()  # [N, 2, H, W]
        self.masks = torch.from_numpy(masks).long()     # [N, H, W]
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment

        # Only geometric augmentations (safe for multi-channel inputs)
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # [2, H, W]
        mask = self.masks[idx]  # [H, W]

        # Resize
        img = F.interpolate(img.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.output_size, mode='nearest').squeeze(0).squeeze(0).long()

        # Apply augmentations (same transform to image and mask)
        if self.augment:
            stacked = torch.cat([img, mask.unsqueeze(0)], dim=0)  # [3, H, W]
            stacked = self.transforms(stacked)
            img, mask = stacked[:-1], stacked[-1].long()  # split back to 2 channels

        return img, mask
    


def Data_loader():
    import torch
    import numpy as np
    import h5py
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # ===============================
    # 1️⃣ Load SAS Dataset
    # ===============================
    file_path = "sassed_V4.h5"  # replace with your path
    with h5py.File(file_path, "r") as f:
        images = f["data"][:]       # complex64
        masks = f["segments"][:]    # uint8

    # Split real and imaginary channels
    images_2ch = np.stack([images.real, images.imag], axis=1)  # shape: (N, 2, H, W)
    print("2-channel images:", images_2ch.shape)
    print("Masks:", masks.shape)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    import random

    # Assuming `images_2ch` and `masks` are NumPy arrays
    num_samples = len(images_2ch)
    all_indices = np.arange(num_samples)

    # Fix random seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(all_indices)

    # 1️⃣ Split off 9 samples for testing
    test_indices = all_indices[:9]
    remaining_indices = all_indices[9:]

    # 2️⃣ Split remaining into 80/20 train/val
    split_point = int(0.8 * len(remaining_indices))
    train_indices = remaining_indices[:split_point]
    val_indices = remaining_indices[split_point:]

    print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

    train_dataset = SASDataset(images_2ch[train_indices], masks[train_indices], augment=True) # loads dataset from images in file path
    val_dataset = SASDataset(images_2ch[val_indices], masks[val_indices], augment=False)
    test_dataset = SASDataset(images_2ch[test_indices], masks[test_indices], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4) # Loads dataset batches of batch_size
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader , val_loader , test_loader
