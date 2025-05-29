import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# === Complex FFT Dataset Wrappers ===

class ComplexFashionMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.fashion_mnist = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.fashion_mnist)

    def __getitem__(self, idx):
        img, label = self.fashion_mnist[idx]
        img = np.array(img, dtype=np.float32) / 255.0
        fft_img = np.fft.fft2(img)

        real = torch.tensor(fft_img.real, dtype=torch.float32)
        imag = torch.tensor(fft_img.imag, dtype=torch.float32)
        fft_tensor = torch.complex(real, imag).unsqueeze(0)  # shape: (1, H, W)

        return fft_tensor, label


class ComplexCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        img = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W, C)
        fft_img = np.fft.fft2(img, axes=(0, 1))  # shape: (H, W, C)

        fft_img = np.transpose(fft_img, (2, 0, 1))  # shape: (C, H, W)
        real = torch.tensor(fft_img.real, dtype=torch.float32)
        imag = torch.tensor(fft_img.imag, dtype=torch.float32)
        fft_tensor = torch.complex(real, imag)  # shape: (C, H, W)

        return fft_tensor, label


# === Generalized DataLoader Function ===

def get_dataloader(dataset_type="fashion", complex_data=False, batch_size=64, shuffle=True,
                   split="train", val_split=0.1, seed=42):
    transform = transforms.ToTensor()
    root = "./data"

    # Load full dataset
    if dataset_type == "fashion":
        if complex_data:
            dataset = ComplexFashionMNIST(root=root, train=(split != "test"))
        else:
            dataset = datasets.FashionMNIST(root=root, train=(split != "test"), download=True, transform=transform)

    elif dataset_type == "cifar":
        if complex_data:
            dataset = ComplexCIFAR10(root=root, train=(split != "test"))
        else:
            dataset = datasets.CIFAR10(root=root, train=(split != "test"), download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset: choose 'fashion' or 'cifar'")

    # Split for val
    if split == "val":
        total_len = len(dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        torch.manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_len, val_len])
        dataset = val_set

    elif split == "train" and val_split > 0:
        total_len = len(dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        torch.manual_seed(seed)
        dataset, _ = random_split(dataset, [train_len, val_len])

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

