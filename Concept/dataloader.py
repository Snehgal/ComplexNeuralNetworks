import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
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
        fft_tensor = torch.tensor(np.stack([fft_img.real, fft_img.imag]), dtype=torch.float32)
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

        # Apply FFT per channel
        fft_img = np.fft.fft2(img, axes=(0, 1))  # shape: (H, W, C), complex
        fft_img = np.transpose(fft_img, (2, 0, 1))  # (C, H, W)

        fft_tensor = torch.tensor(np.stack([fft_img.real, fft_img.imag]), dtype=torch.float32)  # shape: (2, C, H, W)
        return fft_tensor, label

# === DataLoader Creators ===

def ComplexDataLoader(dataset="fashion", batchSize=64, shuffle=True):
    if dataset == "fashion":
        trainset = ComplexFashionMNIST(root='./data', train=True)
    elif dataset == "cifar":
        trainset = ComplexCIFAR10(root='./data', train=True)
    else:
        raise ValueError("Unknown dataset: choose 'fashion' or 'cifar'")
    return DataLoader(trainset, batch_size=batchSize, shuffle=shuffle)

def RealDataLoader(dataset="fashion", batchSize=64, shuffle=True):
    transform = transforms.ToTensor()
    if dataset == "fashion":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset == "cifar":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset: choose 'fashion' or 'cifar'")
    return DataLoader(trainset, batch_size=batchSize, shuffle=shuffle)
