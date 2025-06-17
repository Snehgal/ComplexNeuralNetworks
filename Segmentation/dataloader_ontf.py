import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader

file_path = "D:\\ChiragSehgal\\IIITD\\ComplexNN\\Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)\\Synthetic Aperture Sonar Seabed Environment Dataset (SASSED)\\MLSP-Net Data\\sassed.h5"

PATCH_SIZE = 128
UNUSED_CLASS = 8
STRIDE = 1
N_FOLDS = 5

def compute_patch_indices_ontf(num_images, img_shape, patch_size=PATCH_SIZE, stride=STRIDE):
    indices = []
    for img_idx in range(num_images):
        for i in range(0, img_shape[0], stride):
            for j in range(0, img_shape[1], stride):
                indices.append((img_idx, i, j))
    return indices

def dominant_label_patch_ontf(mask_patch, unused_class=UNUSED_CLASS):
    vals, counts = np.unique(mask_patch, return_counts=True)
    valid = vals != unused_class
    if np.any(valid):
        vals = vals[valid]
        counts = counts[valid]
    return vals[np.argmax(counts)]

class SASSEDPatchDataset_ontf(Dataset):
    def __init__(self, h5_path, patch_indices, patch_size=PATCH_SIZE, unused_class=UNUSED_CLASS, transform=None, target_transform=None):
        self.h5_path = h5_path
        self.patch_indices = patch_indices  # list of (img_idx, i, j)
        self.patch_size = patch_size
        self.unused_class = unused_class
        self.transform = transform
        self.target_transform = target_transform
        # Get image shape and number of images
        with h5py.File(self.h5_path, 'r') as f:
            self.num_images = f['data'].shape[0]
            self.img_shape = f['data'].shape[1:]

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        img_idx, i, j = self.patch_indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            img = f['data'][img_idx]
            mask = f['segments'][img_idx]
        img_patch = img[i:i+self.patch_size, j:j+self.patch_size]
        mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
        # Pad if needed
        if img_patch.shape != (self.patch_size, self.patch_size):
            pad_h = self.patch_size - img_patch.shape[0]
            pad_w = self.patch_size - img_patch.shape[1]
            img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.unused_class)
        patch = np.stack([img_patch.real, img_patch.imag], axis=0)
        mask_patch = mask_patch.astype(np.int64)
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask_patch = self.target_transform(mask_patch)
        return patch, mask_patch

def get_patch_indices_and_labels_ontf(h5_path, patch_size=PATCH_SIZE, stride=STRIDE, unused_class=UNUSED_CLASS):
    with h5py.File(h5_path, 'r') as f:
        num_images = f['data'].shape[0]
        img_shape = f['data'].shape[1:]
    patch_indices = compute_patch_indices_ontf(num_images, img_shape, patch_size, stride)
    # Compute dominant label for each patch (for stratification)
    dominant_labels = []
    with h5py.File(h5_path, 'r') as f:
        for img_idx, i, j in patch_indices:
            mask = f['segments'][img_idx]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            if mask_patch.shape != (patch_size, patch_size):
                pad_h = patch_size - mask_patch.shape[0]
                pad_w = patch_size - mask_patch.shape[1]
                mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=unused_class)
            dominant_labels.append(dominant_label_patch_ontf(mask_patch, unused_class))
    dominant_labels = np.array(dominant_labels)
    return patch_indices, dominant_labels

def get_fold_dataloader_ontf(fold=0, split='train', batch_size=256, transform=None, target_transform=None, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False):
    patch_indices, dominant_labels = get_patch_indices_and_labels_ontf(file_path)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(patch_indices)), dominant_labels))
    train_idx, val_idx = splits[fold]
    indices = train_idx if split == 'train' else val_idx
    selected_patch_indices = [patch_indices[i] for i in indices]
    dataset = SASSEDPatchDataset_ontf(file_path, selected_patch_indices, transform=transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return loader

# Functions to retrieve individual samples
def get_patch_dataset_and_indices_ontf(fold=0, split='train'):
    patch_indices, dominant_labels = get_patch_indices_and_labels_ontf(file_path)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(patch_indices)), dominant_labels))
    train_idx, val_idx = splits[fold]
    indices = train_idx if split == 'train' else val_idx
    selected_patch_indices = [patch_indices[i] for i in indices]
    dataset = SASSEDPatchDataset_ontf(file_path, selected_patch_indices)
    return dataset, indices

def get_train_sample_ontf(idx, fold=0):
    dataset, _ = get_patch_dataset_and_indices_ontf(fold=fold, split='train')
    return dataset[idx]

def get_val_sample_ontf(idx, fold=0):
    dataset, _ = get_patch_dataset_and_indices_ontf(fold=fold, split='val')
    return dataset[idx]