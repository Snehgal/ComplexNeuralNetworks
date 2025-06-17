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

def extract_patches(img, mask, patch_size=PATCH_SIZE, stride=STRIDE, unused_class=UNUSED_CLASS):
    h, w = img.shape
    patches = []
    mask_patches = []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            img_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            # Pad if patch is smaller than patch_size (bottom/right edges)
            if img_patch.shape != (patch_size, patch_size):
                pad_h = patch_size - img_patch.shape[0]
                pad_w = patch_size - img_patch.shape[1]
                img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=unused_class)
            patches.append(img_patch)
            mask_patches.append(mask_patch)
    return patches, mask_patches

def dominant_label(mask_patch, unused_class=UNUSED_CLASS):
    vals, counts = np.unique(mask_patch, return_counts=True)
    valid = vals != unused_class
    if np.any(valid):
        vals = vals[valid]
        counts = counts[valid]
    return vals[np.argmax(counts)]

def prepare_patch_data():
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
        segments = f['segments'][:]

    all_patches = []
    all_mask_patches = []
    patch_img_indices = []

    for idx in range(len(data)):
        img = data[idx]
        mask = segments[idx]
        patches, mask_patches = extract_patches(img, mask)
        all_patches.extend(patches)
        all_mask_patches.extend(mask_patches)
        patch_img_indices.extend([idx] * len(patches))

    all_patches = np.stack(all_patches)
    all_mask_patches = np.stack(all_mask_patches)
    patch_img_indices = np.array(patch_img_indices)
    dominant_labels = np.array([dominant_label(mask) for mask in all_mask_patches])
    return all_patches, all_mask_patches, patch_img_indices, dominant_labels

class SASSEDPatchDataset(Dataset):
    def __init__(self, patches, masks, indices, transform=None, target_transform=None):
        self.patches = patches
        self.masks = masks
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patch = self.patches[self.indices[idx]]
        mask = self.masks[self.indices[idx]]
        # Convert complex to 2 channels (real, imag)
        patch = np.stack([patch.real, patch.imag], axis=0)  # shape: (2, H, W)
        mask = mask.astype(np.int64)
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask = self.target_transform(mask)
        return patch, mask

def get_fold_dataloader(fold=0, split='train', batch_size=256, transform=None, target_transform=None, shuffle=False, num_workers=0, pin_memory=False):
    all_patches, all_mask_patches, _, dominant_labels = prepare_patch_data()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(all_patches)), dominant_labels))
    train_idx, val_idx = splits[fold]
    indices = train_idx if split == 'train' else val_idx
    dataset = SASSEDPatchDataset(all_patches, all_mask_patches, indices, transform=transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return loader

# Example usage in another file:
# from ComplexNeuralNetworks.Segmentation.dataloader import get_fold_dataloader
# train_loader = get_fold_dataloader(fold=0, split='train', batch_size=32)
# val_loader = get_fold_dataloader(fold=0, split='val', batch_size=32, shuffle=False)