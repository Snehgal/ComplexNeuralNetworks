import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import concurrent.futures
import traceback
import sys


file_path = "sassed.h5"
PATCH_SIZE = 128
UNUSED_CLASS = 8
STRIDE = 1
N_FOLDS = 5

PREPROCESSED_DIR = os.path.join(os.path.dirname(file_path), "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
PREPROCESSED_FILE = os.path.join(
    PREPROCESSED_DIR,
    f"patches_ps{PATCH_SIZE}_stride{STRIDE}_unused{UNUSED_CLASS}_fast.npz"
)

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

def process_image(idx, img, mask):
    try:
        patches, mask_patches = extract_patches(img, mask)
        # Convert to 2-channel (real, imag) and float32 for speed
        patches = [np.stack([p.real, p.imag], axis=0).astype(np.float32) for p in patches]
        # Convert mask to uint8 for memory efficiency
        mask_patches = [m.astype(np.uint8) for m in mask_patches]
        indices = [idx] * len(patches)
        return patches, mask_patches, indices
    except Exception as e:
        print(f"[ERROR] Exception in process_image idx={idx}: {e}", file=sys.stderr)
        traceback.print_exc()
        return [], [], []

# ...existing code...

def per_image_patch_file(idx):
    return os.path.join(PREPROCESSED_DIR, f"patches_img{idx}_ps{PATCH_SIZE}_stride{STRIDE}_unused{UNUSED_CLASS}.npz")

def process_and_save_image(idx, img, mask):
    try:
        patches, mask_patches = extract_patches(img, mask)
        patches = [np.stack([p.real, p.imag], axis=0).astype(np.float32) for p in patches]
        mask_patches = [m.astype(np.uint8) for m in mask_patches]
        indices = [idx] * len(patches)
        dominant_labels = [dominant_label(m) for m in mask_patches]
        np.savez(
            per_image_patch_file(idx),
            patches=np.stack(patches),
            masks=np.stack(mask_patches),
            indices=np.array(indices),
            dominant_labels=np.array(dominant_labels)
        )
        return True
    except Exception as e:
        print(f"[ERROR] Exception in process_and_save_image idx={idx}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

def prepare_patch_data():
    try:
        # Check if all per-image files exist
        with h5py.File(file_path, 'r') as f:
            n_images = f['data'].shape[0]

        all_exist = all(os.path.exists(per_image_patch_file(idx)) for idx in range(n_images))
        if not all_exist:
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]
                segments = f['segments'][:]
            max_workers = min(2, os.cpu_count() or 2)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_and_save_image, idx, data[idx], segments[idx])
                    for idx in range(n_images)
                ]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting & saving per-image patches"):
                    future.result()  # errors are printed in process_and_save_image

        # Now load all indices, dominant_labels for stratified split
        all_indices = []
        all_dominant_labels = []
        for idx in range(n_images):
            with np.load(per_image_patch_file(idx)) as npz:
                n_patches = npz['indices'].shape[0]
                all_indices.extend([(idx, i) for i in range(n_patches)])
                all_dominant_labels.extend(npz['dominant_labels'])

        return all_indices, all_dominant_labels, n_images

    except Exception as e:
        print(f"[FATAL ERROR] prepare_patch_data failed: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

class SASSEDPatchDataset(Dataset):
    def __init__(self, all_indices, n_images, preload_ram=False, transform=None, target_transform=None):
        self.all_indices = all_indices  # list of (img_idx, patch_idx)
        self.n_images = n_images
        self.transform = transform
        self.target_transform = target_transform
        self.preload_ram = preload_ram

        # Optionally, preload all patches for selected indices (not recommended for large datasets)
        self.preloaded = None
        if preload_ram:
            # Group indices by image
            from collections import defaultdict
            img_to_patch_indices = defaultdict(list)
            for idx, (img_idx, patch_idx) in enumerate(self.all_indices):
                img_to_patch_indices[img_idx].append(patch_idx)
            self.preloaded = {}
            for img_idx in img_to_patch_indices:
                with np.load(per_image_patch_file(img_idx)) as npz:
                    self.preloaded[img_idx] = (
                        npz['patches'][img_to_patch_indices[img_idx]],
                        npz['masks'][img_to_patch_indices[img_idx]]
                    )

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        img_idx, patch_idx = self.all_indices[idx]
        if self.preload_ram and self.preloaded is not None:
            patch, mask = self.preloaded[img_idx][0][patch_idx], self.preloaded[img_idx][1][patch_idx]
        else:
            with np.load(per_image_patch_file(img_idx)) as npz:
                patch = npz['patches'][patch_idx]
                mask = npz['masks'][patch_idx]
        patch = torch.from_numpy(patch)
        mask = torch.from_numpy(mask.astype(np.int64))
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask = self.target_transform(mask)
        return patch, mask

def get_fold_dataloader(
    fold=0,
    split='train',
    batch_size=32,
    num_workers=4,
    transform=None,
    target_transform=None,
    shuffle=None,
    pin_memory=True,
    preload_ram=False
):
    all_indices, all_dominant_labels, n_images = prepare_patch_data()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(all_indices)), all_dominant_labels))
    train_idx, val_idx = splits[fold]
    indices = [all_indices[i] for i in (train_idx if split == 'train' else val_idx)]
    if shuffle is None:
        shuffle = (split == 'train')
    dataset = SASSEDPatchDataset(
        indices, n_images,
        preload_ram=preload_ram,
        transform=transform,
        target_transform=target_transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return loader

# ...rest of the code...
# Example usage in another file:
# from ComplexNeuralNetworks.Segmentation.dataloader import get_fold_dataloader
# train_loader = get_fold_dataloader(fold=0, split='train', batch_size=32)
# val_loader = get_fold_dataloader(fold=0, split='val', batch_size=32, shuffle=False)