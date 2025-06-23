import os
import numpy as np
import pickle
import h5py
from sklearn.model_selection import train_test_split, StratifiedKFold
from patchify import patchify
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import multiprocessing as mp
import random
from tqdm import tqdm
import time

PATCH_SIZES = [32, 64, 96, 128]
PATCH_STEPS = [32]*len(PATCH_SIZES)

H5_FILE = "sassed_V4.h5"

def ensure_preprocessed(out_root="preprocessed_stride", mode="real"):
    # Check if any patch dir exists, else preprocess
    for ps in PATCH_SIZES:
        patch_dir = os.path.join(out_root, mode, f"patch_{ps}")
        if not os.path.exists(patch_dir):
            preprocess_stride_pipeline(H5_FILE, PATCH_SIZES, PATCH_STEPS, out_root, mode=mode)
            break

def preprocess_stride_pipeline(h5_file, patch_sizes, steps, out_root, mode="real"):
    with h5py.File(h5_file, "r") as f:
        images = np.array(f["data"])
        masks = np.array(f["segments"])
    stratify_labels = np.array([np.bincount(mask.flatten()).argmax() for mask in masks])
    folds, test_idx = split_indices(len(images), stratify_labels)
    stats = {}
    for patch_size, step in zip(patch_sizes, steps):
        patch_dir = os.path.join(out_root, mode, f"patch_{patch_size}")
        for fold_num, (train_idx, val_idx) in enumerate(folds):
            fold_dir = os.path.join(patch_dir, f"fold_{fold_num}")
            for split_name, idxs in zip(['train', 'val'], [train_idx, val_idx]):
                stat = extract_and_save_patches(
                    images, masks, idxs, fold_dir, patch_size, step, split_name, mode=mode
                )
                stats[f"{split_name}_fold{fold_num}_size{patch_size}"] = stat
        stat = extract_and_save_patches(
            images, masks, test_idx, patch_dir, patch_size, step, 'test', mode=mode
        )
        stats[f"test_size{patch_size}"] = stat
    print("Patch extraction stats:", stats)
    return stats

def split_indices(num_images, stratify_labels, test_size=0.1, n_folds=3, random_state=42):
    all_indices = np.arange(num_images)
    train_idx, test_idx = train_test_split(
        all_indices, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_fold_idx, val_fold_idx in skf.split(train_idx, stratify_labels[train_idx]):
        folds.append((train_idx[train_fold_idx], train_idx[val_fold_idx]))
    return folds, test_idx

def extract_and_save_patches(images, masks, indices, out_dir, patch_size, step, split_name, mode="real"):
    import numpy as np
    import os, pickle
    os.makedirs(out_dir, exist_ok=True)
    patch_list, mask_list, meta_list = [], [], []
    t0 = time.time()
    # Wrap the entire image loop in a single tqdm bar
    with tqdm(total=len(indices), desc=f"{split_name} {patch_size}", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for img_idx in indices:
            img = images[img_idx]
            mask = masks[img_idx]
            # If image is complex, convert to 2 channels for patchify
            if mode == "complex":
                if not np.iscomplexobj(img):
                    raise ValueError("Input image is not complex for complex mode.")
                img = np.stack([img.real, img.imag], axis=-1)  # shape: (H, W, 2)
            else:
                if np.iscomplexobj(img):
                    img = np.abs(img)
                    img = np.expand_dims(img, axis=-1)  # shape: (H, W, 1)
                elif img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
            patches = patchify(img, (patch_size, patch_size, img.shape[-1]), step=step)
            patch_reshaped = patches.reshape(-1, patch_size, patch_size, img.shape[-1])
            mask_patches = patchify(mask, (patch_size, patch_size), step=step)
            mask_reshaped = mask_patches.reshape(-1, patch_size, patch_size)
            n_patches = patch_reshaped.shape[0]
            patch_list.append(patch_reshaped.astype(np.float32))
            mask_list.append(mask_reshaped.astype(np.uint8))
            meta_list.extend([
                {
                    'img_idx': img_idx,
                    'patch_idx': i,
                    'orig_shape': img.shape,
                    'patch_size': patch_size
                }
                for i in range(n_patches)
            ])
            pbar.update(1)
    patch_arr = np.concatenate(patch_list, axis=0)
    mask_arr = np.concatenate(mask_list, axis=0)
    np.save(os.path.join(out_dir, f"{split_name}_patches.npy"), patch_arr)
    np.save(os.path.join(out_dir, f"{split_name}_masks.npy"), mask_arr)
    with open(os.path.join(out_dir, f"{split_name}_meta.pkl"), "wb") as f:
        pickle.dump(meta_list, f)
    t1 = time.time()
    stats = {
        "num_patches": patch_arr.shape[0],
        "num_masks": mask_arr.shape[0],
        "time_sec": round(t1 - t0, 2),
        "patch_size": patch_size,
        "split": split_name,
        "dir": out_dir,
    }
    # print(f"[{split_name}] {patch_arr.shape[0]} patches, {mask_arr.shape[0]} masks, time: {stats['time_sec']}s, dir: {out_dir}")
    return stats

class RandomPatchSizeBatchSampler(Sampler):
    """Yields indices for batches, each batch has only one patch size, but patch size is random per batch."""
    def __init__(self, meta, batch_size):
        self.meta = meta
        self.batch_size = batch_size
        # Group indices by patch size
        self.size_to_indices = {}
        for idx, m in enumerate(meta):
            ps = m['patch_size']
            self.size_to_indices.setdefault(ps, []).append(idx)
        self.available_sizes = list(self.size_to_indices.keys())

    def __iter__(self):
        # Shuffle indices within each patch size
        size_to_indices = {ps: idxs.copy() for ps, idxs in self.size_to_indices.items()}
        for idxs in size_to_indices.values():
            random.shuffle(idxs)
        # While there are enough samples for a batch in any size
        while True:
            sizes_with_enough = [ps for ps, idxs in size_to_indices.items() if len(idxs) >= self.batch_size]
            if not sizes_with_enough:
                break
            ps = random.choice(sizes_with_enough)
            batch = [size_to_indices[ps].pop() for _ in range(self.batch_size)]
            yield batch

    def __len__(self):
        return sum(len(idxs) // self.batch_size for idxs in self.size_to_indices.values())

class MemoryMappedPatchDataset(Dataset):
    def __init__(self, patch_file, mask_file, meta_file, complex_mode=False, transform=None, target_transform=None):
        self.patches = np.load(patch_file, mmap_mode='r')
        self.masks = np.load(mask_file, mmap_mode='r')
        with open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)
        self.complex_mode = complex_mode
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.complex_mode:
            # Always return [H, W] complex64
            if patch.ndim == 3 and patch.shape[-1] == 2:
                patch = patch[..., 0] + 1j * patch[..., 1]
            elif patch.ndim == 2:
                patch = patch.astype(np.complex64)
            patch = torch.from_numpy(patch).to(torch.complex64)
        else:
            # Always return [2, H, W] float32: [real, imag]
            if patch.ndim == 3 and patch.shape[-1] == 2:
                patch = np.transpose(patch, (2, 0, 1))  # [H, W, 2] -> [2, H, W]
            elif patch.ndim == 3 and patch.shape[-1] == 1:
                # Only real part, add imag=0
                real = patch[..., 0]
                imag = np.zeros_like(real)
                patch = np.stack([real, imag], axis=0)  # [2, H, W]
            elif patch.ndim == 2:
                real = patch
                imag = np.zeros_like(real)
                patch = np.stack([real, imag], axis=0)
            patch = torch.from_numpy(patch).float()
        mask = torch.from_numpy(self.masks[idx])
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask = self.target_transform(mask)
        return patch, mask
    
def get_fold_dataloader(
    fold_dir=None,
    split='train',
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride",
    mode="real"
):
    ensure_preprocessed(out_root, mode=mode)
    patch_dirs = [os.path.join(out_root, mode, f"patch_{ps}", fold_dir or "fold_0") for ps in PATCH_SIZES]
  
    datasets, metas = [], []
    for pd in patch_dirs:
        patch_file = os.path.join(pd, f"{split}_patches.npy")
        mask_file = os.path.join(pd, f"{split}_masks.npy")
        meta_file = os.path.join(pd, f"{split}_meta.pkl")
        if os.path.exists(patch_file) and os.path.exists(mask_file) and os.path.exists(meta_file):
            ds = MemoryMappedPatchDataset(patch_file, mask_file, meta_file, complex_mode=(mode == "complex"), transform=transform, target_transform=target_transform)
            datasets.append(ds)
            with open(meta_file, 'rb') as f:
                metas.extend(pickle.load(f))
    # Concatenate all datasets
    class ConcatDataset(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.cum_lengths = np.cumsum([len(d) for d in datasets])
        def __len__(self):
            return self.cum_lengths[-1]
        def __getitem__(self, idx):
            ds_idx = np.searchsorted(self.cum_lengths, idx, side='right')
            if ds_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cum_lengths[ds_idx-1]
            return self.datasets[ds_idx][sample_idx]
    full_dataset = ConcatDataset(datasets)
    sampler = RandomPatchSizeBatchSampler(metas, batch_size)
    return DataLoader(
        full_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

def get_complex_fold_dataloader(
    fold_dir=None,
    split='train',
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride"
):
    return get_fold_dataloader(
        fold_dir=fold_dir,
        split=split,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        preload_ram=preload_ram,
        prefetch_factor=prefetch_factor,
        out_root=out_root,
        mode="complex"
    )