import os
import numpy as np
import pickle
import h5py
from sklearn.model_selection import StratifiedKFold
from patchify import patchify
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import multiprocessing as mp
import random
from tqdm import tqdm
import time

PATCH_SIZES = [96, 128, 160, 256]
PATCH_STEPS = [64,96,128,160]

H5_FILE = "sassed_V4.h5"

N_FOLDS = 3
def ensure_preprocessed(out_root="preprocessed_stride", mode="real", n_folds=N_FOLDS):  # Add parameter
    # Check if any patch dir exists, else preprocess
    for ps in PATCH_SIZES:
        patch_dir = os.path.join(out_root, mode, f"patch_{ps}")
        if not os.path.exists(patch_dir):
            preprocess_stride_pipeline(H5_FILE, PATCH_SIZES, PATCH_STEPS, out_root, mode=mode, n_folds=n_folds)  # Pass n_folds
            break

def preprocess_stride_pipeline(h5_file, patch_sizes, steps, out_root, mode="real", n_folds=N_FOLDS):  # Add parameter
    """
    Preprocess data for 5-fold cross-validation only.
    
    Creates patches from all available data and splits them into 5 folds for
    cross-validation training. No separate test set is created or saved.
    
    Args:
        h5_file: Path to the HDF5 file containing images and masks
        patch_sizes: List of patch sizes to extract
        steps: List of step sizes for patch extraction  
        out_root: Root directory for saving preprocessed data
        mode: "real" or "complex" mode for data processing
    
    Returns:
        dict: Statistics about patch extraction for each fold and patch size
    """
    with h5py.File(h5_file, "r") as f:
        images = np.array(f["data"])
        masks = np.array(f["segments"])
    stratify_labels = np.array([np.bincount(mask.flatten()).argmax() for mask in masks])
    folds = split_indices(len(images), stratify_labels, n_folds=n_folds)  # Pass n_folds
    stats = {}

    # --- Patch extraction for cross-validation folds only ---
    for patch_size, step in zip(patch_sizes, steps):
        patch_dir = os.path.join(out_root, mode, f"patch_{patch_size}")
        for fold_num, (train_idx, val_idx) in enumerate(folds):
            fold_dir = os.path.join(patch_dir, f"fold_{fold_num}")
            for split_name, idxs in zip(['train', 'val'], [train_idx, val_idx]):
                stat = extract_and_save_patches(
                    images, masks, idxs, fold_dir, patch_size, step, split_name, mode=mode
                )
                stats[f"{split_name}_fold{fold_num}_size{patch_size}"] = stat
    print("Patch extraction stats:", stats)
    return stats

def split_indices(num_images, stratify_labels, n_folds=N_FOLDS, random_state=42):
    """
    Split all indices into n_folds for cross-validation only.
    No separate test set is created.
    """
    all_indices = np.arange(num_images)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_fold_idx, val_fold_idx in skf.split(all_indices, stratify_labels):
        folds.append((train_fold_idx, val_fold_idx))
    return folds

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
            patch = torch.from_numpy(patch).to(torch.complex64)  # shape [H, W]
            patch = patch.unsqueeze(0)  # shape [1, H, W]
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
        mask = torch.from_numpy(self.masks[idx].copy())
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

def get_complex_cross_validation_dataloaders(
    test_fold=0,
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride",
    train_val_split=0.8
):
    """Convenience function for complex cross-validation dataloaders."""
    return get_cross_validation_dataloaders(
        test_fold=test_fold,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        preload_ram=preload_ram,
        prefetch_factor=prefetch_factor,
        out_root=out_root,
        mode="complex",
        train_val_split=train_val_split
    )

def get_complex_test_dataloader(
    test_fold=0,
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride"
):
    """Convenience function for complex test dataloader."""
    return get_test_dataloader(
        test_fold=test_fold,
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

def get_cross_validation_dataloaders(
    test_fold=0,
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride",
    mode="real",
    train_val_split=0.8,
    n_folds=N_FOLDS
):
    """
    Create train and validation dataloaders for cross-validation.
    
    Args:
        test_fold: Which fold to use as test (0-4), remaining folds are used for train/val
        train_val_split: Fraction of non-test data to use for training (0.8 = 80% train, 20% val)
        Other args: Same as get_fold_dataloader
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    ensure_preprocessed(out_root, mode=mode)
    
    # Get all available folds except the test fold
    available_folds = [i for i in range(n_folds) if i != test_fold]
    
    # Collect datasets and metadata from all non-test folds
    all_datasets = []
    all_metas = []
    
    for fold_num in available_folds:
        fold_dir = f"fold_{fold_num}"
        patch_dirs = [os.path.join(out_root, mode, f"patch_{ps}", fold_dir) for ps in PATCH_SIZES]
        
        for pd in patch_dirs:
            # Combine both train and val from each fold
            for split in ['train', 'val']:
                patch_file = os.path.join(pd, f"{split}_patches.npy")
                mask_file = os.path.join(pd, f"{split}_masks.npy")
                meta_file = os.path.join(pd, f"{split}_meta.pkl")
                
                if os.path.exists(patch_file) and os.path.exists(mask_file) and os.path.exists(meta_file):
                    ds = MemoryMappedPatchDataset(
                        patch_file, mask_file, meta_file, 
                        complex_mode=(mode == "complex"), 
                        transform=transform, 
                        target_transform=target_transform
                    )
                    all_datasets.append(ds)
                    with open(meta_file, 'rb') as f:
                        all_metas.extend(pickle.load(f))
    
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
    
    full_dataset = ConcatDataset(all_datasets)
    total_size = len(full_dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create samplers for each split
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    train_metas = [all_metas[i] for i in train_indices]
    val_metas = [all_metas[i] for i in val_indices]
    
    train_sampler = RandomPatchSizeBatchSampler(train_metas, batch_size)
    val_sampler = RandomPatchSizeBatchSampler(val_metas, batch_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    print(f"Cross-validation setup: test_fold={test_fold}")
    print(f"Training folds: {available_folds}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def get_test_dataloader(
    test_fold=0,
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
    preload_ram=False,
    prefetch_factor=4,
    out_root="preprocessed_stride",
    mode="real"
):
    """
    Create test dataloader for the specified test fold.
    
    Args:
        test_fold: Which fold to use as test (0-4)
        Other args: Same as get_fold_dataloader
    
    Returns:
        DataLoader: Test dataloader for the specified fold
    """
    ensure_preprocessed(out_root, mode=mode)
    
    fold_dir = f"fold_{test_fold}"
    patch_dirs = [os.path.join(out_root, mode, f"patch_{ps}", fold_dir) for ps in PATCH_SIZES]
    
    datasets, metas = [], []
    for pd in patch_dirs:
        # Combine both train and val from the test fold
        for split in ['train', 'val']:
            patch_file = os.path.join(pd, f"{split}_patches.npy")
            mask_file = os.path.join(pd, f"{split}_masks.npy")
            meta_file = os.path.join(pd, f"{split}_meta.pkl")
            
            if os.path.exists(patch_file) and os.path.exists(mask_file) and os.path.exists(meta_file):
                ds = MemoryMappedPatchDataset(
                    patch_file, mask_file, meta_file, 
                    complex_mode=(mode == "complex"), 
                    transform=transform, 
                    target_transform=target_transform
                )
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
    
    test_loader = DataLoader(
        full_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    print(f"Test dataloader created for fold {test_fold}: {len(full_dataset)} samples")
    
    return test_loader

def precompute_cross_validation_data(out_root="preprocessed_stride", mode="real", target_patch_size=128, n_folds=N_FOLDS):
    """
    Precompute and cache all cross-validation splits once.
    Uses only one patch size to avoid dimension mismatch.
    """
    ensure_preprocessed(out_root, mode=mode)
    
    cv_cache_dir = os.path.join(out_root, mode, "cv_cache")
    os.makedirs(cv_cache_dir, exist_ok=True)
    
    # Check if already cached
    cache_complete_file = os.path.join(cv_cache_dir, "cache_complete.txt")
    if os.path.exists(cache_complete_file):
        print(f"Cross-validation cache already exists for {mode} mode")
        return
    
    print(f"Precomputing cross-validation data for {mode} mode using patch size {target_patch_size}...")
    
    # For each test fold configuration
    for test_fold in range(n_folds):
        print(f"Processing test_fold={test_fold}")
        
        # Collect all data from non-test folds using only target_patch_size
        all_patches = []
        all_masks = []
        available_folds = [i for i in range(n_folds) if i != test_fold]
        
        for fold_num in available_folds:
            # Use only the target patch size
            fold_dir = os.path.join(out_root, mode, f"patch_{target_patch_size}", f"fold_{fold_num}")
            
            # Load both train and val from this fold
            for split in ['train', 'val']:
                patch_file = os.path.join(fold_dir, f"{split}_patches.npy")
                mask_file = os.path.join(fold_dir, f"{split}_masks.npy")
                
                if os.path.exists(patch_file) and os.path.exists(mask_file):
                    patches = np.load(patch_file)
                    masks = np.load(mask_file)
                    all_patches.append(patches)
                    all_masks.append(masks)
                    print(f"  Loaded {len(patches)} patches from fold_{fold_num}/{split}")
        
        if not all_patches:
            print(f"No data found for test_fold={test_fold}, skipping...")
            continue
            
        # Concatenate all data
        combined_patches = np.concatenate(all_patches, axis=0)
        combined_masks = np.concatenate(all_masks, axis=0)
        
        # Split into train/val (80/20)
        total_size = len(combined_patches)
        train_size = int(0.8 * total_size)
        
        # Shuffle indices for random split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data
        train_patches = combined_patches[train_indices]
        train_masks = combined_masks[train_indices]
        val_patches = combined_patches[val_indices]
        val_masks = combined_masks[val_indices]
        
        # Save consolidated data for this test fold
        fold_cache_dir = os.path.join(cv_cache_dir, f"test_fold_{test_fold}")
        os.makedirs(fold_cache_dir, exist_ok=True)
        
        np.save(os.path.join(fold_cache_dir, "train_patches.npy"), train_patches)
        np.save(os.path.join(fold_cache_dir, "train_masks.npy"), train_masks)
        np.save(os.path.join(fold_cache_dir, "val_patches.npy"), val_patches)
        np.save(os.path.join(fold_cache_dir, "val_masks.npy"), val_masks)
        
        print(f"  Saved: {len(train_patches)} train, {len(val_patches)} val samples")
        
        # Clean up memory
        del combined_patches, combined_masks, train_patches, train_masks, val_patches, val_masks
        import gc
        gc.collect()
    
    # Create test fold data (using target patch size only)
    for test_fold in range(n_folds):
        test_patches = []
        test_masks = []
        
        fold_dir = os.path.join(out_root, mode, f"patch_{target_patch_size}", f"fold_{test_fold}")
        
        for split in ['train', 'val']:
            patch_file = os.path.join(fold_dir, f"{split}_patches.npy")
            mask_file = os.path.join(fold_dir, f"{split}_masks.npy")
            
            if os.path.exists(patch_file) and os.path.exists(mask_file):
                patches = np.load(patch_file)
                masks = np.load(mask_file)
                test_patches.append(patches)
                test_masks.append(masks)
        
        if test_patches:
            combined_test_patches = np.concatenate(test_patches, axis=0)
            combined_test_masks = np.concatenate(test_masks, axis=0)
            
            fold_cache_dir = os.path.join(cv_cache_dir, f"test_fold_{test_fold}")
            np.save(os.path.join(fold_cache_dir, "test_patches.npy"), combined_test_patches)
            np.save(os.path.join(fold_cache_dir, "test_masks.npy"), combined_test_masks)
    
    # Mark cache as complete
    with open(cache_complete_file, 'w') as f:
        f.write("Cross-validation cache complete")
    
    print(f"Cross-validation preprocessing complete for {mode} mode")

def get_efficient_cross_validation_dataloaders(
    test_fold=0,
    batch_size=32,
    num_workers=8,
    transform=None,
    out_root="preprocessed_stride",
    mode="real",
    preload_to_ram=True,
    patch_size=128  # Add patch size parameter
):
    """
    Efficient cross-validation dataloaders using precomputed consolidated data.
    """
    # Ensure cross-validation data is precomputed
    precompute_cross_validation_data(out_root, mode, target_patch_size=patch_size)
    
    cv_cache_dir = os.path.join(out_root, mode, "cv_cache")
    fold_cache_dir = os.path.join(cv_cache_dir, f"test_fold_{test_fold}")
    
    # Load precomputed data
    if preload_to_ram:
        print(f"Loading cross-validation data for test_fold={test_fold} into RAM...")
        train_patches = np.load(os.path.join(fold_cache_dir, "train_patches.npy"))
        train_masks = np.load(os.path.join(fold_cache_dir, "train_masks.npy"))
        val_patches = np.load(os.path.join(fold_cache_dir, "val_patches.npy"))
        val_masks = np.load(os.path.join(fold_cache_dir, "val_masks.npy"))
    else:
        # Use memory mapping for very large datasets
        train_patches = np.load(os.path.join(fold_cache_dir, "train_patches.npy"), mmap_mode='r')
        train_masks = np.load(os.path.join(fold_cache_dir, "train_masks.npy"), mmap_mode='r')
        val_patches = np.load(os.path.join(fold_cache_dir, "val_patches.npy"), mmap_mode='r')
        val_masks = np.load(os.path.join(fold_cache_dir, "val_masks.npy"), mmap_mode='r')
    
    # Create datasets
    train_dataset = SimplePatchDataset(train_patches, train_masks, complex_mode=(mode=="complex"), transform=transform)
    val_dataset = SimplePatchDataset(val_patches, val_masks, complex_mode=(mode=="complex"), transform=transform)
    
    # Create dataloaders with simple batch sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False
    )
    
    print(f"Loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader

class SimplePatchDataset(Dataset):
    """Simplified dataset that loads pre-consolidated data."""
    def __init__(self, patches, masks, complex_mode=False, transform=None):
        self.patches = patches
        self.masks = masks
        self.complex_mode = complex_mode
        self.transform = transform
        print(f"Dataset initialized - Complex mode: {complex_mode}, Patch shape: {patches.shape}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask = self.masks[idx]
        
        if self.complex_mode:
            # For complex mode: Convert to [1, H, W] complex tensor for CUnet
            if patch.ndim == 3 and patch.shape[-1] == 2:
                # If patch is [H, W, 2] -> convert to complex [H, W]
                real_part = patch[..., 0]  # [H, W]
                imag_part = patch[..., 1]  # [H, W]
                patch = real_part + 1j * imag_part  # [H, W] complex
            elif patch.ndim == 3 and patch.shape[-1] == 1:
                # Single channel [H, W, 1] -> [H, W]
                patch = patch.squeeze(-1)
            
            # Convert to tensor and add channel dimension: [H, W] -> [1, H, W]
            patch = torch.from_numpy(patch).to(torch.complex64)
            patch = patch.unsqueeze(0)  # [H, W] -> [1, H, W] for CUnet
            
        else:
            # For real mode: Convert to [2, H, W] format for UNet
            if patch.ndim == 3 and patch.shape[-1] == 2:
                # [H, W, 2] -> [2, H, W]
                patch = np.transpose(patch, (2, 0, 1))
            elif patch.ndim == 3 and patch.shape[-1] == 1:
                # [H, W, 1] -> [2, H, W] with zero imaginary
                real = patch.squeeze(-1)  # [H, W]
                imag = np.zeros_like(real)  # [H, W]
                patch = np.stack([real, imag], axis=0)  # [2, H, W]
            elif patch.ndim == 2:
                # [H, W] -> [2, H, W] with zero imaginary
                real = patch
                imag = np.zeros_like(real)
                patch = np.stack([real, imag], axis=0)  # [2, H, W]
                
            patch = torch.from_numpy(patch).float()
        
        mask = torch.from_numpy(mask.copy()).long()
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, mask

def get_efficient_test_dataloader(
    test_fold=0,
    batch_size=32,
    num_workers=8,
    transform=None,
    out_root="preprocessed_stride",
    mode="real",
    preload_to_ram=True
):
    """
    Efficient test dataloader using precomputed data.
    """
    precompute_cross_validation_data(out_root, mode)
    
    cv_cache_dir = os.path.join(out_root, mode, "cv_cache")
    fold_cache_dir = os.path.join(cv_cache_dir, f"test_fold_{test_fold}")
    
    if preload_to_ram:
        test_patches = np.load(os.path.join(fold_cache_dir, "test_patches.npy"))
        test_masks = np.load(os.path.join(fold_cache_dir, "test_masks.npy"))
    else:
        test_patches = np.load(os.path.join(fold_cache_dir, "test_patches.npy"), mmap_mode='r')
        test_masks = np.load(os.path.join(fold_cache_dir, "test_masks.npy"), mmap_mode='r')
    
    test_dataset = SimplePatchDataset(test_patches, test_masks, complex_mode=(mode=="complex"), transform=transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"Loaded test data: {len(test_dataset)} samples")
    return test_loader