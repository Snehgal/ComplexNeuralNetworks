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
import pickle
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

file_path = "sassed_V4.h5"
PATCH_SIZE = 128
STRIDE = 32
N_FOLDS = 5
NUM_CLASSES = 9
PREPROCESSED_DIR = os.path.join(os.path.dirname(file_path), f"preprocessed-speed-optimized-{STRIDE}")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# Use uncompressed formats for speed
FOLD_SPLIT_PATH = os.path.join(PREPROCESSED_DIR, f"fold_splits_{N_FOLDS}.pkl")

print(f"NUM_CLASSES = {NUM_CLASSES}")

def save_fold_indices(splits):
    """Save using pickle for faster serialization"""
    with open(FOLD_SPLIT_PATH, 'wb') as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_fold_indices():
    """Load using pickle for faster deserialization"""
    if not os.path.exists(FOLD_SPLIT_PATH):
        return None
    with open(FOLD_SPLIT_PATH, 'rb') as f:
        return pickle.load(f)

def save_fold_dataset(fold=0, split='train'):
    """Save fold dataset using uncompressed .npy for maximum speed"""
    all_indices, all_dominant_labels, n_images = prepare_patch_data()
    splits = load_fold_indices()
    if splits is None:
        print("[INFO] Generating new fold splits...")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        raw_splits = list(skf.split(np.arange(len(all_indices)), all_dominant_labels))
        save_fold_indices(raw_splits)
        splits = raw_splits

    split_indices = splits[fold][0] if split == 'train' else splits[fold][1]
    chosen_indices = [all_indices[i] for i in split_indices]

    # Pre-allocate arrays for maximum speed
    total_patches = len(chosen_indices)
    patches = np.empty((total_patches, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    masks = np.empty((total_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int64)  # Use int64 for PyTorch compatibility
    
    # Group by img_idx for efficient loading
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, (img_idx, patch_idx) in enumerate(chosen_indices):
        grouped[img_idx].append((i, patch_idx))

    # Load data in batches
    current_idx = 0
    for img_idx in tqdm(grouped, desc=f"Loading {split}_fold{fold} (speed optimized)"):
        patch_data = np.load(per_image_patch_file(img_idx))
        mask_data = np.load(per_image_mask_file(img_idx))
        
        for global_idx, patch_idx in grouped[img_idx]:
            patches[global_idx] = patch_data[patch_idx]
            masks[global_idx] = mask_data[patch_idx]

    # Save as uncompressed .npy for fastest loading
    patches_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_patches.npy")
    masks_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_masks.npy")
    
    np.save(patches_file, patches)
    np.save(masks_file, masks)
    
    print(f"[OK] Saved {split}_fold{fold} with {len(patches)} patches")

def save_all_folds():
    """Generate all folds with parallel processing"""
    with ThreadPoolExecutor(max_workers=min(4, N_FOLDS * 2)) as executor:
        futures = []
        for fold in range(N_FOLDS):
            futures.append(executor.submit(save_fold_dataset, fold, 'train'))
            futures.append(executor.submit(save_fold_dataset, fold, 'val'))
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc="Processing all folds"):
            future.result()

def extract_patches_vectorized(img, mask, patch_size=PATCH_SIZE, stride=STRIDE):
    """Vectorized patch extraction for maximum speed"""
    # Truncate and align to stride
    h, w = img.shape
    h = min(h, 1000)
    w = min(w, 1000)
    h = h - (h % stride)
    w = w - (w % stride)
    img = img[:h, :w]
    mask = mask[:h, :w]

    # Calculate number of patches
    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1
    n_patches = n_h * n_w

    # Pre-allocate arrays
    patches = np.empty((n_patches, 2, patch_size, patch_size), dtype=np.float32)
    mask_patches = np.empty((n_patches, patch_size, patch_size), dtype=np.int64)

    # Vectorized extraction
    idx = 0
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            img_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            
            # Stack real/imag directly into pre-allocated array
            patches[idx, 0] = img_patch.real
            patches[idx, 1] = img_patch.imag
            mask_patches[idx] = mask_patch.astype(np.int64)
            idx += 1

    return patches, mask_patches

def dominant_label_vectorized(mask_patches):
    """Vectorized dominant label calculation"""
    batch_size = mask_patches.shape[0]
    dominant_labels = np.empty(batch_size, dtype=np.int64)
    
    for i in range(batch_size):
        vals, counts = np.unique(mask_patches[i], return_counts=True)
        dominant_labels[i] = vals[np.argmax(counts)]
    
    return dominant_labels

def per_image_patch_file(idx):
    return os.path.join(PREPROCESSED_DIR, f"patches_img{idx}_ps{PATCH_SIZE}_stride{STRIDE}.npy")

def per_image_mask_file(idx):
    return os.path.join(PREPROCESSED_DIR, f"masks_img{idx}_ps{PATCH_SIZE}_stride{STRIDE}.npy")

def per_image_meta_file(idx):
    return os.path.join(PREPROCESSED_DIR, f"meta_img{idx}_ps{PATCH_SIZE}_stride{STRIDE}.pkl")

def process_and_save_image_optimized(idx, img, mask):
    """Optimized processing with separate files for different data types"""
    try:
        patches, mask_patches = extract_patches_vectorized(img, mask)
        dominant_labels = dominant_label_vectorized(mask_patches)
        
        # Save as separate uncompressed .npy files for speed
        np.save(per_image_patch_file(idx), patches)
        np.save(per_image_mask_file(idx), mask_patches)
        
        # Save metadata as pickle
        metadata = {
            'indices': np.full(len(patches), idx, dtype=np.int32),
            'dominant_labels': dominant_labels,
            'n_patches': len(patches)
        }
        with open(per_image_meta_file(idx), 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return True
    except Exception as e:
        print(f"[ERROR] Exception in process_and_save_image_optimized idx={idx}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

def prepare_patch_data():
    """Prepare patch data with optimized I/O"""
    try:
        # Load dataset dimensions
        with h5py.File(file_path, 'r') as f:
            n_images = f['data'].shape[0]

        # Check if all files exist
        all_exist = all(
            os.path.exists(per_image_patch_file(idx)) and 
            os.path.exists(per_image_mask_file(idx)) and 
            os.path.exists(per_image_meta_file(idx))
            for idx in range(n_images)
        )
        
        if not all_exist:
            print("[INFO] Processing images with optimized pipeline...")
            # Use more workers for CPU-intensive tasks
            max_workers = min(mp.cpu_count(), 8)
            
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]  # Load all data into memory for speed
                segments = f['segments'][:]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_and_save_image_optimized, idx, data[idx], segments[idx])
                    for idx in range(n_images)
                ]
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                 total=len(futures), desc="Processing images (speed optimized)"):
                    future.result()

        # Fast metadata loading
        all_indices = []
        all_dominant_labels = []
        
        for idx in range(n_images):
            with open(per_image_meta_file(idx), 'rb') as f:
                meta = pickle.load(f)
            n_patches = meta['n_patches']
            all_indices.extend([(idx, i) for i in range(n_patches)])
            all_dominant_labels.extend(meta['dominant_labels'])

        return all_indices, all_dominant_labels, n_images

    except Exception as e:
        print(f"[FATAL ERROR] prepare_patch_data failed: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

class MemoryMappedFoldDataset(Dataset):
    """Memory-mapped dataset for ultra-fast access"""
    def __init__(self, fold, split, transform=None, target_transform=None):
        self.fold = fold
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data using memory mapping for speed
        patches_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_patches.npy")
        masks_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_masks.npy")
        
        if not os.path.exists(patches_file) or not os.path.exists(masks_file):
            print(f"[INFO] Generating {split}_fold{fold} data...")
            save_fold_dataset(fold, split)
        
        # Memory-map the arrays for fastest access
        self.patches = np.load(patches_file, mmap_mode='r')
        self.masks = np.load(masks_file, mmap_mode='r')
        
        print(f"[INFO] Loaded {split}_fold{fold} with {len(self.patches)} samples via memory mapping")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Direct memory access - no copying until necessary
        patch = torch.from_numpy(self.patches[idx].copy())  # Copy only when accessed
        mask = torch.from_numpy(self.masks[idx].copy())
        
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return patch, mask

class PreloadedRAMDataset(Dataset):
    """Fully preloaded dataset in RAM for maximum speed"""
    def __init__(self, fold, split, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
        patches_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_patches.npy")
        masks_file = os.path.join(PREPROCESSED_DIR, f"{split}_fold{fold}_masks.npy")
        
        if not os.path.exists(patches_file) or not os.path.exists(masks_file):
            print(f"[INFO] Generating {split}_fold{fold} data...")
            save_fold_dataset(fold, split)
        
        print(f"[INFO] Loading {split}_fold{fold} into RAM...")
        # Load everything into RAM as PyTorch tensors
        self.patches = torch.from_numpy(np.load(patches_file)).pin_memory()
        self.masks = torch.from_numpy(np.load(masks_file)).pin_memory()
        
        print(f"[INFO] Loaded {len(self.patches)} samples into RAM")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask = self.masks[idx]
        
        if self.transform:
            patch = self.transform(patch)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return patch, mask

def get_fold_dataloader(
    fold=0,
    split='train',
    batch_size=32,
    transform=None,
    target_transform=None,
    num_workers=0,  # Set to 0 for memory-mapped data
    pin_memory=False,  # Disabled for pre-pinned data
    shuffle=True,
    preload_ram=False,
    prefetch_factor=4
):
    """
    Get optimized dataloader
    
    Args:
        preload_ram: If True, loads entire dataset into RAM for maximum speed
                    If False, uses memory mapping for balanced speed/memory usage
    """
    if preload_ram:
        dataset = PreloadedRAMDataset(fold, split, transform, target_transform)
        # Use more workers when data is in RAM
        num_workers = min(8, mp.cpu_count()) if num_workers == 0 else num_workers
        pin_memory = False  # Already pinned
    else:
        dataset = MemoryMappedFoldDataset(fold, split, transform, target_transform)
        # Fewer workers for memory-mapped data to avoid overhead
        num_workers = min(2, mp.cpu_count()) if num_workers == 0 else num_workers
        pin_memory = True
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=(split == 'train')  # Drop last incomplete batch for training
    )

# Utility functions for performance monitoring
def benchmark_loading_speed(fold=0, split='train', batch_size=32, num_batches=10):
    """Benchmark data loading speed"""
    import time
    
    print(f"\n=== Benchmarking {split}_fold{fold} ===")
    
    # Test memory-mapped version
    start_time = time.time()
    loader_mmap = get_fold_dataloader(fold, split, batch_size, preload_ram=False)
    setup_time_mmap = time.time() - start_time
    
    start_time = time.time()
    for i, (patches, masks) in enumerate(loader_mmap):
        if i >= num_batches:
            break
    loading_time_mmap = time.time() - start_time
    
    # Test RAM version
    start_time = time.time()
    loader_ram = get_fold_dataloader(fold, split, batch_size, preload_ram=True)
    setup_time_ram = time.time() - start_time
    
    start_time = time.time()
    for i, (patches, masks) in enumerate(loader_ram):
        if i >= num_batches:
            break
    loading_time_ram = time.time() - start_time
    
    print(f"Memory-mapped: Setup {setup_time_mmap:.2f}s, Loading {loading_time_mmap:.2f}s")
    print(f"RAM preloaded: Setup {setup_time_ram:.2f}s, Loading {loading_time_ram:.2f}s")
    print(f"Speed improvement: {loading_time_mmap/loading_time_ram:.2f}x faster with RAM preloading")

if __name__ == "__main__":
    # Example usage
    print("Speed-optimized data loader initialized")
    # Uncomment to run benchmark
#     benchmark_loading_speed()
