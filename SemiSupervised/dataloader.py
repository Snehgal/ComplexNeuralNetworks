import os
import numpy as np
from patchify import patchify
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PATCH_SIZES = [96, 128, 160]  # List of patch sizes
STRIDES = [64, 96, 128]       # Corresponding strides

def create_patches_ssl_dataset(dataset_dir, patch_sizes, strides):
    """
    Create patches for annotated and unannotated datasets within ssl_dataset.
    
    Args:
        dataset_dir: Path to the ssl_dataset directory
        patch_sizes: List of patch sizes
        strides: List of corresponding strides
    
    Returns:
        Summary of the dataset creation process.
    """
    # Directories for patches
    annotated_dir = os.path.join(dataset_dir, "annotated_patches")
    unannotated_dir = os.path.join(dataset_dir, "unannotated_patches")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(unannotated_dir, exist_ok=True)
    
    summary = {
        "test_images": len(np.load(os.path.join(dataset_dir, "test_ids.npy"))),
        "annotated_images": 0,
        "annotated_patches": 0,
        "unannotated_images": 0,
        "unannotated_patches": 0
    }
    
    # Process annotated and unannotated datasets
    for split_name in ["annotated", "unannotated"]:
        split_dir = os.path.join(dataset_dir, split_name)
        ids = np.load(os.path.join(dataset_dir, f"{split_name}_ids.npy"))
        
        print(f"Creating patches for {split_name}...")
        total_patches = 0
        for patch_size, stride in zip(patch_sizes, strides):
            patch_dir = os.path.join(dataset_dir, f"{split_name}_patches", f"patch_{patch_size}")
            os.makedirs(patch_dir, exist_ok=True)
            
            patch_list, mask_list, meta_list = [], [], []
            for img_id in tqdm(ids, desc=f"{split_name} - Patch Size {patch_size}"):
                # Load image and mask
                img = np.load(os.path.join(split_dir, f"img_{img_id}_data.npy"))
                mask = np.load(os.path.join(split_dir, f"img_{img_id}_segments.npy"))
                
                # Extract patches
                patches = patchify(img, (patch_size, patch_size, img.shape[-1]), step=stride)
                patch_reshaped = patches.reshape(-1, patch_size, patch_size, img.shape[-1])
                mask_patches = patchify(mask, (patch_size, patch_size), step=stride)
                mask_reshaped = mask_patches.reshape(-1, patch_size, patch_size)
                
                # Append patches and metadata
                patch_list.append(patch_reshaped.astype(np.float32))
                mask_list.append(mask_reshaped.astype(np.uint8))
                meta_list.extend([
                    {
                        'img_id': img_id,
                        'patch_idx': i,
                        'orig_shape': img.shape,
                        'patch_size': patch_size
                    }
                    for i in range(patch_reshaped.shape[0])
                ])
                
                total_patches += patch_reshaped.shape[0]
            
            # Save patches and metadata
            patch_arr = np.concatenate(patch_list, axis=0)
            mask_arr = np.concatenate(mask_list, axis=0)
            np.save(os.path.join(patch_dir, f"{split_name}_patches.npy"), patch_arr)
            np.save(os.path.join(patch_dir, f"{split_name}_masks.npy"), mask_arr)
            with open(os.path.join(patch_dir, f"{split_name}_meta.pkl"), "wb") as f:
                pickle.dump(meta_list, f)
            
            print(f"Saved {patch_arr.shape[0]} patches for {split_name} - Patch Size {patch_size}")
        
        # Update summary
        if split_name == "annotated":
            summary["annotated_images"] = len(ids)
            summary["annotated_patches"] += total_patches
        elif split_name == "unannotated":
            summary["unannotated_images"] = len(ids)
            summary["unannotated_patches"] += total_patches
    
    return summary

def load_ssl_patches(patch_dir, split_name, patch_size):
    """
    Load patches and masks for training.
    
    Args:
        patch_dir: Path to the patch directory
        split_name: Name of the split ("annotated" or "unannotated")
        patch_size: Patch size to load
    
    Returns:
        Dataset object for training
    """
    patch_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_patches.npy")
    mask_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_masks.npy")
    meta_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_meta.pkl")
    
    patches = np.load(patch_file, mmap_mode='r')
    masks = np.load(mask_file, mmap_mode='r')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    return SSLPatchDataset(patches, masks, meta)

class SSLPatchDataset(Dataset):
    """Dataset for SSL patches."""
    def __init__(self, patches, masks, meta):
        self.patches = patches
        self.masks = masks
        self.meta = meta

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask = self.masks[idx]
        return torch.from_numpy(patch).float(), torch.from_numpy(mask).long()

def get_ssl_dataloader(patch_dir, split_name, patch_size, batch_size=32, num_workers=4, shuffle=True):
    """
    Create a dataloader for SSL patches.
    
    Args:
        patch_dir: Path to the patch directory
        split_name: Name of the split ("annotated" or "unannotated")
        patch_size: Patch size to load
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object
    """
    dataset = load_ssl_patches(patch_dir, split_name, patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_ssl_patches_complex(patch_dir, split_name, patch_size):
    """
    Load complex-valued patches and masks for training.
    
    Args:
        patch_dir: Path to the patch directory
        split_name: Name of the split ("annotated" or "unannotated")
        patch_size: Patch size to load
    
    Returns:
        Dataset object for training
    """
    patch_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_patches.npy")
    mask_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_masks.npy")
    meta_file = os.path.join(patch_dir, f"patch_{patch_size}", f"{split_name}_meta.pkl")
    
    patches = np.load(patch_file, mmap_mode='r')
    masks = np.load(mask_file, mmap_mode='r')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    return SSLPatchDatasetComplex(patches, masks, meta)

class SSLPatchDatasetComplex(Dataset):
    """Dataset for complex-valued SSL patches."""
    def __init__(self, patches, masks, meta):
        self.patches = patches
        self.masks = masks
        self.meta = meta

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask = self.masks[idx]
        # Return complex-valued patch as real and imaginary parts
        real = torch.from_numpy(patch[..., 0]).float()
        imag = torch.from_numpy(patch[..., 1]).float()
        complex_patch = torch.complex(real, imag)
        return complex_patch, torch.from_numpy(mask).long()

def get_ssl_dataloader_complex(patch_dir, split_name, patch_size, batch_size=32, num_workers=4, shuffle=True):
    """
    Create a dataloader for complex-valued SSL patches.
    
    Args:
        patch_dir: Path to the patch directory
        split_name: Name of the split ("annotated" or "unannotated")
        patch_size: Patch size to load
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object
    """
    dataset = load_ssl_patches_complex(patch_dir, split_name, patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

PATCH_SIZES = [64, 128]
STRIDES = [64, 128]  

ssl_dataset_dir = "ssl_dataset"
summary = create_patches_ssl_dataset(ssl_dataset_dir, PATCH_SIZES, STRIDES)

print("\n=== Dataset Summary ===")
print(f"Number of test images: {summary['test_images']}")
print(f"Number of annotated images: {summary['annotated_images']}")
print(f"Number of annotated patches: {summary['annotated_patches']}")
print(f"Number of unannotated images: {summary['unannotated_images']}")
print(f"Number of unannotated patches: {summary['unannotated_patches']}")

#EXAMPLE
"""annotated_loader_complex_128 = get_ssl_dataloader_complex(
    patch_dir=os.path.join(ssl_dataset_dir, "annotated_patches"),
    split_name="annotated",
    patch_size=128,
    batch_size=16
)

unannotated_loader_complex_128 = get_ssl_dataloader_complex(
    patch_dir=os.path.join(ssl_dataset_dir, "unannotated_patches"),
    split_name="unannotated",
    patch_size=128,
    batch_size=16
)

annotated_loader_complex_64 = get_ssl_dataloader_complex(
    patch_dir=os.path.join(ssl_dataset_dir, "annotated_patches"),
    split_name="annotated",
    patch_size=64,
    batch_size=16
)

unannotated_loader_complex_64 = get_ssl_dataloader_complex(
    patch_dir=os.path.join(ssl_dataset_dir, "unannotated_patches"),
    split_name="unannotated",
    patch_size=64,
    batch_size=16
)"""