import os
import glob
import torch
import numpy as np
import pickle
from tqdm import tqdm
from model_unet import UNet
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import random

# Add at the top of your file (after imports)
def compute_miou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(np.nan)  # Ignore this class in MIoU
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious  # mean IoU, per-class IoU

def compute_classwise_pixel_accuracy(pred, target, num_classes):
    accs = []
    for cls in range(num_classes):
        mask = (target == cls)
        if mask.sum() == 0:
            accs.append(np.nan)
        else:
            accs.append((pred[mask] == cls).sum() / mask.sum())
    return accs

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def depatchify_patches(patches, image_shape, patch_size, step):
    # patches: [N_patches, patch_size, patch_size]
    # image_shape: (H, W)
    # This is a simple version for non-overlapping patches
    # For overlapping, you need to average overlapping regions
    reconstructed = np.zeros(image_shape, dtype=patches.dtype)
    count = np.zeros(image_shape, dtype=np.int32)
    idx = 0
    H, W = image_shape
    for i in range(0, H - patch_size + 1, step):
        for j in range(0, W - patch_size + 1, step):
            reconstructed[i:i+patch_size, j:j+patch_size] += patches[idx]
            count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    reconstructed = reconstructed / np.maximum(count, 1)
    return reconstructed

def visualize_prediction(image, gt_mask, pred_mask, idx=0):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    plt.subplot(1,3,2)
    plt.imshow(gt_mask, cmap='tab20')
    plt.title('Ground Truth')
    plt.subplot(1,3,3)
    plt.imshow(pred_mask, cmap='tab20')
    plt.title('Prediction')
    plt.suptitle(f"Test Image {idx}")
    plt.show()

def evaluate_on_test(fold, patch_size, out_root="preprocessed_random", checkpoint_dir="checkpoints", device="cuda"):
    # Load model and weights
    model = UNet(n_channels=2, n_classes=3, n_out_channels=32).to(device)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_fold_{fold}.pth")
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    # Load test patches and meta
    patch_dir = os.path.join(out_root, "real", f"patch_{patch_size}")
    patch_file = os.path.join(patch_dir, "test_patches.npy")
    mask_file = os.path.join(patch_dir, "test_masks.npy")
    meta_file = os.path.join(patch_dir, "test_meta.pkl")
    patches = np.load(patch_file)
    masks = np.load(mask_file)
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    # Group patches by image
    img_to_patches = {}
    img_to_masks = {}
    for i, m in enumerate(meta):
        img_idx = m['img_idx']
        if img_idx not in img_to_patches:
            img_to_patches[img_idx] = []
            img_to_masks[img_idx] = []
        img_to_patches[img_idx].append(patches[i])
        img_to_masks[img_idx].append(masks[i])

    all_metrics = []
    for img_idx in img_to_patches:
        img_patches = np.stack(img_to_patches[img_idx])  # [N_patches, patch_size, patch_size, 2]
        mask_patches = np.stack(img_to_masks[img_idx])   # [N_patches, patch_size, patch_size]
        # Prepare input for model: [N, 2, patch_size, patch_size]
        img_patches_torch = torch.from_numpy(np.transpose(img_patches, (0,3,1,2))).float().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(img_patches_torch), 32):
                batch = img_patches_torch[i:i+32]
                out = model(batch)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
        preds = np.concatenate(preds, axis=0)  # [N_patches, patch_size, patch_size]
        # Depatchify
        orig_shape = meta[0]['orig_shape'][:2]
        pred_full = depatchify_patches(preds, orig_shape, patch_size, PATCH_STEPS[PATCH_SIZES.index(patch_size)])
        mask_full = depatchify_patches(mask_patches, orig_shape, patch_size, PATCH_STEPS[PATCH_SIZES.index(patch_size)])
        # Visualize
        visualize_prediction(np.zeros(orig_shape), mask_full, pred_full, idx=img_idx)
        # Compute metrics (example: pixel accuracy)
        pa = np.mean(pred_full == mask_full)
        all_metrics.append(pa)
        print(f"Test image {img_idx}: Pixel Accuracy={pa:.4f}")

    print(f"Fold {fold}, Patch size {patch_size}: Mean Pixel Accuracy={np.mean(all_metrics):.4f}")

def evaluate_on_test_for_all_checkpoints(patch_size, out_root="preprocessed_random", checkpoint_dir="checkpoints", device="cuda"):
    # Make checkpoint_dir absolute
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    print("Looking for checkpoints with pattern:", pattern)
    checkpoint_files = sorted(glob.glob(pattern))
    print(f"Found {len(checkpoint_files)} checkpoints.")

    for checkpoint_path in checkpoint_files:
        print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
        model = UNet(n_channels=2, n_classes=9, n_out_channels=16).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Load test patches and meta
        patch_dir = os.path.join(out_root, "real", f"patch_{patch_size}")
        patch_file = os.path.join(patch_dir, "test_patches.npy")
        mask_file = os.path.join(patch_dir, "test_masks.npy")
        meta_file = os.path.join(patch_dir, "test_meta.pkl")
        patches = np.load(patch_file)
        masks = np.load(mask_file)
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

        # Group patches by image
        img_to_patches = {}
        img_to_masks = {}
        for i, m in enumerate(meta):
            img_idx = m['img_idx']
            if img_idx not in img_to_patches:
                img_to_patches[img_idx] = []
                img_to_masks[img_idx] = []
            img_to_patches[img_idx].append(patches[i])
            img_to_masks[img_idx].append(masks[i])
        
        all_metrics = []
        for img_idx in img_to_patches:
            img_patches = np.stack(img_to_patches[img_idx])  # [N_patches, patch_size, patch_size, 2]
            
            img_patches = np.stack(img_to_patches[img_idx])  # [N_patches, patch_size, patch_size, C]

            # Ensure 2 channels for model input
            if img_patches.ndim == 3:  # [N_patches, H, W]
                img_patches = np.stack([img_patches, np.zeros_like(img_patches)], axis=-1)  # [N_patches, H, W, 2]
            elif img_patches.shape[-1] == 1:  # [N_patches, H, W, 1]
                img_patches = np.concatenate([img_patches, np.zeros_like(img_patches)], axis=-1)  # [N_patches, H, W, 2]
            # If already [N_patches, H, W, 2], do nothing

            img_patches_torch = torch.from_numpy(np.transpose(img_patches, (0,3,1,2))).float().to(device)  # [N, 2, H, W]
            mask_patches = np.stack(img_to_masks[img_idx])   # [N_patches, patch_size, patch_size]
            img_patches_torch = torch.from_numpy(np.transpose(img_patches, (0,3,1,2))).float().to(device)
            preds = []
            with torch.no_grad():
                for i in range(0, len(img_patches_torch), 32):
                    batch = img_patches_torch[i:i+32]
                    out = model(batch)
                    pred = torch.argmax(out, dim=1).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            orig_shape = meta[0]['orig_shape'][:2]
            pred_full = depatchify_patches(preds, orig_shape, patch_size, PATCH_STEPS[PATCH_SIZES.index(patch_size)])
            mask_full = depatchify_patches(mask_patches, orig_shape, patch_size, PATCH_STEPS[PATCH_SIZES.index(patch_size)])
            visualize_prediction(np.zeros(orig_shape), mask_full, pred_full, idx=img_idx)
            pa = np.mean(pred_full == mask_full)
            all_metrics.append(pa)
            print(f"Test image {img_idx}: Pixel Accuracy={pa:.4f}")

        print(f"Checkpoint {os.path.basename(checkpoint_path)}, Patch size {patch_size}: Mean Pixel Accuracy={np.mean(all_metrics):.4f}")

def evaluate_on_full_test_images(
    checkpoint_path,
    test_images_path="preprocessed_stride/real/test/test_images.npy",
    test_masks_path="preprocessed_stride/real/test/test_masks.npy",
    patch_size=64,
    device="cuda"
):
    # Load test images and masks
    test_images = np.load(test_images_path)  # [N, H, W, 2]
    test_masks = np.load(test_masks_path)    # [N, H, W]
    n_images = test_images.shape[0]

    # Handle random patch size
    PATCH_SIZES = [32, 64, 96, 128]
    PATCH_STEPS = [32, 64, 96, 128]
    if patch_size == 0:
        patch_size = random.choice(PATCH_SIZES)
        print(f"Randomly selected patch size: {patch_size}")
    if patch_size in PATCH_SIZES:
        step = PATCH_STEPS[PATCH_SIZES.index(patch_size)]
    else:
        step = 1 if patch_size >= 1000 else patch_size

    # Load model
    model = UNet(n_channels=2, n_classes=9, n_out_channels=16).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    num_classes = 9  # Set this to your number of classes
    all_mious = []
    all_class_accs = []
    all_metrics = []
    for img_idx in range(n_images):
        img = test_images[img_idx]  # [H, W, 2]
        mask = test_masks[img_idx]  # [H, W]
        orig_shape = img.shape[:2]

        # Patchify image and mask
        patches = patchify(img, (patch_size, patch_size, 2), step=step)
        patches = patches.reshape(-1, patch_size, patch_size, 2)
        mask_patches = patchify(mask, (patch_size, patch_size), step=step)
        mask_patches = mask_patches.reshape(-1, patch_size, patch_size)

        # Prepare input for model: [N, 2, patch_size, patch_size]
        img_patches_torch = torch.from_numpy(np.transpose(patches, (0,3,1,2))).float().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(img_patches_torch), 32):
                batch = img_patches_torch[i:i+32]
                out = model(batch)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
        preds = np.concatenate(preds, axis=0)  # [N_patches, patch_size, patch_size]

        # Depatchify
        pred_full = depatchify_patches(preds, orig_shape, patch_size, step)
        mask_full = depatchify_patches(mask_patches, orig_shape, patch_size, step)

        # Visualize
        visualize_prediction(np.zeros(orig_shape), mask_full, pred_full, idx=img_idx)
        pa = np.mean(pred_full == mask_full)
        miou, class_ious = compute_miou(pred_full, mask_full, num_classes)
        class_accs = compute_classwise_pixel_accuracy(pred_full, mask_full, num_classes)
        all_metrics.append(pa)
        all_mious.append(miou)
        all_class_accs.append(class_accs)
        print(f"Test image {img_idx}: Pixel Accuracy={pa:.4f}, MIoU={miou:.4f}")

    mean_pa = np.mean(all_metrics)
    mean_miou = np.nanmean(all_mious)
    mean_class_accs = np.nanmean(np.array(all_class_accs), axis=0)
    print(f"Checkpoint {os.path.basename(checkpoint_path)}, Patch size {patch_size}:")
    print(f"  Mean Pixel Accuracy={mean_pa:.4f}")
    print(f"  Mean MIoU={mean_miou:.4f}")
    print(f"  Class-wise Pixel Accuracy: {mean_class_accs}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = r"checkpoints_stride-64_16-out_modified(wt. loss + sgd(0.1lr)) [best perf on old dice loss]/best2"
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    pattern = os.path.join(glob.escape(checkpoint_dir), "checkpoint_epoch_*.pt")
    print("Looking for files with pattern:", pattern)
    checkpoint_files = sorted(glob.glob(pattern))
    print(f"Found {len(checkpoint_files)} checkpoints.")

    for patch_size in [1000,128, 0]:  # 0 means random
        for checkpoint_path in checkpoint_files:
            print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)} with patch size {patch_size}")
            evaluate_on_full_test_images(
                checkpoint_path,
                test_images_path="dataset/real/test/test_images.npy",
                test_masks_path="dataset/real/test/test_masks.npy",
                patch_size=patch_size,
                device=device
            )
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # for patch_size in PATCH_SIZES:
    #     print(f"\nEvaluating all checkpoints for patch size {patch_size}")
    #     evaluate_on_test_for_all_checkpoints(patch_size, out_root="preprocessed_random", checkpoint_dir=r"checkpoints_stride-64_16-out_modified(wt. loss + sgd(0.1lr)) [best perf on old dice loss]", device=device)
    
