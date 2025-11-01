#!/usr/bin/env python3
"""
Fine-tuning SAM on complex-valued domain-specific dataset.
This script modifies SAM to work with 2-channel (real+imaginary) complex data
and fine-tunes the model using domain-expert prompts from labeled regions.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# Check if running in Jupyter notebook
try:
    from IPython import get_ipython
    if get_ipython() is not None and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        # Running in Jupyter notebook
        matplotlib.use('inline')
        plt.ion()  # Turn on interactive mode
        IN_NOTEBOOK = True
    else:
        # Not in notebook, use non-interactive backend
        matplotlib.use('Agg')
        IN_NOTEBOOK = False
except (ImportError, NameError):
    # IPython not available, assume not in notebook
    matplotlib.use('Agg')
    IN_NOTEBOOK = False

# Import SAM components
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, ImageEncoderViT
from segment_anything.modeling.image_encoder import PatchEmbed

# Import your dataset class here
# Assuming the dataset class is in a separate file
# from your_dataset_file import SASDomainDataset

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import h5py

class SASDomainDataset(Dataset):
    """
    Fine-tuning dataset for MSA with domain expertise: 
    Prompts = random points inside each labeled region (1–8).
    """
    def __init__(self, args, data_path, images, masks, 
                 input_size=(992, 992), output_size=(992, 992), 
                 augment=False, prompt='click'):
        
        # Ensure complex channels → 2-channel real+imag representation
        images = np.stack([images.real, images.imag], axis=1)
        self.images = torch.from_numpy(images).float()       # [N, 2, H, W]
        self.masks = torch.from_numpy(masks).long()          # [N, H, W]
        
        self.args = args
        self.data_path = data_path
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.prompt = prompt
        
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15, fill=0),
        ])

    def __len__(self):
        return len(self.images)

    def _generate_prompts_from_regions(self, mask_np):
        """
        Given a mask (H, W) with regions labeled 0–8,
        returns list of (x, y) click prompts inside each region (1–8).
        """
        pts = []
        labels = []
        unique_regions = np.unique(mask_np)
        unique_regions = [r for r in unique_regions if r != 0]  # exclude background
        
        for region in unique_regions:
            indices = np.argwhere(mask_np == region)
            if len(indices) == 0:
                continue
            # Randomly sample one pixel inside the region
            y, x = indices[random.randint(0, len(indices) - 1)]
            pts.append([x, y])  # SAM expects [x, y]
            labels.append(1)    # positive click
        if len(pts) == 0:
            # no regions found, use dummy center point
            H, W = mask_np.shape
            pts = [[W // 2, H // 2]]
            labels = [0]
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        img = self.images[idx]  # [C, H, W]
        mask = self.masks[idx]  # [H, W]
        
        # --- Resize ---
        img = F.interpolate(img.unsqueeze(0), size=self.input_size, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.output_size, mode='nearest').squeeze(0).squeeze(0).long()
        
        # --- Augment (optional) ---
        if self.augment:
            stacked = torch.cat([img, mask.unsqueeze(0).float()], dim=0)
            stacked = self.transforms(stacked)
            img = stacked[:-1]
            mask = stacked[-1].long()
        
        # Ensure mask values are in valid range [0, 9] after all transforms
        mask = torch.clamp(mask, 0, 9)
        
        # --- Generate domain-expert prompts ---
        if self.prompt == 'click':
            pts, p_labels = self._generate_prompts_from_regions(mask.numpy())
        else:
            pts = torch.empty(0, 2)
            p_labels = torch.tensor([0])
        
        image_meta_dict = {'filename_or_obj': f'image_{idx}'}
        
        return {
            'image': img.unsqueeze(0),          # [1, C, H, W]
            'label': mask.unsqueeze(0),         # [1, H, W]
            'p_label': p_labels,                # [num_points]
            'pt': pts,                          # [num_points, 2]
            'image_meta_dict': image_meta_dict
        }


class ComplexSAM(Sam):
    """
    Modified SAM for 2-channel complex data input.
    Changes the first conv layer to accept 2 channels instead of 3.
    """
    def __init__(self, sam_model, freeze_components=None):
        """
        Args:
            sam_model: Original SAM model
            freeze_components: List of components to freeze ['image_encoder', 'prompt_encoder', 'mask_decoder']
        """
        # Copy the original model structure
        super().__init__(
            image_encoder=sam_model.image_encoder,
            prompt_encoder=sam_model.prompt_encoder,
            mask_decoder=sam_model.mask_decoder,
            pixel_mean=[0.0, 0.0],  # Updated for 2-channel data
            pixel_std=[1.0, 1.0],   # Updated for 2-channel data
        )
        
        # Modify the patch embedding to accept 2 channels
        original_patch_embed = self.image_encoder.patch_embed
        self.image_encoder.patch_embed = PatchEmbed(
            kernel_size=(16, 16),
            stride=(16, 16),
            in_chans=2,  # Changed from 3 to 2
            embed_dim=original_patch_embed.proj.out_channels,
        )
        
        # Initialize new conv layer weights from the original RGB weights
        with torch.no_grad():
            # Average the RGB weights to initialize the 2-channel version
            rgb_weights = original_patch_embed.proj.weight.data  # [embed_dim, 3, 16, 16]
            # Use first 2 channels and average the first two RGB channels for initialization
            new_weights = torch.zeros(rgb_weights.shape[0], 2, rgb_weights.shape[2], rgb_weights.shape[3])
            new_weights[:, 0] = rgb_weights[:, 0]  # Real part gets R channel
            new_weights[:, 1] = rgb_weights[:, 1]  # Imaginary part gets G channel
            self.image_encoder.patch_embed.proj.weight.data = new_weights
            
            if original_patch_embed.proj.bias is not None:
                self.image_encoder.patch_embed.proj.bias.data = original_patch_embed.proj.bias.data
        
        # Freeze components if specified
        if freeze_components:
            for component_name in freeze_components:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    for param in component.parameters():
                        param.requires_grad = False
                    print(f"Frozen {component_name}")

    def forward(self, batched_input: List[Dict[str, Any]], multimask_output: bool) -> List[Dict[str, torch.Tensor]]:
        """
        A forward pass that tracks gradients, unlike the original SAM model.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length prompt points.
    Pads prompt points and labels to the same length within a batch.
    """
    # Separate the batch items
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    p_labels_list = [item['p_label'] for item in batch]
    pts_list = [item['pt'] for item in batch]
    image_meta_list = [item['image_meta_dict'] for item in batch]
    
    # Stack images and labels (these should be the same size)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    
    # Find the maximum number of points in this batch
    max_points = max(len(pts) for pts in pts_list)
    
    if max_points == 0:
        # Handle edge case where no points exist
        max_points = 1
        pts_list = [torch.tensor([[0, 0]], dtype=torch.float32) for _ in pts_list]
        p_labels_list = [torch.tensor([0], dtype=torch.long) for _ in p_labels_list]
    
    # Pad points and labels to max_points
    padded_pts = []
    padded_p_labels = []
    
    for pts, p_labels in zip(pts_list, p_labels_list):
        num_pts = len(pts)
        
        if num_pts < max_points:
            # Pad with dummy points (0, 0) and negative labels (-1)
            pad_pts = torch.zeros(max_points - num_pts, 2, dtype=torch.float32)
            pad_labels = torch.full((max_points - num_pts,), -1, dtype=torch.long)
            
            pts = torch.cat([pts, pad_pts], dim=0)
            p_labels = torch.cat([p_labels, pad_labels], dim=0)
        
        padded_pts.append(pts)
        padded_p_labels.append(p_labels)
    
    # Stack padded points and labels
    pts_batch = torch.stack(padded_pts, dim=0)
    p_labels_batch = torch.stack(padded_p_labels, dim=0)
    
    return {
        'image': images,
        'label': labels,
        'p_label': p_labels_batch,
        'pt': pts_batch,
        'image_meta_dict': image_meta_list
    }


class SAMTrainer:
    """Fine-tuning trainer for SAM model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
        # Initialize model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Loss functions
        self.setup_loss_functions()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.train_metrics = []
        self.val_metrics = []
        
        # Create plots directory
        self.plots_dir = Path(self.config['checkpoint_dir']) / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if running in notebook
        self.in_notebook = IN_NOTEBOOK
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['checkpoint_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def build_model(self) -> ComplexSAM:
        """Build the modified SAM model for complex data."""
        # Load pretrained SAM
        sam = sam_model_registry[self.config['model_type']](
            checkpoint=self.config.get('pretrained_checkpoint', None)
        )
        
        # Create complex version
        model = ComplexSAM(sam, freeze_components=self.config.get('freeze_components', None))
        # Count and log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"Built {self.config['model_type']} model with complex input support")
        self.logger.info(f"📊 Parameter Summary:")
        self.logger.info(f"   Total parameters: {total_params:,}")
        self.logger.info(f"   Trainable parameters: {trainable_params:,}")
        self.logger.info(f"   Frozen parameters: {frozen_params:,}")
        self.logger.info(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
                # Log which components are frozen
        freeze_components = self.config.get('freeze_components', None)
        if freeze_components:
            self.logger.info(f"🔒 Frozen components: {freeze_components}")
        else:
            self.logger.info(f"🔓 No components frozen - training entire model")
        
        # Detailed parameter breakdown by component
        self.logger.info(f"📋 Parameter breakdown by component:")
        
        # Image encoder parameters
        image_encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
        image_encoder_trainable = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        self.logger.info(f"   Image Encoder: {image_encoder_trainable:,}/{image_encoder_params:,} trainable")
        
        # Prompt encoder parameters
        prompt_encoder_params = sum(p.numel() for p in model.prompt_encoder.parameters())
        prompt_encoder_trainable = sum(p.numel() for p in model.prompt_encoder.parameters() if p.requires_grad)
        self.logger.info(f"   Prompt Encoder: {prompt_encoder_trainable:,}/{prompt_encoder_params:,} trainable")
        
        # Mask decoder parameters
        mask_decoder_params = sum(p.numel() for p in model.mask_decoder.parameters())
        mask_decoder_trainable = sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad)
        self.logger.info(f"   Mask Decoder: {mask_decoder_trainable:,}/{mask_decoder_params:,} trainable")
        
        self.logger.info(f"Built {self.config['model_type']} model with complex input support")
        return model
        
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config['optimizer'] == 'adamw':
            optimizer = AdamW(
                trainable_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=self.config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
            
        self.logger.info(f"Built {self.config['optimizer']} optimizer with lr={self.config['learning_rate']}")
        return optimizer
        
    def build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler."""
        if self.config['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif self.config['scheduler'] == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif self.config['scheduler'] is None:
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config['scheduler']}")
            
        if scheduler:
            self.logger.info(f"Built {self.config['scheduler']} scheduler")
        return scheduler
        
    def setup_loss_functions(self):
        """Setup loss functions."""
        # Segmentation loss (Focal + Dice)
        self.focal_loss = self.focal_loss_fn
        self.dice_loss = self.dice_loss_fn
        
        # IoU prediction loss
        self.iou_loss = nn.MSELoss()
        
    @staticmethod
    def focal_loss_fn(pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance."""
        # Ensure target values are valid for cross entropy
        num_classes = pred.shape[1]
        target = torch.clamp(target, 0, num_classes - 1)
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
        
    @staticmethod
    def dice_loss_fn(pred, target, smooth=1e-5):
        """Dice loss for segmentation."""
        pred_prob = torch.softmax(pred, dim=1)
        
        # Ensure target values are valid for one-hot encoding
        num_classes = pred.shape[1]
        target = torch.clamp(target, 0, num_classes - 1)
        
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred_prob * target_onehot).sum()
        dice = (2. * intersection + smooth) / (pred_prob.sum() + target_onehot.sum() + smooth)
        return 1 - dice
        
    def compute_loss(self, predictions, targets):
        """Compute combined loss for a batch of predictions and targets."""
        # predictions: list of dicts, one per batch item
        # targets: dict with batched tensors
        # Stack predictions into batch tensors
        # Squeeze leading singleton dimension if present
        masks_pred = torch.stack([p['masks'].squeeze(0) if p['masks'].ndim == 4 and p['masks'].shape[0] == 1 else p['masks'] for p in predictions], dim=0)  # [B, C, H, W]
        iou_pred = torch.stack([p['iou_predictions'].squeeze(0) if p['iou_predictions'].ndim == 2 and p['iou_predictions'].shape[0] == 1 else p['iou_predictions'] for p in predictions], dim=0)  # [B, ...]
        targets_mask = targets['label']  # [B, 1, H, W] or [B, H, W]

        # Reshape for loss computation
        B, C, H, W = masks_pred.shape

        # If target mask has shape [B, 1, H, W] or [B, H, W], squeeze safely
        if targets_mask.ndim == 4 and targets_mask.shape[1] == 1:
            targets_mask = targets_mask.squeeze(1)
        elif targets_mask.ndim == 3 and targets_mask.shape[0] == B:
            pass  # already correct
        elif targets_mask.ndim == 3 and targets_mask.shape[0] != B:
            print(f"[DEBUG] targets_mask shape mismatch: {targets_mask.shape}, expected batch {B}")
            targets_mask = targets_mask[:B]

        # Ensure mask is not boolean (can happen if mask is binary)
        if targets_mask.dtype == torch.bool:
            print(f"[DEBUG] targets_mask was bool, converting to long. Shape: {targets_mask.shape}")
            targets_mask = targets_mask.to(torch.long)
        else:
            targets_mask = targets_mask.long()

        # CRITICAL: Clamp mask values to valid range [0, num_classes-1]
        # This prevents the CUDA assertion error
        num_classes = masks_pred.shape[1]  # Usually 10 for classes 0-9
        targets_mask = torch.clamp(targets_mask, 0, num_classes - 1)
        
        # Validate mask values are in expected range
        mask_min = targets_mask.min().item()
        mask_max = targets_mask.max().item()
        if mask_min < 0 or mask_max >= num_classes:
            print(f"[WARNING] Mask values out of range after clamping: [{mask_min}, {mask_max}], expected [0, {num_classes-1}]")
            # Additional safety clamp
            targets_mask = torch.clamp(targets_mask, 0, num_classes - 1)
        
        # Remove debug prints to clean up output

        # Ensure masks_pred is float32 for loss computation
        if masks_pred.dtype != torch.float32:
            print(f"[DEBUG] masks_pred was {masks_pred.dtype}, converting to float32")
            masks_pred = masks_pred.float()

        # Segmentation losses (input: [B, C, H, W], target: [B, H, W])
        focal = self.focal_loss(masks_pred, targets_mask)
        dice = self.dice_loss(masks_pred, targets_mask)

        # Compute IoU targets for display only (no gradients)
        with torch.no_grad():
            iou_targets = self.compute_iou_targets(masks_pred, targets_mask)
            
            # Handle IoU prediction shape: iou_pred is [B, C], iou_targets is [B]
            # Compute IoU loss for display/monitoring only (detached from computation graph)
            try:
                if iou_pred.ndim == 2:  # [B, C]
                    B, C = iou_pred.shape
                    iou_targets_expanded = iou_targets.unsqueeze(1).expand(-1, C).contiguous()  # [B, C]
                    # Detach iou_pred to prevent gradients
                    iou_pred_detached = iou_pred.detach().contiguous().reshape(-1)
                    iou_targets_flat = iou_targets_expanded.reshape(-1)
                    iou_loss = self.iou_loss(iou_pred_detached, iou_targets_flat)
                else:
                    # Detach iou_pred to prevent gradients
                    iou_loss = self.iou_loss(iou_pred.detach().contiguous().reshape(-1), iou_targets.contiguous().reshape(-1))
            except RuntimeError as e:
                print(f"[ERROR] IoU loss computation failed: {e}")
                # Fallback: set IoU loss to zero for display
                iou_loss = torch.tensor(0.0, device=masks_pred.device)

        # Combine losses: 0.5 focal + 0.5 dice
        total_loss = (
            self.config['focal_weight'] * focal +
            self.config['dice_weight'] * dice
        )

        return {
            'total_loss': total_loss,
            'focal_loss': focal,
            'dice_loss': dice,
        }
        
    def compute_iou_targets(self, pred_masks, target_masks):
        """Compute IoU targets for the IoU prediction head."""
        # pred_masks: [B, C, H, W] where C is number of mask predictions (usually 3)
        # target_masks: [B, H, W]
        
        B, C, H, W = pred_masks.shape
        
        # Convert predictions to binary (sigmoid + threshold)
        pred_binary = torch.sigmoid(pred_masks) > 0.5  # [B, C, H, W]
        pred_binary = pred_binary.float().contiguous()  # Ensure contiguous
        
        # Convert target to binary and expand to match prediction channels
        target_binary = (target_masks > 0).float()  # [B, H, W]
        target_binary = target_binary.unsqueeze(1).expand(-1, C, -1, -1).contiguous()  # [B, C, H, W]
        
        # Compute IoU for each mask prediction
        intersection = (pred_binary * target_binary).sum(dim=[2, 3])  # [B, C]
        union = pred_binary.sum(dim=[2, 3]) + target_binary.sum(dim=[2, 3]) - intersection  # [B, C]
        
        iou = intersection / (union + 1e-7)  # [B, C]
        
        # Take the maximum IoU across the 3 predictions for each batch item
        iou_targets = iou.max(dim=1)[0]  # [B]
        
        return iou_targets.contiguous()  # Ensure result is contiguous
    
    def compute_metrics(self, pred_masks, target_masks):
        """
        Compute mIoU, mCPA, and overall Pixel Accuracy.
        pred_masks: [B, C, H, W] (logits)
        target_masks: [B, H, W] or [B, 1, H, W] (labels)
        """
        with torch.no_grad():
            # Ensure target_masks has the right shape [B, H, W]
            if target_masks.ndim == 4 and target_masks.shape[1] == 1:
                target_masks = target_masks.squeeze(1)  # Remove singleton dimension
            
            pred_labels = torch.argmax(pred_masks, dim=1)  # [B, H, W]
            
            # Ensure both tensors have the same shape before flattening
            if pred_labels.shape != target_masks.shape:
                print(f"[DEBUG] Resizing target_masks from {target_masks.shape} to match pred_labels {pred_labels.shape}")
                target_masks = F.interpolate(
                    target_masks.unsqueeze(1).float(), 
                    size=pred_labels.shape[-2:], 
                    mode='nearest'
                ).squeeze(1).long()

            # Verify alignment after resizing
            if pred_labels.shape != target_masks.shape:
                print(f"[ERROR] Shape mismatch persists: pred_labels {pred_labels.shape}, target_masks {target_masks.shape}")
                return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

            # Flatten for easier computation
            pred_flat = pred_labels.flatten()
            target_flat = target_masks.flatten()

            # Double check sizes match
            if pred_flat.shape[0] != target_flat.shape[0]:
                print(f"[DEBUG] Size mismatch: pred_flat {pred_flat.shape}, target_flat {target_flat.shape}")
                print(f"[DEBUG] pred_labels shape: {pred_labels.shape}, target_masks shape: {target_masks.shape}")
                # Fallback: return zeros
                return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

            # Overall pixel accuracy
            correct_pixels = (pred_flat == target_flat).sum().float()
            total_pixels = target_flat.numel()
            pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else torch.tensor(0.0)

            # Find all classes present in target (including background)
            unique_classes = torch.unique(target_flat)
            
            iou_per_class = []
            cpa_per_class = []

            # Compute metrics for each class (including background class 0)
            for cls in unique_classes:
                cls_int = cls.item()
                
                # True Positive: both pred and target are this class
                tp = ((pred_flat == cls_int) & (target_flat == cls_int)).sum().float()
                
                # False Positive: pred is this class but target is not
                fp = ((pred_flat == cls_int) & (target_flat != cls_int)).sum().float()
                
                # False Negative: target is this class but pred is not
                fn = ((pred_flat != cls_int) & (target_flat == cls_int)).sum().float()

                # Class Pixel Accuracy (Recall/Sensitivity)
                if (tp + fn) > 0:
                    cpa = tp / (tp + fn)
                    cpa_per_class.append(cpa)

                # IoU (Intersection over Union)
                if (tp + fp + fn) > 0:
                    iou = tp / (tp + fp + fn)
                    iou_per_class.append(iou)

            # Mean IoU and Mean Class Pixel Accuracy
            mIoU = torch.mean(torch.stack(iou_per_class)) if iou_per_class else torch.tensor(0.0)
            mCPA = torch.mean(torch.stack(cpa_per_class)) if cpa_per_class else torch.tensor(0.0)

        return mIoU, mCPA, pixel_acc

    def train_epoch(self, dataloader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0, 'focal_loss': 0, 'dice_loss': 0}
        epoch_metrics = {'mIoU': 0, 'mCPA': 0, 'pixel_acc': 0}
        num_samples = 0
        
        pbar = dataloader
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Prepare SAM input format
                sam_input = []
                for i in range(batch['image'].shape[0]):
                    # Filter out padded points (those with label -1)
                    valid_mask = batch['p_label'][i] != -1
                    valid_pts = batch['pt'][i][valid_mask]
                    valid_labels = batch['p_label'][i][valid_mask]
                    
                    # Ensure we have at least one valid point
                    if len(valid_pts) == 0:
                        # Use center point as fallback
                        H, W = self.config['input_size'], self.config['input_size']
                        valid_pts = torch.tensor([[W // 2, H // 2]], dtype=torch.float32, device=self.device)
                        valid_labels = torch.tensor([1], dtype=torch.long, device=self.device)
                    
                    input_dict = {
                        'image': batch['image'][i].squeeze(0),  # [C, H, W]
                        'original_size': (self.config['input_size'], self.config['input_size']),
                        'point_coords': valid_pts.unsqueeze(0),  # [1, N, 2]
                        'point_labels': valid_labels.unsqueeze(0),  # [1, N]
                    }
                    sam_input.append(input_dict)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Ensure model is in training mode
                self.model.train()
                
                # Process each sample individually to ensure gradients
                predictions = []
                for i, input_dict in enumerate(sam_input):
                    # Call model for each sample
                    pred = self.model.forward(
                        [input_dict],  # Wrap in list as expected by SAM
                        multimask_output=True
                    )
                    predictions.extend(pred)  # pred is a list, so extend
                
                # Compute loss for the whole batch
                loss_dict = self.compute_loss(predictions, batch)
                loss = loss_dict['total_loss']
                
                # Compute metrics
                # p['masks'] has shape [1, num_mask_outputs, H, W], squeeze to [num_mask_outputs, H, W]
                # then stack all batch items to get [batch_size, num_mask_outputs, H, W]
                masks_pred = torch.stack([p['masks'].squeeze(0) for p in predictions], dim=0)
                mIoU, mCPA, pixel_acc = self.compute_metrics(masks_pred, batch['label'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('clip_grad_norm', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['clip_grad_norm']
                    )
                
                self.optimizer.step()
                
                # Update metrics
                batch_size = batch['image'].shape[0]
                num_samples += batch_size
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item() * batch_size
                
                epoch_metrics['mIoU'] += mIoU.item() * batch_size
                epoch_metrics['mCPA'] += mCPA.item() * batch_size
                epoch_metrics['pixel_acc'] += pixel_acc.item() * batch_size
                
                # Remove batch-level logging to clean up output
                    
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                if batch_idx == 0:  # If first batch fails, raise the error
                    raise e
                else:
                    # Skip this batch and continue
                    continue
        
        # Average losses and metrics
        if num_samples > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_samples
            for key in epoch_metrics:
                epoch_metrics[key] /= num_samples
        else:
            self.logger.warning("No successful batches in epoch - all losses are zero")
            
        return epoch_losses, epoch_metrics
        
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'focal_loss': 0, 'dice_loss': 0}
        epoch_metrics = {'mIoU': 0, 'mCPA': 0, 'pixel_acc': 0}
        num_samples = 0
        
        with torch.no_grad():
            pbar = dataloader
            for batch in pbar:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Prepare SAM input format
                sam_input = []
                for i in range(batch['image'].shape[0]):
                    # Filter out padded points (those with label -1)
                    valid_mask = batch['p_label'][i] != -1
                    valid_pts = batch['pt'][i][valid_mask]
                    valid_labels = batch['p_label'][i][valid_mask]
                    
                    # Ensure we have at least one valid point
                    if len(valid_pts) == 0:
                        # Use a dummy point if no valid points are found
                        H, W = self.config['input_size'], self.config['input_size']
                        valid_pts = torch.tensor([[W // 2, H // 2]], dtype=torch.float32, device=self.device)
                        valid_labels = torch.tensor([1], dtype=torch.long, device=self.device)
                    
                    input_dict = {
                        'image': batch['image'][i].squeeze(0),  # [C, H, W]
                        'original_size': (self.config['input_size'], self.config['input_size']),
                        'point_coords': valid_pts.unsqueeze(0),  # [1, num_points, 2]
                        'point_labels': valid_labels.unsqueeze(0),  # [1, num_points]
                    }
                    sam_input.append(input_dict)
                
                # Forward pass
                # Process each sample individually to ensure gradients
                predictions = []
                for i, input_dict in enumerate(sam_input):
                    # Call model for each sample
                    pred = self.model.forward(
                        [input_dict],  # Pass as a list of one
                        multimask_output=True
                    )
                    predictions.extend(pred)  # pred is a list, so extend
                
                # Compute loss for the whole batch
                loss_dict = self.compute_loss(predictions, batch)
                
                # Compute metrics
                # p['masks'] has shape [1, num_mask_outputs, H, W], squeeze to [num_mask_outputs, H, W]
                # then stack all batch items to get [batch_size, num_mask_outputs, H, W]
                masks_pred = torch.stack([p['masks'].squeeze(0) for p in predictions], dim=0)
                mIoU, mCPA, pixel_acc = self.compute_metrics(masks_pred, batch['label'])

                # Update metrics
                batch_size = batch['image'].shape[0]
                num_samples += batch_size
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item() * batch_size
                
                epoch_metrics['mIoU'] += mIoU.item() * batch_size
                epoch_metrics['mCPA'] += mCPA.item() * batch_size
                epoch_metrics['pixel_acc'] += pixel_acc.item() * batch_size
        
        # Average losses and metrics
        if num_samples > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_samples
            for key in epoch_metrics:
                epoch_metrics[key] /= num_samples
            
        return epoch_losses, epoch_metrics
        
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch} with val_loss {val_loss:.4f}")
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
    def plot_losses(self, epoch: int):
        """Plot training and validation losses."""
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            return
            
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create subplots for different loss components
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
        
        # Total loss
        axes[0, 0].plot(epochs, [loss['total_loss'] for loss in self.train_losses], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, [loss['total_loss'] for loss in self.val_losses], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Focal loss
        axes[0, 1].plot(epochs, [loss['focal_loss'] for loss in self.train_losses], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, [loss['focal_loss'] for loss in self.val_losses], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Focal Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice loss
        axes[1, 0].plot(epochs, [loss['dice_loss'] for loss in self.train_losses], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, [loss['dice_loss'] for loss in self.val_losses], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Dice Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # mIoU
        axes[1, 1].plot(epochs, [m['mIoU'] for m in self.train_metrics], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, [m['mIoU'] for m in self.val_metrics], 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title('Mean IoU (mIoU)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mIoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to file
        plot_path = self.plots_dir / f'losses_epoch_{epoch:03d}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Show plot in notebook if running in one
        if self.in_notebook:
            plt.show()
        else:
            plt.close()
        
        # Also create and show a summary plot
        plt.figure(figsize=(15, 5))
        
        # Total loss comparison
        plt.subplot(1, 3, 1)
        plt.plot(epochs, [loss['total_loss'] for loss in self.train_losses], 'b-', label='Train', linewidth=2)
        plt.plot(epochs, [loss['total_loss'] for loss in self.val_losses], 'r-', label='Validation', linewidth=2)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate if available
        plt.subplot(1, 3, 2)
        if hasattr(self, 'optimizer') and len(self.train_losses) > 1:
            # Approximate learning rate curve for cosine scheduler
            if self.config.get('scheduler') == 'cosine':
                lrs = []
                for e in epochs:
                    lr = self.config.get('min_lr', 1e-6) + 0.5 * (self.config['learning_rate'] - self.config.get('min_lr', 1e-6)) * (1 + np.cos(np.pi * (e-1) / self.config['num_epochs']))
                    lrs.append(lr)
                plt.plot(epochs, lrs, 'g-', linewidth=2)
                plt.title('Learning Rate Schedule')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'LR Schedule\nNot Visualized', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Learning Rate')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nInfo N/A', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate')
        
        # Loss components
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [loss['focal_loss'] for loss in self.train_losses], 'g-', label='Focal (Train)', linewidth=2, alpha=0.8)
        plt.plot(epochs, [loss['dice_loss'] for loss in self.train_losses], 'b-', label='Dice (Train)', linewidth=2, alpha=0.8)
        plt.title('Training Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary plot
        latest_plot_path = self.plots_dir / 'losses_latest.png'
        plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
        
        # Show in notebook
        if self.in_notebook:
            plt.show()
            # Print current status
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {self.train_losses[-1]['total_loss']:.4f}")
            print(f"   Val Loss: {self.val_losses[-1]['total_loss']:.4f}")
            print(f"   Best Val Loss: {self.best_val_loss:.4f}")
            if hasattr(self, 'optimizer'):
                print(f"   Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   Plots saved to: {self.plots_dir}")
        else:
            plt.close()
        
        self.logger.info(f"Loss plots saved to {self.plots_dir}")
        
    def show_progress_summary(self, epoch: int):
        """Show a compact progress summary for notebook display."""
        if not self.in_notebook or len(self.train_losses) == 0:
            return
            
        from IPython.display import clear_output, display, HTML
        
        # Create a nice HTML summary
        recent_epochs = min(5, len(self.train_losses))
        
        html_summary = f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
            <h3 style="color: #2E7D32; margin-top: 0;">🚀 Training Progress - Epoch {epoch}</h3>
            
            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                <div style="text-align: center;">
                    <h4 style="color: #1976D2; margin: 5px 0;">Current Loss</h4>
                    <p style="font-size: 18px; font-weight: bold; color: #0D47A1;">
                        Train: {self.train_losses[-1]['total_loss']:.4f}<br>
                        Val: {self.val_losses[-1]['total_loss']:.4f}
                    </p>
                </div>
                
                <div style="text-align: center;">
                    <h4 style="color: #E65100; margin: 5px 0;">Best Val Loss</h4>
                    <p style="font-size: 18px; font-weight: bold; color: #BF360C;">
                        {self.best_val_loss:.4f}
                    </p>
                </div>
                
                <div style="text-align: center;">
                    <h4 style="color: #7B1FA2; margin: 5px 0;">Learning Rate</h4>
                    <p style="font-size: 18px; font-weight: bold; color: #4A148C;">
                        {self.optimizer.param_groups[0]['lr']:.2e}
                    </p>
                </div>
            </div>
            
            <div style="margin: 15px 0;">
                <h4 style="color: #388E3C;">Recent Epochs:</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #E8F5E8;">
                        <th style="padding: 8px; border: 1px solid #ddd;">Epoch</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Train Loss</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Val Loss</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Status</th>
                    </tr>
        """
        
        for i in range(max(0, len(self.train_losses) - recent_epochs), len(self.train_losses)):
            ep = i + 1
            train_loss = self.train_losses[i]['total_loss']
            val_loss = self.val_losses[i]['total_loss']
            is_best_str = "🏆 Best!" if abs(val_loss - self.best_val_loss) < 1e-6 else ""
            
            html_summary += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{ep}</td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{train_loss:.4f}</td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{val_loss:.4f}</td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{is_best_str}</td>
                    </tr>
            """
        
        html_summary += """
                </table>
            </div>
            
            <div style="text-align: center; margin-top: 15px;">
                <p style="color: #666; font-style: italic;">
                    📈 Detailed plots shown every 10 epochs | 💾 Checkpoints saved automatically
                </p>
            </div>
        </div>
        """
        
        display(HTML(html_summary))
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training on {len(train_loader.dataset)} samples")
        self.logger.info(f"Validation on {len(val_loader.dataset)} samples")
        
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_losses, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_losses, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_losses)
            self.val_metrics.append(val_metrics)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Clean logging - only epoch summary
            train_log = (
                f"Train - Total: {train_losses['total_loss']:.4f} | "
                f"Dice: {train_losses['dice_loss']:.4f} | "
                f"Focal: {train_losses['focal_loss']:.4f} | "
                f"mIoU: {train_metrics['mIoU']:.4f} | "
                f"Pixel Acc: {train_metrics['pixel_acc']:.4f} | "
                f"Class Pixel Acc: {train_metrics['mCPA']:.4f}"
            )
            
            val_log = (
                f"Val - Total: {val_losses['total_loss']:.4f} | "
                f"Dice: {val_losses['dice_loss']:.4f} | "
                f"Focal: {val_losses['focal_loss']:.4f} | "
                f"mIoU: {val_metrics['mIoU']:.4f} | "
                f"Pixel Acc: {val_metrics['pixel_acc']:.4f} | "
                f"Class Pixel Acc: {val_metrics['mCPA']:.4f}"
            )
            
            self.logger.info(f"Epoch {epoch + 1:03d}")
            self.logger.info(train_log)
            self.logger.info(val_log)
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if (epoch + 1) % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_losses['total_loss'], is_best)
            
            # Plot losses every 10 epochs or if best model
            if (epoch + 1) % self.config.get('plot_interval', 10) == 0 or is_best:
                self.plot_losses(epoch + 1)
            
            # Show progress in notebook every epoch
            if self.in_notebook and (epoch + 1) % 5 == 0:
                self.show_progress_summary(epoch + 1)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def load_data(data_path: str, args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load your complex-valued images and masks.
    Modify this function according to your data format.
    """
    # This is a placeholder - replace with your actual data loading logic
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        with h5py.File(data_path, 'r') as f:
            # Assuming your HDF5 file has 'images' and 'masks' datasets
            images = f['data'][:]  # Complex-valued images
            masks = f['segments'][:]    # Integer masks with regions 0-8
    else:
        # Add other data loading methods as needed
        raise NotImplementedError(f"Data loading for {data_path} not implemented")
    
    # Clamp mask values to valid range [0, 9] and convert invalid values
    print(f"Original mask value range: [{masks.min()}, {masks.max()}]")
    unique_vals = np.unique(masks)
    print(f"Unique mask values: {unique_vals}")
    
    # Clamp values to [0, 9] range
    masks = np.clip(masks, 0, 9)
    
    # Convert any remaining invalid values (like 255 for background) to 0
    masks[masks > 9] = 0
    masks[masks < 0] = 0
    
    print(f"After processing mask value range: [{masks.min()}, {masks.max()}]")
    unique_vals_after = np.unique(masks)
    print(f"Unique mask values after processing: {unique_vals_after}")
    
    return images, masks


def create_config() -> Dict[str, Any]:
    """Create training configuration with hyperparameters."""
    config = {
        # Model configuration
        'model_type': 'vit_b',  # 'vit_b', 'vit_l', 'vit_h'
        'pretrained_checkpoint': None,  # Path to pretrained SAM checkpoint
        'freeze_components': None,  # ['image_encoder'] to freeze encoder, None to train all
        
        # Data configuration
        'input_size': 992,
        'output_size': 992,
        'augment': True,
        'prompt_type': 'click',
        'train_split': 0.8,
        'val_split': 0.2,
        
        # Training hyperparameters
        'batch_size': 1,  # Small batch size due to large input size for ViT-H
        'num_epochs': 200,
        'learning_rate': 5e-5,  # Lower learning rate for fine-tuning
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'betas': (0.9, 0.999),
        
        # Learning rate scheduler
        'scheduler': 'cosine',  # 'cosine', 'step', None
        'min_lr': 1e-6,
        'step_size': 30,  # For StepLR
        'gamma': 0.1,     # For StepLR
        
        # Loss weights
        'focal_weight': 0.5,
        'dice_weight': 0.5,
        'iou_weight': 0.0,  # Exclude IoU from backprop, keep for display only
        
        # Training settings
        'clip_grad_norm': 1.0,
        'num_workers': 4,
        'pin_memory': True,
        
        # Logging and saving
        'checkpoint_dir': './checkpoints_complex_sam',
        'log_interval': 50,
        'save_interval': 10,
        'plot_interval': 10,  # Plot losses every 10 epochs
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SAM on complex domain data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset (HDF5 file)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file (optional)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained SAM checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_config()
    
    # Override with command line arguments
    if args.checkpoint:
        config['pretrained_checkpoint'] = args.checkpoint
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    # Print configuration
    if IN_NOTEBOOK:
        try:
            from IPython.display import display, HTML
            
            # Create nice HTML configuration display
            config_html = """
            <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8f9fa;">
                <h2 style="color: #1976D2; margin-top: 0;">⚙️ SAM Fine-tuning Configuration</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            """
            
            # Model settings
            model_settings = {
                'model_type': config.get('model_type', 'N/A'),
                'freeze_components': config.get('freeze_components', 'None'),
                'input_size': config.get('input_size', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A')
            }
            
            config_html += """
                    <div>
                        <h3 style="color: #388E3C;">🏗️ Model Settings</h3>
                        <ul style="list-style-type: none; padding-left: 0;">
            """
            for key, value in model_settings.items():
                config_html += f"<li><strong>{key}:</strong> {value}</li>"
            
            # Training settings
            training_settings = {
                'learning_rate': config.get('learning_rate', 'N/A'),
                'optimizer': config.get('optimizer', 'N/A'),
                'scheduler': config.get('scheduler', 'N/A'),
                'num_epochs': config.get('num_epochs', 'N/A'),
                'weight_decay': config.get('weight_decay', 'N/A')
            }
            
            config_html += """
                        </ul>
                    </div>
                    <div>
                        <h3 style="color: #E65100;">🎯 Training Settings</h3>
                        <ul style="list-style-type: none; padding-left: 0;">
            """
            for key, value in training_settings.items():
                config_html += f"<li><strong>{key}:</strong> {value}</li>"
            
            config_html += """
                        </ul>
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 10px; background-color: #E3F2FD; border-radius: 5px;">
                    <h4 style="color: #0D47A1; margin: 0 0 10px 0;">📁 Paths</h4>
                    <p><strong>Data:</strong> """ + args.data_path + """</p>
                    <p><strong>Checkpoint:</strong> """ + (args.checkpoint or "None") + """</p>
                    <p><strong>Output:</strong> """ + config.get('checkpoint_dir', 'N/A') + """</p>
                </div>
            </div>
            """
            
            display(HTML(config_html))
        except ImportError:
            # Fallback to regular print
            print("=" * 50)
            print("Training Configuration:")
            print("=" * 50)
            for key, value in config.items():
                print(f"{key}: {value}")
            print("=" * 50)
    else:
        print("=" * 50)
        print("Training Configuration:")
        print("=" * 50)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    # Load data
    print("Loading data...")
    images, masks = load_data(args.data_path, args)
    print(f"Loaded {len(images)} samples")
    print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
    
    # Split data
    n_samples = len(images)
    n_train = int(config['train_split'] * n_samples)
    
    train_images = images[:n_train]
    train_masks = masks[:n_train]
    val_images = images[n_train:]
    val_masks = masks[n_train:]
    
    print(f"Train samples: {len(train_images)}, Val samples: {len(val_images)}")
    
    # Create datasets
    train_dataset = SASDomainDataset(
        args, args.data_path, train_images, train_masks,
        input_size=(config['input_size'], config['input_size']),
        output_size=(config['output_size'], config['output_size']),
        augment=config['augment'],
        prompt=config['prompt_type']
    )
    
    val_dataset = SASDomainDataset(
        args, args.data_path, val_images, val_masks,
        input_size=(config['input_size'], config['input_size']),
        output_size=(config['output_size'], config['output_size']),
        augment=False,  # No augmentation for validation
        prompt=config['prompt_type']
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    # Create trainer
    trainer = SAMTrainer(config)
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()