import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import numpy as np
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def compute_metrics(pred, true, num_classes):
    correct = (pred == true).sum()
    total = np.prod(true.shape)
    CPA = []
    IoU = []
    for c in range(num_classes):
        tp = ((pred==c)&(true==c)).sum()
        fp = ((pred==c)&(true!=c)).sum()
        fn = ((pred!=c)&(true==c)).sum()
        cpa = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        IoU.append(iou)
        CPA.append(cpa)
    return np.nanmean(IoU), CPA

# Metrics
def compute_metrics_per_image(preds, masks, num_classes):
    """
    preds: [B, H, W] long tensor
    masks: [B, H, W] long tensor
    Returns: mean_cpa, mean_iou over batch (per-image averaged)
    """
    batch_cpa = []
    batch_iou = []

    for i in range(preds.shape[0]):
        img_pred = preds[i]
        img_mask = masks[i]

        img_cpa = []
        img_iou = []

        for cls in range(num_classes):
            mask_cls = (img_mask == cls)
            pred_cls = (img_pred == cls)

            # Skip class if not present in mask
            if mask_cls.sum().item() == 0:
                continue

            intersect = (mask_cls & pred_cls).sum().item()
            union = (mask_cls | pred_cls).sum().item()

            # CPA
            img_cpa.append(intersect / mask_cls.sum().item())

            # IoU
            if union > 0:
                img_iou.append(intersect / union)

        # Average over classes **present in this image**
        if img_cpa:
            batch_cpa.append(np.mean(img_cpa))
        if img_iou:
            batch_iou.append(np.mean(img_iou))

    # Average over batch
    mean_cpa = np.mean(batch_cpa)
    mean_iou = np.mean(batch_iou)

    return mean_cpa, mean_iou

def train_segformer(model, train_loader, val_loader, num_classes=9, epochs=50, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_miou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            logits = outputs.logits

            # Resize logits to mask size
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===============================
        # Validation
        # ===============================
        model.eval()
        val_loss = 0.0
        all_preds, all_masks = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())

        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        mean_cpa, mean_iou = compute_metrics_per_image(all_preds, all_masks, num_classes)
    

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| CPA: {mean_cpa:.4f} | mIoU: {mean_iou:.4f}")

        # Save best model
        if mean_iou > best_val_miou:
            best_val_miou = mean_iou
            torch.save(model.state_dict(), "best_segformer_sas.pth")
            print("Saved Best Model ✅")


def train_model_UNet(model, train_loader, val_loader, *, NUM_CLASSES=9, NUM_EPOCHS=500, LR=1e-5, WEIGHT_DECAY=0.01):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)

    # ==== Training ====
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.long().to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        val_miou_list = []
        val_cpa_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.long().to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, masks)
                epoch_val_loss += loss.item()

                preds = logits.argmax(dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                for p,t in zip(preds, masks_np):
                    miou, cpa = compute_metrics(p,t,NUM_CLASSES)
                    val_miou_list.append(miou)
                    val_cpa_list.append(cpa)

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        mean_val_miou = np.nanmean(val_miou_list)
        mean_val_cpa = np.nanmean(val_cpa_list)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {mean_val_miou:.4f} | Val mCPA: {mean_val_cpa:.4f}")

        if (epoch+1)%10 == 0:

            # Plot losses
            plt.figure(figsize=(8,5))
            plt.plot(range(1,len(train_losses)+1), train_losses, label="Train Loss")
            plt.plot(range(1,len(val_losses)+1), val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training & Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Save model
            save_path = f"unet_epoch_{epoch+1}.pth"
            torch.save({
             'epoch': epoch+1,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'train_losses': train_losses,
             'val_losses': val_losses
                }, save_path)
            print(f"Saved model checkpoint at {save_path}")
