import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import time

from model_unet import UNet, ComplexUNet
from ComplexNeuralNetworks.Segmentation.updatedDataloader import get_fold_dataloader, get_complex_fold_dataloader

class SimpleNorm(object):
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)

def train_model(num_epochs=5, num_classes=9, n_out=32, batch_size=16, nw=4, complex_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)

    for fold in range(3):
        print(f"\n========== Fold {fold} ==========")
        if complex_mode:
            model = ComplexUNet(n_channels=1, n_classes=num_classes, n_out_channels=n_out).to(device)
            loader_fn = get_complex_fold_dataloader
        else:
            model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out).to(device)
            loader_fn = get_fold_dataloader

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_loader = loader_fn(
            fold_dir=f"fold_{fold}",
            split='train',
            batch_size=batch_size,
            num_workers=nw,
            transform=SimpleNorm(),
            out_root="preprocessed_random"
        )
        val_loader = loader_fn(
            fold_dir=f"fold_{fold}",
            split='val',
            batch_size=batch_size,
            num_workers=nw,
            transform=SimpleNorm(),
            out_root="preprocessed_random"
        )

        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            model.train()
            total_train_loss = 0
            t0 = time.time()
            for inputs, targets in tqdm(train_loader, desc="Training", leave=True):
                inputs, targets = inputs.to(device), targets.to(device)
                if complex_mode and inputs.ndim == 3:
                    inputs = inputs.unsqueeze(1)  # [B, 1, H, W] complex
                if targets.ndim == 4 and targets.shape[1] == 1:
                    targets = targets[:, 0, :, :]
                targets = targets.long()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            t1 = time.time()
            print(f"Train Loss: {avg_train_loss:.4f} | Time: {t1-t0:.2f}s")

            # ---------- Validation ----------
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validation", leave=True):
                    inputs, targets = inputs.to(device), targets.to(device)
                    if complex_mode and inputs.ndim == 3:
                        inputs = inputs.unsqueeze(1)
                    if targets.ndim == 4 and targets.shape[1] == 1:
                        targets = targets[:, 0, :, :]
                    targets = targets.long()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Val   Loss: {avg_val_loss:.4f}")
            
            del train_loader
            del val_loader
            torch.cuda.empty_cache()
            import gc
            gc.collect()

if __name__ == "__main__":
    # Train real-valued UNet on all folds
    train_model(num_epochs=2, num_classes=9, n_out=16, batch_size=16, nw=8, complex_mode=False)
    # Train complex-valued ComplexUNet on all folds
    train_model(num_epochs=2, num_classes=9, n_out=16, batch_size=16, nw=8, complex_mode=True)