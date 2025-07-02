from test import SimpleNorm,CustomLoss, UNet, ComplexUNet, load_latest_checkpoint, save_checkpoint, plot_losses, compute_metrics, get_confusion_matrix
from updatedDataloader import get_efficient_cross_validation_dataloaders

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
def train_model(test_fold=0, num_epochs=25, plot_every=5, num_classes=3, n_out=32, save_every=5,
                checkpoint_dir='checkpoints', nw=4, w1_=0.5, w2_=0.5, t_max=40, lr=0.001, patch_size=128):
    """
    Train model using efficient 5-fold cross-validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)
    print(f"Using fold {test_fold} as test set, training on remaining folds")

    model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CustomLoss(alpha=1.0, w1=w1_, w2=w2_)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-4)
    # Load checkpoint if available - Fix: include scheduler parameter (None for real model)
    model, optimizer, _, start_epoch, train_losses, val_losses = load_latest_checkpoint(model, optimizer, None, checkpoint_dir)
    print(f"🔁 Resuming training from epoch {start_epoch + 1} to {num_epochs + start_epoch}.")

    FOLDER_DIR = "crossVal_Dataset"

    
    # Create efficient dataloaders (loads once, uses efficiently)
    train_loader, val_loader = get_efficient_cross_validation_dataloaders(
        test_fold=test_fold,
        batch_size=16,
        num_workers=8,
        transform=SimpleNorm(),
        out_root=FOLDER_DIR,
        mode="real",
        preload_to_ram=True,
        patch_size=patch_size  # Specify patch size
    )
    
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f"\n[Epoch {epoch}/{num_epochs + start_epoch}]")
        model.train()
        total_train_loss = 0
        train_conf_matrix = torch.zeros(num_classes, num_classes, device=device)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            train_conf_matrix += get_confusion_matrix(outputs.detach(), targets, num_classes=num_classes)

        avg_train_loss = total_train_loss / len(train_loader)
        PA, CPA, MIoU = compute_metrics(train_conf_matrix)
        print(f"Train Loss: {avg_train_loss:.4f} | PA: {PA:.4f} | CPA: {CPA:.4f} | MIoU: {MIoU:.4f}")
        train_losses.append(avg_train_loss)

        # ---------- Validation ----------
        model.eval()
        total_val_loss = 0
        val_conf_matrix = torch.zeros(num_classes, num_classes, device=device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                if outputs.shape[-2:] != targets.shape[-2:]:
                    targets = targets[:, :outputs.shape[2], :outputs.shape[3]]

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                val_conf_matrix += get_confusion_matrix(outputs, targets, num_classes=num_classes)

        avg_val_loss = total_val_loss / len(val_loader)
        PA, CPA, MIoU = compute_metrics(val_conf_matrix)
        print(f"Val   Loss: {avg_val_loss:.4f} | PA: {PA:.4f} | CPA: {CPA:.4f} | MIoU: {MIoU:.4f}")
        val_losses.append(avg_val_loss)

        # Save checkpoint & optionally plot
        if epoch % save_every == 0 or epoch == num_epochs + start_epoch:
            save_checkpoint(epoch, model, optimizer, scheduler,train_losses, val_losses, checkpoint_dir)
        if epoch % plot_every == 0 or epoch == num_epochs + plot_every:
            plot_losses(train_losses, val_losses, 100)
            
        torch.cuda.empty_cache()
        import gc
        gc.collect()     

def train_modelComplex(test_fold=0, num_epochs=25, plot_every=5, num_classes=9, n_out=32, save_every=5,
                checkpoint_dir='checkpoints', nw=4, w1_=0.5, w2_=0.5, t_max=40, lr=0.001, patch_size=128):
    """
    Train complex model using efficient 5-fold cross-validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)
    print(f"Using fold {test_fold} as test set, training on remaining folds")

    # CUnet expects [B, H, W] input, so n_channels is not used in the same way
    model = ComplexUNet(n_channels=1, n_classes=num_classes, n_out_channels=n_out).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = CustomLoss(alpha=1.0, w1=w1_, w2=w2_)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-4)

    # Load checkpoint if available - Fix: include scheduler parameter
    model, optimizer, scheduler, start_epoch, train_losses, val_losses = load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir)
    print(f"🔁 Resuming training from epoch {start_epoch + 1} to {num_epochs + start_epoch}.")
    
    FOLDER_DIR = "crossVal_Dataset"
    
    # Create efficient dataloaders (loads once, uses efficiently)
    train_loader, val_loader = get_efficient_cross_validation_dataloaders(
        test_fold=test_fold,
        batch_size=16,
        num_workers=8,
        transform=SimpleNorm(),
        out_root=FOLDER_DIR,
        mode="complex",
        preload_to_ram=True,
        patch_size=patch_size  # Specify patch size
    )
    
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f"\n[Epoch {epoch}/{num_epochs + start_epoch}]")
        model.train()
        total_train_loss = 0
        train_conf_matrix = torch.zeros(num_classes, num_classes, device=device)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Debug: Check input shape
            print(f"Debug: Input shape: {inputs.shape}, dtype: {inputs.dtype}")
            
            # CUnet now expects [B, 1, H, W] complex input - dataloader provides correct format
            # No need to squeeze or modify inputs here
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            train_conf_matrix += get_confusion_matrix(outputs.detach(), targets, num_classes=num_classes)

        avg_train_loss = total_train_loss / len(train_loader)
        PA, CPA, MIoU = compute_metrics(train_conf_matrix)
        print(f"Train Loss: {avg_train_loss:.4f} | PA: {PA:.4f} | CPA: {CPA:.4f} | MIoU: {MIoU:.4f}")
        train_losses.append(avg_train_loss)

        # ---------- Validation ----------
        model.eval()
        total_val_loss = 0
        val_conf_matrix = torch.zeros(num_classes, num_classes, device=device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # No need to modify inputs - dataloader provides correct [B, 1, H, W] format
                
                outputs = model(inputs)
                if outputs.shape[-2:] != targets.shape[-2:]:
                    targets = targets[:, :outputs.shape[2], :outputs.shape[3]]

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                val_conf_matrix += get_confusion_matrix(outputs, targets, num_classes=num_classes)

        avg_val_loss = total_val_loss / len(val_loader)
        PA, CPA, MIoU = compute_metrics(val_conf_matrix)
        print(f"Val   Loss: {avg_val_loss:.4f} | PA: {PA:.4f} | CPA: {CPA:.4f} | MIoU: {MIoU:.4f}")
        val_losses.append(avg_val_loss)
        
        # Step scheduler & print learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint & optionally plot
        if epoch % save_every == 0 or epoch == num_epochs + start_epoch:
            save_checkpoint(epoch, model, optimizer, scheduler,train_losses, val_losses, checkpoint_dir)
        if epoch % plot_every == 0 or epoch == num_epochs + plot_every:
            plot_losses(train_losses, val_losses, 100)
            
        torch.cuda.empty_cache()
        import gc
        gc.collect()
              
train_model(test_fold=0, num_epochs=2, plot_every=2, num_classes=9, n_out=16, save_every=1,
                checkpoint_dir='testR', nw=4, w1_=0.5, w2_=0.5, t_max=2, lr=0.001, patch_size=128)

train_modelComplex(test_fold=0, num_epochs=2, plot_every=5, num_classes=9, n_out=16, save_every=2,
                checkpoint_dir='testC', nw=4, w1_=0.5, w2_=0.5, t_max=2, lr=0.001, patch_size=128)