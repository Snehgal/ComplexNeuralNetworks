def train_model(num_epochs=25, plot_every=5, num_classes=3, n_out=32, save_every=5,
                checkpoint_dir='checkpoints', nw=4, w1_=0.5, w2_=0.5, t_max=40,lr = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)

    model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CustomLoss(alpha=1.0, w1=w1_, w2=w2_)

    # Load checkpoint if available
    model, optimizer, start_epoch, train_losses, val_losses = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    remaining_epochs = num_epochs - start_epoch
    print(f"🔁 Resuming training from epoch {start_epoch + 1} to {num_epochs + start_epoch}.")

    # Add Cosine Annealing LR scheduler
   # scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-5)
    FOLDER_DIR = "dataset"
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        
        # Loaders
        train_loader =  get_fold_dataloader(
                    fold_dir=f"fold_{(epoch-1)%3}",
                    split='train',
                    batch_size=16,
                    num_workers=8,
                    transform=SimpleNorm(),
                    out_root=FOLDER_DIR,
                    mode = "real"
                )
        val_loader =  get_fold_dataloader(
                    fold_dir=f"fold_{(epoch-1)%3}",
                    split='val',
                    batch_size=16,
                    num_workers=8,
                    transform=SimpleNorm(),
                    out_root=FOLDER_DIR,
                    mode = "real"
                )
    
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
            for i, (inputs, targets) in enumerate(val_loader):
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

        # Step scheduler & print learning rate
        #scheduler.step()


        # Save checkpoint & optionally plot
        if epoch % save_every == 0 or epoch == num_epochs + start_epoch:
            save_checkpoint(epoch, model, optimizer, train_losses, val_losses, checkpoint_dir)
            plot_losses(train_losses, val_losses, 100)
            
        del train_loader
        del val_loader
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
def train_modelComplex(num_epochs=25, plot_every=5, num_classes=3, n_out=32, save_every=5,
                checkpoint_dir='checkpoints', nw=4, w1_=0.5, w2_=0.5, t_max=40,lr = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)

    model = CUnet(n_channels=1, n_classes=9, n_out_channels=n_out).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = CustomLoss(alpha=1.0, w1=w1_, w2=w2_)

    # Load previous checkpoint if exists
    model, optimizer, start_epoch, train_losses, val_losses = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    remaining_epochs = num_epochs - start_epoch
    print(f"🔁 Resuming training from epoch {start_epoch + 1} to {num_epochs+start_epoch}.")
    
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-4)
    FOLDER_DIR = "dataset"
    # Data
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        
        train_loader =  get_fold_dataloader(
                    fold_dir=f"fold_{(epoch-1)%3}",
                    split='train',
                    batch_size=16,
                    num_workers=8,
                    transform=SimpleNorm(),
                    out_root=FOLDER_DIR,
                    mode = "complex"
                )
        val_loader =  get_fold_dataloader(
                    fold_dir=f"fold_{(epoch-1)%3}",
                    split='val',
                    batch_size=16,
                    num_workers=8,
                    transform=SimpleNorm(),
                    out_root=FOLDER_DIR,
                    mode = "complex"
                )
        
        print(f"\n[Epoch {epoch}/{num_epochs+start_epoch}]")
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
        
        # Step scheduler & print learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        

        # Save checkpoint
        if epoch % save_every == 0 or epoch == num_epochs + start_epoch:
            save_checkpoint(epoch, model, optimizer, train_losses, val_losses, checkpoint_dir)
            
        del train_loader
        del val_loader
        torch.cuda.empty_cache()
        import gc
        gc.collect()
