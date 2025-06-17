from dataloader import get_fold_dataloader
#testing the dataloaders
def test_loader():
    train_loader = train_loader = get_fold_dataloader(
    fold=0,
    split='train',
    batch_size=64,
    shuffle=False,
    num_workers=16,
    pin_memory=True
)
    val_loader = train_loader = get_fold_dataloader(
    fold=0,
    split='val',
    batch_size=64,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

    print("Testing train_loader...")
    for i, (patch, mask) in enumerate(train_loader):
        print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
        print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
        # Print unique mask values in the batch
        print(f"Batch {i}: unique mask values {mask.unique()}")
        if i == 1:  # Just check first two batches
            break

    print("\nTesting val_loader...")
    for i, (patch, mask) in enumerate(val_loader):
        print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
        print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
        print(f"Batch {i}: unique mask values {mask.unique()}")
        if i == 1:
            break

if __name__ == "__main__":
    test_loader()
    
'''example train cycle
from dataloader import get_fold_dataloader, prepare_patch_data, SASSEDPatchDataset
import torch

def test_loader():
    train_loader = get_fold_dataloader(
        fold=0,
        split='train',
        batch_size=64,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = get_fold_dataloader(
        fold=0,
        split='val',
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print("Testing train_loader...")
    for i, (patch, mask) in enumerate(train_loader):
        print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
        print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
        print(f"Batch {i}: unique mask values {mask.unique()}")
        if i == 1:
            break

    print("\nTesting val_loader...")
    for i, (patch, mask) in enumerate(val_loader):
        print(f"Batch {i}: patch shape {patch.shape}, dtype {patch.dtype}")
        print(f"Batch {i}: mask shape {mask.shape}, dtype {mask.dtype}")
        print(f"Batch {i}: unique mask values {mask.unique()}")
        if i == 1:
            break

def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (patch, mask) in enumerate(loader):
        patch = patch.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        patch_complex = torch.complex(patch[:,0], patch[:,1])
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(patch_complex)
            if isinstance(output, tuple):
                output = output[1]
            if torch.is_complex(output):
                output = output.real
            loss = criterion(output, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / (batch_idx + 1)

def validate_amp(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (patch, mask) in enumerate(loader):
            patch = patch.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            patch_complex = torch.complex(patch[:,0], patch[:,1])
            with torch.cuda.amp.autocast():
                output = model(patch_complex)
                if isinstance(output, tuple):
                    output = output[1]
                if torch.is_complex(output):
                    output = output.real
                loss = criterion(output, mask)
            total_loss += loss.item()
    return total_loss / (batch_idx + 1)

def train_complex_model_amp(model_class, n_classes=9, epochs=5, fold=0, batch_size=32, lr=1e-3, device='cuda'):
    train_loader = get_fold_dataloader(fold=fold, split='train', batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = get_fold_dataloader(fold=fold, split='val', batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = model_class(n_channels=2, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=8)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        train_loss = train_one_epoch_amp(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate_amp(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

# --- Functions to retrieve individual samples from splits ---

def get_patch_dataset_and_indices(fold=0, split='train'):
    all_patches, all_mask_patches, _, dominant_labels = prepare_patch_data()
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(all_patches)), dominant_labels))
    train_idx, val_idx = splits[fold]
    indices = train_idx if split == 'train' else val_idx
    dataset = SASSEDPatchDataset(all_patches, all_mask_patches, indices)
    return dataset, indices

def get_train_sample(idx, fold=0):
    dataset, _ = get_patch_dataset_and_indices(fold=fold, split='train')
    return dataset[idx]

def get_val_sample(idx, fold=0):
    dataset, _ = get_patch_dataset_and_indices(fold=fold, split='val')
    return dataset[idx]

# Example usage:
# patch, mask = get_train_sample(0)
# patch, mask = get_val_sample(0)

if __name__ == "__main__":
    test_loader()
    # Example: train_complex_model_amp(ComplexUNet, n_classes=9, epochs=2, device='cuda')
    '''