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
import torch
from dataloader import get_fold_dataloader
from UNET.model import ComplexUNet
from LadderNet.model import ComplexLadderNet

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (patch, mask) in enumerate(loader):
        patch = patch.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        # Convert (batch, 2, H, W) to complex: (batch, C, H, W)
        patch_complex = torch.complex(patch[:,0], patch[:,1])
        # Forward
        output = model(patch_complex)
        # If model returns a tuple, take the segmentation output
        if isinstance(output, tuple):
            output = output[1]
        # If output is complex, use real part for loss
        if torch.is_complex(output):
            output = output.real
        # output: (batch, n_classes, H, W), mask: (batch, H, W)
        loss = criterion(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (batch_idx + 1)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (patch, mask) in enumerate(loader):
            patch = patch.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            patch_complex = torch.complex(patch[:,0], patch[:,1])
            output = model(patch_complex)
            if isinstance(output, tuple):
                output = output[1]
            if torch.is_complex(output):
                output = output.real
            loss = criterion(output, mask)
            total_loss += loss.item()
    return total_loss / (batch_idx + 1)

def train_complex_model(model_class, n_classes=9, epochs=5, fold=0, batch_size=32, lr=1e-3, device='cuda'):
    train_loader = get_fold_dataloader(fold=fold, split='train', batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = get_fold_dataloader(fold=fold, split='val', batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = model_class(n_channels=2, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=8)  # ignore unused class

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training ComplexUNet...")
    train_complex_model(ComplexUNet, n_classes=9, epochs=2, device=device)
    print("Training ComplexLadderNet...")
    train_complex_model(ComplexLadderNet, n_classes=9, epochs=2, device=device)
    '''