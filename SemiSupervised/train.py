import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from dataloader import get_ssl_dataloader
from model_local import UNet  # Ensure you are using the correct UNet model

# Path to the dataset directory
ssl_dataset_dir = "ssl_dataset"

# Parameters for the dataloaders
patch_size = 256  # Using the patch size defined in your dataloader
batch_size = 4    # Small batch size as requested

# Set up device - FORCE CPU
device = torch.device("cpu")
print(f"Using device: {device} (Forced CPU)")

# Create dataloaders for annotated and unannotated data
annotated_loader = get_ssl_dataloader(
    patch_dir=os.path.join(ssl_dataset_dir, "annotated_patches"),
    split_name="annotated",
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=0  # Set to 0 to avoid memory issues
)

unannotated_loader = get_ssl_dataloader(
    patch_dir=os.path.join(ssl_dataset_dir, "unannotated_patches"),
    split_name="unannotated",
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=0  # Set to 0 to avoid memory issues
)

# Initialize model, loss function, and optimizer
model = UNet(n_channels=2, n_classes=2).to(device)  # 2 channels for real & imaginary parts
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, annotated_loader, unannotated_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Create iterators for both loaders
    annotated_iter = iter(annotated_loader)
    unannotated_iter = iter(unannotated_loader)
    
    # Calculate total batches to process (limited by smaller dataset)
    total_batches = min(len(annotated_loader), len(unannotated_loader))
    
    # Process batches
    for _ in tqdm(range(total_batches), desc="Training batches"):
        # Get annotated batch
        try:
            ann_patches, ann_masks, _, _ = next(annotated_iter)
        except StopIteration:
            # Restart if we've gone through all batches
            annotated_iter = iter(annotated_loader)
            ann_patches, ann_masks, _, _ = next(annotated_iter)
            
        # Get unannotated batch
        try:
            unann_patches, _, unann_pseudo1, _ = next(unannotated_iter)
        except StopIteration:
            # Restart if we've gone through all batches
            unannotated_iter = iter(unannotated_loader)
            unann_patches, _, unann_pseudo1, _ = next(unannotated_iter)
            
        # Check shapes before moving to device
        print(f"ann_patches shape before: {ann_patches.shape}")
        print(f"unann_patches shape before: {unann_patches.shape}")

        # Permute the dimensions if necessary
        if len(ann_patches.shape) == 4 and ann_patches.shape[-1] == 2:
            ann_patches = ann_patches.permute(0, 3, 1, 2)  # [B, C, H, W]
        if len(unann_patches.shape) == 4 and unann_patches.shape[-1] == 2:
            unann_patches = unann_patches.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Check shapes after permuting
        print(f"ann_patches shape after: {ann_patches.shape}")
        print(f"unann_patches shape after: {unann_patches.shape}")
            
        # Combine data and MOVE TO DEVICE FIRST
        ann_patches = ann_patches.to(device)
        ann_masks = ann_masks.to(device)
        unann_patches = unann_patches.to(device)
        unann_pseudo1 = unann_pseudo1.to(device)
        
        # Check the target values
        print(f"Unique values in ann_masks: {torch.unique(ann_masks)}")
        print(f"Unique values in unann_pseudo1: {torch.unique(unann_pseudo1)}")

        # Clamp the target values to be within the valid range [0, n_classes-1]
        n_classes = 2  # Assuming binary segmentation
        ann_masks = torch.clamp(ann_masks, 0, n_classes - 1)
        unann_pseudo1 = torch.clamp(unann_pseudo1, 0, n_classes - 1)
        
        # Concatenate after moving to device
        patches = torch.cat([ann_patches, unann_patches], dim=0)
        targets = torch.cat([ann_masks, unann_pseudo1], dim=0)
        
        # Forward pass
        outputs = model(patches)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        batch_count += 1
        
        # Print batch stats
        if batch_count % 5 == 0:
            print(f"  Batch {batch_count}/{total_batches}, Loss: {loss.item():.4f}")
    
    # Return average loss
    return total_loss / batch_count if batch_count > 0 else 0

# Training loop
num_epochs = 2
print("Starting training...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train the model for one epoch
    avg_loss = train_epoch(model, annotated_loader, unannotated_loader, criterion, optimizer, device)
    
    # Print epoch stats
    print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "unet_ssl_model.pth")

print("Training completed and model saved!")