import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataloader import get_fold_dataloader

# Simple normalization transform for 2-channel input
class SimpleNorm(object):
    def __call__(self, x):
        # x: Tensor of shape (2, H, W)
        return (x - x.mean()) / (x.std() + 1e-6)

# Simplest model: 1 conv layer, output 1 channel (for demonstration)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # Use CPU for simplicity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get DataLoader with transforms
    transform = SimpleNorm()
    train_loader = get_fold_dataloader(
        fold=0,
        split='train',
        batch_size=64,
        num_workers=0,
        transform=transform,
        target_transform=None,
        shuffle=True,
        pin_memory=False
    )

    # Instantiate model, loss, optimizer
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()  # For demonstration; use appropriate loss for your task
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run one epoch
    model.train()
    for batch_idx, (inputs, masks) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32).unsqueeze(1)  # (B, 1, H, W)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Resize masks if needed to match outputs
        if outputs.shape != masks.shape:
            masks = masks[:, :, :outputs.shape[2], :outputs.shape[3]]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        # if batch_idx >= 2:  # Just run a few batches for demonstration
        #     break