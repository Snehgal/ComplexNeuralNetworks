import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu
from torchsummary import summary
from torchvision import models
from torch.nn.functional import relu
from torchsummary import summary
'''In the U-Net paper they used 0 padding and applied post-processing 
techniques to restore the original size of the image, 
however here, we uses 1 padding so that final feature map is not cropped and
 to eliminate any need to apply post-processing to our output image.'''
class PseudoComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None, padding=0, ceil_mode=False, count_include_pad=True, output_size=None):
        super().__init__()
        if output_size is not None:
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif kernel_size is not None:
            self.pool = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        else:
            # Default to adaptive pooling if nothing is specified
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        real = self.pool(x.real)
        imag = self.pool(x.imag)
        return torch.complex(real, imag)
class ComplexDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = ComplexConv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(mid_ch)
        self.conv2 = ComplexConv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_ch)

    def forward(self, x):
        x = complex_relu(self.bn1(self.conv1(x)))
        x = complex_relu(self.bn2(self.conv2(x)))
        return x

class ComplexDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ComplexDoubleConv(in_ch, out_ch)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class ComplexUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(out_ch * 2, out_ch)
        ''' if x1.size() != x2.size():
               diffY = x2.size()[2] - x1.size()[2]
               diffX = x2.size()[3] - x1.size()[3]
               x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
   diffY // 2, diffY - diffY // 2])'''
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Resize x1 if needed to match x2

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ComplexUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2 , n_out_channels = 64):
        super().__init__()
        self.down1 = ComplexDoubleConv(n_channels, n_out_channels)
        self.down2 = ComplexDoubleConv(n_out_channels, n_out_channels*2)
        self.down3 = ComplexDoubleConv(n_out_channels*2, n_out_channels*4 )
        self.down4 = ComplexDoubleConv(n_out_channels*4 , n_out_channels*8)
        self.pool = PseudoComplexAvgPool2d(kernel_size=2,stride=2)
        self.bottleneck = ComplexDoubleConv(n_out_channels*8, n_out_channels*16)
        self.up1 = ComplexUp(n_out_channels*16, n_out_channels*8)
        self.up2 = ComplexUp(n_out_channels*8, n_out_channels*4 )
        self.up3 = ComplexUp(n_out_channels*4 , n_out_channels*2)
        self.up4 = ComplexUp(n_out_channels*2, n_out_channels)
        self.out_conv = ComplexConv2d(n_out_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x = self.pool(x1)
        x2 = self.down2(x)
        x = self.pool(x2)
        x3 = self.down3(x)
        x = self.pool(x3)
        x4 = self.down4(x)
        x = self.pool(x4)
        x = self.bottleneck(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x.real

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,mid_ch = None):
        super(DoubleConv, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down_conv = DoubleConv(in_ch, out_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.down_conv(x)
        x = self.maxpool(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels*2, out_channels)  # receives concatenated tensor (in_ch = 2*out_ch)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # Concatenate along channel axis
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2 , n_out_channels = 64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = DoubleConv(n_channels, n_out_channels)
        self.down2 = DoubleConv(n_out_channels, n_out_channels*2)
        self.down3 = DoubleConv(n_out_channels*2, n_out_channels*4)
        self.down4 = DoubleConv(n_out_channels*4, n_out_channels*8)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(n_out_channels*8, n_out_channels*16)
        self.up1 = Up(n_out_channels*16, n_out_channels*8, bilinear=False)
        self.up2 = Up(n_out_channels*8, n_out_channels*4, bilinear=False)
        self.up3 = Up(n_out_channels*4, n_out_channels*2, bilinear=False)
        self.up4 = Up(n_out_channels*2, n_out_channels, bilinear=False)
        self.out_conv = nn.Conv2d(n_out_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x = self.pool(x1)
        x2 = self.down2(x)
        x = self.pool(x2)
        x3 = self.down3(x)
        x = self.pool(x3)
        x4 = self.down4(x)
        x = self.pool(x4)
        x = self.bottleneck(x)
        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.out_conv(x)
        return x
def testUnet(complex = False, out_ch = 64):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if complex:
        real = torch.randn(2, 3, 128, 128)
        imag = torch.randn(2, 3, 128, 128)
        dummy_input = torch.complex(real, imag)
        model = ComplexUNet(n_channels=3, n_classes=5,n_out_channels=out_ch).to(device)
    else:
        dummy_input = torch.randn(2, 3, 128, 128)
        model = UNet(n_channels=3, n_classes=5,n_out_channels=out_ch).to(device)



    # Forward pass
    output = model(dummy_input)

    # Dummy target with same shape
    target = torch.randn_like(output)

    # Loss (MSE for simplicity)
    criterion = nn.MSELoss()
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    print("✅ Forward and backward pass completed successfully.")
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

# Run the test
# testUnet(complex = False,out_ch=16)

'''model = ComplexUNet()
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNorm(object):
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, epsilon=1e-6, w1=0.5, w2=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.w1 = w1  # Weight for WCE
        self.w2 = w2  # Weight for Dice

    def forward(self, preds, targets):
        """
        preds: (N, C, H, W) logits
        targets: (N, H, W) class indices
        """
        N, C, H, W = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        targets_flat = targets.view(-1)                        # (N*H*W,)

        # One-hot encode targets
        targets_flat = targets.view(-1).long()  # Added this line
        y_onehot = F.one_hot(targets_flat, num_classes=C).float()  # (N*H*W, C)
        probs = F.softmax(preds_flat, dim=1)                       # (N*H*W, C)

        # ---------- WCE ----------
        exp_term = torch.exp(-self.alpha * probs)
        log_preds = torch.log(probs + self.epsilon)
        wce = -torch.sum(exp_term * y_onehot * log_preds, dim=1)
        LWCE = torch.mean(wce)

        # ---------- Standard Dice Loss ----------
        probs_dice = probs.view(-1, C).T            # (C, N*H*W)
        y_onehot_dice = y_onehot.view(-1, C).T      # (C, N*H*W)

        intersection = (probs_dice * y_onehot_dice).sum(dim=1)
        union = probs_dice.sum(dim=1) + y_onehot_dice.sum(dim=1)

        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1 - dice_score
        dice_loss = dice_loss.mean()  # average across classes

        return self.w1 * LWCE + self.w2 * dice_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
 # assumes you have this
# ------------------------- Metrics -------------------------
def get_confusion_matrix(preds, targets, num_classes):
    # Ensure everything is on the same device (GPU preferred)
   

    # Get predicted class per pixel
    preds = torch.argmax(preds, dim=1).view(-1)
    targets = targets.view(-1)

    # Mask out invalid labels
    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    # Compute confusion matrix
    conf_vector = num_classes * targets + preds
    conf_matrix = torch.bincount(conf_vector, minlength=num_classes ** 2).reshape(num_classes, num_classes).float()

    return conf_matrix


def compute_metrics(conf_matrix):
    K = conf_matrix.shape[0]
    PA = torch.sum(torch.diag(conf_matrix)) / torch.sum(conf_matrix)
    CPA = torch.mean(torch.diag(conf_matrix) / (conf_matrix.sum(dim=1) + 1e-6))
    IoUs = torch.diag(conf_matrix) / (
        conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - torch.diag(conf_matrix) + 1e-6
    )
    MIoU = torch.mean(IoUs)
    return PA.item(), CPA.item(), MIoU.item()

# ------------------------- Plotting -------------------------
def plot_losses(train_losses, val_losses, epoch):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss up to Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------ Visualizing output ---------------------
import matplotlib.pyplot as plt
import numpy as np
def visualize_prediction(inputs, targets, predictions, idx=0):
    input_np = inputs[idx].cpu().numpy()
    target_np = targets[idx].cpu().numpy()
    pred_np = predictions[idx].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(np.abs(input_np[0]), cmap='gray')  # Channel 1 magnitude
    axes[0].set_title('Input |Channel 1|')

    
    

    axes[1].imshow(target_np, cmap='nipy_spectral')
    axes[1].set_title('Ground Truth Mask')

    axes[2].imshow(pred_np, cmap='nipy_spectral')
    axes[2].set_title('Predicted Mask')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

import os
import torch

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),  # ✅ Save scheduler
        'train_losses': train_losses,
        'val_losses': val_losses
    }, save_path)
    print(f"✅ Saved checkpoint at epoch {epoch} → {save_path}")


def load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return model, optimizer, scheduler, 0, [], []

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        return model, optimizer, scheduler, 0, [], []

    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])  # ✅ Restore scheduler
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    print(f"📦 Loaded checkpoint from {checkpoint_path}")
    return model, optimizer, scheduler, checkpoint['epoch'], train_losses, val_losses