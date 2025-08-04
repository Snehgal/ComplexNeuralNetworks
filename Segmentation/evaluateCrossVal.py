import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

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

# ================================================================================================
# DATASET CLASS
# ================================================================================================

class H5Dataset(Dataset):
    def __init__(self, h5_file_path: str, mode: str = "real"):
        """
        Args:
            h5_file_path: Path to the .h5 file
            mode: "real" or "complex"
        """
        self.h5_file_path = h5_file_path
        self.mode = mode.lower()
        
        # Load data from h5 file
        with h5py.File(h5_file_path, 'r') as f:
            self.data = f['data'][:]  # Assuming data key
            self.labels = f['segments'][:]  # Assuming labels key
        
        print(f"Loaded dataset: {self.data.shape}, Labels: {self.labels.shape}")
        print(f"Mode: {self.mode}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_sample = self.data[idx]  # Shape: (H, W) or (H, W, 2) if complex
        label_sample = self.labels[idx]  # Shape: (H, W)
        
        # Convert to tensors
        data_tensor = torch.from_numpy(data_sample).float()
        label_tensor = torch.from_numpy(label_sample).long()
        
        if self.mode == "real":
            # For real mode, expect 2 channels (real + imag) or convert complex to 2-channel
            if len(data_tensor.shape) == 2:  # (H, W)
                # Assume it's magnitude, create dummy imaginary
                data_tensor = torch.stack([data_tensor, torch.zeros_like(data_tensor)], dim=0)
            elif len(data_tensor.shape) == 3 and data_tensor.shape[0] == 2:  # (2, H, W)
                # Already in correct format
                pass
            elif len(data_tensor.shape) == 3 and data_tensor.shape[-1] == 2:  # (H, W, 2)
                # Transpose to (2, H, W)
                data_tensor = data_tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected data shape for real mode: {data_tensor.shape}")
                
        elif self.mode == "complex":
            # For complex mode, create complex tensor
            if len(data_tensor.shape) == 2:  # (H, W)
                # Assume it's magnitude, create complex
                data_tensor = torch.complex(data_tensor, torch.zeros_like(data_tensor))
                data_tensor = data_tensor.unsqueeze(0)  # Add channel dimension
            elif len(data_tensor.shape) == 3 and data_tensor.shape[-1] == 2:  # (H, W, 2)
                # Create complex from real and imaginary parts
                real_part = data_tensor[:, :, 0]
                imag_part = data_tensor[:, :, 1]
                data_tensor = torch.complex(real_part, imag_part)
                data_tensor = data_tensor.unsqueeze(0)  # Add channel dimension
            elif len(data_tensor.shape) == 3 and data_tensor.shape[0] == 2:  # (2, H, W)
                # Create complex from 2-channel format
                real_part = data_tensor[0]
                imag_part = data_tensor[1]
                data_tensor = torch.complex(real_part, imag_part)
                data_tensor = data_tensor.unsqueeze(0)  # Add channel dimension
            else:
                raise ValueError(f"Unexpected data shape for complex mode: {data_tensor.shape}")
        
        return data_tensor, label_tensor

# ================================================================================================
# METRICS COMPUTATION
# ================================================================================================

def robust_compute_metrics(conf_matrix, eps=1e-10):
    """Robust metrics computation with error handling"""
    
    # Ensure conf_matrix is valid
    assert conf_matrix.shape[0] == conf_matrix.shape[1], "Confusion matrix must be square"
    assert torch.all(conf_matrix >= 0), "Confusion matrix cannot have negative values"
    
    num_classes = conf_matrix.shape[0]
    
    # Pixel Accuracy
    correct = torch.diag(conf_matrix).sum()
    total = conf_matrix.sum()
    PA = correct / (total + eps)
    
    # Class Pixel Accuracy
    per_class_total = conf_matrix.sum(1)  # Ground truth count per class
    per_class_correct = torch.diag(conf_matrix)
    
    # Only compute CPA for classes that exist in ground truth
    valid_classes = per_class_total > 0
    if valid_classes.sum() > 0:
        per_class_acc = per_class_correct[valid_classes] / per_class_total[valid_classes]
        CPA = per_class_acc.mean()
    else:
        CPA = torch.tensor(0.0)
    
    # Mean IoU
    TP = torch.diag(conf_matrix)
    FP = conf_matrix.sum(0) - TP  # False positives per class
    FN = conf_matrix.sum(1) - TP  # False negatives per class
    
    # Union = TP + FP + FN
    union = TP + FP + FN
    
    # Only compute IoU for classes with non-zero union
    valid_classes = union > 0
    if valid_classes.sum() > 0:
        iou_per_class = TP[valid_classes] / union[valid_classes]
        MIoU = iou_per_class.mean()
    else:
        MIoU = torch.tensor(0.0)
    
    return PA.item(), CPA.item(), MIoU.item()

def get_confusion_matrix(predictions, targets, num_classes):
    """Generate confusion matrix"""
    # Flatten predictions and targets
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Ensure predictions and targets are in valid range
    predictions = torch.clamp(predictions, 0, num_classes - 1)
    targets = torch.clamp(targets, 0, num_classes - 1)
    
    # Create confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    for t, p in zip(targets, predictions):
        conf_matrix[t.long(), p.long()] += 1
    
    return conf_matrix

# ================================================================================================
# ENSEMBLE EVALUATOR
# ================================================================================================

class EnsembleEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = []
        
    def load_models(self, checkpoint_paths: List[str], mode: str, num_classes: int = 9, n_out_channels: int = 16):
        """
        Load 3 models from checkpoint paths
        
        Args:
            checkpoint_paths: List of 3 paths to .pt files
            mode: "real" or "complex"
            num_classes: Number of segmentation classes
            n_out_channels: Number of output channels for model architecture
        """
        print(f"Loading {len(checkpoint_paths)} models in {mode.upper()} mode...")
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"  Loading model {i+1}: {checkpoint_path}")
            
            # Initialize model
            if mode.lower() == 'real':
                model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out_channels)
            elif mode.lower() == 'complex':
                model = ComplexUNet(n_channels=1, n_classes=num_classes, n_out_channels=n_out_channels)
            else:
                raise ValueError("mode must be 'real' or 'complex'")
            
            # Load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Extract model state dict (handle different checkpoint formats)
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'model_state' in checkpoint:
                    model_state = checkpoint['model_state']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    # Assume the entire checkpoint is the state dict
                    model_state = checkpoint
                
                model.load_state_dict(model_state)
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                print(f"    ✓ Model {i+1} loaded successfully")
                
            except Exception as e:
                print(f"    ✗ Error loading model {i+1}: {e}")
                raise
        
        print(f"All {len(self.models)} models loaded successfully!\n")
    
    def predict_ensemble(self, data_batch):
        """
        Get ensemble predictions using majority voting
        
        Args:
            data_batch: Input batch tensor
            
        Returns:
            ensemble_predictions: Majority vote predictions
        """
        batch_size = data_batch.shape[0]
        
        # Get predictions from all models
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(data_batch)
                predictions = torch.argmax(outputs, dim=1)  # Shape: (B, H, W)
                all_predictions.append(predictions)
        
        # Stack predictions: (num_models, B, H, W)
        all_predictions = torch.stack(all_predictions, dim=0)
        
        # Majority voting
        ensemble_predictions = torch.mode(all_predictions, dim=0)[0]  # Shape: (B, H, W)
        
        return ensemble_predictions
    
    def evaluate_dataset(self, dataset_path: str, mode: str, num_classes: int = 9, batch_size: int = 16):
        """
        Evaluate ensemble on entire dataset
        
        Args:
            dataset_path: Path to .h5 dataset file
            mode: "real" or "complex"
            num_classes: Number of segmentation classes
            batch_size: Batch size for evaluation
            
        Returns:
            PA, CPA, MIoU: Evaluation metrics
        """
        print(f"{'='*80}")
        print(f"ENSEMBLE EVALUATION")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_path}")
        print(f"Mode: {mode.upper()}")
        print(f"Number of models: {len(self.models)}")
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*80}\n")
        
        # Create dataset and dataloader
        dataset = H5Dataset(dataset_path, mode=mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize confusion matrix
        total_conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        # Evaluate
        print("Evaluating ensemble...")
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data_batch, label_batch) in enumerate(dataloader):
                # Move to device
                data_batch = data_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                # Get ensemble predictions
                ensemble_preds = self.predict_ensemble(data_batch)
                
                # Update confusion matrix
                batch_conf_matrix = get_confusion_matrix(ensemble_preds, label_batch, num_classes)
                total_conf_matrix += batch_conf_matrix
                
                total_samples += data_batch.shape[0]
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches ({total_samples} samples)")
        
        print(f"Evaluation completed! Total samples: {total_samples}\n")
        
        # Compute metrics
        PA, CPA, MIoU = robust_compute_metrics(total_conf_matrix)
        
        # Print results
        print(f"{'='*80}")
        print(f"ENSEMBLE RESULTS")
        print(f"{'='*80}")
        print(f"📊 ENSEMBLE PERFORMANCE:")
        print(f"  Pixel Accuracy (PA): {PA:.4f}")
        print(f"  Class Pixel Accuracy (CPA): {CPA:.4f}")
        print(f"  Mean IoU (MIoU): {MIoU:.4f}")
        print(f"{'='*80}")
        
        return PA, CPA, MIoU, total_conf_matrix

# ================================================================================================
# MAIN EXECUTION FUNCTION
# ================================================================================================

def evaluate_ensemble(
    checkpoint_paths: List[str],
    dataset_path: str,
    mode: str,
    num_classes: int = 9,
    n_out_channels: int = 16,
    batch_size: int = 16
):
    """
    Main function to evaluate ensemble of 3 UNets
    
    Args:
        checkpoint_paths: List of 3 paths to .pt checkpoint files
        dataset_path: Path to .h5 dataset file  
        mode: "real" or "complex"
        num_classes: Number of segmentation classes
        n_out_channels: Number of output channels for model architecture
        batch_size: Batch size for evaluation
        
    Returns:
        PA, CPA, MIoU: Evaluation metrics
    """
    
    if len(checkpoint_paths) != 3:
        raise ValueError("Exactly 3 checkpoint paths must be provided")
    
    # Verify files exist
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Create evaluator
    evaluator = EnsembleEvaluator()
    
    # Load models
    evaluator.load_models(checkpoint_paths, mode, num_classes, n_out_channels)
    
    # Evaluate dataset
    PA, CPA, MIoU, conf_matrix = evaluator.evaluate_dataset(
        dataset_path, mode, num_classes, batch_size
    )
    
    return PA, CPA, MIoU, conf_matrix

checkpoint_paths = [
    "./realCV/r0_checkpoint_epoch_150.pt",
    "./realCV/r1_checkpoint_epoch_150.pt", 
    "./realCV/r2_checkpoint_epoch_150.pt"
]

dataset_path = "./sassed_V4.h5"
mode = "real"  # or "complex"

# Run evaluation
try:
    PA, CPA, MIoU, conf_matrix = evaluate_ensemble(
        checkpoint_paths=checkpoint_paths,
        dataset_path=dataset_path,
        mode=mode,
        num_classes=9,
        n_out_channels=16,
        batch_size=16
    )

    print(f"\nFinal Results:")
    print(f"PA: {PA:.4f}")
    print(f"CPA: {CPA:.4f}")
    print(f"MIoU: {MIoU:.4f}")

except Exception as e:
    print(f"Error during evaluation: {e}")