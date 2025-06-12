import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu
from torchsummary import summary
from torchvision import models
from torch.nn.functional import relu
from torchsummary import summary
import torch.nn.functional as F

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
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # If needed, pad x1 to match x2's size
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='nearest')
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class LadderNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,num_classes=6):
        super(LadderNet, self).__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        self.bottleneck = DoubleConv(512, 1024)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classify = nn.Linear(1024, num_classes)  # num_classes = number of image-level classes
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up1 = Up(in_channels=1024, skip_channels=512, out_channels=512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, in_channels)
        
        self.seg1 = Up(1024, 512, 512)
        self.seg2 = Up(512, 256, 256)
        self.seg3 = Up(256, 128, 128)
        self.seg4 = Up(128, 64, 64)
        
        self.rec_out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)      # [B, 64, H, W]
        x2 = self.down2(self.maxpool(x1))  # [B, 128, H/2, W/2]
        x3 = self.down3(self.maxpool(x2))  # [B, 256, H/4, W/4]
        x4 = self.down4(self.maxpool(x3))  # [B, 512, H/8, W/8]
        x_bottleneck = self.bottleneck(self.maxpool(x4))  # [B, 1024, H/16, W/16]

        # Classification head
        x_class = self.global_pool(x_bottleneck)
        x_class = torch.flatten(x_class, 1)
        x_classification_output = self.classify(x_class)

        # Reconstruction decoder
        x_rec = self.up1(x_bottleneck, x4)
        x_rec = self.up2(x_rec, x3)
        x_rec = self.up3(x_rec, x2)
        x_rec = self.up4(x_rec, x1)
        x_rec_output = self.rec_out_conv(x_rec)
        
        # Segmentation decoder
        x_seg1 = self.seg1(x_bottleneck, x4)
        x_seg2 = self.seg2(x_seg1, x3)
        x_seg3 = self.seg3(x_seg2, x2)
        x_seg4 = self.seg4(x_seg3, x1)
        x_seg_out = self.out_conv(x_seg4)

        return x_classification_output, x_seg_out, x_rec_output

import torch

# Example input: batch size 2, 3 channels, 256x256 image
dummy_input = torch.randn(2, 3, 256, 256)

model = LadderNet()  # Make sure your model is correctly defined and imported
model.eval()         # Set to eval mode (not strictly necessary for shape test)

with torch.no_grad():
    outputs = model(dummy_input)

# If your model returns multiple outputs (classification, segmentation, reconstruction):
x_classification_output, x_seg_out, x_rec_output = outputs

print("Classification output shape:", x_classification_output.shape)  # Should be [2, num_classes]
print("Segmentation output shape:", x_seg_out.shape)                  # Should be [2, num_seg_classes, 256, 256]
print("Reconstruction output shape:", x_rec_output.shape) 

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, "laddernet.onnx", opset_version=11)