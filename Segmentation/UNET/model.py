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
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.down1 = ComplexDoubleConv(n_channels, 64)
        self.down2 = ComplexDoubleConv(64, 128)
        self.down3 = ComplexDoubleConv(128, 256)
        self.down4 = ComplexDoubleConv(256, 512)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.bottleneck = ComplexDoubleConv(512, 1024)
        self.up1 = ComplexUp(1024, 512)
        self.up2 = ComplexUp(512, 256)
        self.up3 = ComplexUp(256, 128)
        self.up4 = ComplexUp(128, 64)
        self.out_conv = ComplexConv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x = self.pool(x1)
        x2 = self.down2(x)
        x = self.pool(x2)
        x3 = self.down3(x)
        x = self.pool(x3)
        x4 = self.down4(x)
        x = self.bottleneck(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)


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
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x3 = self.down3(x)
        x = self.maxpool(x3)
        x4 = self.down4(x)
        x = self.maxpool(x4)
        x = self.bottleneck(x)
        x = self.up1(x,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.out_conv(x)
        return x

model = ComplexUNet()
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")