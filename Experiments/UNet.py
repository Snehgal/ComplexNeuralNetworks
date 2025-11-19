import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.functional import relu


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
