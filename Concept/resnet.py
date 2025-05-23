import torch
from torch.autograd.function import _SingleLevelFunction
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import (
    ComplexConv2d, ComplexBatchNorm2d, ComplexLinear
)
from complexPyTorch.complexFunctions import (
    complex_avg_pool2d, complex_max_pool2d, complex_relu
)
import math

# Basic Block for ResNet18,34
class BasicBlock(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, inChannels, outChannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)

        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.downsample = downsample  # This helps adjust dimensions when needed

    def forward(self, x):
        residual = x  # Store input for residual connection

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add residual connection
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, inChannels=1, numClasses=1000, width_multiplier=1):
        super(ResNet, self).__init__()

        wm = width_multiplier  # width multiplier alias
        
        # Apply width multiplier to the initial conv as well
        initial_channels = int(64 * wm)

        # Initial conv now scales with width multiplier
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inChannels = initial_channels

        self.layer1 = self.makeLayer(block, int(64 * wm), layers[0], stride=1)
        self.layer2 = self.makeLayer(block, int(128 * wm), layers[1], stride=2)
        self.layer3 = self.makeLayer(block, int(256 * wm), layers[2], stride=2)
        self.layer4 = self.makeLayer(block, int(512 * wm), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(512 * block.expansion * wm), numClasses)

        self.initialiseWeights()

    def makeLayer(self, block, outChannels, numBlocks, stride=1):
        downsample = None
        if stride != 1 or self.inChannels != outChannels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inChannels, outChannels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels * block.expansion)
            )

        layers = []
        layers.append(block(self.inChannels, outChannels, stride, downsample))
        self.inChannels = outChannels * block.expansion

        for _ in range(1, numBlocks):
            layers.append(block(self.inChannels, outChannels))

        return nn.Sequential(*layers)

    def initialiseWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(numClasses=10, inChannels=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], inChannels, numClasses)

def ResNet18x2(numClasses=10, inChannels=1):
    """ResNet18 with approximately 2x parameters using sqrt(2) width multiplier"""
    return ResNet(BasicBlock, [2, 2, 2, 2], inChannels, numClasses, width_multiplier=math.sqrt(2))
