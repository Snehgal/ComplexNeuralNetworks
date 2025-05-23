import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexMaxPool2d,ComplexAvgPool2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import math

# LENET
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet2x(nn.Module):
    def __init__(self):
        super(LeNet2x, self).__init__()
        # √2 scaling
        conv1_out = round(6 * math.sqrt(2))    # 8
        conv2_out = round(16 * math.sqrt(2))   # 23
        fc1_out = round(120 * math.sqrt(2))    # 170
        fc2_out = round(84 * math.sqrt(2))     # 119

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=5)
        self.fc1 = nn.Linear(in_features=conv2_out * 5 * 5, out_features=fc1_out)
        self.fc2 = nn.Linear(in_features=fc1_out, out_features=fc2_out)
        self.fc3 = nn.Linear(in_features=fc2_out, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ComplexLeNet(nn.Module):
    def __init__(self):
        super(ComplexLeNet, self).__init__()

        # Convolutional layers (with doubled channels)
        self.conv1 = ComplexConv2d(in_channels=1, out_channels=6, kernel_size=5)  # 6→12
        self.pool1 = ComplexAvgPool2d(2, 2)
        self.conv2 = ComplexConv2d(in_channels=6, out_channels=16, kernel_size=5)  # 16→32
        self.pool2 = ComplexAvgPool2d(2, 2)

        # Fully connected layers
        self.fc1 = ComplexLinear(in_features=16*5*5, out_features=120)  # 120→240
        self.fc2 = ComplexLinear(in_features=120, out_features=84)     # 84→168
        self.fc3 = ComplexLinear(in_features=84, out_features=10)      # Final output layer (real output optional)

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_relu(x)
        x = self.fc3(x)
        return x






# CUSTOM CNN

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Conv Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Conv Layer 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 4x4 -> 1x1

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class CustomCNN2x(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN2x, self).__init__()

        # Conv layers scaled by ~1.414
        self.conv1 = nn.Conv2d(3, 90, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(90)

        self.conv2 = nn.Conv2d(90, 181, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(181)

        self.conv3 = nn.Conv2d(181, 362, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(362)

        self.conv4 = nn.Conv2d(362, 724, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(724)

        # FC layers adjusted carefully to hit ~8M total params
        self.fc1 = nn.Linear(724 * 1 * 1, 2880)  # scaled input & output
        self.fc2 = nn.Linear(2880, 720)          # smaller FC2 to avoid explosion
        self.fc3 = nn.Linear(720, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ComplexCustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ComplexCustomCNN, self).__init__()

        # Complex Conv Layers
        self.conv1 = ComplexConv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm2d(64)

        self.conv2 = ComplexConv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm2d(128)

        self.conv3 = ComplexConv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm2d(256)

        self.conv4 = ComplexConv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = ComplexBatchNorm2d(512)

        # Fully connected layers
        self.fc1 = ComplexLinear(512 * 1 * 1, 2048)
        self.fc2 = ComplexLinear(2048, 512)
        self.fc3 = ComplexLinear(512, num_classes)

        # Complex MaxPool2d (you can also use functional version)
        self.pool = ComplexMaxPool2d(2)

    def forward(self, x):
        # Convert input to complex (imag part zero)
        x = x.type(torch.complex64)

        # Conv block 1
        x = complex_relu(self.bn1(self.conv1(x)))
        x = complex_max_pool2d(x, kernel_size=2)

        # Conv block 2
        x = complex_relu(self.bn2(self.conv2(x)))
        x = complex_max_pool2d(x, kernel_size=2)

        # Conv block 3
        x = complex_relu(self.bn3(self.conv3(x)))
        x = complex_max_pool2d(x, kernel_size=2)

        # Conv block 4
        x = complex_relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Use real pooling for spatial collapse

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = complex_relu(self.fc1(x))
        x = complex_relu(self.fc2(x))
        x = self.fc3(x)  # No activation on last layer

        # Return real part for classification logits
        return x.real


def count_parameters(model,p=True):
    total = sum(p.numel() for p in model.parameters())
    if p:
        print(f'Total parameters: {total:,}')
    return total
