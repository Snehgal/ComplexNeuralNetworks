import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexMaxPool2d,ComplexAvgPool2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
import math

# LENET
class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Placeholder: will be defined after we see an input
        self.flatten_dim = None

        self.fc1 = None  # lazy init
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self.flatten_dim is None:
            self.flatten_dim = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(self.flatten_dim, 120)
            # Move to same device
            self.fc1.to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet2x(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet2x, self).__init__()

        # √2 scaling
        conv1_out = round(6 * math.sqrt(2))    # ~8
        conv2_out = round(16 * math.sqrt(2))   # ~23
        fc1_out = round(120 * math.sqrt(2))    # ~170
        fc2_out = round(84 * math.sqrt(2))     # ~119

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=5)

        # Lazy FC layer initialization
        self.flatten_dim = None
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.num_classes = num_classes

        self.fc1 = None  # initialized in first forward pass
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self.flatten_dim is None:
            self.flatten_dim = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(self.flatten_dim, self.fc1_out).to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ComplexLeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ComplexLeNet, self).__init__()

        # Complex Conv layers
        self.conv1 = ComplexConv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.pool1 = ComplexMaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ComplexConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = ComplexMaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (lazy init)
        self.flatten_dim = None
        self.fc1 = None
        self.fc1_out = 120
        self.fc2 = ComplexLinear(self.fc1_out, 84)
        self.fc3 = ComplexLinear(84, num_classes)

    def forward(self, x):
        # Ensure input is complex
        x = x.to(torch.complex64) if not x.is_complex() else x

        x = self.pool1(complex_relu(self.conv1(x)))
        x = self.pool2(complex_relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        if self.flatten_dim is None:
            self.flatten_dim = x.size(1)
            self.fc1 = ComplexLinear(self.flatten_dim, self.fc1_out).to(x.device)

        x = complex_relu(self.fc1(x))
        x = complex_relu(self.fc2(x))
        x = self.fc3(x)
        return x.real  # Return real logits for classification




# CUSTOM CNN
class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomCNN, self).__init__()

        # Conv layers (now using in_channels)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Adaptive avg pool will reduce to (1,1) no matter the input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)  # Output shape: (B, 512, 1, 1)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class CustomCNN2x(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomCNN2x, self).__init__()

        # Conv layers scaled by √2 (~1.414)
        self.conv1 = nn.Conv2d(in_channels, 90, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(90)

        self.conv2 = nn.Conv2d(90, 181, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(181)

        self.conv3 = nn.Conv2d(181, 362, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(362)

        self.conv4 = nn.Conv2d(362, 724, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(724)

        # Adaptive pooling handles variable image sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(724 * 1 * 1, 2880)
        self.fc2 = nn.Linear(2880, 720)
        self.fc3 = nn.Linear(720, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ComplexCustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):  # ✅ added in_channels
        super(ComplexCustomCNN, self).__init__()

        # Complex Conv Layers
        self.conv1 = ComplexConv2d(in_channels, 64, kernel_size=3, padding=1)  # ✅ use in_channels
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

        # Pooling (still functional is fine)
        self.pool = ComplexMaxPool2d(2)

    def forward(self, x):
        # Ensure complex input
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
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = complex_relu(self.fc1(x))
        x = complex_relu(self.fc2(x))
        x = self.fc3(x)

        return x.real  # Return logits (real part)


def count_parameters(model,p=True):
    total = sum(p.numel() for p in model.parameters())
    if p:
        print(f'Total parameters: {total:,}')
    return total