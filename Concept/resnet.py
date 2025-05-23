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

# Basic Block for ComplexResNet18,34
class ComplexBasicBlock(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, inChannels, outChannels, stride=1, downsample=None):
        super(ComplexBasicBlock, self).__init__()
        self.conv1 = ComplexConv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(outChannels)

        self.conv2 = ComplexConv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(outChannels)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = complex_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = complex_relu(out + residual)
        return out

# ComplexResNet backbone
class ComplexResNet(nn.Module):
    def __init__(self, block, layers, inChannels=2, numClasses=1000, width_multiplier=1):
        super(ComplexResNet, self).__init__()

        self.conv1 = nn.Sequential(
            ComplexConv2d(inChannels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            ComplexBatchNorm2d(64),
        )

        self.inChannels = 64 * width_multiplier
        wm = width_multiplier
        self.layer1 = self.makeLayer(block, int(64 * wm), layers[0], stride=1)
        self.layer2 = self.makeLayer(block, int(128 * wm), layers[1], stride=2)
        self.layer3 = self.makeLayer(block, int(256 * wm), layers[2], stride=2)
        self.layer4 = self.makeLayer(block, int(512 * wm), layers[3], stride=2)

        self.fc = ComplexLinear(512 * block.expansion, numClasses)

        self.initialiseWeights()

    def makeLayer(self, block, outChannels, numBlocks, stride=1):
        downsample = None
        if stride != 1 or self.inChannels != outChannels * block.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(self.inChannels, outChannels * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(outChannels * block.expansion)
            )

        layers = []
        layers.append(block(self.inChannels, outChannels, stride, downsample))
        self.inChannels = outChannels * block.expansion

        for _ in range(1, numBlocks):
            layers.append(block(self.inChannels, outChannels))

        return nn.Sequential(*layers)

    def initialiseWeights(self):
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                # Complex layers typically have separate real and imaginary weight parameters
                if hasattr(m, 'weight_r') and hasattr(m, 'weight_i'):
                    nn.init.kaiming_normal_(m.weight_r, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.weight_i, mode='fan_out', nonlinearity='relu')
                elif hasattr(m, 'weight'):
                    # If weight is a complex tensor, initialize both parts
                    if torch.is_complex(m.weight):
                        nn.init.kaiming_normal_(m.weight.real, mode='fan_out', nonlinearity='relu')
                        nn.init.kaiming_normal_(m.weight.imag, mode='fan_out', nonlinearity='relu')
                    else:
                        # Standard initialization if it's a regular tensor
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        
            elif isinstance(m, ComplexBatchNorm2d):
                # Similar approach for batch norm layers
                if hasattr(m, 'weight_r') and hasattr(m, 'weight_i'):
                    m.weight_r.data.fill_(1)
                    m.weight_i.data.fill_(0)  # Imaginary part typically starts at 0
                    if hasattr(m, 'bias_r') and hasattr(m, 'bias_i'):
                        m.bias_r.data.zero_()
                        m.bias_i.data.zero_()
                elif hasattr(m, 'weight'):
                    if torch.is_complex(m.weight):
                        m.weight.real.data.fill_(1)
                        m.weight.imag.data.fill_(0)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.real.data.zero_()
                            m.bias.imag.data.zero_()
                            
            elif isinstance(m, ComplexLinear):
                # Initialize complex linear layers
                if hasattr(m, 'weight_r') and hasattr(m, 'weight_i'):
                    nn.init.kaiming_normal_(m.weight_r, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.weight_i, mode='fan_out', nonlinearity='relu')
                elif hasattr(m, 'weight'):
                    if torch.is_complex(m.weight):
                        nn.init.kaiming_normal_(m.weight.real, mode='fan_out', nonlinearity='relu')
                        nn.init.kaiming_normal_(m.weight.imag, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        h, w = x.shape[-2:]
        x = complex_avg_pool2d(x, kernel_size=(h, w))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ComplexResNet18(numClasses=10, inChannels=2):
    return ComplexResNet(ComplexBasicBlock, [2, 2, 2, 2], inChannels, numClasses)

def get_detailed_summary(model, input_size=(2, 28, 28)):
    """
    Create a detailed summary in torchsummary format for ComplexResNet
    """
    model.eval()
    
    # Create complex dummy input
    dummy_input = torch.randn(1, *input_size, dtype=torch.complex64)
    
    print("=" * 64)
    print(f"{'Layer (type)':>20} {'Output Shape':>20} {'Param #':>12}")
    print("=" * 64)
    
    layer_count = 1
    total_params = 0
    activations = []
    
    # Hook function to capture layer outputs
    def hook_fn(module, input, output):
        nonlocal layer_count, total_params
        
        # Get module name
        module_name = module.__class__.__name__
        
        # Count parameters
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        
        # Get output shape
        if torch.is_complex(output):
            output_shape = list(output.shape)
        else:
            output_shape = list(output.shape)
        
        # Format output shape
        shape_str = str(output_shape).replace('1, ', '-1, ')
        
        # Print layer info
        layer_type = f"{module_name}-{layer_count}"
        print(f"{layer_type:>20} {shape_str:>20} {params:>12,}")
        
        layer_count += 1
        activations.append(output)
    
    # Register hooks for all modules
    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print("=" * 64)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
        print("-" * 64)
        
        # Calculate memory usage (approximate)
        input_size_mb = (dummy_input.numel() * 8) / (1024**2)  # Complex64 = 8 bytes
        param_size_mb = (total_params * 8) / (1024**2)  # Assuming complex parameters
        
        # Estimate forward pass memory (rough approximation)
        forward_pass_mb = sum(act.numel() * 8 for act in activations[-5:]) / (1024**2)
        
        print(f"Input size (MB): {input_size_mb:.2f}")
        print(f"Forward/backward pass size (MB): {forward_pass_mb:.2f}")
        print(f"Params size (MB): {param_size_mb:.2f}")
        print(f"Estimated Total Size (MB): {input_size_mb + forward_pass_mb + param_size_mb:.2f}")
        print("-" * 64)
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()