import torch
from torch.autograd.function import _SingleLevelFunction
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexBatchNorm2d,ComplexReLU,ComplexMaxPool2d,ComplexAvgPool2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_avg_pool2d
import math
from model import count_parameters

class PseudoComplexAvgPool2d(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        real = self.pool(x.real)
        imag = self.pool(x.imag)
        return torch.complex(real, imag)

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
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print(f"Model is Orthogonally initialized")


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

# def ResNet18x2(numClasses=10, inChannels=1):
#     """ResNet18 with approximately 4x parameters using 2 width multiplier"""
#     return ResNet(BasicBlock, [2, 2, 2, 2], inChannels, numClasses, width_multiplier=2)

def ResNet18x2(numClasses=10, inChannels=1):
    """ResNet18 with approximately 2x parameters using sqrt(2) width multiplier"""
    return ResNet(BasicBlock, [2, 2, 2, 2], inChannels, numClasses, width_multiplier=math.sqrt(2))
    
# Basic Block for ComplexResNet18,34
class ComplexBasicBlock(nn.Module):
    expansion = 1  # No expansion in BasicBlock
    
    def __init__(self, inChannels, outChannels, stride=1, downsample=None):
        super(ComplexBasicBlock, self).__init__()
        self.conv1 = ComplexConv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(outChannels)

        self.conv2 = ComplexConv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(outChannels)
        self.relu = ComplexReLU()
        self.downsample = downsample  # This helps adjust dimensions when needed

    def forward(self, x):
        residual = x  # Store input for residual connection

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add residual connection
        out = self.relu(out)

        return out

def complex_kaiming_normal_(tensor, mode='fan_in'):
    """Kaiming initialization for complex weights"""
    nn.init.kaiming_normal_(tensor.real, mode=mode)
    nn.init.kaiming_normal_(tensor.imag, mode=mode)

class ComplexResNet(nn.Module):
    def __init__(self, block, layers, inChannels=1, numClasses=1000, width_multiplier=1):
        super(ComplexResNet, self).__init__()
        
        wm = width_multiplier
        initial_channels = int(64 * wm)

        self.conv1 = nn.Sequential(
            ComplexConv2d(inChannels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            ComplexBatchNorm2d(initial_channels),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inChannels = initial_channels

        self.layer1 = self.makeLayer(block, int(64 * wm), layers[0], stride=1)
        self.layer2 = self.makeLayer(block, int(128 * wm), layers[1], stride=2)
        self.layer3 = self.makeLayer(block, int(256 * wm), layers[2], stride=2)
        self.layer4 = self.makeLayer(block, int(512 * wm), layers[3], stride=2)

        self.avgpool = PseudoComplexAvgPool2d()
        self.fc = ComplexLinear(int(512 * block.expansion * wm), numClasses)

        self.initialiseWeights()

    def initialiseWeights(self):
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                nn.init.orthogonal_(m.conv_r.weight)
                nn.init.orthogonal_(m.conv_i.weight)
                if m.conv_r.bias is not None:
                    nn.init.constant_(m.conv_r.bias, 0)
                    nn.init.constant_(m.conv_i.bias, 0)
                    
            elif isinstance(m, ComplexBatchNorm2d):
                # Initialize batch norm parameters
                if hasattr(m, 'bn_r') and hasattr(m, 'bn_i'):
                    nn.init.constant_(m.bn_r.weight, 1)
                    nn.init.constant_(m.bn_i.weight, 1)
                    nn.init.constant_(m.bn_r.bias, 0)
                    nn.init.constant_(m.bn_i.bias, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        print(f"Model is Orthogonally initialized")

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

def ComplexResNet18(numClasses = 10,inChannels = 2):
     return ComplexResNet(ComplexBasicBlock, [2, 2, 2, 2], inChannels, numClasses)
 
def get_model_summary(model, input_size=(2, 224, 224), device='cpu'):
    """
    Get a detailed summary of a model (e.g., ComplexResNet) that accepts complex inputs.
    The input should have 2 channels: real and imaginary parts.

    Args:
        model (torch.nn.Module): The model to summarize.
        input_size (tuple): Shape of the input (channels, height, width), where channels should be 2 for complex input.
        device (str): Device to run the model on.
    """
    model.eval()
    model = model.to(device)

    c, h, w = input_size
    assert c == 2, "Input must have 2 channels (real and imaginary) for complex models."

    # Create complex dummy input
    real = torch.randn(1, 1, h, w, dtype=torch.float32).to(device)
    imag = torch.randn(1, 1, h, w, dtype=torch.float32).to(device)
    dummy_input = torch.complex(real, imag)  # shape: (1, 1, H, W), dtype: complex64

    print("=" * 80)
    print(f"{'Layer (type)':>25} {'Output Shape':>25} {'Param #':>15}")
    print("=" * 80)

    total_params = 0
    trainable_params = 0
    summary = {}
    hooks = []

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params, trainable_params

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {}

            # Get output shape
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [list(o.size()) for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())

            # Count parameters
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[m_key]["nb_params"] = params
            total_params += params
            trainable_params += params

            # Print line
            output_shape = summary[m_key]["output_shape"]
            if isinstance(output_shape[0], list):
                shape_str = str(output_shape)
            else:
                shape_str = str(output_shape).replace('1, ', '-1, ')
            print(f"{m_key:>25} {shape_str:>25} {params:>15,}")

        if not isinstance(module, nn.Sequential) and not isinstance(module, ComplexResNet) and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook))

    # Register hooks
    model.apply(register_hook)

    # Run forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)

        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {count_parameters(model):,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("-" * 80)

        # Estimate memory usage
        input_size_mb = dummy_input.numel() * 8 / (1024**2)  # complex64 = 8 bytes
        param_size_mb = total_params * 8 / (1024**2)  # assuming complex params
        print(f"Input size (MB): {input_size_mb:.2f}")
        print(f"Params size (MB): {param_size_mb:.2f}")
        print(f"Output shape: {list(output.shape)}")
        print(f"Output dtype: {output.dtype}")
        print("=" * 80)

    except Exception as e:
        print(f"Error during forward pass: {e}")

    finally:
        for hook in hooks:
            hook.remove()
