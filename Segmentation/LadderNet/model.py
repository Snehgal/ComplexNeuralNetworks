import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexConvTranspose2d,ComplexLinear,ComplexMaxPool2d,ComplexAvgPool2d
from complexPyTorch.complexFunctions import complex_relu
from torchsummary import summary
from torchvision import models
from torch.nn.functional import relu
from torchsummary import summary
import torch.nn.functional as F

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

class Up2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up2, self).__init__()
        # Always use ConvTranspose2d for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # If needed, pad x1 to match x2's size
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='nearest')
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

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
        self.pool = PseudoComplexAvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class ComplexUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = ComplexConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ComplexDoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # If needed, pad x1 to match x2's size
        if x1.shape[-2:] != x2.shape[-2:]:
            # Use real-valued interpolation as a fallback (not recommended), or crop
            # Here, we crop x1 to match x2
            diffY = x2.size(-2) - x1.size(-2)
            diffX = x2.size(-1) - x1.size(-1)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

#DISCLAIMER: Works only in very specific input dimensions cases because of UpSample instead of ConvTranspose2d
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
        
        self.seg_out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)      # [B, 64, H, W]
        x2 = self.down2(x1) # [B, 128, H/2, W/2]
        x3 = self.down3(x2) # [B, 256, H/4, W/4]
        x4 = self.down4(x3) # [B, 512, H/8, W/8]
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
        x_seg_out = self.seg_out_conv(x_seg4)

        return x_classification_output, x_seg_out, x_rec_output

class LadderNetEQ(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_classes=6):
        super(LadderNetEQ, self).__init__()
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classify = nn.Linear(1024, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = Up2(1024, 512, 512)
        self.up2 = Up2(512, 256, 256)
        self.up3 = Up2(256, 128, 128)
        self.up4 = Up2(128, 64, in_channels)

        self.seg1 = Up2(1024, 512, 512)
        self.seg2 = Up2(512, 256, 256)
        self.seg3 = Up2(256, 128, 128)
        self.seg4 = Up2(128, 64, 64)

        self.rec_out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.seg_out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x_bottleneck = self.bottleneck(self.maxpool(x4))

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
        if x_rec_output.shape[-2:] != x.shape[-2:]:
            x_rec_output = F.interpolate(x_rec_output, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Segmentation decoder
        x_seg1 = self.seg1(x_bottleneck, x4)
        x_seg2 = self.seg2(x_seg1, x3)
        x_seg3 = self.seg3(x_seg2, x2)
        x_seg4 = self.seg4(x_seg3, x1)
        x_seg_out = self.seg_out_conv(x_seg4)
        if x_seg_out.shape[-2:] != x.shape[-2:]:
            x_seg_out = F.interpolate(x_seg_out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return x_classification_output, x_seg_out, x_rec_output
    
class ComplexLadderNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,num_classes=6):
        super(ComplexLadderNet, self).__init__()
        self.down1 = ComplexDown(in_channels, 64)
        self.down2 = ComplexDown(64, 128)
        self.down3 = ComplexDown(128, 256)
        self.down4 = ComplexDown(256, 512)
        self.bottleneck = ComplexDoubleConv(512, 1024)
        self.global_pool = PseudoComplexAvgPool2d(output_size=(1, 1))
        self.classify = ComplexLinear(1024, num_classes)  # num_classes = number of image-level classes
        self.maxpool = ComplexMaxPool2d(kernel_size=2, stride=2)
        
        self.up1 = ComplexUp(1024, 512, 512)
        self.up2 = ComplexUp(512, 256, 256)
        self.up3 = ComplexUp(256, 128, 128)
        self.up4 = ComplexUp(128, 64, in_channels)
        
        self.seg1 = ComplexUp(1024, 512, 512)
        self.seg2 = ComplexUp(512, 256, 256)
        self.seg3 = ComplexUp(256, 128, 128)
        self.seg4 = ComplexUp(128, 64, 64)
        
        self.rec_out_conv = ComplexConv2d(in_channels, in_channels, kernel_size=1)
        self.seg_out_conv = ComplexConv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)      # [B, 64, H, W]
        x2 = self.down2(x1)     # [B, 128, H/2, W/2]
        x3 = self.down3(x2)     # [B, 256, H/4, W/4]
        x4 = self.down4(x3)     # [B, 512, H/8, W/8]
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
        x_seg_out = self.seg_out_conv(x_seg4)

        # Ensure output sizes match input size COMPLEX VALUED
        # if x_rec_output.shape[-2:] != x.shape[-2:]:
        #     x_rec_output = F.interpolate(x_rec_output.real, size=x.shape[-2:], mode='bilinear', align_corners=False) + \
        #                 1j * F.interpolate(x_rec_output.imag, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     x_rec_output = torch.complex(x_rec_output.real, x_rec_output.imag)

        # if x_seg_out.shape[-2:] != x.shape[-2:]:
        #     x_seg_out = F.interpolate(x_seg_out.real, size=x.shape[-2:], mode='bilinear', align_corners=False) + \
        #                 1j * F.interpolate(x_seg_out.imag, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     x_seg_out = torch.complex(x_seg_out.real, x_seg_out.imag)
        
        # Ensure output sizes match input size ONLY REAL PART
        if x_rec_output.shape[-2:] != x.shape[-2:]:
            x_rec_output = F.interpolate(x_rec_output.real, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if x_seg_out.shape[-2:] != x.shape[-2:]:
            x_seg_out = F.interpolate(x_seg_out.real, size=x.shape[-2:], mode='bilinear', align_corners=False)
    
        return x_classification_output, x_seg_out, x_rec_output

def testNormal(onnx = False):
    import torch

    dummy_input = torch.randn(2, 3, 256, 256)
    model = LadderNet()
    model.train()

    # Forward pass
    x_classification_output, x_seg_out, x_rec_output = model(dummy_input)

    # Example targets
    classification_target = torch.randint(0, 6, (2,))
    segmentation_target = torch.randint(0, 1, (2, 1, 256, 256))
    reconstruction_target = dummy_input
    print("Input shape:", dummy_input.shape)
    print("Classification output shape:", x_classification_output.shape)
    print("Segmentation output shape:", x_seg_out.shape)
    print("Reconstruction output shape:", x_rec_output.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Losses
    classification_loss = torch.nn.CrossEntropyLoss()(x_classification_output, classification_target)
    segmentation_loss = torch.nn.BCEWithLogitsLoss()(x_seg_out, segmentation_target.float())
    reconstruction_loss = torch.nn.MSELoss()(x_rec_output, reconstruction_target)

    # Total loss and backward
    loss = classification_loss + segmentation_loss + reconstruction_loss
    loss.backward()
    
    print("Forward and backward pass completed for LadderNet.")

    if onnx:
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(model, dummy_input, "laddernet.onnx", opset_version=11)
def testNormalEQ(onnx = False):
    import torch

    dummy_input = torch.randn(2, 3, 128, 128)
    model = LadderNetEQ()
    model.train()

    # Forward pass
    x_classification_output, x_seg_out, x_rec_output = model(dummy_input)

    # Example targets
    classification_target = torch.randint(0, 6, (2,))
    segmentation_target = torch.randint(0, 1, (2, 1, 128, 128))
    reconstruction_target = dummy_input

    # Losses
    classification_loss = torch.nn.CrossEntropyLoss()(x_classification_output, classification_target)
    segmentation_loss = torch.nn.BCEWithLogitsLoss()(x_seg_out, segmentation_target.float())
    reconstruction_loss = torch.nn.MSELoss()(x_rec_output, reconstruction_target)

    # Total loss and backward
    loss = classification_loss + segmentation_loss + reconstruction_loss
    loss.backward()
    print("Input shape:", dummy_input.shape)
    print("Classification output shape:", x_classification_output.shape)
    print("Segmentation output shape:", x_seg_out.shape)
    print("Reconstruction output shape:", x_rec_output.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("Forward and backward pass completed for LadderNetEQ.")

    if onnx:
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(model, dummy_input, "laddernet_eq.onnx", opset_version=11)
def testComplex():
    import torch

    real = torch.randn(2, 3, 128, 128)
    imag = torch.randn(2, 3, 128, 128)
    complex_input = torch.complex(real, imag)

    model = ComplexLadderNet()
    model.train()

    # Forward pass
    x_classification_output, x_seg_out, x_rec_output = model(complex_input)

    # Example targets
    classification_target = torch.randint(0, 6, (2,))
    segmentation_target = torch.randint(0, 1, (2, 1, 128, 128))
    reconstruction_target = complex_input

    # Losses (real part only)
    classification_loss = torch.nn.CrossEntropyLoss()(x_classification_output.real, classification_target)
    segmentation_loss = torch.nn.BCEWithLogitsLoss()(x_seg_out.real, segmentation_target.float())
    reconstruction_loss = torch.nn.MSELoss()(x_rec_output.real, reconstruction_target.real)

    # Total loss and backward
    loss = classification_loss + segmentation_loss + reconstruction_loss
    loss.backward()
    print("Input shape:", complex_input.shape)
    print("Classification output shape:", x_classification_output.shape)
    print("Segmentation output shape:", x_seg_out.shape)
    print("Reconstruction output shape:", x_rec_output.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("Forward and backward pass completed for ComplexLadderNet.")

# testNormal() #preferably dont use
testNormalEQ()
testComplex()