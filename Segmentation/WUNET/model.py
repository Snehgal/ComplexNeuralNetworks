import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class WaveletDown(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)

    def forward(self, x):
        Yl, Yh = self.dwt(x)
        return Yl  # Discard high-frequency for now


class WaveletUp(nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.iwt = DWTInverse(mode='zero', wave=wave)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        B, C, H, W = x.shape
        # High-pass coefficients as zeros (3 subbands × C channels)
        Yh = [x.new_zeros((B, C, 3, H, W))]

        # Perform inverse wavelet transform with zeroed high-frequency detail
        x = self.iwt((x, Yh))

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetWavelet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, base_ch=64, wavelet=True, wave='haar'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.wavelet = wavelet
        self.wave = wave

        # Encoder
        self.down1 = DoubleConv(n_channels, base_ch)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.down4 = DoubleConv(base_ch * 4, base_ch * 8)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if wavelet:
            self.pool = WaveletDown(wave=wave)

        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder with wavelet upsampling
        self.up1 = WaveletUp(base_ch * 16 + base_ch * 8, base_ch * 8, wave=wave)
        self.up2 = WaveletUp(base_ch * 8 + base_ch * 4, base_ch * 4, wave=wave)
        self.up3 = WaveletUp(base_ch * 4 + base_ch * 2, base_ch * 2, wave=wave)
        self.up4 = WaveletUp(base_ch * 2 + base_ch, base_ch, wave=wave)

        self.out_conv = nn.Conv2d(base_ch, n_classes, kernel_size=1)

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

        return self.out_conv(x)

def testUnet(complex = False, out_ch = 64):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if complex:
        real = torch.randn(2, 3, 128, 128)
        imag = torch.randn(2, 3, 128, 128)
        dummy_input = torch.complex(real, imag)
        model = ComplexUNet(n_channels=3, n_classes=5,n_out_channels=out_ch).to(device)
    else:
        dummy_input = torch.randn(2, 3, 128, 128)
        model = UNetWavelet(n_channels=3, n_classes=5,base_ch=out_ch).to(device)



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
