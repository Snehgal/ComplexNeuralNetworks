import os

from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.notebook import tqdm
from train import train_model

import resnet as res
import model as mod
from dataloader import get_dataloader

# === Custom CNN Models ===
custom_models = {
    # Custom CNNs
    "LeNet": mod.LeNet(),
    "LeNet2x": mod.LeNet2x(),
    "ComplexLeNet": mod.ComplexLeNet(),
    "CustomCNN": mod.CustomCNN(),
    "CustomCNN2x": mod.CustomCNN2x(),
    "ComplexCustomCNN": mod.ComplexCustomCNN(),
    # ResNet variants
    "ResNet18": res.ResNet18(),
    "ResNet18x2": res.ResNet18x2(),
    "ComplexResNet18": res.ComplexResNet18(inChannels=1)
}
    
def modelSizes():
    # === ResNet Models for FashionMNIST ===
    print("ResNet18 Summary")
    resnet18 = res.ResNet18()
    summary(resnet18, (1, 28, 28))

    print("\nResNet18x2 Summary")
    resnet18x2 = res.ResNet18x2()
    summary(resnet18x2, (1, 28, 28))

    print("\nComplex ResNet18 Summary")
    complex_resnet18 = res.ComplexResNet18(inChannels=1)
    res.get_model_summary(complex_resnet18, (2, 28, 28))

    # Display parameter counts for each model
    print("\nModel Parameter Counts:")
    for name, model_instance in custom_models.items():
        params = mod.count_parameters(model_instance,False)
        print(f"\n{name}: {params:,} parameters")


def show_images(images, title, is_complex=False):
    num_samples = min(5, images.shape[0])
    rows = 2 if is_complex else 1
    plt.figure(figsize=(num_samples * 2.4, 2.5 * rows))

    for i in range(num_samples):
        img = images[i]

        if is_complex:
            if img.ndim == 3 and img.shape[0] == 2:
                # Fashion Complex: (2, H, W)
                real = img[0]
                imag = img[1]
                magnitude = torch.log1p((real**2 + imag**2).sqrt())
                phase = torch.atan2(imag, real)

                # Magnitude
                plt.subplot(rows, num_samples, i + 1)
                plt.imshow(magnitude, cmap='gray')
                plt.title(f"Mag {i+1}")
                plt.axis('off')

                # Phase
                plt.subplot(rows, num_samples, i + 1 + num_samples)
                plt.imshow(phase, cmap='twilight_shifted')
                plt.title(f"Phase {i+1}")
                plt.axis('off')

            elif img.ndim == 4 and img.shape[0] == 2:
                # CIFAR Complex: (2, 3, H, W)
                real = img[0]
                imag = img[1]
                magnitude = torch.log1p((real**2 + imag**2).sqrt())
                phase = torch.atan2(imag, real)

                # Magnitude
                plt.subplot(rows, num_samples, i + 1)
                rgb_mag = magnitude.permute(1, 2, 0).clamp(0, 1)
                plt.imshow(rgb_mag)
                plt.title(f"Mag {i+1}")
                plt.axis('off')

                # Phase
                plt.subplot(rows, num_samples, i + 1 + num_samples)
                rgb_phase = phase.permute(1, 2, 0).clamp(0, 1)
                plt.imshow(rgb_phase, cmap='twilight_shifted')
                plt.title(f"Phase {i+1}")
                plt.axis('off')
        else:
            plt.subplot(1, num_samples, i + 1)
            if images.shape[1] == 3:
                # CIFAR Real
                img = images[i].permute(1, 2, 0)
                plt.imshow(img)
            else:
                # Fashion Real
                plt.imshow(images[i][0], cmap='gray')
            plt.title(f"Image {i+1}")
            plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# example
# model = mod.LeNet2x(in_channels=1)  # or 3 for CIFAR
# train_model("LeNet2x", model, epochs=1, learning_rate=0.01, dataset="fashion", complex_data=False)

print(mod.count_parameters(res.FashionResNet18(), False))
