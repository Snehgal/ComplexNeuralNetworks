from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import resnet as res
import model as mod
from dataloader import RealDataLoader, ComplexDataLoader

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

def trainOnce(dataset="fashion"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n====================Train once on {dataset}===================")
    print(f"\nRunning on {device}")
    real_model_names = {"ResNet18", "ResNet18x2", "LeNet", "LeNet2x", "CustomCNN", "CustomCNN2x"}

    # Update models for CIFAR input channels
    if dataset == "cifar":
        custom_models["ResNet18"] = res.ResNet18(inChannels=3)
        custom_models["ResNet18x2"] = res.ResNet18x2(inChannels=3)
        custom_models["ComplexResNet18"] = res.ComplexResNet18(inChannels=3)
        custom_models["CustomCNN"] = mod.CustomCNN(in_channels=3)
        custom_models["CustomCNN2x"] = mod.CustomCNN2x(in_channels=3)
        custom_models["ComplexCustomCNN"] = mod.ComplexCustomCNN(in_channels=3)
        custom_models["LeNet"] = mod.LeNet(in_channels=3)
        custom_models["LeNet2x"] = mod.LeNet2x(in_channels=3)
        custom_models["ComplexLeNet"] = mod.ComplexLeNet(in_channels=3)
    else:
        custom_models["ResNet18"] = res.ResNet18(inChannels=1)
        custom_models["ResNet18x2"] = res.ResNet18x2(inChannels=1)
        custom_models["ComplexResNet18"] = res.ComplexResNet18(inChannels=1)
        custom_models["CustomCNN"] = mod.CustomCNN(in_channels=1)
        custom_models["CustomCNN2x"] = mod.CustomCNN2x(in_channels=1)
        custom_models["ComplexCustomCNN"] = mod.ComplexCustomCNN(in_channels=1)
        custom_models["LeNet"] = mod.LeNet(in_channels=1)
        custom_models["LeNet2x"] = mod.LeNet2x(in_channels=1)
        custom_models["ComplexLeNet"] = mod.ComplexLeNet(in_channels=1)

    rL = RealDataLoader(dataset=dataset, batchSize=16, shuffle=True)
    cL = ComplexDataLoader(dataset=dataset, batchSize=16, shuffle=True)
    for name, model_instance in custom_models.items():
        print(f"\n{name}:")

        # Choose the correct DataLoader
        if name in real_model_names:
            loader = rL
        else:
            loader = cL

        data_iter = iter(loader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        print(f"Dataset: {dataset}, Input shape: {inputs.shape}")

        model = model_instance.to(device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Prepare input
        if 'Complex' in name:
            if len(inputs.shape) == 5:  # CIFAR (B, 2, 3, H, W)
                B, two, C, H, W = inputs.shape
                x = inputs.reshape(B, two * C, H, W)
                x = torch.complex(x[:, 0::2, :, :], x[:, 1::2, :, :])
            else:  # Fashion (B, 2, H, W)
                x = torch.complex(inputs[:, 0, :, :], inputs[:, 1, :, :]).unsqueeze(1)
        else:
            if len(inputs.shape) == 5:  # Complex CIFAR, real model
                x = inputs[:, 0, :, :, :]  # take only real part
            else:
                x = inputs  # already real (B, 1, H, W) or (B, 3, H, W)

        try:
            outputs = model(x)
            if torch.is_complex(outputs):
                outputs = outputs.abs()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error in {name}: {e}")

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

def showData():
    # Load 1 batch from each type
    real_fashion_loader = RealDataLoader("fashion", batchSize=5, shuffle=False)
    complex_fashion_loader = ComplexDataLoader("fashion", batchSize=5, shuffle=False)
    real_cifar_loader = RealDataLoader("cifar", batchSize=5, shuffle=False)
    complex_cifar_loader = ComplexDataLoader("cifar", batchSize=5, shuffle=False)

    # Get one batch each
    real_fashion_imgs, _ = next(iter(real_fashion_loader))
    complex_fashion_imgs, _ = next(iter(complex_fashion_loader))
    real_cifar_imgs, _ = next(iter(real_cifar_loader))
    complex_cifar_imgs, _ = next(iter(complex_cifar_loader))

    # Show the images
    show_images(real_fashion_imgs, "FashionMNIST (Real)")
    show_images(complex_fashion_imgs, "FashionMNIST (Complex)", is_complex=True)
    show_images(real_cifar_imgs, "CIFAR10 (Real)")
    show_images(complex_cifar_imgs, "CIFAR10 (Complex)", is_complex=True)

# showData()
# modelSizes()
if __name__ == "__main__":
    trainOnce("fashion")
    trainOnce("cifar")