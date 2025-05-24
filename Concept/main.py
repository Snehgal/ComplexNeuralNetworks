from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import resnet as res
import model as mod
from dataloader import ComplexDataLoader

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

def trainOnce():
    dataset = "fashion"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning on {device}")

    # Data
    loader = ComplexDataLoader(dataset = "fashion",batchSize=16, shuffle=True)
    data_iter = iter(loader)
    inputs, targets = next(data_iter)
    inputs, targets = inputs.to(device), targets.to(device)

    # Model Forward-Backward loop
    loss_fn = nn.CrossEntropyLoss()

    print("\n=== One Forward-Backward Iteration Per Model ===")


    for name, model_instance in custom_models.items():
        print(f"\n{name}:")
        model = model_instance.to(device)
        model.train()

        # Prepare optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # For complex inputs: cast to complex dtype
        if inputs.shape[1] == 2 and 'Complex' in name:
            x = torch.complex(inputs[:, 0, :, :], inputs[:, 1, :, :]).unsqueeze(1)  # (B, 1, H, W)
        else:
            x = inputs if inputs.shape[1] == 1 else inputs[:, :1, :, :]  # fallback if input is 2-ch
        # Forward
        try:
            outputs = model(x)
            # =============================
            # Fix for complex-valued output -> to change and decide later
            if torch.is_complex(outputs):
                outputs = outputs.abs()  # or outputs.real
            # =============================
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error in {name}: {e}")
            
trainOnce()