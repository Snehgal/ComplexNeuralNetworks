from torchsummary import summary

import resnet as res
import model as mod

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

# Display parameter counts for each model
print("\nModel Parameter Counts:")
for name, model_instance in custom_models.items():
    params = mod.count_parameters(model_instance,False)
    print(f"\n{name}: {params:,} parameters")
