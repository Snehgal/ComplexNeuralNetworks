from torchsummary import summary

import resnet as m
# models for FashionMNIST
R18_FMNIST = m.ResNet18()
R18x2_FMNIST = m.ResNet18x2()
iR18_FMNIST = m.ComplexResNet18()
print("ResNet18 Summary")
summary(R18_FMNIST, (1, 28, 28))
print("\nResNet18x2 Summary")
summary(R18x2_FMNIST, (1, 28, 28))
print("\nComplex ResNet18 Summary")
m.get_detailed_summary(iR18_FMNIST, (2, 28, 28))