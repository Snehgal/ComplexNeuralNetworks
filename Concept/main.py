from torchsummary import summary

import resnet as m
# models for FashionMNIST
R18_FMNIST = m.ResNet18()
R18x2_FMNIST = m.ResNet18x2()
iR18_FMNIST = m.ComplexResNet18()
summary(R18_FMNIST, (1, 28, 28))
summary(R18x2_FMNIST, (1, 28, 28))
m.get_detailed_summary(iR18_FMNIST, (2, 28, 28))