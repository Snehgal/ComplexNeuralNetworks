import resnet as m
# models for FashionMNIST
R18_FMNIST = m.ResNet18()
R18x2_FMNIST = m.ResNet18x2()
summary(R18_FMNIST, (1, 28, 28))
summary(R18x2_FMNIST, (1, 28, 28))