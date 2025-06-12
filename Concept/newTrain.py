import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from scipy.signal import hilbert
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import os
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from resnet import PseudoComplexAvgPool2d 

torch.backends.cudnn.benchmark = True

experiment_results = []
# =========================
# Complex Dataset Classes
# =========================
class ComplexFashionMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.fashion_mnist = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.fashion_mnist)

    def __getitem__(self, idx):
        img, label = self.fashion_mnist[idx]
        if self.transform:
            img = self.transform(img)
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()
        img = img.astype(np.float32)
        analytic_signal = hilbert(img, axis=1)
        complex_tensor = torch.from_numpy(analytic_signal).to(torch.complex64).unsqueeze(0)
        return complex_tensor, label

class ComplexCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=None
        )
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        img = self.transform(img)
        img_np = img.numpy()
        complex_channels = []
        for c in range(3):
            real_channel = img_np[c]
            analytic_signal = hilbert(real_channel, axis=1)
            complex_tensor = torch.from_numpy(analytic_signal).to(torch.complex64)
            complex_channels.append(complex_tensor)
        complex_tensor = torch.stack(complex_channels)
        return complex_tensor, label

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.base_lrs]
# =========================
# Dataset registry
# =========================
def get_dataset(name, root, train, transform):
    if name == "CIFAR10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif name == "FashionMNIST":
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    elif name == "ComplexFashionMNIST":
        return ComplexFashionMNIST(root=root, train=train, transform=transform)
    elif name == "ComplexCIFAR10":
        return ComplexCIFAR10(root=root, train=train, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# =========================
# Model registry
# =========================
import resnet as res
import model as mdl

MODEL_REGISTRY = {
    # ResNet18
    "ResNet18-CIFAR10": lambda: res.ResNet18(numClasses=10, inChannels=3),
    "ResNet18-FashionMNIST": lambda: res.ResNet18(numClasses=10, inChannels=1),
    "ResNet18x2-CIFAR10": lambda: res.ResNet18x2(numClasses=10, inChannels=3),
    "ResNet18x2-FashionMNIST": lambda: res.ResNet18x2(numClasses=10, inChannels=1),
    "ComplexResNet18-ComplexCIFAR10": lambda: res.ComplexResNet18(numClasses=10, inChannels=3),
    "ComplexResNet18-ComplexFashionMNIST": lambda: res.ComplexResNet18(numClasses=10, inChannels=1),

    # CustomCNN
    "CustomCNN-CIFAR10": lambda: mdl.CustomCNN(in_channels=3, num_classes=10),
    "CustomCNN-FashionMNIST": lambda: mdl.CustomCNN(in_channels=1, num_classes=10),
    "CustomCNN2x-CIFAR10": lambda: mdl.CustomCNN2x(in_channels=3, num_classes=10),
    "CustomCNN2x-FashionMNIST": lambda: mdl.CustomCNN2x(in_channels=1, num_classes=10),
    "ComplexCustomCNN-ComplexCIFAR10": lambda: mdl.ComplexCustomCNN(in_channels=3, num_classes=10),
    "ComplexCustomCNN-ComplexFashionMNIST": lambda: mdl.ComplexCustomCNN(in_channels=1, num_classes=10),
}

experiments = [
    {
    "model_name": "ComplexResNet18-ComplexFashionMNIST",
    "dataset": "ComplexFashionMNIST",
    "save_path": "best_complexresnet18_fashionmnist.pth",
    "hyperparams": {
        "batch_size": 128,
        "epochs": 200,
        "learning_rate": 0.1,
        "warmup_lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9,
        "val_split": 0.1,
        "warmup_epochs": 5
    }
},
    # ResNet18
    {
        "model_name": "ResNet18-CIFAR10",
        "dataset": "CIFAR10",
        "save_path": "best_resnet18_cifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ResNet18-FashionMNIST",
        "dataset": "FashionMNIST",
        "save_path": "best_resnet18_fashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ResNet18x2-CIFAR10",
        "dataset": "CIFAR10",
        "save_path": "best_resnet18x2_cifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ResNet18x2-FashionMNIST",
        "dataset": "FashionMNIST",
        "save_path": "best_resnet18x2_fashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ComplexResNet18-ComplexCIFAR10",
        "dataset": "ComplexCIFAR10",
        "save_path": "best_complexresnet18_complexcifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ComplexResNet18-ComplexFashionMNIST",
        "dataset": "ComplexFashionMNIST",
        "save_path": "best_complexresnet18_complexfashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    # CustomCNN
    {
        "model_name": "CustomCNN-CIFAR10",
        "dataset": "CIFAR10",
        "save_path": "best_customcnn_cifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "CustomCNN-FashionMNIST",
        "dataset": "FashionMNIST",
        "save_path": "best_customcnn_fashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "CustomCNN2x-CIFAR10",
        "dataset": "CIFAR10",
        "save_path": "best_customcnn2x_cifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "CustomCNN2x-FashionMNIST",
        "dataset": "FashionMNIST",
        "save_path": "best_customcnn2x_fashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ComplexCustomCNN-ComplexCIFAR10",
        "dataset": "ComplexCIFAR10",
        "save_path": "best_complexcustomcnn_complexcifar10.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 20,
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
    {
        "model_name": "ComplexCustomCNN-ComplexFashionMNIST",
        "dataset": "ComplexFashionMNIST",
        "save_path": "best_complexcustomcnn_complexfashionmnist.pth",
        "hyperparams": {
            "batch_size": 128,
            "epochs": 10,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "val_split": 0.1,
        }
    },
]
# =========================
# Utility functions
# =========================
def print_experiment_header(model_name, save_path, hyperparams, device, dataset):
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model:         {model_name}")
    print(f"Dataset:       {dataset}")
    print(f"Device:        {device}")
    print(f"Batch_size:    {hyperparams['batch_size']}")
    print(f"Epochs:        {hyperparams['epochs']}")
    print(f"Weight_decay:  {hyperparams['weight_decay']}")
    print(f"Momentum:      {hyperparams['momentum']}")
    print(f"Val_split:     {hyperparams['val_split']}")
    # Print transforms used
    if dataset in ["CIFAR10", "ComplexCIFAR10"]:
        transforms_used = "RandAugment, RandomCrop, RandomHorizontalFlip, Normalise"
    elif dataset in ["FashionMNIST", "ComplexFashionMNIST"]:
        transforms_used = "RandomCrop, RandomHorizontalFlip, Normalise"
    else:
        transforms_used = "Unknown"
    print(f"Transforms:    {transforms_used}")
    print(f"AMP:           {'Enabled' if torch.cuda.is_available() else 'Disabled'}")
    print(f"Optimiser:     SGD")
    # Scheduler details
    if 'warmup_epochs' in hyperparams and hyperparams['warmup_epochs'] > 0:
        print(f"Scheduler:     Warmup ({hyperparams['warmup_epochs']} epochs, Linear {hyperparams.get('warmup_lr', 0.01)} → {hyperparams['learning_rate']}) + CosineAnnealingLR ({hyperparams['epochs']-hyperparams['warmup_epochs']} epochs)")
    else:
        print(f"Scheduler:     CosineAnnealingLR (T_max={hyperparams['epochs']})")
    print(f"Loss:          CrossEntropyLoss")
    print(f"Best model save: {save_path}")
    # Data sizes
    trainloader, valloader, testloader = get_data_loaders(dataset, hyperparams["batch_size"], hyperparams["val_split"])
    print(f"Train Samples: {len(trainloader.dataset)}")
    print(f"Val Samples:   {len(valloader.dataset)}")
    print(f"Test Samples:  {len(testloader.dataset)}")
    print("="*60)
    print()

def get_data_loaders(dataset_name, batch_size, val_split):
    if dataset_name in ["CIFAR10", "ComplexCIFAR10"]:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name in ["FashionMNIST", "ComplexFashionMNIST"]:
        mean, std = (0.2860,), (0.3530,)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    root = './data'
    full_trainset = get_dataset(dataset_name, root, train=True, transform=transform_train)
    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    testset = get_dataset(dataset_name, root, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    return trainloader, valloader, testloader

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Acc')
    plt.plot(epochs_range, val_accuracies, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE of Validation Set")
    plt.show()

def evaluate(model, loader, criterion, device, name="Validation"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if torch.is_complex(outputs):
                outputs_for_loss = outputs.real
            else:
                outputs_for_loss = outputs
            loss = criterion(outputs_for_loss, targets)
            running_loss += loss.item()  # <-- FIXED
            _, predicted = outputs_for_loss.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    print(f"🔎 {name} Loss: {avg_loss:.4f} | {name} Acc: {acc:.2f}%")
    return avg_loss, acc

def extract_customcnn_features(model, inputs):
    # Detect if model is pseudo-complex by checking the type of global_pool
    is_pseudo_complex = hasattr(model, 'global_pool') and isinstance(model.global_pool, PseudoComplexAvgPool2d)
    relu_fn = complex_relu if is_pseudo_complex else F.relu

    x = relu_fn(model.bn1(model.conv1(inputs)))
    x = F.max_pool2d(x, 2) if not is_pseudo_complex else complex_max_pool2d(x, kernel_size=2)
    x = relu_fn(model.bn2(model.conv2(x)))
    x = F.max_pool2d(x, 2) if not is_pseudo_complex else complex_max_pool2d(x, kernel_size=2)
    feats = relu_fn(model.bn3(model.conv3(x)))
    feats = torch.flatten(feats, 1)
    return feats

def run_experiment(model_name, save_path, hyperparams, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_experiment_header(model_name, save_path, hyperparams, device, dataset_name)
    trainloader, valloader, testloader = get_data_loaders(dataset_name, hyperparams["batch_size"], hyperparams["val_split"])
    model = MODEL_REGISTRY[model_name]().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"]
    )
    epochs = hyperparams["epochs"]
    warmup_epochs = hyperparams.get("warmup_epochs", 0)
    base_lr = hyperparams.get("warmup_lr", 0.01)
    max_lr = hyperparams["learning_rate"]

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        base_lr=base_lr,
        max_lr=max_lr
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_test_acc = 0.0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, hyperparams["epochs"] + 1):
        # Train
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                if torch.is_complex(outputs):
                    outputs_for_loss = outputs.real  # Use real part only
                else:
                    outputs_for_loss = outputs
                loss = criterion(outputs_for_loss, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, predicted = outputs_for_loss.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation
        val_loss, val_acc = evaluate(model, valloader, criterion, device, name="Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
        scheduler.step()
        if epoch % 10 == 0 or epoch == hyperparams["epochs"]:
            _, test_acc = evaluate(model, testloader, criterion, device, name="Test")
            if test_acc > best_test_acc:
                best_test_acc = test_acc

    # Final test accuracy with best model
    model.load_state_dict(torch.load(save_path))
    _, final_test_acc = evaluate(model, testloader, criterion, device, name="Test")
    if final_test_acc > best_test_acc:
        best_test_acc = final_test_acc

    print(f"\n🏁 Best Validation Accuracy: {best_val_acc:.2f}%\n")
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, hyperparams["epochs"])
    
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs = inputs.to(device)
            # For ResNet-like models (has layer1)
            if hasattr(model, 'layer1'):
                x = model.conv1(inputs)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                feats = x
            # For CustomCNN and variants (has conv3)
            elif hasattr(model, 'conv3'):
                feats = extract_customcnn_features(model, inputs)
            else:
                # Fallback: flatten input
                feats = inputs.view(inputs.size(0), -1)
            if torch.is_complex(feats):
                feats = feats.real
            features.append(feats.cpu().numpy())
            labels.append(targets.cpu().numpy())
    # Before calling plot_tsne -> prurely a time optimisation step, will prolly give worse results faster
    if len(features) > 1000:
        features_all = np.concatenate(features, axis=0)
        labels_all = np.concatenate(labels, axis=0)
        idx = np.random.choice(len(features_all), 1000, replace=False)
        features_sub = features_all[idx]
        labels_sub = labels_all[idx]
    else:
        features_sub = np.concatenate(features, axis=0)
        labels_sub = np.concatenate(labels, axis=0)
    plot_tsne(features_sub, labels_sub)

    # Store results for summary
    experiment_results.append({
        "Model": model_name,
        "Dataset": dataset_name,
        "Best Train Acc": max(train_accuracies),
        "Best Val Acc": best_val_acc,
        "Best Test Acc": best_test_acc,
        "Save Path": save_path
    })


    from torchviz import make_dot
    import os

    os.makedirs(save_dir, exist_ok=True)

    default_shapes = {
        "CIFAR10": (1, 3, 32, 32),
        "FashionMNIST": (1, 1, 28, 28),
        "ComplexCIFAR10": (1, 3, 32, 32),
        "ComplexFashionMNIST": (1, 1, 28, 28),
    }
    for model_name, model_fn in MODEL_REGISTRY.items():
        for key in default_shapes:
            if key in model_name:
                input_shape = default_shapes[key]
                break
        else:
            input_shape = (1, 3, 32, 32)
        if input_shapes and model_name in input_shapes:
            input_shape = input_shapes[model_name]
        model = model_fn()
        model.eval()
        dummy_input = torch.randn(*input_shape)
        try:
            output = model(dummy_input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_complex(output):
                output = output.real
            dot = make_dot(output, params=dict(model.named_parameters()))
            dot.format = "png"
            dot.render(os.path.join(save_dir, f"{model_name}_graph"), cleanup=True)
            print(f"Saved graph for {model_name} to {save_dir}")
        except Exception as e:
            print(f"Could not visualize {model_name}: {e}")
    """
    Visualize all models in MODEL_REGISTRY using torchviz.
    Args:
        input_shapes (dict): Optional dict mapping model_name to input shape tuple.
        save_dir (str): Directory to save the generated graphs.
    """
    from torchviz import make_dot
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Default input shapes if not provided
    default_shapes = {
        "CIFAR10": (1, 3, 32, 32),
        "FashionMNIST": (1, 1, 28, 28),
        "ComplexCIFAR10": (1, 3, 32, 32),
        "ComplexFashionMNIST": (1, 1, 28, 28),
    }
    for model_name, model_fn in MODEL_REGISTRY.items():
        # Infer dataset type from model name
        for key in default_shapes:
            if key in model_name:
                input_shape = default_shapes[key]
                break
        else:
            input_shape = (1, 3, 32, 32)
        if input_shapes and model_name in input_shapes:
            input_shape = input_shapes[model_name]
        model = model_fn()
        model.eval()
        dummy_input = torch.randn(*input_shape)
        try:
            output = model(dummy_input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_complex(output):
                output = output.real
            dot = make_dot(output, params=dict(model.named_parameters()))
            dot.format = "png"
            dot.render(os.path.join(save_dir, f"{model_name}_graph"), cleanup=True)
            print(f"Saved graph for {model_name} to {save_dir}")
        except Exception as e:
            print(f"Could not visualize {model_name}: {e}")
# =========================
# Main loop for all experiments
# =========================
if __name__ == "__main__":
    for exp in experiments:
        run_experiment(exp["model_name"], exp["save_path"], exp["hyperparams"], exp["dataset"])

    # Print summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"{'Model':30} {'Dataset':20} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Save Path'}")
    print("-"*80)
    for res in experiment_results:
        print(f"{res['Model']:30} {res['Dataset']:20} {res['Best Train Acc']:10.2f} {res['Best Val Acc']:10.2f} {res['Best Test Acc']:10.2f} {res['Save Path']}")
    print("="*80)
