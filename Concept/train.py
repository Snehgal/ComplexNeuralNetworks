import os

from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.notebook import tqdm
import resnet as res
import model as mod
from dataloader import get_dataloader

# model name(just for printing) and model instance needs to be created while passing params to the function
def train_model(model_name, model_instance, epochs, learning_rate, dataset="fashion", batch_size=64,
                val_split=0.1, path=None, complex_data=False, resume=False,
                plot=True, plot_every=2, optimizer_type="adam", momentum=0.9, T_max=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_instance.to(device)
    # === Directory Setup ===
    save_dir = path or f"{model_name}_{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")
    best_checkpoint_path = os.path.join(save_dir, "best_checkpoint.pt")

    # === Optimizer Selection ===
    if optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        print(f"Using SGD optimizer with momentum={momentum}")
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Using Adam optimizer")
    else:
        raise ValueError("Unsupported optimizer type. Choose 'adam' or 'sgd'.")

    loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0

    # === Scheduler Setup ===
    T_max = T_max or epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # === Resume Checkpoint ===
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    print(f"""
    ========== Training Configuration ==========
    Model Name      : {model_name}
    Dataset         : {dataset.upper()} ({'Complex' if complex_data else 'Real'})
    Device          : {device}
    Epochs          : {epochs}
    Batch Size      : {batch_size}
    Learning Rate   : {learning_rate}
    Optimizer       : {optimizer_type.upper()} {'(Momentum: ' + str(momentum) + ')' if optimizer_type.lower() == 'sgd' else ''}
    Scheduler       : CosineAnnealingLR (T_max = {T_max or epochs})
    Validation Split: {val_split}
    Resume Training : {resume}
    Save Directory  : {save_dir}
    ============================================
    """)
    # === Data Loaders ===
    train_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                  batch_size=batch_size, split="train", val_split=val_split)
    val_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                batch_size=batch_size, split="val", val_split=val_split)
    test_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                 batch_size=batch_size, split="test")

    # === History for Plotting ===
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    # === Training Loop ===
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if torch.is_complex(outputs):
                    outputs = torch.abs(outputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step()

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # === Save Checkpoints ===
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss
        }, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, best_checkpoint_path)
            print(f"New best model saved! Val Acc: {val_acc:.4f}")

        # === Plotting ===
        if plot and (epoch + 1) % plot_every == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, epoch + 2), train_losses, label="Train Loss", marker='o')
            plt.plot(range(1, epoch + 2), val_losses, label="Val Loss", marker='o')
            plt.title("Loss vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epoch + 2), train_accs, label="Train Acc", marker='o')
            plt.plot(range(1, epoch + 2), val_accs, label="Val Acc", marker='o')
            plt.title("Accuracy vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    # === Test Evaluation ===
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if torch.is_complex(outputs):
                outputs = torch.abs(outputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= test_total
    test_acc = test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
