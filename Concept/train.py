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
                val_split=0.1, path=None, complex_data=False, resume=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_instance.to(device)
    print(f"\nTraining {model_name} on {dataset.upper()} ({'Complex' if complex_data else 'Real'}) using {device}")

    # Create directory to save checkpoints
    save_dir = path or f"{model_name}_{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0

    # === Resume from checkpoint ===
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # === Data Loaders ===
    train_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                  batch_size=batch_size, split="train", val_split=val_split)
    val_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                batch_size=batch_size, split="val", val_split=val_split)
    test_loader = get_dataloader(dataset_type=dataset, complex_data=complex_data,
                                 batch_size=batch_size, split="test")

    # === Training Loop ===
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if torch.is_complex(outputs):  # if output is complex, convert to magnitude
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

        print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # === Save checkpoint ===
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, checkpoint_path)

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
    print(f"\n Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

