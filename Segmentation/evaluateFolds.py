import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Import your models and functions
from test import UNet, ComplexUNet, compute_metrics, get_confusion_matrix, CustomLoss, SimpleNorm
from updatedDataloader import (
    get_efficient_cross_validation_dataloaders, 
    get_efficient_test_dataloader
)

class CheckpointEvaluator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and extract all information"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            return {
                'epoch': checkpoint.get('epoch', 0),
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'model_state': checkpoint.get('model_state_dict', checkpoint.get('model_state', None))
            }
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
    
    def evaluate_model_on_data(self, model, dataloader, num_classes, criterion):
        """Evaluate model on given dataloader and return all metrics"""
        model.eval()
        total_loss = 0
        conf_matrix = torch.zeros(num_classes, num_classes, device=self.device)
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # Handle size mismatch
                if outputs.shape[-2:] != targets.shape[-2:]:
                    targets = targets[:, :outputs.shape[2], :outputs.shape[3]]
                
                # Calculate loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Update confusion matrix
                conf_matrix += get_confusion_matrix(outputs, targets, num_classes=num_classes)
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        PA, CPA, MIoU = compute_metrics(conf_matrix)
        
        # Handle CPA - check if it's a scalar or array
        CPA_mean = CPA.mean() if hasattr(CPA, 'mean') else CPA
        
        return avg_loss, PA, CPA_mean, MIoU, conf_matrix.cpu().numpy()
    
    def evaluate_checkpoint(self, checkpoint_path, data_folder, model_type, fold_number, num_classes=9, n_out_channels=16):
        """
        Main evaluation function
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            data_folder: Name of the data folder (e.g., 'crossVal_Dataset')
            model_type: 'real' or 'complex'
            fold_number: Which fold to use as test (0, 1, 2, etc.)
            num_classes: Number of classes for segmentation
            n_out_channels: Number of output channels for the model architecture
        """
        print(f"{'='*80}")
        print(f"EVALUATING CHECKPOINT: {checkpoint_path}")
        print(f"Data Folder: {data_folder}")
        print(f"Model Type: {model_type.upper()}")
        print(f"Test Fold: {fold_number}")
        print(f"Number of Classes: {num_classes}")
        print(f"Output Channels: {n_out_channels}")
        print(f"{'='*80}")
        
        # Load checkpoint
        checkpoint_data = self.load_checkpoint(checkpoint_path)
        if checkpoint_data is None:
            print("Failed to load checkpoint!")
            return None
        
        print(f"Checkpoint loaded successfully - Epoch: {checkpoint_data['epoch']}")
        
        # Initialize model
        if model_type.lower() == 'real':
            model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out_channels).to(self.device)
            mode = "real"
        elif model_type.lower() == 'complex':
            model = ComplexUNet(n_channels=1, n_classes=num_classes, n_out_channels=n_out_channels).to(self.device)
            mode = "complex"
        else:
            raise ValueError("model_type must be 'real' or 'complex'")
        
        # Load model weights
        if checkpoint_data['model_state'] is not None:
            model.load_state_dict(checkpoint_data['model_state'])
            print("Model weights loaded successfully")
        else:
            print("Warning: No model state found in checkpoint!")
            return None
        
        # Initialize loss function
        criterion = CustomLoss(alpha=1.0, w1=0.5, w2=0.5)
        
        # Get dataloaders
        try:
            print(f"\nCreating dataloaders for fold {fold_number}...")
            train_loader, val_loader = get_efficient_cross_validation_dataloaders(
                test_fold=fold_number,
                batch_size=16,
                num_workers=4,
                transform=SimpleNorm(),  # Apply the same transform as during training
                out_root=data_folder,
                mode=mode,
                preload_to_ram=True,
                patch_size=128
            )
            
            test_loader = get_efficient_test_dataloader(
                test_fold=fold_number,
                batch_size=16,
                num_workers=4,
                transform=SimpleNorm(),  # Apply the same transform as during training
                out_root=data_folder,
                mode=mode,
                preload_to_ram=True,
            )
            print("Dataloaders created successfully")
            
        except Exception as e:
            print(f"Error creating dataloaders: {e}")
            return None
        
        # Evaluate on all datasets
        print("\nEvaluating model performance...")
        
        print("  Evaluating on training set...")
        train_loss, train_pa, train_cpa, train_miou, train_conf = self.evaluate_model_on_data(
            model, train_loader, num_classes, criterion
        )
        
        print("  Evaluating on validation set...")
        val_loss, val_pa, val_cpa, val_miou, val_conf = self.evaluate_model_on_data(
            model, val_loader, num_classes, criterion
        )
        
        print("  Evaluating on test set...")
        test_loss, test_pa, test_cpa, test_miou, test_conf = self.evaluate_model_on_data(
            model, test_loader, num_classes, criterion
        )
        
        # Compile results
        results = {
            'checkpoint_path': checkpoint_path,
            'epoch': checkpoint_data['epoch'],
            'data_folder': data_folder,
            'model_type': model_type,
            'fold_number': fold_number,
            'num_classes': num_classes,
            'n_out_channels': n_out_channels,
            
            # Training history
            'train_losses_history': checkpoint_data['train_losses'],
            'val_losses_history': checkpoint_data['val_losses'],
            
            # Current epoch results
            'train_loss': train_loss,
            'train_pa': train_pa,
            'train_cpa': train_cpa,
            'train_miou': train_miou,
            'train_conf_matrix': train_conf,
            
            'val_loss': val_loss,
            'val_pa': val_pa,
            'val_cpa': val_cpa,
            'val_miou': val_miou,
            'val_conf_matrix': val_conf,
            
            'test_loss': test_loss,
            'test_pa': test_pa,
            'test_cpa': test_cpa,
            'test_miou': test_miou,
            'test_conf_matrix': test_conf,
        }
        
        # Print results
        self.print_results(results)
        
        # Generate plots
        self.plot_training_history(results)
        self.plot_confusion_matrices(results)
        self.plot_metrics_comparison(results)
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS - EPOCH {results['epoch']}")
        print(f"{'='*80}")
        
        print(f"\n📊 TRAINING SET RESULTS:")
        print(f"  Loss: {results['train_loss']:.4f}")
        print(f"  Pixel Accuracy (PA): {results['train_pa']:.4f}")
        print(f"  Class Pixel Accuracy (CPA): {results['train_cpa']:.4f}")
        print(f"  Mean IoU (MIoU): {results['train_miou']:.4f}")
        
        print(f"\n📊 VALIDATION SET RESULTS:")
        print(f"  Loss: {results['val_loss']:.4f}")
        print(f"  Pixel Accuracy (PA): {results['val_pa']:.4f}")
        print(f"  Class Pixel Accuracy (CPA): {results['val_cpa']:.4f}")
        print(f"  Mean IoU (MIoU): {results['val_miou']:.4f}")
        
        print(f"\n📊 TEST SET RESULTS:")
        print(f"  Loss: {results['test_loss']:.4f}")
        print(f"  Pixel Accuracy (PA): {results['test_pa']:.4f}")
        print(f"  Class Pixel Accuracy (CPA): {results['test_cpa']:.4f}")
        print(f"  Mean IoU (MIoU): {results['test_miou']:.4f}")
        
        print(f"\n{'='*80}")
    
    def plot_training_history(self, results):
        """Plot loss vs epochs graph"""
        train_losses = results['train_losses_history']
        val_losses = results['val_losses_history']
        
        if not train_losses or not val_losses:
            print("No training history available for plotting")
            return
        
        epochs = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Loss vs Epochs
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.axvline(x=results['epoch'], color='g', linestyle='--', alpha=0.7, 
                   label=f'Current Epoch ({results["epoch"]})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Current Metrics Comparison
        plt.subplot(2, 2, 2)
        datasets = ['Train', 'Val', 'Test']
        pas = [results['train_pa'], results['val_pa'], results['test_pa']]
        cpas = [results['train_cpa'], results['val_cpa'], results['test_cpa']]
        mious = [results['train_miou'], results['val_miou'], results['test_miou']]
        
        x = np.arange(len(datasets))
        width = 0.25
        
        plt.bar(x - width, pas, width, label='PA', alpha=0.8)
        plt.bar(x, cpas, width, label='CPA', alpha=0.8)
        plt.bar(x + width, mious, width, label='MIoU', alpha=0.8)
        
        plt.xlabel('Dataset')
        plt.ylabel('Score')
        plt.title(f'Metrics Comparison - Epoch {results["epoch"]}')
        plt.xticks(x, datasets)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss Comparison
        plt.subplot(2, 2, 3)
        losses = [results['train_loss'], results['val_loss'], results['test_loss']]
        colors = ['blue', 'red', 'green']
        plt.bar(datasets, losses, color=colors, alpha=0.7)
        plt.xlabel('Dataset')
        plt.ylabel('Loss')
        plt.title(f'Loss Comparison - Epoch {results["epoch"]}')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training Progress Highlight
        plt.subplot(2, 2, 4)
        if len(train_losses) > 1:
            plt.plot(epochs, train_losses, 'b-', alpha=0.5, linewidth=1)
            plt.plot(epochs, val_losses, 'r-', alpha=0.5, linewidth=1)
            
            # Highlight current epoch
            if results['epoch'] <= len(train_losses):
                plt.scatter([results['epoch']], [train_losses[results['epoch']-1]], 
                           color='blue', s=100, label=f'Train Loss @ Epoch {results["epoch"]}')
                plt.scatter([results['epoch']], [val_losses[results['epoch']-1]], 
                           color='red', s=100, label=f'Val Loss @ Epoch {results["epoch"]}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Current Epoch Highlight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{results['model_type']}_fold{results['fold_number']}_epoch{results['epoch']}_training_history.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved as: {plot_name}")
        plt.show()
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        datasets = ['Train', 'Val', 'Test']
        conf_matrices = [results['train_conf_matrix'], results['val_conf_matrix'], results['test_conf_matrix']]
        pas = [results['train_pa'], results['val_pa'], results['test_pa']]
        mious = [results['train_miou'], results['val_miou'], results['test_miou']]
        
        for i, (dataset, conf_mat, pa, miou) in enumerate(zip(datasets, conf_matrices, pas, mious)):
            sns.heatmap(conf_mat, annot=True, fmt='.0f', cmap='Blues', 
                       square=True, ax=axes[i], cbar_kws={'label': 'Count'})
            axes[i].set_title(f'{dataset} Set Confusion Matrix\nPA: {pa:.4f}, MIoU: {miou:.4f}')
            axes[i].set_xlabel('Predicted Class')
            axes[i].set_ylabel('True Class')
        
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{results['model_type']}_fold{results['fold_number']}_epoch{results['epoch']}_confusion_matrices.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices plot saved as: {plot_name}")
        plt.show()
    
    def plot_metrics_comparison(self, results):
        """Plot detailed metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        datasets = ['Train', 'Val', 'Test']
        pas = [results['train_pa'], results['val_pa'], results['test_pa']]
        cpas = [results['train_cpa'], results['val_cpa'], results['test_cpa']]
        mious = [results['train_miou'], results['val_miou'], results['test_miou']]
        losses = [results['train_loss'], results['val_loss'], results['test_loss']]
        
        # Plot 1: Accuracy Metrics
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, pas, width, label='Pixel Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, cpas, width, label='Class Pixel Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MIoU
        axes[0, 1].bar(datasets, mious, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Mean IoU')
        axes[0, 1].set_title('Mean IoU Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss
        axes[1, 0].bar(datasets, losses, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: All Metrics Combined
        metrics_data = np.array([pas, cpas, mious])
        metrics_names = ['PA', 'CPA', 'MIoU']
        
        for i, (metric_data, metric_name) in enumerate(zip(metrics_data, metrics_names)):
            axes[1, 1].plot(datasets, metric_data, 'o-', label=metric_name, linewidth=2, markersize=8)
        
        axes[1, 1].set_xlabel('Dataset')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('All Metrics Overview')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{results['model_type']}_fold{results['fold_number']}_epoch{results['epoch']}_metrics_comparison.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved as: {plot_name}")
        plt.show()

# Example usage functions
def evaluate_single_checkpoint(checkpoint_path, data_folder="crossVal_Dataset", 
                             model_type="real", fold_number=0, num_classes=9, n_out_channels=16):
    """
    Convenience function to evaluate a single checkpoint
    
    Args:
        checkpoint_path: Path to the .pt file
        data_folder: Name of the data folder
        model_type: 'real' or 'complex'
        fold_number: Which fold to use as test
        num_classes: Number of segmentation classes
        n_out_channels: Number of output channels for the model architecture
    """
    evaluator = CheckpointEvaluator()
    results = evaluator.evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        data_folder=data_folder,
        model_type=model_type,
        fold_number=fold_number,
        num_classes=num_classes,
        n_out_channels=n_out_channels
    )
    return results

results2 = evaluate_single_checkpoint(
    checkpoint_path="./real_16_fold2/checkpoint_epoch_200.pt",
    data_folder="crossVal_Dataset",
    model_type="real",
    fold_number=2,
    num_classes=9,
    n_out_channels=16
)