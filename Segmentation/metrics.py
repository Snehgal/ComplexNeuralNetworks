import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CrossValidationMetrics:
    def __init__(self, device=None, disable_progress=True):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.disable_progress = disable_progress
        print(f"Using device: {self.device}")
    
    def load_model(self, checkpoint_path, model_type, num_classes=9, n_out_channels=16):
        """Load a single model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize model
            if model_type.lower() == 'real':
                model = UNet(n_channels=2, n_classes=num_classes, n_out_channels=n_out_channels).to(self.device)
            elif model_type.lower() == 'complex':
                model = ComplexUNet(n_channels=1, n_classes=num_classes, n_out_channels=n_out_channels).to(self.device)
            else:
                raise ValueError("model_type must be 'real' or 'complex'")
            
            # Load model weights
            model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state', None))
            if model_state is not None:
                model.load_state_dict(model_state)
                print(f"✓ Loaded model from {checkpoint_path}")
            else:
                raise ValueError("No model state found in checkpoint!")
            
            return model
        
        except Exception as e:
            print(f"✗ Error loading model from {checkpoint_path}: {e}")
            return None
    
    def get_test_dataloader(self, fold_number, data_folder, model_type, batch_size=32):
        """Get test dataloader for a specific fold"""
        try:
            mode = "real" if model_type.lower() == 'real' else "complex"
            
            test_loader = get_efficient_test_dataloader(
                test_fold=fold_number,
                batch_size=batch_size,  # Increased default batch size
                num_workers=4,  # Increased num_workers for faster data loading
                transform=SimpleNorm(),
                out_root=data_folder,
                mode=mode,
                preload_to_ram=True,
            )
            print(f"✓ Created test dataloader for fold {fold_number}")
            return test_loader
        
        except Exception as e:
            print(f"✗ Error creating test dataloader for fold {fold_number}: {e}")
            return None
    
    def compute_confusion_matrix_for_fold(self, model, dataloader, num_classes):
        """Compute confusion matrix for one fold"""
        model.eval()
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Add progress bar for batches
            pbar = tqdm(dataloader, desc=f"   Processing batches", unit="batch", 
                       leave=False, disable=self.disable_progress)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # Handle size mismatch if needed
                if outputs.shape[-2:] != targets.shape[-2:]:
                    targets = targets[:, :outputs.shape[2], :outputs.shape[3]]
                
                # Get predictions (apply argmax to logits)
                predictions = torch.argmax(outputs, dim=1)
                
                # Flatten arrays
                targets_flat = targets.view(-1)
                predictions_flat = predictions.view(-1)
                
                # Optimized confusion matrix update using bincount
                # This is much faster than the loop
                valid_mask = (targets_flat < num_classes) & (predictions_flat < num_classes)
                targets_flat = targets_flat[valid_mask]
                predictions_flat = predictions_flat[valid_mask]
                
                # Create indices for bincount: target * num_classes + prediction
                indices = targets_flat * num_classes + predictions_flat
                cm_update = torch.bincount(indices, minlength=num_classes*num_classes)
                cm_update = cm_update.view(num_classes, num_classes)
                confusion_matrix += cm_update
                
                # Update progress bar with current batch info
                pbar.set_postfix({
                    'batch_size': inputs.shape[0],
                    'pixels': f"{inputs.shape[0] * inputs.shape[2] * inputs.shape[3]:,}"
                })
        
        return confusion_matrix.cpu().numpy()
    
    def compute_metrics_from_confusion_matrix(self, confusion_matrix):
        """Compute all metrics from confusion matrix"""
        # Pixel Accuracy (PA)
        total_pixels = np.sum(confusion_matrix)
        correct_pixels = np.trace(confusion_matrix)
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Class-wise Pixel Accuracy (CPA)
        class_pixel_accuracy = []
        for i in range(confusion_matrix.shape[0]):
            class_total = np.sum(confusion_matrix[i, :])
            if class_total > 0:
                cpa = confusion_matrix[i, i] / class_total
            else:
                cpa = 0.0
            class_pixel_accuracy.append(cpa)
        
        # IoU per class
        iou_per_class = []
        for i in range(confusion_matrix.shape[0]):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        # Mean IoU
        mean_iou = np.mean(iou_per_class)
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'class_pixel_accuracy': np.array(class_pixel_accuracy),
            'iou_per_class': np.array(iou_per_class),
            'mean_iou': mean_iou,
            'confusion_matrix': confusion_matrix
        }
    
    def evaluate_cross_validation(self, model_paths, data_folder, model_type, 
                                 num_classes=9, n_out_channels=16, batch_size=32):
        """
        Evaluate 3-fold cross-validation results
        
        Args:
            model_paths: List of 3 .pt file paths (one for each fold)
            data_folder: Name of the data folder (e.g., 'crossVal_Dataset')
            model_type: 'real' or 'complex'
            num_classes: Number of segmentation classes
            n_out_channels: Number of output channels for model architecture
            batch_size: Batch size for evaluation (increased default for speed)
        
        Returns:
            dict: Combined metrics across all folds
        """
        print(f"{'='*80}")
        print(f"CROSS-VALIDATION EVALUATION")
        print(f"Model Type: {model_type.upper()}")
        print(f"Number of Classes: {num_classes}")
        print(f"Output Channels: {n_out_channels}")
        print(f"{'='*80}")
        
        if len(model_paths) != 3:
            raise ValueError("Expected exactly 3 model paths for 3-fold cross-validation")
        
        # Initialize combined confusion matrix
        combined_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # Process each fold
        fold_results = []
        print(f"\n🚀 Starting evaluation of {len(model_paths)} folds...")
        
        # Add overall progress bar for folds
        fold_pbar = tqdm(enumerate(model_paths), total=len(model_paths), 
                        desc="📊 Overall Progress", unit="fold", disable=self.disable_progress)
        
        for fold_idx, model_path in fold_pbar:
            fold_pbar.set_description(f"📊 Processing Fold {fold_idx}")
            print(f"\n📊 Processing Fold {fold_idx}...")
            print(f"   Model: {model_path}")
            
            # Load model
            model = self.load_model(model_path, model_type, num_classes, n_out_channels)
            if model is None:
                continue
            
            # Get test dataloader
            test_loader = self.get_test_dataloader(fold_idx, data_folder, model_type, batch_size)
            if test_loader is None:
                continue
            
            # Compute confusion matrix for this fold
            print(f"   Computing metrics for fold {fold_idx}...")
            fold_confusion_matrix = self.compute_confusion_matrix_for_fold(model, test_loader, num_classes)
            
            # Add to combined matrix
            combined_confusion_matrix += fold_confusion_matrix
            
            # Store fold results
            fold_metrics = self.compute_metrics_from_confusion_matrix(fold_confusion_matrix)
            fold_results.append({
                'fold': fold_idx,
                'model_path': model_path,
                'confusion_matrix': fold_confusion_matrix,
                'metrics': fold_metrics
            })
            
            print(f"   ✓ Fold {fold_idx} - PA: {fold_metrics['pixel_accuracy']:.4f}, mIoU: {fold_metrics['mean_iou']:.4f}")
            
            # Update fold progress bar
            fold_pbar.set_postfix({
                'PA': f"{fold_metrics['pixel_accuracy']:.3f}",
                'mIoU': f"{fold_metrics['mean_iou']:.3f}"
            })
        
        fold_pbar.close()
        
        # Compute combined metrics
        print(f"\n🎯 Computing combined metrics...")
        combined_metrics = self.compute_metrics_from_confusion_matrix(combined_confusion_matrix)
        
        # Compile final results
        results = {
            'model_type': model_type,
            'num_classes': num_classes,
            'n_out_channels': n_out_channels,
            'fold_results': fold_results,
            'combined_confusion_matrix': combined_confusion_matrix,
            'combined_metrics': combined_metrics,
            'model_paths': model_paths
        }
        
        # Print and visualize results
        self.print_results(results)
        self.plot_results(results)
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        metrics = results['combined_metrics']
        
        print(f"\n{'='*80}")
        print(f"FINAL CROSS-VALIDATION RESULTS")
        print(f"{'='*80}")
        
        # Overall Pixel Accuracy
        print(f"\n🎯 Overall Pixel Accuracy (PA): {metrics['pixel_accuracy']*100:.2f}%")
        print(f"🎯 Overall Mean IoU (mIoU): {metrics['mean_iou']*100:.2f}%")
        
        # Per-class metrics table
        print(f"\n📊 Per-Class Metrics:")
        
        # Prepare data for table
        table_data = []
        for i in range(results['num_classes']):
            cpa = metrics['class_pixel_accuracy'][i] * 100
            iou = metrics['iou_per_class'][i] * 100
            table_data.append([f"Class {i}", f"{cpa:.2f}%", f"{iou:.2f}%"])
        
        # Add mean row
        mean_cpa = np.mean(metrics['class_pixel_accuracy']) * 100
        mean_iou = metrics['mean_iou'] * 100
        table_data.append(["Mean", f"{mean_cpa:.2f}%", f"{mean_iou:.2f}%"])
        
        headers = ["Class", "CPA (%)", "IoU (%)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Individual fold summary
        print(f"\n📈 Individual Fold Performance:")
        fold_table_data = []
        for fold_result in results['fold_results']:
            fold_metrics = fold_result['metrics']
            pa = fold_metrics['pixel_accuracy'] * 100
            miou = fold_metrics['mean_iou'] * 100
            fold_table_data.append([f"Fold {fold_result['fold']}", f"{pa:.2f}%", f"{miou:.2f}%"])
        
        # Add combined row
        combined_pa = metrics['pixel_accuracy'] * 100
        combined_miou = metrics['mean_iou'] * 100
        fold_table_data.append(["Combined", f"{combined_pa:.2f}%", f"{combined_miou:.2f}%"])
        
        fold_headers = ["Fold", "PA (%)", "mIoU (%)"]
        print(tabulate(fold_table_data, headers=fold_headers, tablefmt="grid"))
        
        print(f"\n{'='*80}")
    
    def plot_results(self, results):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Combined Confusion Matrix
        cm = results['combined_confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                   cbar_kws={'label': 'Count'})
        axes[0, 0].set_title('Combined Confusion Matrix\n(All Folds)')
        axes[0, 0].set_xlabel('Predicted Class')
        axes[0, 0].set_ylabel('True Class')
        
        # Plot 2: Per-Class IoU
        iou_values = results['combined_metrics']['iou_per_class'] * 100
        class_names = [f'Class {i}' for i in range(results['num_classes'])]
        axes[0, 1].bar(class_names, iou_values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('IoU per Class')
        axes[0, 1].set_ylabel('IoU (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Per-Class CPA
        cpa_values = results['combined_metrics']['class_pixel_accuracy'] * 100
        axes[0, 2].bar(class_names, cpa_values, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Class Pixel Accuracy per Class')
        axes[0, 2].set_ylabel('CPA (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Fold-wise Performance Comparison
        fold_pas = [fold['metrics']['pixel_accuracy']*100 for fold in results['fold_results']]
        fold_mious = [fold['metrics']['mean_iou']*100 for fold in results['fold_results']]
        fold_names = [f'Fold {fold["fold"]}' for fold in results['fold_results']]
        
        x = np.arange(len(fold_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, fold_pas, width, label='PA (%)', alpha=0.8)
        axes[1, 0].bar(x + width/2, fold_mious, width, label='mIoU (%)', alpha=0.8)
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Performance (%)')
        axes[1, 0].set_title('Fold-wise Performance')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(fold_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: IoU vs CPA scatter
        iou_vals = results['combined_metrics']['iou_per_class'] * 100
        cpa_vals = results['combined_metrics']['class_pixel_accuracy'] * 100
        axes[1, 1].scatter(iou_vals, cpa_vals, s=100, alpha=0.7, c=range(len(iou_vals)), cmap='viridis')
        for i, (iou, cpa) in enumerate(zip(iou_vals, cpa_vals)):
            axes[1, 1].annotate(f'C{i}', (iou, cpa), xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('IoU (%)')
        axes[1, 1].set_ylabel('CPA (%)')
        axes[1, 1].set_title('IoU vs CPA per Class')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Overall Summary
        overall_metrics = ['PA', 'mIoU', 'Mean CPA']
        overall_values = [
            results['combined_metrics']['pixel_accuracy'] * 100,
            results['combined_metrics']['mean_iou'] * 100,
            np.mean(results['combined_metrics']['class_pixel_accuracy']) * 100
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = axes[1, 2].bar(overall_metrics, overall_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Overall Performance Summary')
        axes[1, 2].set_ylabel('Score (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{results['model_type']}_cross_validation_results.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        print(f"\n📈 Results visualization saved as: {plot_name}")
        plt.show()

# Convenience function for easy usage
def evaluate_cross_validation_models(model_paths, data_folder="crossVal_Dataset", 
                                   model_type="real", num_classes=9, n_out_channels=16, 
                                   batch_size=32):
    """
    Convenience function to evaluate cross-validation results
    
    Args:
        model_paths: List of 3 .pt file paths (one for each fold)
        data_folder: Name of the data folder
        model_type: 'real' or 'complex' 
        num_classes: Number of segmentation classes
        n_out_channels: Number of output channels for model architecture
        batch_size: Batch size for evaluation (larger = faster)
    
    Returns:
        dict: Complete evaluation results
    """
    evaluator = CrossValidationMetrics(disable_progress=True)
    results = evaluator.evaluate_cross_validation(
        model_paths=model_paths,
        data_folder=data_folder,
        model_type=model_type,
        num_classes=num_classes,
        n_out_channels=n_out_channels,
        batch_size=batch_size
    )
    return results