import matplotlib.pyplot as plt
import os
import numpy as np

# Path to the dataset directory
ssl_dataset_dir = "ssl_dataset"

# Parameters for the dataloader - REDUCE THESE VALUES
patch_size = 256  # Changed from 256 to 128
batch_size = 4    # Keep small to avoid memory issues

# Reduce number of workers to avoid memory issues
annotated_loader = get_ssl_dataloader(
    patch_dir=os.path.join(ssl_dataset_dir, "annotated_patches"),
    split_name="annotated",
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=1  # Reduce workers to 1 to avoid memory issues
)

unannotated_loader = get_ssl_dataloader(
    patch_dir=os.path.join(ssl_dataset_dir, "unannotated_patches"),
    split_name="unannotated",
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=1  # Reduce workers to 1 to avoid memory issues
)

# Function to visualize a batch with memory management
def visualize_batch(loader, title, split_name):
    for patches, masks, pseudo1, pseudo2 in loader:  # Now correctly unpacking all returns
        fig, axes = plt.subplots(batch_size, 4, figsize=(15, batch_size * 3))
        
        # Handle case where batch_size is 1
        if batch_size == 1:
            axes = np.array([axes])
            
        for i in range(batch_size):
            # Extract magnitude for visualization
            magnitude = (patches[i, ..., 0]**2 + patches[i, ..., 1]**2).sqrt().numpy()  # Magnitude
            mask = masks[i].numpy()
            pseudo1_img = pseudo1[i].numpy()  # Use the pseudo1 from dataloader
            pseudo2_img = pseudo2[i].numpy()  # Use the pseudo2 from dataloader

            # Plot magnitude
            axes[i, 0].imshow(magnitude, cmap='gray')
            axes[i, 0].set_title(f"Patch {i} (Magnitude)")
            axes[i, 0].axis('off')

            # Plot mask
            axes[i, 1].imshow(mask, cmap='jet')
            axes[i, 1].set_title(f"Mask {i}")
            axes[i, 1].axis('off')

            # Plot pseudo label 1
            axes[i, 2].imshow(pseudo1_img, cmap='jet')
            axes[i, 2].set_title(f"Pseudo Label 1 {i}")
            axes[i, 2].axis('off')

            # Plot pseudo label 2
            axes[i, 3].imshow(pseudo2_img, cmap='jet')
            axes[i, 3].set_title(f"Pseudo Label 2 {i}")
            axes[i, 3].axis('off')

        plt.suptitle(f"{title} - {split_name.capitalize()}", fontsize=16)
        
        # Try to avoid the tight_layout error by using a more basic approach
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.show()
        
        # Clear memory
        plt.close(fig)
        break  # Only visualize one batch

# Visualize annotated patches
try:
    print("Visualizing annotated patches...")
    visualize_batch(annotated_loader, "Patches", "annotated")
except Exception as e:
    print(f"Error visualizing annotated patches: {e}")

# Visualize unannotated patches
try:
    print("Visualizing unannotated patches...")
    visualize_batch(unannotated_loader, "Patches", "unannotated")
except Exception as e:
    print(f"Error visualizing unannotated patches: {e}")