import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed, felzenszwalb, slic
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from skimage.color import label2rgb
from collections import defaultdict

# === Utility Functions ===

def compute_magnitude(complex_img):
    real = complex_img[..., 0]
    imag = complex_img[..., 1]
    return np.sqrt(real**2 + imag**2)

def normalize(img):
    return (img - img.min()) / (img.ptp() + 1e-8)

# === Segmentation Methods ===

def felzenszwalb_segmentation(complex_img, scale=50, sigma=0.5, min_size=10000):
    magnitude = normalize(compute_magnitude(complex_img))
    return felzenszwalb(magnitude, scale=scale, sigma=sigma, min_size=min_size)

def felzenszwalb_segmentation_with_phase(complex_img, scale=50, sigma=0.5, min_size=10000):
    # Extract magnitude and phase
    magnitude = normalize(compute_magnitude(complex_img))
    phase = np.arctan2(complex_img[..., 1], complex_img[..., 0])
    phase_normalized = normalize(phase)  # Scale to [0,1]
    
    # Create multichannel image (magnitude, phase)
    multichannel_img = np.stack([magnitude, phase_normalized], axis=-1)
    
    # Updated parameter: use channel_axis instead of multichannel
    return felzenszwalb(multichannel_img, scale=scale, sigma=sigma, 
                        min_size=min_size, channel_axis=-1)
    
def felzenszwalb_feature_fusion(complex_img, scale=50, sigma=0.5, min_size=10000, 
                              mag_weight=0.7, phase_weight=0.3):
    # Extract magnitude and phase
    magnitude = normalize(compute_magnitude(complex_img))
    phase = np.arctan2(complex_img[..., 1], complex_img[..., 0])
    phase_normalized = normalize(phase)
    
    # Create a weighted fusion of features
    fused_feature = mag_weight * magnitude + phase_weight * phase_normalized
    
    # Apply segmentation on the fused feature
    return felzenszwalb(fused_feature, scale=scale, sigma=sigma, min_size=min_size)

def felzenszwalb_tri_channel(complex_img, scale=50, sigma=0.5, min_size=10000):
    """
    Apply Felzenszwalb segmentation using real, imaginary and magnitude channels.
    
    Args:
        complex_img: Input image with shape [H, W, 2] where channel 0 is real, channel 1 is imaginary
        scale: Free parameter for segmentation. Higher means larger clusters.
        sigma: Width of Gaussian kernel for pre-processing
        min_size: Minimum component size
        
    Returns:
        Label image
    """
    # Extract components
    real_part = normalize(complex_img[..., 0])  # Real part
    imag_part = normalize(complex_img[..., 1])  # Imaginary part
    magnitude = normalize(compute_magnitude(complex_img))  # Magnitude
    
    # Create 3-channel image
    tri_channel_img = np.stack([real_part, imag_part, magnitude], axis=-1)
    
    # Apply segmentation with 3 channels
    return felzenszwalb(tri_channel_img, scale=scale, sigma=sigma, 
                       min_size=min_size, channel_axis=-1)

# === Visualization ===

def show_labels(labels, title, subplot_index):
    plt.subplot(1, 5, subplot_index)
    plt.imshow(label2rgb(labels, bg_label=0))
    plt.title(title)
    plt.axis('off')

# === Main Loop ===

# Load your test data
test_images = np.load("test/test_images.npy")  # shape [N, H, W, 2]
test_masks = np.load("test/test_masks.npy")    # shape [N, H, W]

for i in range(len(test_images)):
    img = test_images[i]
    true_mask = test_masks[i]
    magnitude = compute_magnitude(img)

    # --- Felzenszwalb Segmentation ---
    felz_labels = felzenszwalb_tri_channel(img)

    # --- Build Region Dictionary ---
    region_dict = defaultdict(list)
    for r in range(felz_labels.shape[0]):
        for c in range(felz_labels.shape[1]):
            region_label = felz_labels[r, c]
            region_dict[region_label].append((r, c))

    print(f"Image {i}: {len(region_dict)} regions")

    # === Plot ===
    plt.figure(figsize=(15, 3))

    plt.subplot(1, 5, 1)
    plt.imshow(magnitude, cmap='gray')
    plt.title(f"Magnitude Image {i}")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    show_labels(felz_labels, "Felzenszwalb", 3)

    plt.tight_layout()
    plt.show()
    
def felzenszwalb_parameter_grid(complex_img, true_mask, method="tri_channel"):
    """
    Generate a grid of segmentation results with different sigma and scale values.
    
    Args:
        complex_img: Input complex image with shape [H, W, 2]
        true_mask: Ground truth segmentation mask
        method: Segmentation method to use ("tri_channel", "magnitude", "mag_phase", "fusion")
    """
    # Define parameter ranges
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3]
    scales = [50, 100, 150, 200, 250, 300]
    min_size = 5000  # Fixed
    
    # Create figure
    fig, axes = plt.subplots(len(sigmas) + 1, len(scales) + 1, figsize=(18, 15))
    
    # Show true mask labels instead of input magnitude
    axes[0, 0].imshow(label2rgb(true_mask, bg_label=0))
    axes[0, 0].set_title("True Mask Labels")
    axes[0, 0].axis('off')
    
    # Add scale labels in first row
    for j, scale in enumerate(scales):
        axes[0, j+1].text(0.5, 0.5, f"Scale = {scale}", ha='center', va='center', fontsize=12)
        axes[0, j+1].axis('off')
    
    # Add sigma labels in first column
    for i, sigma in enumerate(sigmas):
        axes[i+1, 0].text(0.5, 0.5, f"Sigma = {sigma}", ha='center', va='center', fontsize=12)
        axes[i+1, 0].axis('off')
    
    # Generate segmentations for each parameter combination
    for i, sigma in enumerate(sigmas):
        for j, scale in enumerate(scales):
            # Choose segmentation method
            if method == "tri_channel":
                labels = felzenszwalb_tri_channel(complex_img, scale=scale, sigma=sigma, min_size=min_size)
            elif method == "magnitude":
                labels = felzenszwalb_segmentation(complex_img, scale=scale, sigma=sigma, min_size=min_size)
            elif method == "mag_phase":
                labels = felzenszwalb_segmentation_with_phase(complex_img, scale=scale, sigma=sigma, min_size=min_size)
            elif method == "fusion":
                labels = felzenszwalb_feature_fusion(complex_img, scale=scale, sigma=sigma, min_size=min_size)
            
            # Display segmentation result
            axes[i+1, j+1].imshow(label2rgb(labels, bg_label=0))
            region_count = len(np.unique(labels))
            axes[i+1, j+1].set_title(f"{region_count} regions")
            axes[i+1, j+1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Felzenszwalb Segmentation with {method.replace('_', ' ').title()} (min_size={min_size})", 
                 fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.96)
    plt.show()
    
    return fig
# Choose one image to visualize parameter grid (e.g., first image)
i = 0  # Change this to any image index you want to analyze
img = test_images[i]
true_mask = test_masks[i]

# Generate parameter grid visualization
# felzenszwalb_parameter_grid(img, true_mask, method="tri_channel")
felzenszwalb_parameter_grid(img, true_mask, method="magnitude")
# felzenszwalb_parameter_grid(img, true_mask, method="mag_phase")
# felzenszwalb_parameter_grid(img, true_mask, method="fusion")