import os
import numpy as np
import h5py
from tqdm import tqdm
import time
from skimage.segmentation import felzenszwalb
from collections import defaultdict
import random

def compute_magnitude(complex_img):
    real = complex_img[..., 0]
    imag = complex_img[..., 1]
    return np.sqrt(real**2 + imag**2)

def normalize(img):
    return (img - img.min()) / (img.ptp() + 1e-8)

def felzenszwalb_segmentation(complex_img, scale=50, sigma=0.5, min_size=10000):
    """Generate pseudo-labels using magnitude-only Felzenszwalb segmentation"""
    magnitude = normalize(compute_magnitude(complex_img))
    return felzenszwalb(magnitude, scale=scale, sigma=sigma, min_size=min_size)

def felzenszwalb_tri_channel(complex_img, scale=50, sigma=0.5, min_size=10000):
    """
    Generate pseudo-labels using real, imaginary and magnitude channels.
    
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

def createDataset(filepath, testIDs, annotatedIDs, 
                 felz_scale=50, felz_sigma=0.5, felz_min_size=10000,
                 tri_scale=50, tri_sigma=0.5, tri_min_size=10000,
                 output_dir="dataset", batch_size=10):
    """
    Create a dataset with pseudo-labels and split into test, annotated, and unannotated sets.
    
    Args:
        filepath: Path to the sassed_V4.h5 file
        testIDs: List of image IDs for test set
        annotatedIDs: List of image IDs for annotated set
        felz_scale: Scale parameter for felzenszwalb_segmentation
        felz_sigma: Sigma parameter for felzenszwalb_segmentation
        felz_min_size: Min_size parameter for felzenszwalb_segmentation
        tri_scale: Scale parameter for felzenszwalb_tri_channel
        tri_sigma: Sigma parameter for felzenszwalb_tri_channel
        tri_min_size: Min_size parameter for felzenszwalb_tri_channel
        output_dir: Directory to save the dataset
        batch_size: Number of images to process at once
        
    Returns:
        Dict with information about the created dataset
    """
    # Load data and segments from the HDF5 file
    print(f"Loading data from {filepath}...")
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]  # Complex-valued images
        segments = f['segments'][:]  # Ground truth segmentation masks
    
    # Convert complex data to 2-channel format
    data_2ch = np.stack([data.real, data.imag], axis=-1)
    
    # Create output directories
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    os.makedirs(f"{output_dir}/annotated", exist_ok=True)
    os.makedirs(f"{output_dir}/unannotated", exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Convert IDs to sets for faster lookup
    testIDs_set = set(testIDs)
    annotatedIDs_set = set(annotatedIDs)
    
    # Validate inputs
    num_images = len(data_2ch)
    assert len(segments) == num_images, "Number of images and segments must match"
    assert max(testIDs_set) < num_images if testIDs_set else True, "Test IDs must be valid image indices"
    assert max(annotatedIDs_set) < num_images if annotatedIDs_set else True, "Annotated IDs must be valid image indices"
    
    # Check for overlap
    overlap = testIDs_set.intersection(annotatedIDs_set)
    if overlap:
        print(f"Warning: {len(overlap)} images appear in both test and annotated sets")
    
    # Track counts for each split
    test_count = 0
    annotated_count = 0
    unannotated_count = 0
    
    # Process each image
    print(f"Processing {num_images} images in batches of {batch_size}...")
    for start_idx in tqdm(range(0, num_images, batch_size)):
        end_idx = min(start_idx + batch_size, num_images)
        batch_indices = list(range(start_idx, end_idx))
        
        for i in batch_indices:
            # Generate pseudo-labels for this image
            try:
                pseudo1 = felzenszwalb_segmentation(
                    data_2ch[i], scale=felz_scale, sigma=felz_sigma, min_size=felz_min_size
                )
                
                pseudo2 = felzenszwalb_tri_channel(
                    data_2ch[i], scale=tri_scale, sigma=tri_sigma, min_size=tri_min_size
                )
                
                # Determine which split this image belongs to
                if i in testIDs_set:
                    save_dir = f"{output_dir}/test"
                    test_count += 1
                elif i in annotatedIDs_set:
                    save_dir = f"{output_dir}/annotated"
                    annotated_count += 1
                else:
                    save_dir = f"{output_dir}/unannotated"
                    unannotated_count += 1
                
                # Save image data and labels
                np.save(f"{save_dir}/img_{i}_data.npy", data_2ch[i])
                np.save(f"{save_dir}/img_{i}_segments.npy", segments[i])
                np.save(f"{save_dir}/img_{i}_pseudo1.npy", pseudo1)
                np.save(f"{save_dir}/img_{i}_pseudo2.npy", pseudo2)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
    
    # Save metadata about the splits
    np.save(f"{output_dir}/test_ids.npy", np.array(list(testIDs_set)))
    np.save(f"{output_dir}/annotated_ids.npy", np.array(list(annotatedIDs_set)))
    unannotated_ids = [i for i in range(num_images) if i not in testIDs_set and i not in annotatedIDs_set]
    np.save(f"{output_dir}/unannotated_ids.npy", np.array(unannotated_ids))
    
    # Calculate and print statistics
    elapsed_time = time.time() - start_time
    
    print(f"\nDataset created successfully in {output_dir}/")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Test set: {test_count} images")
    print(f"Annotated set: {annotated_count} images")
    print(f"Unannotated set: {unannotated_count} images")
    
    # Return dataset info
    return {
        "output_dir": output_dir,
        "test_count": test_count,
        "annotated_count": annotated_count,
        "unannotated_count": unannotated_count,
        "total_images": num_images,
        "processing_time": elapsed_time,
        "felz_params": {"scale": felz_scale, "sigma": felz_sigma, "min_size": felz_min_size},
        "tri_params": {"scale": tri_scale, "sigma": tri_sigma, "min_size": tri_min_size}
    }
    
def loadDataset(dataset_dir, split_name):
    """
    Load data from a specific split of the dataset.
    
    Args:
        dataset_dir: Directory containing the dataset
        split_name: Name of the split ("test", "annotated", or "unannotated")
        
    Returns:
        Dict containing the data for this split
    """
    valid_splits = ["test", "annotated", "unannotated"]
    assert split_name in valid_splits, f"Split must be one of {valid_splits}"
    
    # Get the list of file IDs for this split
    ids = np.load(f"{dataset_dir}/{split_name}_ids.npy")
    
    result = {
        "ids": ids,
        "data": [],
        "segments": [],
        "pseudo1": [],
        "pseudo2": []
    }
    
    # Load data for each ID
    for img_id in ids:
        result["data"].append(np.load(f"{dataset_dir}/{split_name}/img_{img_id}_data.npy"))
        result["pseudo1"].append(np.load(f"{dataset_dir}/{split_name}/img_{img_id}_pseudo1.npy"))
        result["pseudo2"].append(np.load(f"{dataset_dir}/{split_name}/img_{img_id}_pseudo2.npy"))
        result["segments"].append(np.load(f"{dataset_dir}/{split_name}/img_{img_id}_segments.npy"))
    
    # Convert lists to numpy arrays
    result["data"] = np.array(result["data"])
    result["segments"] = np.array(result["segments"])
    result["pseudo1"] = np.array(result["pseudo1"])
    result["pseudo2"] = np.array(result["pseudo2"])
    
    return result

def getIDs(TOTAL,numTest,numAnnotated):
    testIDs = random.sample(range(TOTAL), numTest)
    available_ids = set(range(TOTAL)) - set(testIDs)
    if numAnnotated > len(available_ids):
        raise ValueError("Not enough IDs left to sample from.")
    annotatedIDs = random.sample(list(available_ids), numAnnotated)
    
    return testIDs,annotatedIDs

TOTAL = 129
numTest = 9
annotatedPercentage = 10
numAnnotated = int((annotatedPercentage/100)*(TOTAL-numTest))

testIDs,annotatedIDs = getIDs(TOTAL,numTest,numAnnotated)
print("Test IDs:", sorted(testIDs))
print("Annotated IDs:", sorted(annotatedIDs))

createDataset(
        filepath="sassed_V4.h5",
        testIDs=testIDs,
        annotatedIDs=annotatedIDs,
        felz_scale=100,felz_sigma=0.5,felz_min_size=10000,
        tri_scale=50,tri_sigma=1.0,tri_min_size=10000,
        output_dir="ssl_dataset",
        batch_size=1
    )