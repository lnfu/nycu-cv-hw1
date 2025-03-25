import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def count_images_per_class(data_dir):
    """
    Count the number of images in each class directory.
    
    Args:
        data_dir (pathlib.Path): Path to the dataset directory (all)
        
    Returns:
        dict: Dictionary mapping class indices to image counts
    """
    # Use ImageFolder to get class information
    dataset = datasets.ImageFolder(root=data_dir)
    
    # Get class to index mapping
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Count images per class
    counts = {idx: 0 for idx in range(len(class_to_idx))}
    
    # Count images in each class directory
    for class_name, idx in class_to_idx.items():
        class_dir = data_dir / class_name
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        counts[idx] = len(image_files)
    
    return counts, idx_to_class

def plot_class_distribution(all_counts, idx_to_class, save_path=None, sort_by_count=True):
    """
    Plot the distribution of images across classes for the all dataset.
    
    Args:
        all_counts (dict): Dictionary mapping class indices to image counts
        idx_to_class (dict): Dictionary mapping class indices to class names
        save_path (pathlib.Path, optional): Path to save the plot. If None, the plot is displayed.
        sort_by_count (bool): If True, sort by count. If False, sort by class index.
    """
    # Create a list of (class_idx, all_count) tuples
    class_data = [(idx, all_counts[idx]) for idx in idx_to_class.keys()]
    
    # Sort by count if requested
    if sort_by_count:
        class_data.sort(key=lambda x: x[1], reverse=True)
    else:
        class_data.sort(key=lambda x: x[0])
    
    # Unpack the sorted data
    classes = [str(item[0]) for item in class_data]  # Convert to strings for labels
    all_values = [item[1] for item in class_data]
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    
    # Create the bars
    x = np.arange(len(classes))
    plt.bar(x, all_values, label='train/val data')
    
    # Add labels, title, and legend
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('# of Images', fontsize=20)
    plt.title('Class Distribution', fontsize=20)
    plt.xticks(x, classes, rotation=90)
    plt.legend()
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the class indices
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # Define data directory paths
    base_dir = pathlib.Path(".")
    data_dir = base_dir / "data"
    all_dir = data_dir / "all"
    
    # Count images in each class
    logging.info("Counting images in all set...")
    all_counts, idx_to_class = count_images_per_class(all_dir)
    
    # Log the counts
    logging.info("All set class distribution:")
    for idx, count in sorted(all_counts.items()):
        logging.info(f"Class {idx}: {count} images")
    
    # Total images
    total_all = sum(all_counts.values())
    logging.info(f"Total images in all dataset: {total_all}")
    
    # Plot the distribution
    save_path = pathlib.Path("class_distribution_all.png")
    plot_class_distribution(all_counts, idx_to_class, save_path, sort_by_count=True)

if __name__ == "__main__":
    main()
