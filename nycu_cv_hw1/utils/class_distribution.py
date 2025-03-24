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
        data_dir (pathlib.Path): Path to the dataset directory (train, val, or all)
        
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

def plot_class_distribution(train_counts, val_counts, all_counts, idx_to_class, save_path=None, sort_by_count=True):
    """
    Plot the distribution of images across classes for train, val, and all datasets.
    
    Args:
        train_counts (dict): Dictionary mapping class indices to image counts for training set
        val_counts (dict): Dictionary mapping class indices to image counts for validation set
        all_counts (dict): Dictionary mapping class indices to image counts for all set
        idx_to_class (dict): Dictionary mapping class indices to class names
        save_path (pathlib.Path, optional): Path to save the plot. If None, the plot is displayed.
        sort_by_count (bool): If True, sort by count. If False, sort by class index.
    """
    # Create a list of (class_idx, train_count, val_count, all_count) tuples
    class_data = [(idx, train_counts[idx], val_counts[idx], all_counts[idx]) for idx in idx_to_class.keys()]
    
    # Sort by train count if requested
    if sort_by_count:
        class_data.sort(key=lambda x: x[1], reverse=True)
    else:
        class_data.sort(key=lambda x: x[0])
    
    # Unpack the sorted data
    classes = [str(item[0]) for item in class_data]  # Convert to strings for labels
    train_values = [item[1] for item in class_data]
    val_values = [item[2] for item in class_data]
    all_values = [item[3] for item in class_data]
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(classes))
    width = 0.25  # Adjusted width for three bars
    
    # Create the bars
    plt.bar(x - width, train_values, width, label='Train')
    plt.bar(x, val_values, width, label='Validation')
    plt.bar(x + width, all_values, width, label='All Data')
    
    # Add labels, title, and legend
    plt.xlabel('Class Index')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class in Train, Validation, and All Datasets')
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
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    all_dir = data_dir / "all"
    
    # Count images in each class
    logging.info("Counting images in training set...")
    train_counts, idx_to_class = count_images_per_class(train_dir)
    
    logging.info("Counting images in validation set...")
    val_counts, _ = count_images_per_class(val_dir)
    
    logging.info("Counting images in all set...")
    all_counts, _ = count_images_per_class(all_dir)
    
    # Log the counts
    logging.info("Training set class distribution:")
    for idx, count in sorted(train_counts.items()):
        logging.info(f"Class {idx}: {count} images")
    
    logging.info("Validation set class distribution:")
    for idx, count in sorted(val_counts.items()):
        logging.info(f"Class {idx}: {count} images")
    
    logging.info("All set class distribution:")
    for idx, count in sorted(all_counts.items()):
        logging.info(f"Class {idx}: {count} images")
    
    # Calculate total counts
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_all = sum(all_counts.values())
    
    logging.info(f"Total training images: {total_train}")
    logging.info(f"Total validation images: {total_val}")
    logging.info(f"Total images in all dataset: {total_all}")
    
    # Plot the distribution
    save_path = pathlib.Path("class_distribution_all.png")
    plot_class_distribution(train_counts, val_counts, all_counts, idx_to_class, save_path, sort_by_count=True)
    
    # Create a second plot showing the distribution as percentages
    plt.figure(figsize=(15, 8))
    
    # Create a list of (class_idx, train_percentage, val_percentage, all_percentage) tuples
    train_pct_by_class = {idx: (train_counts[idx] / total_train * 100) for idx in idx_to_class.keys()}
    val_pct_by_class = {idx: (val_counts[idx] / total_val * 100) for idx in idx_to_class.keys()}
    all_pct_by_class = {idx: (all_counts[idx] / total_all * 100) for idx in idx_to_class.keys()}
    
    # Create a list of (class_idx, train_pct, val_pct, all_pct) tuples
    pct_data = [(idx, train_pct_by_class[idx], val_pct_by_class[idx], all_pct_by_class[idx]) for idx in idx_to_class.keys()]
    
    # Sort by train percentage if requested
    pct_data.sort(key=lambda x: x[1], reverse=True)
    
    # Unpack the sorted data
    pct_classes = [str(item[0]) for item in pct_data]  # Convert to strings for labels
    train_pct = [item[1] for item in pct_data]
    val_pct = [item[2] for item in pct_data]
    all_pct = [item[3] for item in pct_data]
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(pct_classes))
    width = 0.25  # Adjusted width for three bars
    
    # Create the bars
    plt.bar(x - width, train_pct, width, label='Train')
    plt.bar(x, val_pct, width, label='Validation')
    plt.bar(x + width, all_pct, width, label='All Data')
    
    # Add labels, title, and legend
    plt.xlabel('Class Index (Sorted by Percentage)')
    plt.ylabel('Percentage of Images (%)')
    plt.title('Percentage of Images per Class in Train, Validation, and All Datasets')
    plt.xticks(x, pct_classes, rotation=90)
    plt.legend()
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for the class indices
    plt.tight_layout()
    
    # Save the percentage plot
    percentage_save_path = pathlib.Path("class_distribution_percentage_all.png")
    plt.savefig(percentage_save_path)
    logging.info(f"Percentage plot saved to {percentage_save_path}")

if __name__ == "__main__":
    main()
