import os
import shutil
import pathlib

DATA_DIR_PATH = pathlib.Path("data")


def aggregate_data(src_dirs, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate through source directories
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist, skipping...")
            continue

        # Iterate through class subdirectories in the source directory
        for label_dir in os.listdir(src_dir):
            src_class_dir = os.path.join(src_dir, label_dir)

            if os.path.isdir(src_class_dir):
                # Create class subdirectory in the destination directory if not exist
                dest_class_dir = os.path.join(dest_dir, label_dir)
                if not os.path.exists(dest_class_dir):
                    os.makedirs(dest_class_dir)

                # Copy files from source class directory to destination class directory
                for file_name in os.listdir(src_class_dir):
                    src_file = os.path.join(src_class_dir, file_name)
                    dest_file = os.path.join(dest_class_dir, file_name)

                    # Copy the file if it is not already copied
                    if os.path.isfile(src_file):
                        shutil.copy(src_file, dest_file)
                        print(f"Copied: {src_file} -> {dest_file}")


if __name__ == "__main__":
    train_dir = DATA_DIR_PATH / "train"
    val_dir = DATA_DIR_PATH / "val"
    all_dir = DATA_DIR_PATH / "all"

    aggregate_data([train_dir, val_dir], all_dir)
