import torch
import pandas as pd
import tqdm
from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model
from torch.utils.data import DataLoader
from PIL import Image
import os
import pathlib
from nycu_cv_hw1.model import Model
import torchvision.transforms as transforms

# Directories for the model and data
DATA_DIR_PATH = pathlib.Path("data")
MODEL_DIR_PATH = pathlib.Path("models")

# Initialize config
config = Config("config.yaml")


# Custom Dataset class for test images
class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img_name = os.path.basename(img_path).split(".")[
            0
        ]  # Extract the image name without extension
        return img, img_name


# Load the model
def load_model(model_path):
    num_classes = 100
    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config.backbone_model, num_classes).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model


# Get test data loader
def get_test_loader():

    tf = transforms.Compose(
        [
            transforms.RandomRotation(10),  # Random rotation between -10 and 10 degrees
            transforms.RandomResizedCrop(
                224
            ),  # Randomly crop and resize to 224x224 (adjust if needed)
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.ColorJitter(brightness=0.3),  # Simulate color jitter (like zoom)
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)
            ),  # Random translation (width_shift and height_shift)
            transforms.GaussianBlur(kernel_size=3),  # 隨機模糊
            transforms.ToTensor(),  # Convert image to PyTorch tensor
        ]
    )

    test_dataset = CustomImageFolder(img_dir=str(DATA_DIR_PATH / "test"), transform=tf)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return test_loader


# Run inference on the test set
def infer(device, model, test_loader):
    predictions = []
    with torch.no_grad():

        for inputs, img_names in tqdm.tqdm(
            test_loader, desc="Running Inference", ncols=100
        ):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Store image names and predictions
            for idx, pred in enumerate(preds):
                image_name = img_names[idx]
                predictions.append([image_name, pred.item()])
    return predictions


# Save predictions to CSV
def save_predictions(predictions):
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])

    # Sort predictions by pred_label in ascending order
    df = df.sort_values(by="pred_label", ascending=True)

    # Save the sorted predictions to CSV
    df.to_csv("prediction.csv", index=False)
    print("Prediction saved to prediction.csv")


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = (
        MODEL_DIR_PATH / "epoch_100_lr_0.0001_batch_128_20250324_041750.pt"
    )  # Replace with the correct model file name
    model = load_model(model_path).to(device)

    test_loader = get_test_loader()

    # Perform inference
    predictions = infer(device, model, test_loader)

    # Save the predictions to a CSV file
    save_predictions(predictions)


if __name__ == "__main__":
    main()
