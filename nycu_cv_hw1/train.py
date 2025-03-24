import datetime
import logging
import pathlib
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
import torchvision
import tqdm
import torchvision.transforms as transforms
import sklearn.utils.class_weight as class_weight

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model

DATA_DIR_PATH = pathlib.Path("data")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)

config = Config("config.yaml")

tf = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.2, saturation=0.2, brightness=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ]
)


def get_data_loaders():

    all_dataset = torchvision.datasets.ImageFolder(
        root=DATA_DIR_PATH / "all",
        transform=tf,
        is_valid_file=lambda path: path.endswith(".jpg"),
    )
    cw = torch.FloatTensor(
        class_weight.compute_class_weight(
            "balanced", classes=np.unique(all_dataset.targets), y=all_dataset.targets
        )
    )
    logging.info(f"dataset size = {len(all_dataset)}")
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_dataset, [train_size, val_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    return train_loader, val_loader, len(all_dataset.class_to_idx), cw


def train(
    device: torch.torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: tensorboard.writer.SummaryWriter,
    cw,  # class_weight TODO
):

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=0.1,
        weight=cw.to(device),
        # reduction="mean" # TODO
    )  # TODO label_smoothing=0.1?

    for epoch in range(config.num_epoch):

        train_loss, val_loss = 0.0, 0.0
        train_correct, val_correct = 0, 0
        train_size, val_size = 0, 0

        model.train()  # 轉成 training mode
        for inputs, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
        ):
            optimizer.zero_grad()

            inputs = inputs.to(device)  # TODO type hint
            labels = labels.to(device)  # TODO type hint torch.Size([32])

            outputs = model(inputs)  # torch.Size([32, 100]) 機率
            preds = torch.argmax(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # * inputs.size(0)
            train_correct += (preds == labels).sum().item()  # 預測正確
            train_size += inputs.size(0)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                loss = loss_fn(outputs, labels)

                val_loss += loss.item()  # * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_size += inputs.size(0)

        train_loss = float(train_loss) / float(train_size)
        val_loss = float(val_loss) / float(val_size)

        train_acc = float(train_correct) / float(train_size)
        val_acc = float(val_correct) / float(val_size)

        logging.info(
            f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

        # writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        # writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        # writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
        # writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)
        writer.add_scalars(
            "Loss", {"Train": train_loss, "Validation": val_loss}, epoch + 1
        )
        writer.add_scalars(
            "Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch + 1
        )


def main():

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    train_loader, val_loader, num_classes, cw = get_data_loaders()

    # Model
    model = Model(config.backbone_model, num_classes).to(device)

    # Training
    writer = tensorboard.writer.SummaryWriter(log_dir=LOG_DIR_PATH)

    optimizer = config.get_optimizer(model.parameters())

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=7, gamma=0.1
    # )  # TODO config

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    try:
        train(device, model, optimizer, scheduler, train_loader, val_loader, writer, cw)
    except KeyboardInterrupt:
        logging.warning("Training interrupted! Saving model before exiting...")
    finally:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epoch_{config.num_epoch}_lr_{config.lr}_batch_{config.batch_size}_{current_time}.pt"

        torch.save(model, MODEL_DIR_PATH / filename)
        logging.info(f"Model saved as {filename}")
        print(filename)
        writer.close()

    # torch.save(model.state_dict(), 'model.pt') # TODO only save weights


if __name__ == "__main__":
    main()
