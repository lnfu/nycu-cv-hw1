import datetime
import logging
import pathlib

import torch
import torch.utils.tensorboard as tensorboard
import torchvision
import tqdm

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model

DATA_DIR_PATH = pathlib.Path("data")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")

config = Config("config.yaml")


def get_data_loaders():
    tf = torchvision.models.ResNet101_Weights.DEFAULT.transforms()
    train_dataset = torchvision.datasets.ImageFolder(
        root=DATA_DIR_PATH / "train", transform=tf
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=DATA_DIR_PATH / "val", transform=tf
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )
    if len(train_dataset.class_to_idx) != len(val_dataset.class_to_idx):
        raise ValueError()
    return train_loader, val_loader, len(train_dataset.class_to_idx)


def main():

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    train_loader, val_loader, num_classes = get_data_loaders()

    backbone = torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.DEFAULT, progress=True
    )
    for param in backbone.parameters():
        param.requires_grad = False

    model = Model(backbone, num_classes).to(device)

    writer = tensorboard.writer.SummaryWriter(log_dir=LOG_DIR_PATH)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(config.num_epoch):

        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0.0
        val_acc = 0.0

        model.train()  # 轉成 training mode
        for inputs, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
        ):
            inputs = inputs.to(device)  # TODO type hint
            labels = labels.to(device)  # TODO type hint

            outputs = model(inputs)

            loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean")  # TODO
            loss = loss_criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for inputs, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
                loss = loss_criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

        logging.info(
            f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

    writer.close()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model, MODEL_DIR_PATH / f"{current_time}.pt")
    # torch.save(model.state_dict(), 'model.pt') # only save weights


if __name__ == "__main__":
    main()
