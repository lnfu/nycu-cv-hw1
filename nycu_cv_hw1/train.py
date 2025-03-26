import click

import datetime
import logging
import pathlib
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
import torchvision
import tqdm
import sklearn.utils.class_weight as class_weight
import seaborn as sns
import matplotlib.pyplot as plt

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model
from nycu_cv_hw1.transform import train_transform
from sklearn.metrics import confusion_matrix


DATA_DIR_PATH = pathlib.Path("data")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)


def get_data_loaders(config: Config):

    all_dataset = torchvision.datasets.ImageFolder(
        root=DATA_DIR_PATH / "all",
        transform=train_transform,
        is_valid_file=lambda path: path.endswith(".jpg"),
    )
    cw = torch.FloatTensor(
        class_weight.compute_class_weight(
            "balanced", classes=np.unique(all_dataset.targets), y=all_dataset.targets
        )
    )
    logging.info(f"dataset size = {len(all_dataset)}")
    train_size = int(0.8 * len(all_dataset))  # TODO config
    val_size = len(all_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_dataset, [train_size, val_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )
    return train_loader, val_loader, len(all_dataset.class_to_idx), cw


def train(
    config: Config,
    device: torch.torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: tensorboard.writer.SummaryWriter,
    cw: torch.Tensor,  # class_weight
) -> int:

    if config.use_class_weight:
        loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            weight=cw.to(device),
            # reduction="mean" # TODO
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    real_num_epoch = 0

    for epoch in range(config.num_epoch):
        real_num_epoch += 1

        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        train_correct, val_correct = 0, 0
        train_size, val_size = 0, 0

        train_all_preds, train_all_labels = [], []
        model.train()
        for inputs, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
        ):
            optimizer.zero_grad()

            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)  # torch.Size([32])

            outputs = model(inputs)  # torch.Size([32, 100]) 機率
            preds = torch.argmax(outputs, dim=1)

            loss: torch.Tensor = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()  # 預測正確
            train_size += inputs.size(0)
            train_all_preds.extend(preds.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

        if train_size > float(0):
            train_loss = float(train_loss) / float(train_size)
            train_acc = float(train_correct) / float(train_size)

        model.eval()
        val_all_preds, val_all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
            ):
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                loss: torch.Tensor = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_size += inputs.size(0)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        scheduler.step()

        if val_size > float(0):
            val_loss = float(val_loss) / float(val_size)
            val_acc = float(val_correct) / float(val_size)

        logging.info(
            f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

        train_cm = confusion_matrix(val_all_labels, val_all_preds)
        val_cm = confusion_matrix(val_all_labels, val_all_preds)
        writer.add_figure(
            "Confusion Matrix (train)",
            get_confusion_matrix_figure(train_cm),
            epoch + 1,
        )
        writer.add_figure(
            "Confusion Matrix (val)",
            get_confusion_matrix_figure(val_cm),
            epoch + 1,
        )

        writer.add_scalars(
            "Loss", {"Train": train_loss, "Validation": val_loss}, epoch + 1
        )
        writer.add_scalars(
            "Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch + 1
        )

        if epoch % 5 == 0:
            torch.save(model, MODEL_DIR_PATH / f"final_{epoch}.pt")

    return real_num_epoch


def get_confusion_matrix_figure(cm):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=False, yticklabels=False
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    return fig


@click.command()
@click.argument("config_file", type=click.Path(exists=True), default="config.yaml")
def main(config_file):

    config = Config(config_file)

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    train_loader, val_loader, num_classes, cw = get_data_loaders(config)

    # Model
    model = Model(config.backbone_model, num_classes).to(device)

    # Training
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_batch_{config.batch_size}_{current_time}.pt"
    writer = tensorboard.writer.SummaryWriter(log_dir=LOG_DIR_PATH / filename)
    logging.info(f"{filename}")

    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.fc.parameters(), "lr": 0.1},
            {"params": model.backbone.layer4.parameters(), "lr": 0.01},
            {"params": model.backbone.layer3.parameters(), "lr": 0.005},
            {"params": model.backbone.layer2.parameters(), "lr": 0.001},
            {"params": model.backbone.layer1.parameters(), "lr": 0.0005},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    real_num_epoch = 0

    try:
        real_num_epoch = train(
            config=config,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
            cw=cw,
        )
    except KeyboardInterrupt:
        logging.warning("Training interrupted! Saving model before exiting...")
    finally:

        torch.save(model, MODEL_DIR_PATH / filename)
        logging.info(f"Model saved as {filename}")
        print(real_num_epoch)  # TODO
        print(filename)
        writer.close()


if __name__ == "__main__":
    main()
