import logging
import pathlib

import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, models, transforms
from tqdm import tqdm

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model

DATA_DIR_PATH = pathlib.Path("data")
config = Config("config.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",
)

# TODO 這是 data augmentation & normalization
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device: {device}")

train_dataset = datasets.ImageFolder(root=DATA_DIR_PATH / "train", transform=transform)
train_num_classes = len(train_dataset.class_to_idx)
train_loader = data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1
)

val_dataset = datasets.ImageFolder(root=DATA_DIR_PATH / "val", transform=transform)
val_num_classes = len(val_dataset.class_to_idx)
val_loader = data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
)

if train_num_classes != val_num_classes:
    raise Exception()  # TODO

backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT, progress=True)
for param in backbone.parameters():
    param.requires_grad = False

model = Model(backbone, train_num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(config.num_epoch):
    model.train()  # 轉成 training mode

    for inputs, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
    ):
        inputs = inputs.to(device)  # TODO type hint
        labels = labels.to(device)  # TODO type hint

        outputs = model(inputs)

        loss_criterion = torch.nn.CrossEntropyLoss()  # TODO
        loss = loss_criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        for inputs, labels in tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{config.num_epoch}", ncols=100
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels)


exit(0)


# print(sum(p.numel() for p in backbone.parameters()))

# input_image = Image.open("data/train/5/00e0f23c-1dfd-4032-a9bb-d5f0c0e346d6.jpg")
# preprocess = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to("cuda")
#     backbone.to("cuda")

# with torch.no_grad():
#     output = backbone(input_batch)

# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# # print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities.shape)
# _, index = torch.max(probabilities, dim=0)
# print(categories[index])


# [
#     "ResNeXt101_32X8D_Weights",
#     "ResNeXt101_64X4D_Weights",
#     "ResNeXt50_32X4D_Weights",
#     "ResNet",
#     "ResNet101_Weights",
#     "ResNet152_Weights",
#     "ResNet18_Weights",
#     "ResNet34_Weights",
#     "ResNet50_Weights",
#     "Wide_ResNet101_2_Weights",
#     "Wide_ResNet50_2_Weights",
#     "resnet",
#     "resnet101",
#     "resnet152",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnext101_32x8d",
#     "resnext101_64x4d",
#     "resnext50_32x4d",
#     "wide_resnet101_2",
#     "wide_resnet50_2",
# ]
