import logging
import pathlib
import typing

import torch
import torch.utils.tensorboard as tensorboard
import torchvision
import tqdm
from PIL import Image, ImageFile

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.model import Model

DATA_DIR_PATH = pathlib.Path("data")

config = Config("config.yaml")


class TestDataset(torch.utils.data.Dataset):
    image_file_paths: typing.List[pathlib.Path]

    def __init__(
        self,
        image_dir_path: typing.Union[str, pathlib.Path],
        transform: typing.Optional[typing.Callable] = None,
    ):
        if isinstance(image_dir_path, str):
            image_dir_path = pathlib.Path(image_dir_path)

        # TODO 如果不是 pathlib.Path raise Exception

        self.image_file_paths = sorted(
            pathlib.Path(image_dir_path).glob("*.jpg")
        )  # 讀取所有圖片
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, idx) -> typing.Tuple[ImageFile.ImageFile, str]:
        img_path = self.image_file_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path.stem


def main():

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    num_classes = 100  # TODO
    backbone = torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.DEFAULT, progress=True
    )
    model = torch.load("models/20250312_064812.pt", weights_only=False)
    model.eval()

    transform = torchvision.models.ResNet101_Weights.DEFAULT.transforms()
    test_dataset = TestDataset(
        image_dir_path=DATA_DIR_PATH / "test", transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1
    )


    print("image_name,pred_label")

    for inputs, image_names in test_loader:  # TODO inputs or images (naming)
        inputs = inputs.to(device)  # TODO type hint

        outputs = model(inputs)

        # TODO 印出來, 一個 row 一筆資料 (output, image_name)
        # image_name,pred_label

        for output, image_name in zip(outputs, image_names):
            _, index = torch.max(output, dim=0)
            print(image_name, end=",")
            print(index.item(), end="\n")


if __name__ == "__main__":
    main()
