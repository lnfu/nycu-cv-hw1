import logging
import pathlib
import typing

import torch
import torchvision
import tqdm
from PIL import Image, ImageFile

from nycu_cv_hw1.config import Config
from nycu_cv_hw1.transform import tta_transforms

DATA_DIR_PATH = pathlib.Path("data")
MODEL_DIR_PATH = pathlib.Path("models")

config = Config("config.yaml")

# INDEX -> CLASS
all_dataset = torchvision.datasets.ImageFolder(
    root=DATA_DIR_PATH / "all",
    is_valid_file=lambda path: path.endswith(".jpg"),
)
idx_to_class = {v: k for k, v in all_dataset.class_to_idx.items()}


class TtaTestDataset(torch.utils.data.Dataset):
    image_file_paths: typing.List[pathlib.Path]

    def __init__(
        self,
        image_dir_path: typing.Union[str, pathlib.Path],
        tta_transforms: typing.Optional[typing.Callable] = None,
    ):
        if isinstance(image_dir_path, str):
            image_dir_path = pathlib.Path(image_dir_path)

        if not isinstance(image_dir_path, pathlib.Path):
            raise ValueError()

        self.image_file_paths = sorted(pathlib.Path(image_dir_path).glob("*.jpg"))
        self.tta_transforms = tta_transforms

    def __len__(self) -> int:
        return len(self.image_file_paths)

    def __getitem__(self, idx) -> typing.Tuple[ImageFile.ImageFile, str]:
        img_path = self.image_file_paths[idx]
        image = Image.open(img_path).convert("RGB")

        image_list = []
        if self.tta_transforms:
            for transform in self.tta_transforms:
                t_image = transform(image)
                image_list.append(t_image)

        return image_list, img_path.stem


def main():

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    model = torch.load(
        MODEL_DIR_PATH / config.inference_model,
        weights_only=False,
    )

    tta_test_dataset = TtaTestDataset(
        image_dir_path=DATA_DIR_PATH / "test", tta_transforms=tta_transforms
    )
    tta_test_loader = torch.utils.data.DataLoader(
        tta_test_dataset, batch_size=config.batch_size, shuffle=False
    )

    print("image_name,pred_label")

    model.eval()
    for image_lists, image_names in tqdm.tqdm(tta_test_loader, desc="", ncols=100):
        outputs_list = []
        for inputs in image_lists:
            inputs: torch.Tensor = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                outputs_list.append(outputs)
        avg_outputs = torch.mean(torch.stack(outputs_list), dim=0)

        for output, image_name in zip(avg_outputs, image_names):
            index = torch.argmax(output, dim=0)
            print(image_name, end=",")
            print(idx_to_class[index.item()], end="\n")


if __name__ == "__main__":
    main()
