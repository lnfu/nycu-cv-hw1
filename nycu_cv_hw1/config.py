import logging
import pathlib
import typing
import torchvision
import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",
)


class Config:
    _config: dict

    def __init__(
        self,
        config_file_path: typing.Union[str, pathlib.Path],
    ):
        if isinstance(config_file_path, str):
            config_file_path = pathlib.Path(config_file_path)

        if not isinstance(config_file_path, pathlib.Path):
            raise TypeError()

        if not config_file_path.exists():
            raise FileNotFoundError()

        with open(config_file_path, "r") as config_file:
            try:
                self._config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logging.error(f"Failed to parse YAML file '{config_file_path}': {exc}")
                raise ValueError()

    @property
    def num_epoch(self) -> int:
        """訓練次數, 預設值 100"""
        return int(self._config.get("training", {}).get("epochs", 100))

    @property
    def batch_size(self) -> int:
        """batch size, 預設值 32"""
        return int(self._config.get("training", {}).get("batch_size", 32))

    @property
    def inference_model(self) -> str:
        return str(self._config["model"]["inference"])

    @property
    def label_smoothing(self) -> float:
        return float(self._config["loss"].get("label_smoothing", 0.0))

    @property
    def use_class_weight(self) -> bool:
        return bool(self._config["loss"].get("class_weight", False))

    @property
    def default_transform(self):
        """
        預設 Resnet 使用的 transform
        """
        backbone = str(self._config["model"]["backbone"]).lower()
        transform_map = {
            "resnet18": torchvision.models.ResNet18_Weights.DEFAULT.transforms(),
            "resnet34": torchvision.models.ResNet34_Weights.DEFAULT.transforms(),
            "resnet50": torchvision.models.ResNet50_Weights.DEFAULT.transforms(),
            "resnet101": torchvision.models.ResNet101_Weights.DEFAULT.transforms(),
            "resnext101_64x4d": torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT.transforms(),
        }

        if backbone not in transform_map:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        return transform_map[backbone]

    @property
    def backbone_model(self) -> torch.nn.Module:
        """
        backbone model

        resnet18 | resnet34 | resnet50 | resnet101
        """
        backbone = str(self._config["model"]["backbone"]).lower()
        pretrained = bool(self._config["model"]["pretrained"])

        model_map = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnext101_64x4d": torchvision.models.resnext101_64x4d,
        }

        if backbone not in model_map:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        weights = None
        if pretrained:
            weights_map = {
                "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
                "resnext101_64x4d": torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT,
            }
            weights = weights_map.get(backbone)

        return model_map[backbone](weights=weights, progress=True)
