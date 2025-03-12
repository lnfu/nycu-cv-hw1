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

        # TODO 如果不是 pathlib.Path raise Exception

        if not config_file_path.exists():
            raise FileNotFoundError()

        if config_file_path.suffix != ".yaml" and config_file_path.suffix != ".yml":
            raise ValueError()

        with open(config_file_path, "r") as config_file:
            try:
                self._config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error parsing YAML file: {exc}")

    @property
    def num_epoch(self) -> int:
        return int(self._config["training"]["epochs"])

    @property
    def batch_size(self) -> int:
        return int(self._config["training"]["batch_size"])

    @property
    def lr(self) -> float:
        return float(self._config["training"]["learning_rate"])

    @property
    def weight_decay(self) -> float:
        return float(self._config["optimizer"]["weight_decay"])

    @property
    def backbone_model(self) -> torch.nn.Module:
        backbone = str(self._config["model"]["backbone"]).lower()
        pretrained = bool(self._config["model"]["pretrained"])

        model_map = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
        }

        if backbone not in model_map:
            raise ValueError()

        weights = None
        if pretrained:
            weights_map = {
                "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
            }
            weights = weights_map.get(backbone)

        return model_map[backbone](weights=weights, progress=True)
