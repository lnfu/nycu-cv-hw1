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

        if config_file_path.suffix != ".yaml" and config_file_path.suffix != ".yml":
            raise ValueError()

        with open(config_file_path, "r") as config_file:
            try:
                self._config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                logging.error(f"Failed to parse YAML file '{config_file_path}': {exc}")
                raise ValueError(f"Error parsing YAML file: {exc}")

    @property
    def num_epoch(self) -> int:
        """返回訓練的 epoch 次數"""
        return int(self._config.get("training", {}).get("epochs", 100))  # 預設值 100

    @property
    def batch_size(self) -> int:
        """返回訓練的 batch size"""
        return int(self._config.get("training", {}).get("batch_size", 32))

    @property
    def lr(self) -> float:
        """返回學習率"""
        return float(self._config.get("optimizer", {}).get("learning_rate", 1e-3))

    @property
    def weight_decay(self) -> float:
        """返回權重衰減"""
        return float(self._config.get("optimizer", {}).get("weight_decay", 0.0))

    @property
    def optimizer_type(self) -> str:
        """
        返回優化器類型，並檢查其是否支持
        :raises ValueError: 若優化器類型無效
        """
        optimizer = str(self._config.get("optimizer", {}).get("type", "adam")).lower()
        if optimizer not in {"adam", "sgd", "rmsprop"}:
            raise ValueError(
                f"Unsupported optimizer '{optimizer}'. Must be one of: adam, sgd, rmsprop"
            )
        return optimizer

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        """
        根據 YAML 設定返回對應的 Optimizer
        :param model_params: 模型的參數
        :return: 相應的優化器實例
        """
        optimizer_type = self.optimizer_type
        lr = self.lr
        weight_decay = self.weight_decay
        momentum = float(self._config.get("optimizer", {}).get("momentum", 0.9))
        epsilon = float(self._config.get("optimizer", {}).get("epsilon", 1e-8))

        # 選擇對應的優化器
        if optimizer_type == "adam":
            return torch.optim.Adam(
                model_params, lr=lr, weight_decay=weight_decay, eps=epsilon
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                model_params, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                model_params, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f"Unexpected optimizer '{optimizer_type}'")

    @property
    def default_transform(self):
        """
        根據 backbone 模型返回預設的數據增強轉換
        :raises ValueError: 若 backbone 無效
        """
        backbone = str(self._config["model"]["backbone"]).lower()
        transform_map = {
            "resnet18": torchvision.models.ResNet18_Weights.DEFAULT.transforms(),
            "resnet34": torchvision.models.ResNet34_Weights.DEFAULT.transforms(),
            "resnet50": torchvision.models.ResNet50_Weights.DEFAULT.transforms(),
            "resnet101": torchvision.models.ResNet101_Weights.DEFAULT.transforms(),
        }

        if backbone not in transform_map:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        return transform_map[backbone]

    @property
    def backbone_model(self) -> torch.nn.Module:
        """
        返回指定 backbone 模型的實例
        :raises ValueError: 若 backbone 無效
        """
        backbone = str(self._config["model"]["backbone"]).lower()
        pretrained = bool(self._config["model"]["pretrained"])

        model_map = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
        }

        if backbone not in model_map:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        # 預訓練權重設置
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
