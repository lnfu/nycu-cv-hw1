import logging
import pathlib
import typing

import yaml

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
    def num_epoch(self):
        return int(self._config["training"]["epochs"])

    @property
    def batch_size(self):
        return int(self._config["training"]["batch_size"])

    @property
    def lr(self):
        return float(self._config["training"]["learning_rate"])

    @property
    def weight_decay(self):
        return float(self._config["optimizer"]["weight_decay"])
