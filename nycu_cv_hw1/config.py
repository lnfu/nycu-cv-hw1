import pathlib

import yaml


class Config:
    _config: dict

    def __init__(self, config_file_path: pathlib.Path | str):
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
        return self._config["training"]["epochs"]

    @property
    def batch_size(self):
        return self._config["training"]["batch_size"]
