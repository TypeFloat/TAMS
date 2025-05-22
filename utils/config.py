import json
from enum import Enum
from threading import RLock
from typing import Any

from easydict import EasyDict


class Singleton(type):
    single_lock = RLock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instance


class Config(metaclass=Singleton):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["_config_type", "_config"]:
            self.__dict__[__name] = __value
        else:
            self._config[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        if __name in ["_config", "_config_type"]:
            raise AttributeError("Uninitialized configuration")
        attr = self._config.get(__name, None)
        if attr is not None:
            return attr
        else:
            raise AttributeError(f"Attempting to access a configuration that does not exist, {__name}")

    def load(self, path: str):
        config_type = path.split(".")[-1]
        with open(path, 'r') as f:
            if config_type == "json":
                self._config_type = "json"
                config = json.load(f)
            else:
                raise NotImplementedError
        self._config = EasyDict(config)

    def dump(self, path: str):
        config_type = path.split(".")[-1]
        with open(path, 'w') as f:
            if self._config_type == config_type:
                json.dump(self._config, f, indent=4)
            else:
                raise NotImplementedError
