import json
import logging
import os
import pickle
import time
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from concurrent_log_handler import ConcurrentRotatingFileHandler
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    ERROR = logging.ERROR
    ROOT = ""

    @staticmethod
    def init_logger(root: str, level: int = INFO) -> None:
        if not os.path.exists(root):
            os.makedirs(root)
        time_str = time.strftime('%m-%d-', time.localtime(time.time()))
        index = 0
        while os.path.exists(f"{root}/{time_str}{index}"):
            index += 1
        root = root + '/' + time_str + str(index)
        os.makedirs(root)
        filename = f'{root}/log.log'
        logging.basicConfig(
            format='【%(levelname)s】%(message)s',
            level=level,
            handlers=[
                ConcurrentRotatingFileHandler(filename, "a", 1024 * 1024 * 10, 10)
            ],
        )
        Logger.ROOT = root

    @staticmethod
    def info(message: str) -> None:
        logging.info(message)

    @staticmethod
    def warning(message: str) -> None:
        logging.warning(message)

    @staticmethod
    def debug(message: str):
        logging.debug(message)

    @staticmethod
    def error(message: str):
        logging.error(message)

    @staticmethod
    def save_data(filename: str, suffix: str, obj: Any) -> None:
        if suffix == "npy":
            with open(f"{Logger.ROOT}/{filename}.npy", "wb") as f:
                np.save(f, obj)
        elif suffix == "pkl":
            with open(f"{Logger.ROOT}/{filename}.pkl", "wb") as f:
                pickle.dump(obj, f)
        elif suffix == "json":
            with open(f"{Logger.ROOT}/{filename}.json", "w") as f:
                json.dump(obj, f, indent=4)
        elif suffix == "pth":
            with open(f"{Logger.ROOT}/{filename}.pth", "wb") as f:
                torch.save(obj, f)
        elif suffix == "txt":
            with open(f"{Logger.ROOT}/{filename}.txt", "w") as f:
                f.write(obj)
        elif suffix == "png":
            plt.clf()
            plt.plot(obj)
            plt.savefig(f"{Logger.ROOT}/{filename}.png")
        else:
            raise NotImplementedError

    @staticmethod
    def load_data(filename: str, suffix: str) -> Any:
        if suffix == "pkl":
            with open(f"{Logger.ROOT}/{filename}.pkl", "rb") as f:
                data = pickle.load(f)
        else:
            raise NotImplementedError
        return data

    @staticmethod
    def log_num_of_parameters(model: nn.Module, model_name: str):
        Logger.info(
            f"{model_name}, the total number of model parameters is：{sum([param.nelement() for param in model.parameters()])}"
        )

    @staticmethod
    def tb_add_scalar(
        writer: SummaryWriter, tag: str, scalar_value: float, global_step: int = None
    ) -> None:
        writer.add_scalar(tag, scalar_value, global_step)

    @staticmethod
    def tb_add_scalars(
        writer: SummaryWriter,
        tag: str,
        scalar_value_dict: Dict[str, float],
        global_step: int = None,
    ) -> None:
        writer.add_scalars(tag, scalar_value_dict, global_step)

    @staticmethod
    def run_time(info: str):
        def outer_wrapper(func: Callable):
            def inner_wrapper(*args, **kwargs):
                start = time.time()
                res = func(*args, **kwargs)
                end = time.time()
                Logger.debug(f"{info} running time：{end - start:.2f}s")
                return res

            return inner_wrapper

        return outer_wrapper
