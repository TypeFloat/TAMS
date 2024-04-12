import os
import sys

sys.path.append(os.getcwd())

from sutils.exp.config import Config
from sutils.exp.logger import Logger
from sutils.network import get_available_device

from src.gvae import train

if __name__ == "__main__":
    root = f"log/pretrain"
    level = Logger.INFO
    Logger.init_logger(root, level)

    config = Config()
    config.load("config/default.json")
    config.dump(Logger.ROOT + "/config.json")
    config.DEVICE = get_available_device()

    train()
