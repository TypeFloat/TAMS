import os
import sys

sys.path.append(os.getcwd())

from utils.config import Config
from utils.logger import Logger
from utils.network import get_available_device

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
