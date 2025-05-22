import argparse
import os
import sys

sys.path.append(os.getcwd())


from src.tams import TAMS
from utils.benchmark import Benchmark
from utils.config import Config
from utils.logger import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_env", action="store_true")
    args = parser.parse_args()

    root = f"log/tams"
    level = Logger.INFO
    Logger.init_logger(root, level)

    config = Config()
    config.load("config/default.json")
    config.dump(Logger.ROOT + "/config.json")
    config.DEVICE = "cpu"

    tams = TAMS()
    benchmark = Benchmark(tams, args.random_env)
    benchmark.run()
