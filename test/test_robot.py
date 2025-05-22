import os
import sys

sys.path.append(os.getcwd())

from src.robot import RobotGenerator
from utils.config import Config
from utils.data_utils import DataUtils


def test():
    cfg = Config()
    cfg.load("config/default.json")
    data_utils = DataUtils()
    rg = RobotGenerator()
    rule_list = [0, 1, 14, 33, 7, 34, 29, 37, 32, 34, 29, 37, 32, 39, 39, 39, 39]
    morph = data_utils.get_morph_from_rule(rule_list)
    rg.generate(morph, [0, 0.2, 0.2], 0, "plane")


if __name__ == "__main__":
    test()
