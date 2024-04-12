import os
from typing import Tuple

import numpy as np

from .mujoco_env import MujocoEnv


class AntEnv(MujocoEnv):
    def __init__(self, n_step: int = 5) -> None:
        super().__init__(
            os.path.join(os.path.dirname(__file__), "../assets/ant.xml"),
            "torso0",
            n_step,
            healthy_range=(0.2, 0.75),
            init_step=32,
        )

    def get_obs(self):
        ...
