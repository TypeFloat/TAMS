from typing import List, Optional, Union

import gym
import numpy as np
import torch
from gym.spaces import Box
from sutils.simulation.env.mujoco_env import MujocoEnv

from src.robot import CodeNode, RobotGenerator

gym.logger.set_level(40)


class ObsType(dict):
    def to(self, device: torch.device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class BaseEnv(MujocoEnv):

    def __init__(
        self,
        target: float,
        random_code: Union[int, str],
        robot_node: Optional[CodeNode],
        generate_xml: bool,
        map_id: Optional[str] = None,
        init_loc: Optional[List[float]] = None,
        n_step: int = 5,
        z_reward_weight: float = 1.0,
    ) -> None:
        self._target = target
        self._last_position = 0.0

        if generate_xml:
            rg = RobotGenerator()
            rg.generate(robot_node, init_loc, random_code, map_id)
        if robot_node is not None:
            use_z_reward = (
                True if robot_node.left_node.left_node.direction == "Z" else False
            )
        else:
            use_z_reward = True
        super().__init__(
            f"assets/robot-{random_code}.xml",
            "body_0",
            n_step,
            use_healthy_reward=True,
            use_ctrl_reward=True,
            use_z_reward=use_z_reward,
            z_reward_weight=z_reward_weight,
            healthy_range=(-0.2, 2),
            init_step=n_step * 20,
        )
        self._actuator_num = self._model.actuator_ctrlrange.shape[0]
        self.action_space = Box(-1, 1, shape=(self._actuator_num,), dtype=np.float32)

    def get_done(self, pos_after: np.ndarray) -> bool:
        return pos_after[1] >= self._target

    def get_forward_reward(
        self, pos_before: np.ndarray, pos_after: np.ndarray
    ) -> float:
        return (pos_after[1] - pos_before[1]) / self._dt

    def get_obs(self):
        sensordata = self._data.sensordata
        loc = self.get_loc()
        return np.concatenate([sensordata, loc])

    def get_loc(self) -> np.ndarray:
        return self._data.xpos[self._head_id]


class SingleEnv(BaseEnv):
    def __init__(
        self,
        random_code: int,
        target: float,
        robot_node: Optional[CodeNode],
        generate_xml: bool,
        map_id: Optional[str] = None,
        n_step: int = 5,
    ) -> None:
        super().__init__(
            target,
            random_code,
            robot_node,
            generate_xml,
            map_id,
            [0, -0.2, 1.0],
            n_step,
        )


class ComplexEnv(BaseEnv):
    def __init__(
        self,
        random_code: int,
        n_step: int = 5,
    ) -> None:
        super().__init__(
            0,
            random_code,
            robot_node=None,
            generate_xml=False,
            n_step=n_step,
            z_reward_weight=2.0,
        )
        self.reset()
        pos = self._data.subtree_com[self._head_id].copy()
        self._target = pos[1] + 2.2
