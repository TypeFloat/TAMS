from copy import deepcopy

import cv2
import mujoco
import numpy as np
from gym import Env, spaces
from mujoco.renderer import Renderer
from typing_extensions import Tuple


class MujocoEnv(Env):
    def __init__(
        self,
        xml_path: str,
        head_name: str,
        n_step: int = 5,
        forward_reward_weight: int = 1,
        healthy_reward_weight: int = 1,
        ctrl_reward_weight: int = 1,
        contact_reward_weight: int = 1,
        z_reward_weight: int = 1,
        use_healthy_reward: bool = False,
        use_ctrl_reward: bool = False,
        use_contact_reward: bool = False,
        use_z_reward: bool = False,
        healthy_range: Tuple[float, float] = (0.2, 1.0),
        init_step: int = 32,
    ) -> None:
        super().__init__()
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._dt = self._model.opt.timestep * n_step
        self._n_step = n_step
        self._tracker = None
        self._tracker_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, "tracker"
        )
        self._fixed_cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed"
        )
        self._forward_reward_reward = forward_reward_weight
        self._healthy_reward_weight = healthy_reward_weight
        self._ctrl_reward_weight = ctrl_reward_weight
        self._contact_reward_weight = contact_reward_weight
        self._z_reward_weight = z_reward_weight
        self._use_healthy_reward = use_healthy_reward
        self._use_ctrl_reward = use_ctrl_reward
        self._use_contact_reward = use_contact_reward
        self._use_z_reward = use_z_reward
        self._healthy_range = healthy_range
        self._head_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, head_name
        )
        self.observation_space = self._set_obs_space()
        self.action_space = self._set_action_space()
        self._init_step = init_step

    def _init_tracker(self):
        width, height = self.render_size
        self._tracker = Renderer(self._model, height=height, width=width)

    def _set_obs_space(self):
        obs = self.get_obs()
        obs = np.zeros_like(obs)
        return spaces.Box(low=obs - np.inf, high=obs + np.inf, dtype=np.float32)

    def _set_action_space(self):
        bounds = self._model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return self.action_space.sample()

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._model

    @property
    def render_size(self) -> Tuple[int, int]:
        return self._model.vis.global_.offwidth, self._model.vis.global_.offheight

    def set_state(self, state: mujoco.MjData):
        self._data = deepcopy(state)

    def get_state(self) -> mujoco.MjData:
        return deepcopy(self._data)

    def get_obs(self):
        raise NotImplementedError

    def get_done(self, pos_after: np.ndarray) -> bool:
        raise NotImplementedError

    def get_forward_reward(
        self, pos_before: np.ndarray, pos_after: np.ndarray
    ) -> float:
        raise NotImplementedError

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        info = {}

        pos_before = self._data.subtree_com[self._head_id].copy()
        self.sim_step(actions)
        pos_after = self._data.subtree_com[self._head_id].copy()

        is_healthy = self._healthy_range[0] <= pos_after[2] <= self._healthy_range[1]

        done = self.get_done(pos_after) or not is_healthy
        reward = 0

        forward_reward = self.get_forward_reward(pos_before, pos_after)
        info["forward_reward"] = forward_reward
        reward += forward_reward

        if self._use_ctrl_reward:
            ctrl_reward = 0.5 * np.square(actions).mean()
            info["ctrl_reward"] = ctrl_reward
            reward -= self._ctrl_reward_weight * ctrl_reward
        if self._use_contact_reward:
            contact_reward = 0.5 * 1e-3 * np.square(self._data.cfrc_ext).mean()
            info["contact_reward"] = contact_reward
            reward -= self._contact_reward_weight * contact_reward
        if self._use_healthy_reward:
            healthy_reward = 0.0 if is_healthy else -100
            info["healthy_reward"] = healthy_reward
            reward += self._healthy_reward_weight * healthy_reward

        if self._use_z_reward:
            z_reward = (pos_after[2] - pos_before[2]) / self._dt
            info["z_reward"] = z_reward
            reward += self._z_reward_weight * z_reward

        info["reward"] = reward
        return self.get_obs(), reward, done, info

    def sim_step(self, action: np.ndarray) -> None:
        self._data.ctrl = action.clip(
            self._model.actuator_ctrlrange[:, 0], self._model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(self._model, self._data, self._n_step)
        if self._use_contact_reward:
            mujoco.mj_rnePostConstraint(self._model, self._data)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_step(self._model, self._data, self._init_step)
        return self.get_obs()

    def render(self, mode: str = 'human', cam_name: str = "tracker") -> np.ndarray:
        if cam_name == "tracker":
            cam_id = self._tracker_id
        else:
            cam_id = self._fixed_cam_id
        if self._tracker is None:
            self._init_tracker()
        self._tracker.update_scene(self._data, cam_id)
        image = self._tracker.render()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mode == 'human':
            cv2.imshow("renderer", image)
            cv2.waitKey(1)
        return image