from typing import Any, Type

import mujoco
import numpy as np

from utils.env import MujocoEnv


class MPPIController:
    def __init__(
        self,
        env_func: Type[MujocoEnv],
        env_arg: Any,
        lambda_: float,
        horizon: int,
        n_samples: int,
        discount: float,
        random_seed: int = 0,
    ):
        np.random.seed(random_seed)
        self._env = env_func(*env_arg)
        self._action_size = self._env.action_space.shape[0]
        self._lambda = lambda_
        self._horizon = horizon
        self._n_samples = n_samples
        self._actions = np.random.uniform(
            -1, 1, size=[self._horizon, self._action_size]
        )
        self._discount = np.power(discount, np.arange(self._horizon))

    def get_action(self, state: mujoco.MjData):
        actions, rewards = self._rollout(state)
        costs = -np.sum(rewards * self._discount, 1)
        weight = np.exp(-self._lambda * costs)
        if np.abs(np.sum(weight) - 0) < 1e-3:
            weight = np.zeros_like(weight)
        else:
            weight = weight / np.sum(weight)
        weight = weight.reshape(-1, 1, 1)
        self._actions = np.sum(actions * weight, axis=0)

        action = self._actions[0]
        self._actions = np.roll(self._actions, -1, axis=0)
        self._actions[-1] = 0
        return action

    def _sample_action(self) -> np.ndarray:
        noise = np.random.normal(
            0, 0.25, size=[self._n_samples, self._horizon, self._action_size]
        )
        actions = self._actions.reshape(1, *self._actions.shape).repeat(
            self._n_samples, axis=0
        )
        actions = np.clip(
            actions + noise, self._env.action_space.low, self._env.action_space.high
        )
        return actions

    def _rollout(self, state: mujoco.MjData):
        actions = self._sample_action()
        all_rewards = []
        for index in range(self._n_samples):
            rewards = []
            self._env.set_state(state)
            for action in actions[index]:
                _, reward, _, _ = self._env.step(action)
                rewards.append(reward)
            all_rewards.append(rewards)
        all_rewards = np.stack(all_rewards)
        return actions, all_rewards


def benchmark(
    env_func: Type[MujocoEnv],
    env_arg: Any,
    lambda_: float,
    horizon: int,
    n_samples: int,
    discount: float,
    eplen: int,
    repeat_time: int = 1,
):
    env = env_func(*env_arg)
    controller = MPPIController(
        env_func, env_arg, lambda_, horizon, n_samples, discount
    )
    rewards = np.zeros(repeat_time)
    for i in range(repeat_time):
        env.reset()
        reward = []
        for _ in range(eplen):
            action = controller.get_action(env.get_state())
            _, r, d, _ = env.step(action)
            reward.append(r)
            if d:
                break
        rewards[i] = sum(reward) / len(reward)
    return rewards.mean()
