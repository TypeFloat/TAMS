import time
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Optional, Type

import numpy as np
import torch

from src.env import MujocoEnv, SingleEnv
from src.mppi import MPPIController
from src.robot import CodeNode
from utils.config import Config
from utils.logger import Logger
from utils.terrain import (
    generate_barrier,
    generate_gap,
    generate_incline,
    generate_plane,
    generate_stair,
)


class Benchmark:
    def __init__(self, algorithm: Any, random: bool = False) -> None:
        self._cfg = Config()
        self._alg = algorithm
        self._terrain_list = []
        if random:
            extension = "_random"
        else:
            extension = ""

        self._terrain_list = [
            {
                "call": generate_plane,
                "args": (random,),
                "name": "plane" + extension,
            },
            {
                "call": generate_stair,
                "args": (self._cfg.SIM.STAIR_FACTOR, random),
                "name": f"stair_{self._cfg.SIM.STAIR_FACTOR}" + extension,
            },
            {
                "call": generate_barrier,
                "args": (self._cfg.SIM.BARRIER_FACTOR, random),
                "name": f"barrier_{self._cfg.SIM.BARRIER_FACTOR}" + extension,
            },
            {
                "call": generate_incline,
                "args": (self._cfg.SIM.INCLINE_FACTOR, random),
                "name": f"incline_{self._cfg.SIM.INCLINE_FACTOR}" + extension,
            },
            {
                "call": generate_gap,
                "args": (self._cfg.SIM.GAP_FACTOR, random),
                "name": f"gap_{self._cfg.SIM.GAP_FACTOR}" + extension,
            },
        ]

    def run(self):
        for i in range(len(self._terrain_list)):
            self.run_per_terrain(i)

    def run_per_terrain(self, index: int):
        st_time = time.time()
        terrain_func = self._terrain_list[index]
        terrain_name = terrain_func["name"]
        Logger.info(f"Cross the {index + 1}-th environment，the terrain type is {terrain_name}")
        target_fitness = np.inf
        best_robot = self._alg.get_best_robot(index, terrain_func, target_fitness)

        env = SingleEnv(
            random_code=0,
            target=self._cfg.SIM.SEMI_LENGTH * 2,
            robot_node=best_robot,
            generate_xml=True,
            map_id=terrain_name,
            n_step=self._cfg.SIM.N_STEP,
        )
        controller = MPPIController(
            SingleEnv,
            (
                0,
                self._cfg.SIM.SEMI_LENGTH * 2,
                best_robot,
                False,
                terrain_name,
                self._cfg.SIM.N_STEP,
            ),
            lambda_=self._cfg.MPPI.LAMBDA,
            horizon=self._cfg.MPPI.HORIZON,
            n_samples=self._cfg.MPPI.N_SAMPLES,
            discount=self._cfg.MPPI.DISCOUNT,
            random_seed=self._cfg.RANDOM_SEED,
        )

        env.reset()
        d = False
        step = 0
        while not d and step < 1000:
            action = controller.get_action(env.get_state())
            _, _, d, _ = env.step(action)
            step += 1
        Logger.info(f"Finish crossing，use {step} steps，running time is {time.time() - st_time}s")

    def get_terrain_list(self):
        return self._terrain_list


def eval_func(
    index: int,
    pop: Optional[CodeNode],
    target: float,
    map_id: str,
    n_step: int,
    lambda_: float,
    horizon: int,
    n_samples: int,
    discount: int,
    eplen: int,
    repeat_time: int,
):
    if pop is None:
        return -100
    else:
        return benchmark(
            SingleEnv,
            (index + 1, target, pop, True, map_id, n_step),
            lambda_=lambda_,
            horizon=horizon,
            n_samples=n_samples,
            discount=discount,
            eplen=eplen,
            repeat_time=repeat_time,
            random_seed=0,
        )


def benchmark(
    env_func: Type[MujocoEnv],
    env_arg: Any,
    lambda_: float,
    horizon: int,
    n_samples: int,
    discount: float,
    eplen: int,
    repeat_time: int = 1,
    random_seed: int = 0,
):
    env = env_func(*env_arg)
    controller = MPPIController(
        env_func, env_arg, lambda_, horizon, n_samples, discount, random_seed
    )
    rewards = np.zeros(repeat_time)
    steps = np.zeros(repeat_time) + eplen
    for i in range(repeat_time):
        env.reset()
        ep_rewards = []
        for j in range(eplen):
            action = controller.get_action(env.get_state())
            _, _, d, info = env.step(action)
            ep_rewards.append(info["forward_reward"] + info["healthy_reward"])
            if d:
                if info["healthy_reward"] < 0:
                    return -100
                else:
                    break

        rewards[i] = sum(ep_rewards) / steps[i]
    return rewards.mean()


def de_fitness_function(
    pops: np.ndarray,
    alg: Any,
    map_id: str,
) -> np.ndarray:
    config = Config()
    collected_pops = []
    for pop in pops:
        pop = torch.as_tensor(pop, dtype=torch.float32)
        rule_list = alg.gvae.decode(pop, alg.data_util)[0]
        Logger.debug(rule_list)
        if rule_list is None:
            collected_pops.append(None)
        else:
            robot_node = alg.data_util.get_morph_from_rule(rule_list)
            collected_pops.append(robot_node)
    y = []
    parallel_fuc = partial(
        eval_func,
        target=config.SIM.SEMI_LENGTH * 2,
        map_id=map_id,
        n_step=config.SIM.N_STEP,
        lambda_=config.MPPI.LAMBDA,
        horizon=config.MPPI.HORIZON,
        n_samples=config.MPPI.N_SAMPLES,
        discount=config.MPPI.DISCOUNT,
        eplen=config.MPPI.EPLEN,
        repeat_time=config.MPPI.REPEAT_TIME,
    )

    with Pool(cpu_count()) as pool:
        for index, pop in enumerate(collected_pops):
            result = pool.apply_async(
                func=parallel_fuc,
                args=(
                    index,
                    pop,
                ),
                error_callback=print,
            )
            y.append(result)
        pool.close()
        pool.join()
    y = [i.get() for i in y]
    y = np.array(y)

    if y.max() > alg.best_fitness:
        alg.best_robot = collected_pops[y.argmax()]
        alg.best_fitness = y.max()

    return np.array(y)
