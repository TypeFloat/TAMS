import time
from functools import partial
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from sutils.exp.config import Config
from sutils.exp.logger import Logger

from src.gvae import GVAE
from src.tan import TAN, patch, train_of_tan
from utils.benchmark import de_fitness_function
from utils.data_utils import DataUtils


class TAMS:
    def __init__(self) -> None:
        self._cfg = Config()
        self.tan = TAN()
        self.tan.to(self._cfg.DEVICE)
        self.data_util = DataUtils()
        self.gvae = GVAE()
        self.gvae.load_state_dict(torch.load("data/gvae.pth", map_location="cpu"))
        self.best_robot = None
        self.best_fitness = -np.inf
        self.initilize_tan()

    def get_best_robot(
        self, index: int, terrain_func: Dict[str, Any], target_fitness: float
    ) -> None:
        self.best_robot = None
        self.best_fitness = -np.inf

        terrain = terrain_func["call"](*terrain_func["args"])

        st_time = time.time()
        (
            best_x,
            best_y,
            population_history,
            fitness_history,
            best_fitness_hitory,
        ) = DE(
            eval_func=partial(
                de_fitness_function,
                alg=self,
                map_id=terrain_func["name"],
            ),
            alg=self,
            n_dim=self._cfg.GVAE.D_MU,
            pop_size=self._cfg.TAMS.DE.POP_SIZE,
            terrain=terrain,
            target_fitness=target_fitness,
        ).run(
            max_iter=self._cfg.TAMS.DE.MAX_ITER, record_history=True
        )
        Logger.info(f"Finish DE, running time is {time.time() - st_time}s")
        Logger.save_data(f"tan_{index}", "pth", self.tan.state_dict())
        log_info = {
            "population_history": population_history,
            "fitness_history": fitness_history,
            "best_fitness_history": best_fitness_hitory,
            "best_robot": self.best_robot,
        }
        Logger.save_data(f"iteration_{index}", "pkl", log_info)
        Logger.save_data(f"best_{index}", "pkl", self.best_robot)

        return self.best_robot

    def initilize_tan(self) -> None:
        self.tan = TAN()
        self.tan.to(self._cfg.DEVICE)
        self.max_val = self._cfg.TAMS.TAN.MAX_VAL
        self.min_val = self._cfg.TAMS.TAN.MIN_VAL

    def _process_terrain(self, terrain: np.ndarray, repeat_time: int) -> torch.Tensor:
        terrain = torch.as_tensor(terrain, dtype=torch.float32, device=self._cfg.DEVICE)
        terrain = terrain.unsqueeze(0)
        terrain.requires_grad = False
        terrain = patch(terrain)
        terrain = terrain.repeat((repeat_time, 1, 1))
        return terrain

    def optimize_tan(
        self, mu: np.ndarray, terrain: np.ndarray, labels: np.ndarray
    ) -> None:
        terrain = self._process_terrain(terrain, mu.shape[0])
        valid_index = labels != -100
        mu = mu[valid_index]
        terrain = terrain[valid_index]
        labels = labels[valid_index]
        mu = torch.as_tensor(mu, dtype=torch.float32)
        terrain = torch.as_tensor(terrain, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        labels = (labels - self._cfg.TAMS.TAN.MIN_VAL) / (
            self._cfg.TAMS.TAN.MAX_VAL - self._cfg.TAMS.TAN.MIN_VAL
        )
        train_of_tan(self.tan, mu, terrain, labels)

    def optimize_population(
        self, terrain: Union[torch.Tensor, np.ndarray], pop: np.ndarray
    ):
        terrain = self._process_terrain(terrain, pop.shape[0])
        mu = torch.as_tensor(pop, dtype=torch.float32)
        return self.tan(terrain, mu).detach().cpu().numpy()


class DE:
    def __init__(
        self,
        alg: TAMS,
        terrain: np.ndarray,
        eval_func: Callable[[np.ndarray], np.ndarray],
        n_dim: int,
        pop_size: int = 50,
        crl: float = 0.1,
        cru: float = 0.6,
        lower_bound: Union[np.ndarray, float] = -1.0,
        higher_bound: Union[np.ndarray, float] = 1.0,
        constraint_eq: List = [],
        constraint_ueq: List = [],
        target_fitness: float = np.inf,
    ):
        assert pop_size >= 3, "min pop_size is three"
        self._eval_func = eval_func
        self._n_dim = n_dim
        self._pop_size = pop_size
        self._crl = crl
        self._cru = cru
        if type(lower_bound) == float:
            self._lower_bound = np.array([lower_bound] * n_dim)
        else:
            self._lower_bound = lower_bound
        if type(higher_bound) == float:
            self._higher_bound = np.array([higher_bound] * n_dim)
        else:
            self._higher_bound = higher_bound

        self._have_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        self._constraint_eq = constraint_eq
        self._constraint_ueq = constraint_ueq
        self._target_fitness = target_fitness
        self._pop = None
        self._tamc = alg
        self._terrain = terrain

    def _create_population(self):
        # create the population
        population = np.random.uniform(
            low=self._lower_bound,
            high=self._higher_bound,
            size=(self._pop_size, self._n_dim),
        )
        return population

    def _get_fitness(self, pop: np.ndarray):
        y = self._eval_func(pop)

        if self._have_constraint:
            penalty_eq = np.array(
                [np.sum(np.abs([c_i(x) for c_i in self._constraint_eq])) for x in pop]
            )
            penalty_ueq = np.array(
                [
                    np.sum(np.abs([max(0, c_i(x)) for c_i in self._constraint_ueq]))
                    for x in pop
                ]
            )
            y -= 1e5 * penalty_eq + 1e5 * penalty_ueq
        return y

    def _mutation(self):
        '''
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        '''
        random_idx = []
        for _ in range(self._pop_size):
            idx = np.random.choice(self._pop_size, size=3, replace=False).tolist()
            idx.sort(reverse=True, key=lambda e: self._pop_fitness[e])
            random_idx.append(idx)
        random_idx = np.array(random_idx)
        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        # NOTE r1, r2, r3 should not be equal and f(r1) > f(r2) > f(r3)
        assert np.sum(r1 == r2) == 0
        assert np.sum(r1 == r3) == 0
        assert np.sum(r2 == r3) == 0

        if (
            np.sum(self._pop_fitness[r1] <= self._pop_fitness[r2]) == 0
            and np.sum(self._pop_fitness[r2] < self._pop_fitness[r3]) == 0
        ):
            factor = (
                0.1
                + 0.8
                * (self._pop_fitness[r2] - self._pop_fitness[r1])
                / (self._pop_fitness[r3] - self._pop_fitness[r1])
            ).reshape((-1, 1))
        else:
            factor = 0.5
        v_pop = self._pop[r1] + factor * (self._pop[r2] - self._pop[r3])

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(
            low=self._lower_bound,
            high=self._higher_bound,
            size=(self._pop_size, self._n_dim),
        )
        v_pop = np.where(v_pop < self._lower_bound, mask, v_pop)
        v_pop = np.where(v_pop > self._higher_bound, mask, v_pop)
        return v_pop

    def _crossover(self, v_pop: np.ndarray):
        '''
        if rand < prob_crossover, use V, else use X
        '''
        fmin = self._pop_fitness.min()
        fmax = self._pop_fitness.max()
        fmean = self._pop_fitness.mean()
        index = self._pop_fitness > fmean
        prob_cr = np.zeros((self._pop_size,)) + self._crl
        if fmax - fmin > 1e-3:
            prob_cr[index] += (
                (self._cru - self._crl)
                * (self._pop_fitness[index] - fmin)
                / (fmax - fmin)
            )
        mask = np.random.rand(self._pop_size, self._n_dim) < prob_cr.reshape(-1, 1)
        u_pop = np.where(mask, v_pop, self._pop)
        return u_pop

    def _selection(self, u_pop: np.ndarray):
        '''
        greedy selection
        '''
        u_fitness = self._get_fitness(u_pop)
        index = self._pop_fitness > u_fitness
        pop = np.where(index.reshape(-1, 1), self._pop, u_pop)
        pop_fitness = np.where(index, self._pop_fitness, u_fitness)
        return pop, pop_fitness, u_fitness

    def _optimize_population(self, terrain: np.ndarray):
        pop = []
        while len(pop) < self._pop_size:
            v_pop = self._mutation()
            u_pop = self._crossover(v_pop)
            predicted_fitness = self._tamc.optimize_population(terrain, u_pop)
            pop.append(u_pop[predicted_fitness.argmax()])
        return np.vstack(pop)

    def run(
        self,
        max_iter: int = 200,
        record_history: bool = False,
    ):
        if record_history:
            best_fitness_history = []
            poplulation_history = []
            fitness_history = []

        st_time = time.time()
        self._pop = self._create_population()
        Logger.info("Finish population initializing")
        self._pop_fitness = self._get_fitness(self._pop)
        Logger.info(
            f"init best_fitness: {self._pop_fitness.max()}, used time: {time.time() - st_time}"
        )

        self._tamc.optimize_tan(
            self._pop.copy(), self._terrain.copy(), self._pop_fitness.copy()
        )

        if record_history:
            fitness_history.append(self._pop_fitness.copy())
            poplulation_history.append(self._pop.copy())
            best_fitness_history.append(self._pop_fitness.max())

        index = 0
        while True:
            st_time = time.time()
            t_pop = self._optimize_population(self._terrain)
            self._pop, self._pop_fitness, t_fitness = self._selection(t_pop)
            self._tamc.optimize_tan(
                t_pop.copy(), self._terrain.copy(), t_fitness.copy()
            )

            if record_history:
                fitness_history.append(t_fitness.copy())
                poplulation_history.append(t_pop.copy())
                best_fitness_history.append(self._pop_fitness.max())
            Logger.info(
                f"iter: {index}, best_fitness: {self._pop_fitness.max()}, used time: {time.time() - st_time}"
            )
            Logger.debug(t_fitness.__str__())
            if self._pop_fitness.max() > self._target_fitness:
                break
            index += 1
            if max_iter > 0 and index > max_iter:
                break

        best_index = self._pop_fitness.argmax()
        best_x = self._pop[best_index]
        if self._have_constraint:
            best_y = self._eval_func(best_x.reshape(1, -1))[0]
        else:
            best_y = self._pop_fitness[best_index]

        if record_history:
            return (
                best_x,
                best_y,
                poplulation_history,
                fitness_history,
                best_fitness_history,
            )
        else:
            return best_x, best_y
