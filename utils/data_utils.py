import json
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import apted
import numpy as np
import torch
from apted.helpers import Tree
from sklearn.model_selection import train_test_split
from utils.config import Config
from torch.utils.data import DataLoader

from src.robot import CodeNode


class DataUtils:

    def __init__(self) -> None:
        config = Config()
        self._n_rule = config.GVAE.NRULE
        self._max_len = config.GVAE.MAX_LEN
        self._min_joints = config.SIM.MIN_JOINTS
        with open("config/rule.json", "r") as f:
            self._rules = json.load(f)

    def _sample(self, weight: np.ndarray) -> int:

        probs = weight / np.sum(weight)
        return np.random.choice(weight.size, size=1, p=probs)[0]

    def _update(
        self,
        node: CodeNode,
        left_attr: str,
        right_attr: str,
        stack: List[CodeNode],
    ) -> None:

        if node.elem_type == "INIT":
            left_node = CodeNode("START", "O", "O")
            right_node = CodeNode("E", "O", "O")
        else:
            left_node = CodeNode(left_attr[0], left_attr[1], left_attr[2])
            right_node = CodeNode(right_attr[0], right_attr[1], right_attr[2])
        node.set_child_node(left_node, right_node)
        if right_node.elem_type != "E" and right_node.elem_type != "W":
            stack.append(right_node)
        if left_node.elem_type != "E" and left_node.elem_type != "W":
            stack.append(left_node)

    def _get_node_info(self, node: CodeNode) -> Tuple[List[List[str]], int, np.ndarray]:

        if node.elem_type == "INIT":
            rules = self._rules["INIT"]["RULE"]
            num = 0
        elif node.elem_type == "START":
            rules = self._rules["START"]["RULE"]
            num = self._rules["INIT"]["RULE_NUM"]
        elif node.elem_type == "B":
            rules = self._rules["B"]["RULE"]
            num = self._rules["INIT"]["RULE_NUM"] + self._rules["START"]["RULE_NUM"]

        elif node.elem_type == "S":
            rules = self._rules["S"]["RULE"]
            num = (
                self._rules["INIT"]["RULE_NUM"]
                + self._rules["START"]["RULE_NUM"]
                + self._rules["B"]["RULE_NUM"]
            )
        else:
            if node.available == "B":
                rules = self._rules["JB"]["RULE"]
                num = (
                    self._rules["INIT"]["RULE_NUM"]
                    + self._rules["START"]["RULE_NUM"]
                    + self._rules["B"]["RULE_NUM"]
                    + self._rules["S"]["RULE_NUM"]
                )
            else:
                rules = self._rules["JS"]["RULE"]
                num = (
                    self._rules["INIT"]["RULE_NUM"]
                    + self._rules["START"]["RULE_NUM"]
                    + self._rules["B"]["RULE_NUM"]
                    + self._rules["S"]["RULE_NUM"]
                    + self._rules["JB"]["RULE_NUM"]
                )
        weight = np.zeros(self._n_rule)
        weight[num : num + len(rules)] = 1
        if node.elem_type == "JS" and node.pre_node == "S":
            direction = node.pre_node.direction
            if direction == "Y":
                weight[37] = 0
            elif direction == "y":
                weight[36] = 0
            elif direction == "Z":
                weight[39] = 0
            elif direction == "z":
                weight[38] = 0
        return rules, num, weight

    def generate_data(self, data_size: int) -> None:


        def is_repeat():
            for i in range(success_num):
                if np.sum(np.abs(datasets[i] - datasets[success_num])) == 0:
                    return True
            return False

        success_num = 0
        datasets = np.zeros((data_size, self._max_len), dtype=np.int64)
        masks = np.zeros(
            (data_size, self._max_len, self._n_rule),
            dtype=np.int64,
        )
        roots = []
        while success_num < data_size:
            datasets[success_num, :] = self._n_rule - 1
            masks[success_num, :, :] = 0
            root = CodeNode("INIT", "O", "O")
            global_stack = [root]
            for i in range(self._max_len):
                node = global_stack.pop()
                rule, num, weight = self._get_node_info(node)
                masks[success_num, i] = weight
                index = self._sample(masks[success_num, i])
                datasets[success_num, i] = index
                left_attr, right_attr = rule[index - num]
                self._update(
                    node,
                    left_attr,
                    right_attr,
                    global_stack,
                )
                if len(global_stack) == 0:
                    if root.get_joints_num() >= self._min_joints and not is_repeat():
                        roots.append(root)
                        masks[success_num, i + 1 :, -1] = 1
                        success_num += 1
                    break
        np.save(f"data/gvae_rules_{data_size}.npy", datasets)
        np.save(f"data/gvae_masks_{data_size}.npy", masks)

        _, similarities = parallel_function(roots)
        np.save(
            f"data/gvae_similarities_{data_size}.npy",
            similarities,
        )

    def load_data(
        self, data_size: int, include_similarity: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        path = f"data/gvae_rules_{data_size}.npy"
        if not os.path.exists(path):
            self.generate_data(data_size, include_similarity)
        data = np.load(path)
        path = f"data/gvae_masks_{data_size}.npy"
        masks = np.load(path)
        if include_similarity:
            path = f"data/gvae_similarities_{data_size}.npy"
            similarities = np.load(path)
        else:
            similarities = None
        return data, masks, similarities

    def get_morph_from_rule(self, rule_list: Union[List[int], np.ndarray]) -> CodeNode:

        root = CodeNode("INIT", "O", "O")
        stack = [root]
        for rule in rule_list:
            node = stack.pop()
            rules, num, _ = self._get_node_info(node)

            left_attr, right_attr = rules[int(rule) - num]
            self._update(node, left_attr, right_attr, stack)
            if len(stack) == 0:
                break
        assert root.get_joints_num() >= self._min_joints

        return root

    def initialize_stack(self, max_iter) -> None:
        self._morph_stacks = []
        self._roots = []
        for i in range(max_iter):
            root = CodeNode("INIT", "O", "O")
            self._roots.append(root)
            self._morph_stacks.append([root])
        init_prob = torch.zeros(max_iter, self._n_rule)
        init_prob[:, 0] = 1
        self.get_rule_from_prob(init_prob)

    def check_available(self, tgt: torch.Tensor) -> None:
        ret_tgt = []
        for i in range(tgt.size(0)):
            if (
                len(self._morph_stacks[i]) == 0
                and self._roots[i].get_joints_num() > self._min_joints
            ):
                ret_tgt.append(tgt[i].numpy())
            else:
                ret_tgt.append(None)
        return ret_tgt

    def get_rule_from_prob(self, probs: torch.Tensor) -> Optional[torch.Tensor]:
        indexs = []
        for i in range(probs.shape[0]):
            if len(self._morph_stacks[i]) == 0:
                index = self._n_rule - 1
            else:
                prob = probs[i]
                node = self._morph_stacks[i].pop()
                rule, num, weight = self._get_node_info(node)
                prob = prob.exp() * weight
                prob = prob / prob.sum()
                index = prob.argmax(dim=-1)
                left_attr, right_attr = rule[index - num]
                self._update(node, left_attr, right_attr, self._morph_stacks[i])
            indexs.append(index)

        return torch.Tensor(indexs).reshape((-1, 1)).to(torch.int64)


def get_str_from_tree(root: CodeNode):
    def recursive(node: CodeNode) -> str:
        if node.left_node is None and node.right_node is None:
            return "{O}"
        else:
            return (
                "{"
                + str_dict[node.elem_type + node.direction + node.available]
                + recursive(node.left_node)
                + recursive(node.right_node)
                + "}"
            )

    str_dict = {
        "STARTOO": "A",
        "BOO": "B",
        "BYO": "C",
        "BZO": "D",
        "SXO": "E",
        "SYO": "F",
        "SyO": "G",
        "SZO": "H",
        "SzO": "I",
        "JOB": "a",
        "JXB": "b",
        "JYB": "c",
        "JZB": "d",
        "JOS": "e",
        "JXS": "f",
        "JYS": "g",
        "JZS": "h",
    }
    return recursive(root.left_node)


def get_ted_from_tree(tree1: CodeNode, tree2: CodeNode) -> int:
    return apted.APTED(
        *map(Tree.from_text, [get_str_from_tree(tree1), get_str_from_tree(tree2)])
    ).compute_edit_distance()


def get_ted_from_strs(args: Tuple[List[str], int]) -> int:
    strs = args[0]
    index = args[1]
    similiraties = np.zeros(len(strs))
    for i, s in enumerate(strs[index:]):
        similiraties[i + index] = apted.APTED(
            *map(Tree.from_text, [s, strs[index]])
        ).compute_edit_distance()

    return similiraties


def parallel_function(robots: List[CodeNode]) -> np.ndarray:
    strs = []
    for robot in robots:
        strs.append(get_str_from_tree(robot))

    args = [(strs, i) for i in range(len(strs))]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        tasks = pool.map_async(get_ted_from_strs, args)
        tasks.wait()
        similarities = tasks.get()
    similarities = np.stack(similarities)
    similarities += similarities.T
    return strs, similarities


def get_data(one_hot: bool) -> Tuple[DataLoader, DataLoader]:

    config = Config()

    path = f"data/gvae_rules_{config.TRAIN_OF_GVAE.NUMSIZE}.npy"
    if not os.path.exists(path):
        DataUtils().generate_data(config.TRAIN_OF_GVAE.NUMSIZE)
    data = torch.from_numpy(np.load(path)).long()
    if one_hot:
        data = torch.nn.functional.one_hot(data, config.GVAE.NRULE).float()
    path = f"data/gvae_masks_{config.TRAIN_OF_GVAE.NUMSIZE}.npy"
    masks = torch.from_numpy(np.load(path)).long()

    index = np.arange(data.shape[0])
    train_index, eval_index = train_test_split(index, test_size=0.2, shuffle=True)
    train_data = data[train_index]
    train_masks = masks[train_index]
    eval_data = data[eval_index]
    eval_masks = masks[eval_index]

    path = f"data/gvae_similarities_{config.TRAIN_OF_GVAE.NUMSIZE}.npy"
    similarities = torch.from_numpy(np.load(path)).float()
    train_similarities = []
    for i in train_index:
        train_similarities.append(similarities[i, train_index])
    train_similarities = torch.stack(train_similarities)
    eval_similarities = []
    for i in eval_index:
        eval_similarities.append(similarities[i, eval_index])
    eval_similarities = torch.stack(eval_similarities)

    return (
        DataLoader(
            train_data, batch_size=config.TRAIN_OF_GVAE.BATCH_SIZE, shuffle=False
        ),
        DataLoader(
            train_masks,
            batch_size=config.TRAIN_OF_GVAE.BATCH_SIZE,
            shuffle=False,
        ),
        train_similarities,
        eval_data,
        eval_masks,
        eval_similarities,
    )
