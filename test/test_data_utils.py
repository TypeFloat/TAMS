import unittest

import numpy as np

from src.robot import CodeNode
from utils.config import Config
from utils.data_utils import (
    DataUtils,
    get_str_from_tree,
    get_ted_from_tree,
    parallel_function,
)


class DataUtilsTest(unittest.TestCase):
    def test_tree_str(self):
        config = Config()
        config.load("config/default.json")
        data_utils = DataUtils()
        rule_list = [0, 1, 14, 33, 7, 34, 29, 37, 32, 34, 29, 37, 32, 39, 39, 39, 39]
        node = data_utils.get_morph_from_rule(rule_list)
        tree_str = get_str_from_tree(node)
        self.assertEqual(
            tree_str,
            "{A{C{d{B{O}{e{E{f{H{O}{O}}{O}}{O}}{O}}}{O}}{e{E{f{H{O}{O}}{O}}{O}}{O}}}{O}}",
        )

    def test_ted(self):
        config = Config()
        config.load("config/default.json")
        data_utils = DataUtils()
        rule_list = [0, 1, 14, 33, 7, 34, 29, 37, 32, 34, 29, 37, 32, 39, 39, 39, 39]
        node1 = data_utils.get_morph_from_rule(rule_list)
        rule_list = [0, 1, 14, 33, 7, 34, 29, 37, 32, 34, 29, 36, 32, 39, 39, 39, 39]
        ndoe2 = data_utils.get_morph_from_rule(rule_list)
        self.assertEqual(get_ted_from_tree(node1, ndoe2), 1)
        node3 = CodeNode("START", "O", "O")
        node3.left_node = CodeNode("E", "O", "O")
        self.assertEqual(get_ted_from_tree(node1, node3), 24)

    def test_parallel_function(self):
        config = Config()
        config.load("config/default.json")
        data_utils = DataUtils()
        data_utils.generate_data(100)
        data = np.load("data/gvae_rules_100.npy")
        strs = []
        nodes = []
        for rule_list in data:
            node = data_utils.get_morph_from_rule(rule_list)
            nodes.append(node)
            strs.append(get_str_from_tree(node))

        similarities = np.load("data/gvae_similarities_100.npy")
        strs_, similarities_ = parallel_function(nodes)
        self.assertEqual(strs, strs_)
        self.assert_(np.array_equal(similarities, similarities_))
