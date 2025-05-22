from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

from utils import mjcf


class CodeNode:
    def __init__(self, elem_type: str, direction: str, available: str) -> None:
        self.elem_type = elem_type
        self.direction = direction
        self.available = available
        self.left_node: Optional[CodeNode] = None
        self.right_node: Optional[CodeNode] = None
        self.pre_node: Optional[CodeNode] = None

    def copy(self) -> "CodeNode":
        return deepcopy(self)

    def pre_order(self) -> List["CodeNode"]:
        nodes = [self]
        if self.left_node is not None:
            nodes += self.left_node.pre_order()
        if self.right_node is not None:
            nodes += self.right_node.pre_order()
        return nodes

    def node_count(self) -> int:
        nodes = self.pre_order()
        return len(nodes)

    def tree2str(self) -> str:

        tree_str = ""
        queue = ["r", self]
        while len(queue) > 0:
            node = queue.pop(0)
            if isinstance(node, CodeNode):
                tree_str += f"{node.elem_type}{node.direction}   "
                queue.append(node.left_node)
                queue.append(node.right_node)
            elif node == "r":
                if len(queue) > 0:
                    queue.append("r")
                    tree_str += "\n"
            elif node == "O":
                tree_str += "OO   "
        return tree_str[1:]

    def set_child_node(self, left_node: "CodeNode", right_node: "CodeNode") -> None:
        self.left_node = left_node
        self.right_node = right_node
        left_node.pre_node = self
        right_node.pre_node = self

    def get_links_num(self) -> int:

        links = 0
        if self.elem_type == "B":
            links += 1
        elif self.elem_type == "S":
            links += 2

        if self.left_node is not None:
            links += self.left_node.get_links_num()
        if self.right_node is not None:
            links += self.right_node.get_links_num()
        return links

    def get_joints_num(self) -> int:
        joints = 0
        if self.elem_type == "J" and self.direction != "O":
            if self.available == "B":
                joints += 1
            else:
                joints += 2
        if self.left_node is not None:
            joints += self.left_node.get_joints_num()
        if self.right_node is not None:
            joints += self.right_node.get_joints_num()
        return joints


class MorphTreeNode:

    def __init__(
        self,
        body_offset: List[int],
        body_euler: List[int],
        geom_class: str,
        geom_euler: List[int],
        joint_class: str,
        joint_axis: List[int],
        joint_offset: List[int],
        have_site: bool,
    ) -> None:
        self.joint_class = joint_class
        self.joint_axis = joint_axis
        self.body_offset = body_offset
        self.body_euler = body_euler
        self.joint_offset = joint_offset
        self.geom_class = geom_class
        self.geom_euler = geom_euler
        self.have_site = have_site
        self.left_node: Optional[MorphTreeNode] = None
        self.right_node: Optional[MorphTreeNode] = None

    def copy(self) -> "MorphTreeNode":
        return deepcopy(self)


class RobotGenerator:

    def __init__(self) -> None:
        self._morph_count = {}
        self._model = mjcf.Mujoco()
        self._worldbody = mjcf.Worldbody()
        self._actuator = mjcf.Actuator()
        self._sensor = mjcf.Sensor()
        self._model.add_children([self._worldbody, self._actuator, self._sensor])
        self._root = None

    def generate(
        self,
        code_tree: CodeNode,
        init_loc: List[float],
        robot_id: Union[int, str],
        map_id: str,
    ) -> None:
        self._model.add_children(
            [mjcf.Include(f"{map_id}.xml"), mjcf.Include("setting.xml")]
        )
        if code_tree.elem_type == "INIT":
            code_tree = code_tree.left_node.left_node
            code_tree.pre_node = None
        root = self._convert(code_tree, init_loc)
        self._add_node(self._worldbody, root)
        with open(f"assets/robot-{robot_id}.xml", "w") as f:
            f.write(self._model.xml())

    def _add_node(
        self,
        parent_node: Union[mjcf.Body, mjcf.Worldbody],
        child_node: MorphTreeNode,
    ) -> None:
        body_order = self._morph_count.get("body", 0)
        self._morph_count["body"] = body_order + 1
        body = mjcf.Body(
            name=f"body_{str(body_order)}",
            pos=child_node.body_offset,
            euler=child_node.body_euler,
        )
        geom = mjcf.Geom(class_=child_node.geom_class, euler=child_node.geom_euler)
        body.add_child(geom)

        if isinstance(parent_node, mjcf.Worldbody):
            if child_node.body_euler[0] == 0:
                body.add_child(
                    mjcf.Camera(
                        name="tracker", mode="track", pos=[3.5, 0, 5], euler=[0, 30, 60]
                    )
                )
            elif child_node.body_euler[0] == -90:
                body.add_child(
                    mjcf.Camera(
                        name="tracker",
                        mode="track",
                        pos=[3.5, -5, 0],
                        euler=[90, 30, 60],
                    )
                )

        if child_node.joint_class != "fixed_joint":
            joint_order = self._morph_count.get("joint", 0)
            joint_name = f"joint_{str(joint_order)}"
            self._morph_count["joint"] = joint_order + 1
            joint = mjcf.Joint(
                name=joint_name,
                class_=child_node.joint_class,
                axis=child_node.joint_axis,
                pos=child_node.joint_offset,
            )
            body.add_child(joint)
            if child_node.joint_class != "free_joint":
                actuator = mjcf.Motor(
                    joint=joint_name, gear=150, name=f"motor_{str(joint_order)}"
                )
                self._actuator.add_child(actuator)

        if child_node.have_site:
            joint_sensor_order = self._morph_count.get("joint_sensor", 0)
            self._morph_count["joint_sensor"] = joint_sensor_order + 1
            euler = [0, 0, 0]
            if child_node.joint_axis[0] == 1:
                euler[1] = 90
            if child_node.joint_axis[1] == 1:
                euler[0] = 90

            site_name = f"joint_sensor_{joint_sensor_order}"
            site = mjcf.Site(
                class_=child_node.joint_class,
                pos=child_node.joint_offset,
                euler=euler,
                name=site_name,
            )
            self._sensor.add_child(mjcf.Jointpos(joint_name, noise=1e-6))
            self._sensor.add_child(mjcf.Jointvel(joint_name, noise=1e-6))
            body.add_child(site)
        parent_node.add_child(body)

        if child_node.left_node is not None:
            self._add_node(body, child_node.left_node)
        if child_node.right_node is not None:
            left_part = child_node.right_node.copy()
            right_part = child_node.right_node.copy()
            self._add_node(body, left_part)

            right_part.body_offset[0] *= -1
            right_part.joint_offset[0] *= -1
            node = right_part.left_node
            while node is not None:
                node.body_offset[0] *= -1
                node.joint_offset[0] *= -1
                node = node.left_node
            self._add_node(body, right_part)

    def _convert(
        self, code_tree: Optional[CodeNode], init_loc: Optional[List[float]] = None
    ) -> Optional[MorphTreeNode]:

        if code_tree.elem_type == "E":
            return None

        if code_tree.elem_type == "J":
            code_tree = code_tree.left_node

        if code_tree.pre_node is None:
            body_offset = init_loc
            if code_tree.direction == "Y":
                body_euler = [0, 0, 0]
            elif code_tree.direction == "Z":
                body_euler = [-90, 0, 0]

        else:
            body_euler = [0, 0, 0]
            pre_gemo = code_tree.pre_node.pre_node.elem_type
            if code_tree.elem_type == "B":
                body_offset = [0, -0.15, 0]
            else:
                if pre_gemo == "B":
                    body_offset = [-0.05, 0, 0]
                else:
                    pre_direction = code_tree.pre_node.pre_node.direction
                    body_offset = [0, 0, 0]
                    if pre_direction == "X":
                        body_offset[0] -= 0.05
                    elif pre_direction == "Y":
                        body_offset[1] -= 0.05
                    elif pre_direction == "y":
                        body_offset[1] += 0.05
                    elif pre_direction == "Z":
                        body_offset[2] -= 0.05
                    elif pre_direction == "z":
                        body_offset[2] += 0.05
                if code_tree.direction == "X":
                    body_offset[0] -= 0.05
                elif code_tree.direction == "Y":
                    body_offset[1] -= 0.05
                elif code_tree.direction == "y":
                    body_offset[1] += 0.05
                elif code_tree.direction == "Z":
                    body_offset[2] -= 0.05
                elif code_tree.direction == "z":
                    body_offset[2] += 0.05

        # geom
        if code_tree.elem_type == "B":
            geom_class = "body_link"
            geom_euler = [90, 0, 0]
        elif code_tree.elem_type == "S":
            geom_class = "limb_small"
            if code_tree.direction == "X":
                geom_euler = [0, 90, 0]
            elif code_tree.direction == "Y":
                geom_euler = [90, 0, 0]
            elif code_tree.direction == "y":
                geom_euler = [90, 0, 0]
            elif code_tree.direction == "Z":
                geom_euler = [0, 0, 0]
            elif code_tree.direction == "z":
                geom_euler = [0, 0, 0]

        # joint_class
        if code_tree.pre_node is None:
            joint_class = "free_joint"
        else:
            if code_tree.pre_node.direction == "O":
                joint_class = "fixed_joint"
            else:
                if code_tree.elem_type == "B":
                    joint_class = "cylinder_joint_big"
                else:
                    joint_class = "cylinder_joint_small"

        # joint_axis
        if code_tree.pre_node is None:
            joint_axis = [0, 0, 0]
        else:
            if code_tree.pre_node.direction == "X":
                joint_axis = [1, 0, 0]
            elif code_tree.pre_node.direction == "Y":
                joint_axis = [0, 1, 0]
            elif code_tree.pre_node.direction == "y":
                joint_axis = [0, 1, 0]
            elif code_tree.pre_node.direction == "Z":
                joint_axis = [0, 0, 1]
            elif code_tree.pre_node.direction == "z":
                joint_axis = [0, 0, 1]
            else:
                joint_axis = [0, 0, 0]

        # joint_offset
        if code_tree.pre_node is None or code_tree.pre_node.direction == "O":
            joint_offset = [0, 0, 0]
        else:
            if code_tree.elem_type == "B":
                joint_offset = [0, 0.075, 0]
            elif code_tree.elem_type == "S":
                offset = 0.05
                if code_tree.direction == "X":
                    joint_offset = [offset, 0, 0]
                elif code_tree.direction == "Y":
                    joint_offset = [0, offset, 0]
                elif code_tree.direction == "y":
                    joint_offset = [0, -offset, 0]
                elif code_tree.direction == "Z":
                    joint_offset = [0, 0, offset]
                elif code_tree.direction == "z":
                    joint_offset = [0, 0, -offset]

        # have_site
        if (code_tree.pre_node is not None) and (code_tree.pre_node.direction != "O"):
            have_site = True
        else:
            have_site = False

        node = MorphTreeNode(
            body_offset=body_offset,
            body_euler=body_euler,
            geom_class=geom_class,
            geom_euler=geom_euler,
            joint_class=joint_class,
            joint_axis=joint_axis,
            joint_offset=joint_offset,
            have_site=have_site,
        )
        node.left_node = self._convert(code_tree.left_node)
        node.right_node = self._convert(code_tree.right_node)

        return node
