import os
from copy import deepcopy
from typing import List, Union

import mujoco
import numpy as np

from utils import mjcf
from utils.config import Config

config = Config()


def generate_random_box(
    children: List[mjcf.Geom], y_offset: float = 0.0, z_offset: float = 0.0
):
    cfg = Config()
    env = mjcf.Mujoco()
    worldbody = mjcf.Worldbody()
    asset = mjcf.Asset()
    env.add_children([worldbody, asset])
    asset.add_child(
        mjcf.Texture(
            builtin="flat",
            height=512,
            width=512,
            name="grid",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.2, 0.3, 0.4],
            type="2d",
        )
    )
    asset.add_child(
        mjcf.Material(
            name="grid",
            reflectance=0.2,
            texrepeat=[3, 3],
            texture="grid",
            texuniform=True,
        )
    )
    worldbody.add_children(children)
    box_size = np.random.rand(cfg.SIM.BOX_NUM, 3) * cfg.SIM.BOX_SEMI_LENGTH
    box_size = box_size.tolist()
    pos = np.zeros((cfg.SIM.BOX_NUM, 3))
    pos[:, 0] += (
        np.random.rand(cfg.SIM.BOX_NUM) * cfg.SIM.SEMI_WIDTH * 2 - cfg.SIM.SEMI_WIDTH
    )
    pos[:, 1] += np.random.rand(cfg.SIM.BOX_NUM) * cfg.SIM.SEMI_LENGTH * 2 + y_offset
    pos[:, 2] = 1 + z_offset
    pos = pos.tolist()

    for i in range(cfg.SIM.BOX_NUM):
        box = mjcf.Body(pos=pos[i])
        box.add_children(
            [
                mjcf.Freejoint(),
                mjcf.Geom(type="box", size=box_size[i], rgba=[0.5, 0.5, 0.5, 1]),
            ]
        )
        worldbody.add_child(box)

    model = mujoco.MjModel.from_xml_string(env.xml())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 600)
    pos = data.xpos.copy()[1:].tolist()
    quat = data.xquat.copy()[1:].tolist()

    for i in range(len(box_size)):
        children.append(
            mjcf.Geom(
                type="box",
                size=box_size[i],
                rgba=[0.5, 0.5, 0.5, 1],
                pos=pos[i],
                quat=quat[i],
            )
        )


def generate_xml(
    children: List[mjcf.Geom],
    filename: str,
    start_z_pos: float,
    final_z_pos: float,
    random: bool = False,
):
    if random:
        filename += "_random"
    if os.path.exists(f"assets/{filename}.npy"):
        return np.load(f"assets/{filename}.npy")

    env = mjcf.Mujoco()
    worldbody = mjcf.Worldbody()
    env.add_child(worldbody)
    worldbody.add_child(
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[0, -config.SIM.SEMI_LENGTH, -0.05 + start_z_pos],
            size=[config.SIM.SEMI_WIDTH, config.SIM.SEMI_LENGTH, 0.1],
        )
    )
    worldbody.add_children(children)
    worldbody.add_child(
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[0, config.SIM.SEMI_LENGTH * 3, -0.05 + final_z_pos],
            size=[config.SIM.SEMI_WIDTH, config.SIM.SEMI_LENGTH, 0.1],
        )
    )

    cfg = Config()

    env_ = mjcf.Mujoco()
    asset = mjcf.Asset()
    asset.add_child(
        mjcf.Texture(
            builtin="flat",
            height=512,
            width=512,
            name="grid",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.2, 0.3, 0.4],
            type="2d",
        )
    )
    asset.add_child(
        mjcf.Material(
            name="grid",
            reflectance=0.2,
            texrepeat=[3, 3],
            texture="grid",
            texuniform=True,
        )
    )
    worldbody_ = deepcopy(worldbody)
    env_.add_children([worldbody_, asset])
    if random:
        box_size = np.random.rand(cfg.SIM.BOX_NUM, 3) * cfg.SIM.BOX_SEMI_LENGTH
        box_size = box_size.tolist()
        pos = np.zeros((cfg.SIM.BOX_NUM, 3))
        pos[:, 0] += (
            np.random.rand(cfg.SIM.BOX_NUM) * cfg.SIM.SEMI_WIDTH * 2
            - cfg.SIM.SEMI_WIDTH
        )
        pos[:, 1] += np.random.rand(cfg.SIM.BOX_NUM) * cfg.SIM.SEMI_LENGTH * 2
        pos[:, 2] = 1
        pos = pos.tolist()

        for i in range(cfg.SIM.BOX_NUM):
            box = mjcf.Body(pos=pos[i])
            box.add_children(
                [
                    mjcf.Freejoint(),
                    mjcf.Geom(type="box", size=box_size[i], rgba=[0.5, 0.5, 0.5, 1]),
                ]
            )
            worldbody_.add_child(box)
    else:
        box_size = []

    sensor = mjcf.Sensor()
    env_.add_child(sensor)
    site_body = mjcf.Body()
    worldbody_.add_child(site_body)
    for i in range(cfg.SIM.STANDARD_LENGTH):
        for j in range(cfg.SIM.STANDARD_LENGTH):
            index = i * cfg.SIM.STANDARD_LENGTH + j
            site_body.add_child(
                mjcf.Site(
                    name=f"hfield_{index}",
                    euler=[180, 0, 0],
                    pos=[
                        -cfg.SIM.SEMI_WIDTH
                        + j * cfg.SIM.SEMI_WIDTH * 2 / cfg.SIM.STANDARD_LENGTH,
                        i * cfg.SIM.SEMI_LENGTH * 2 / cfg.SIM.STANDARD_LENGTH,
                        1,
                    ],
                )
            )
            sensor.add_child(mjcf.Rangefinder(site=f"hfield_{index}"))

    model = mujoco.MjModel.from_xml_string(env_.xml())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 600)
    pos = data.xpos.copy()[1:].tolist()
    quat = data.xquat.copy()[1:].tolist()
    terrain = data.sensordata.copy()
    terrain = 1 - terrain.reshape((cfg.SIM.STANDARD_LENGTH, cfg.SIM.STANDARD_LENGTH))

    for i in range(len(box_size)):
        worldbody.add_child(
            mjcf.Geom(
                type="box",
                size=box_size[i],
                rgba=[0.5, 0.5, 0.5, 1],
                pos=pos[i],
                quat=quat[i],
            )
        )

    with open(f"assets/{filename}.xml", "w") as f:
        f.write(env.xml())
    np.save(f"assets/{filename}.npy", terrain)
    return terrain


def generate_plane(
    random: bool = False, y_offset: float = 0.0, z_offset: float = 0.0
) -> Union[np.ndarray, List[mjcf.Geom]]:
    children = [
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[0, config.SIM.SEMI_LENGTH + y_offset, -0.05 + z_offset],
            size=[config.SIM.SEMI_WIDTH, config.SIM.SEMI_LENGTH, 0.1],
        )
    ]
    if y_offset == 0.0 and z_offset == 0.0:
        terrain = generate_xml(children, "plane", 0, 0, random)
        return terrain
    else:
        if random:
            generate_random_box(children, y_offset, z_offset)
        return children


def generate_stair(
    factor: int, random: bool = False, y_offset: float = 0.0, z_offset: float = 0.0
) -> Union[np.ndarray, List[mjcf.Geom]]:
    diff = config.SIM.SEMI_HEIGHT / factor * 1.2

    children = []
    for i in range(factor):
        children.append(
            mjcf.Geom(
                type="box",
                material="grid",
                pos=[
                    0,
                    config.SIM.SEMI_LENGTH / factor * (2 * i + 1) + y_offset,
                    -0.05 + diff * (i + 1) + z_offset,
                ],
                size=[
                    config.SIM.SEMI_WIDTH,
                    config.SIM.SEMI_LENGTH / factor,
                    0.1,
                ],
            )
        )
    if y_offset == 0.0 and z_offset == 0.0:
        terrain = generate_xml(
            children, f"stair_{factor}", 0, config.SIM.SEMI_HEIGHT * 1.2, random
        )
        return terrain
    else:
        if random:
            generate_random_box(children, y_offset, z_offset)
        return children


def generate_incline(
    factor: int, random: bool = False, y_offset: float = 0.0, z_offset: float = 0.0
) -> Union[np.ndarray, List[mjcf.Geom]]:
    angle = factor * np.pi / 180
    height = 2 * config.SIM.SEMI_LENGTH * np.tan(factor * np.pi / 180)
    children = [
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[
                0,
                config.SIM.SEMI_LENGTH + y_offset,
                config.SIM.SEMI_LENGTH * np.tan(angle) - 0.05 + z_offset,
            ],
            size=[
                config.SIM.SEMI_WIDTH,
                config.SIM.SEMI_LENGTH / np.cos(angle),
                0.1,
            ],
            euler=[factor, 0, 0],
        )
    ]

    if y_offset == 0.0 and z_offset == 0.0:
        terrain = generate_xml(children, f"incline_{factor}", 0, height, random)
        return terrain
    else:
        if random:
            generate_random_box(children, y_offset, z_offset)
        return children


def generate_gap(
    factor: int, random: bool = False, y_offset: float = 0.0, z_offset: float = 0.0
) -> Union[np.ndarray, List[mjcf.Geom]]:
    factor *= 0.1

    children = []
    children.append(
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[
                0,
                +config.SIM.SEMI_LENGTH * (1 - factor) / 2 + y_offset,
                -0.05 + z_offset,
            ],
            size=[
                config.SIM.SEMI_WIDTH,
                config.SIM.SEMI_LENGTH * (1 - factor) / 2,
                0.1,
            ],
        )
    )
    children.append(
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[
                0,
                config.SIM.SEMI_LENGTH
                + config.SIM.SEMI_LENGTH * (1 - factor) / 2
                + y_offset,
                -0.05 + z_offset,
            ],
            size=[
                config.SIM.SEMI_WIDTH,
                config.SIM.SEMI_LENGTH * (1 - factor) / 2,
                0.1,
            ],
        )
    )

    if y_offset == 0.0 and z_offset == 0.0:
        terrain = generate_xml(children, f"gap_{int(factor * 10)}", 0, 0, random)
        return terrain
    else:
        if random:
            generate_random_box(children, y_offset, z_offset)
        return children


def generate_barrier(
    factor: int, random: bool = False, y_offset: float = 0.0, z_offset: float = 0.0
) -> Union[np.ndarray, List[mjcf.Geom]]:
    factor *= 0.1

    children = [
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[0, config.SIM.SEMI_LENGTH + y_offset, -0.05 + z_offset],
            size=[config.SIM.SEMI_WIDTH, config.SIM.SEMI_LENGTH, 0.1],
        ),
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[
                0,
                config.SIM.SEMI_LENGTH / 2
                + config.SIM.SEMI_LENGTH / config.SIM.STANDARD_LENGTH
                + y_offset,
                0.05 + config.SIM.SEMI_HEIGHT / 2 * factor + z_offset,
            ],
            size=[
                config.SIM.SEMI_WIDTH,
                config.SIM.SEMI_LENGTH / config.SIM.STANDARD_LENGTH * 10,
                config.SIM.SEMI_HEIGHT / 2 * factor,
            ],
        ),
        mjcf.Geom(
            type="box",
            material="grid",
            pos=[
                0,
                config.SIM.SEMI_LENGTH / 2 * 3
                + config.SIM.SEMI_LENGTH / config.SIM.STANDARD_LENGTH
                + y_offset,
                0.05 + config.SIM.SEMI_HEIGHT / 2 * factor + z_offset,
            ],
            size=[
                config.SIM.SEMI_WIDTH,
                config.SIM.SEMI_LENGTH / config.SIM.STANDARD_LENGTH * 10,
                config.SIM.SEMI_HEIGHT / 2 * factor,
            ],
        ),
    ]

    if y_offset == 0.0 and z_offset == 0.0:
        terrain = generate_xml(children, f"barrier_{int(factor * 10)}", 0, 0, random)
        return terrain
    else:
        if random:
            generate_random_box(children, y_offset, z_offset)
        return children
