import time
from typing import Dict

import pybullet as p


def get_joints_index(
    body_uniqueId: int, include_fixed_joint: bool = True
) -> Dict[str, int]:

    joints_index = {}

    for i in range(p.getNumJoints(body_uniqueId)):
        joint_info = p.getJointInfo(body_uniqueId, i)
        if not include_fixed_joint and joint_info[2] == p.JOINT_FIXED:
            continue
        joint_name = str(joint_info[1], 'utf-8')
        joints_index[joint_name] = i

    return joints_index


def get_links_index(body_uniqueId: int) -> Dict[str, int]:

    links_index = {}

    for i in range(p.getNumJoints(body_uniqueId)):
        link_name = str(p.getJointInfo(body_uniqueId, i)[12], 'utf-8')
        links_index[link_name] = i

    return links_index


def make_object_visual_center(
    body_uniqueId: int,
    camera_distance: int = 3,
    camera_yaw: int = 110,
    camera_pitch: int = 30,
) -> None:

    location, _ = p.getBasePositionAndOrientation(body_uniqueId)

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=-camera_pitch,
        cameraTargetPosition=location,
    )


def show_urdf(model_path: str) -> None:

    p.connect(p.GUI)
    urdf = p.loadURDF(
        model_path,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
    )
    debug_index = []

    for i in range(p.getNumJoints(urdf)):
        joint_info = p.getJointInfo(urdf, i)

        if joint_info[2] == p.JOINT_FIXED:
            continue
        else:
            debug_index.append(
                (
                    p.addUserDebugParameter(
                        paramName=bytes.decode(joint_info[1]),
                        rangeMin=joint_info[8],
                        rangeMax=joint_info[9],
                        startValue=(joint_info[8] + joint_info[9]) / 2,
                    ),
                    i,
                )
            )

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1 / 240)

        for i in range(len(debug_index)):
            p.setJointMotorControl2(
                urdf,
                debug_index[i][1],
                p.POSITION_CONTROL,
                targetPosition=p.readUserDebugParameter(debug_index[i][0]),
                force=300,
            )
