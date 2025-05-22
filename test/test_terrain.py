import os
import sys

sys.path.append(os.getcwd())

from utils.config import Config
from utils.terrain import (
    generate_barrier,
    generate_gap,
    generate_incline,
    generate_plane,
    generate_stair,
)

if __name__ == "__main__":
    config = Config()
    config.load("config/default.json")
    terrain_list = []
    terrain_list.append({"call": generate_plane, "args": (False,), "name": "plane"})
    terrain_list.extend(
        [
            {
                "call": generate_barrier,
                "args": (i, False),
                "name": f"barrier_{i}",
            }
            for i in range(
                config.SIM.BARRIER_FACTOR - 2,
                config.SIM.BARRIER_FACTOR + 1,
            )
        ]
    )
    terrain_list.extend(
        [
            {"call": generate_gap, "args": (i, False), "name": f"gap_{i}"}
            for i in range(config.SIM.GAP_FACTOR - 2, config.SIM.GAP_FACTOR + 1)
        ]
    )
    terrain_list.extend(
        [
            {
                "call": generate_incline,
                "args": (i, False),
                "name": f"incline_{i}",
            }
            for i in range(
                config.SIM.INCLINE_FACTOR - 10,
                config.SIM.INCLINE_FACTOR + 1,
                5,
            )
        ]
    )
    terrain_list.extend(
        [
            {"call": generate_stair, "args": (i, False), "name": f"stair_{i}"}
            for i in range(
                config.SIM.STAIR_FACTOR + 2,
                config.SIM.STAIR_FACTOR - 1,
                -1,
            )
        ]
    )
    for terrain in terrain_list:
        if terrain["name"] == "plane":
            t = terrain["call"](True)
        else:
            t = terrain["call"](terrain["args"][0], True)
        assert t.shape[0] == 50 and t.shape[1] == 50
    # for i in range(5, 16, 5):
    #     generate_incline(factor=i, xml=True)
