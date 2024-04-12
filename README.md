# Terrain-Aware Morphology Search Algorithm (TAMS)

## Installation

You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

The science-utils is a util package for the project. You can install it by running the following command:

```bash
pip install -e science-utils
```

## Code Structure

```
.
├── README.md
├── assets
│   └── setting.xml
├── config
│   ├── default.json
│   └── rule.json
├── data
├── log
├── requirements.txt
├── science-utils
├── scripts
│   ├── pretrain.py
│   └── run.py
├── src
│   ├── env.py
│   ├── tan.py
│   ├── gvae.py
│   ├── mppi.py
│   ├── robot.py
│   └── tams.py
└── utils
    ├── benchmark.py
    ├── data_utils.py
    └── terrain.py
```

The main algorithm code is stored in `./src/`. The `env.py` is the interaction environment. The `tan.py` defines the terrain-aware neural network. The `gvae.py` is the morphology embedding module. The `mppi.py` is the MPPI controller. The `robot.py` is a util to generate robot tree stucture and convert it to mujoco mjcf file. The `tams.py` is the TAMS's main code.

The `./scripts/` directory contains the main scripts to run the algorithm. And the `./utils/` directory contains some util functions. Note, all the parameters used in code is defined in `./config/default.json`.
And the module joint rules is defined in `./config/rule.json`.

## Running

### Pre-training

To pre-train the morphology embedding module, you could run the following command:

```bash
python scripts/pretrain.py
```

It will generate datasets firstly, and train the morphology embedding module. The weights of the trained model will be saved in `./log/pretrain/*/gvae.pth`. Then you need to copy the `gvae.pth` to the `./data/` dictionary.

### Search

Once get the pre-trained model, you could run the following command to search the morphology:

```bash
python scripts/run.py
```

All the search results will be saved in `./log/tams/*/`. Note, in the search, the map file will be generated automiclly. To accelerate the program, each map only consiste of one terrain.