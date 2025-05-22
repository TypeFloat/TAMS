# 地形安置的形态搜索算法（Terrain-Aware Morphology Search Algorithm, TAMS）

## 环境配置

执行一下命令即可在python3.8环境下安装所需的依赖包：

```bash
pip install -r requirements.txt
```

## 目录结构

```
.
├── README.md
├── assets
│   ├── README.md
│   ├── setting.xml
├── config
│   ├── README.md
│   ├── default.json
│   └── rule.json
├── data
│   ├── README.md
├── log
│   └── README.md
├── requirements.txt
├── scripts
│   ├── pretrain.py
│   └── run.py
├── src
│   ├── env.py
│   ├── gvae.py
│   ├── mppi.py
│   ├── robot.py
│   ├── tams.py
│   └── tan.py
├── test
│   ├── test_data_utils.py
│   ├── test_gvae.py
│   ├── test_robot.py
│   └── test_terrain.py
└── utils
    ├── benchmark.py
    ├── config.py
    ├── data_utils.py
    ├── env.py
    ├── logger.py
    ├── mjcf
    │   ├── __init__.py
    │   ├── default.py
    │   ├── element.py
    │   ├── elements.py
    │   ├── equality.py
    │   ├── fixed.py
    │   ├── sensor.py
    │   ├── spatial.py
    │   ├── visual.py
    │   └── xmltodict.py
    ├── network.py
    └── terrain.py
```

TAMS主要的代码存储在`./src/`目录下。`env.py`是交互环境，`tan.py`定义了地形感知神经网络，`gvae.py`是形态嵌入模块，`mppi.py`是MPPI控制器，`robot.py`是生成机器人树结构并转换为mujoco mjcf文件的工具，`tams.py`是TAMS的主代码。

`./scripts/`目录包含运行算法的主要脚本，`./utils/`目录包含一些工具函数。注意，代码中使用的所有参数都在 `./config/default.json` 中定义。

## 运行

### 预训练

在运行TAMS之前，你需要预训练形态嵌入模块。预训练的目的是为了生成一个形态嵌入模型，该模型可以将离散形态结构嵌入到一个低维连续空间中，以便于后续的搜索和优化。

```bash
python scripts/pretrain.py
```

它首先会生成数据集，然后训练形态嵌入模块。训练好的模型权重将保存在 `./log/pretrain/*/gvae.pth` 中。然后你需要将 `gvae.pth` 复制到 `./assets/` 目录中。

### 形态搜索

在预训练完成后，你可以运行以下命令来进行形态搜索：

```bash
python scripts/run.py
```

所有的搜索结果将保存在 `./log/tams/*/` 中。注意，在搜索过程中，地图文件将自动生成。为了加速仿真程序的执行，每个地图文件只包含一个地形。