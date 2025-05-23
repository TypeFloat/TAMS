import random
from typing import List, Type

import numpy as np
import torch
import torch.nn as nn


def get_available_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# def weight_initialize(layer: nn.Module) -> None:
#     if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
#         torch.nn.init.orthogonal_(layer.weight, 1)
#         torch.nn.init.constant_(layer.bias, 0)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def make_conv(
#     channels: List[int],
#     conv: Type[nn.Module],
#     pool: Type[nn.Module],
#     dropout: Type[nn.Module],
#     kernel_size: int,
#     dropout_p: float = 0.0,
#     activation: Type[nn.Module] = nn.Tanh,
# ) -> nn.Sequential:
#     assert len(channels) >= 2, "Must have input and output channel!"
#     layers = []
#     for i in range(len(channels) - 1):
#         layers.append(conv(channels[i], channels[i + 1], kernel_size))
#         layers.append(activation())
#         layers.append(pool(kernel_size))
#         layers.append(dropout(p=dropout_p))
#     return nn.Sequential(*layers)


# def get_conv_out(forward_net: nn.Module, shape: torch.Size, expand: bool):
#     if expand:
#         fake_data = torch.zeros(1, *shape)
#     else:
#         fake_data = torch.zeros(*shape)
#     output: torch.Tensor = forward_net(fake_data)
#     return int(np.prod(output.shape))

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * -(torch.log(torch.Tensor([10000.0])) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

        
def make_mlp(
    sizes: List[int],
    activation: Type[nn.Module],
    output_activation: Type[nn.Module] = nn.Identity,
):
    assert len(sizes) >= 2, "Must have input and output dim!"
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)