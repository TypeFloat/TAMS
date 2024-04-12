import torch
import torch.nn as nn
from sutils.exp.config import Config
from sutils.network.transformer import PositionalEmbedding
from sutils.network.utils import make_mlp


class TAN(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Config()
        self.batch_norm1 = nn.BatchNorm1d(cfg.GVAE.D_MU)
        self.batch_norm2 = nn.BatchNorm1d(cfg.GVAE.D_MU)
        self.terrain_embedding = nn.Linear(100, cfg.TAN.TRANSFORMER.D_MODEL)
        self.pos_embedding = PositionalEmbedding(cfg.TAN.TRANSFORMER.D_MODEL)
        self.terrain_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.TAN.TRANSFORMER.D_MODEL,
                cfg.TAN.TRANSFORMER.NHEAD,
                cfg.TAN.TRANSFORMER.DIM_FEEDFORWARD,
                batch_first=True,
            ),
            cfg.TAN.TRANSFORMER.NUM_ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.terrain_linear_encoder = nn.Linear(
            cfg.TAN.TRANSFORMER.D_MODEL * cfg.TAN.MAX_LEN,
            cfg.TAN.TRANSFORMER.D_MODEL,
            cfg.GVAE.D_MU,
        )
        self.fc = make_mlp(
            [
                cfg.GVAE.D_MU,
                cfg.GVAE.D_MU * 2,
                cfg.GVAE.D_MU * 3,
                cfg.GVAE.D_MU * 2,
                cfg.GVAE.D_MU,
                1,
            ],
            nn.ReLU,
            nn.Identity,
        )

    def forward(self, terrain: torch.Tensor, mu: torch.Tensor):
        terrain = self.terrain_embedding(terrain)
        terrain = self.pos_embedding(terrain)
        terrain = self.terrain_transformer_encoder(terrain)
        terrain = terrain.reshape((terrain.size(0), -1))
        terrain = self.terrain_linear_encoder(terrain)
        terrain = self.batch_norm1(terrain)
        mu = self.batch_norm2(mu)
        embedding = mu + terrain
        return self.fc(embedding).squeeze(1)


def patch(terrain: torch.Tensor) -> torch.Tensor:
    # batch, 50, 50 -> batch, 25, 100
    patched_terrain = []
    for t in terrain:
        t_ = []
        for i in range(0, t.size(0), 10):
            for j in range(0, t.size(1), 10):
                t_.append(t[i : i + 10, j : j + 10].reshape((1, -1)))
        t_ = torch.vstack(t_)
        patched_terrain.append(t_)
    patched_terrain = torch.stack(patched_terrain)
    return patched_terrain


def train_of_tan(
    tan,
    mu: torch.Tensor,
    terrain: torch.Tensor,
    labels: torch.Tensor,
):
    cfg = Config()
    mu = mu.to(torch.float32).to(cfg.DEVICE)
    terrain = terrain.to(torch.float32).to(cfg.DEVICE)
    labels = labels.to(torch.float32).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(tan.parameters(), lr=cfg.TAMS.TAN.LR)
    loss_f = nn.MSELoss()
    epochs = cfg.TAMS.TAN.EPOCHS
    for i in range(epochs):
        pred = tan(terrain, mu)
        loss = loss_f(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
