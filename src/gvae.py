import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config
from utils.logger import Logger
from utils.network import PositionalEmbedding
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Tuple

from utils.data_utils import DataUtils, get_data


def sample(mu, var):
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return mu + eps * std


class TransformerModule(nn.Module):
    def __init__(
        self,
        rule_embedding: nn.Embedding,
        pos_embedding: PositionalEmbedding,
    ) -> None:
        super().__init__()
        self._rule_embedding = rule_embedding
        self._pos_embedding = pos_embedding


class TransformerEncoder(TransformerModule):
    def __init__(
        self,
        rule_embedding: nn.Embedding,
        pos_embedding: PositionalEmbedding,
    ) -> None:
        super().__init__(rule_embedding, pos_embedding)
        config = Config()
        self._transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.GVAE.TRANSFORMER.D_MODEL,
                config.GVAE.TRANSFORMER.NHEAD,
                config.GVAE.TRANSFORMER.DIM_FEEDFORWARD,
                batch_first=True,
            ),
            config.GVAE.TRANSFORMER.NUM_ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self._linear_encoder = nn.Linear(
            config.GVAE.TRANSFORMER.D_MODEL * config.GVAE.MAX_LEN,
            config.GVAE.TRANSFORMER.D_MODEL,
        )
        self._mu_fc = nn.Linear(config.GVAE.TRANSFORMER.D_MODEL, config.GVAE.D_MU)
        self._var_fc = nn.Linear(config.GVAE.TRANSFORMER.D_MODEL, config.GVAE.D_MU)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor]:
        src = self._rule_embedding(src)
        src = self._pos_embedding(src)
        out = self._transformer_encoder(src)
        out = self._linear_encoder(out.reshape(out.size(0), -1))
        mu, var = self._mu_fc(out), self._var_fc(out)
        return mu, var

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        mu, var = self.forward(src)
        return mu


class TransformerDecoder(TransformerModule):
    def __init__(
        self,
        rule_embedding: nn.Embedding,
        pos_embedding: PositionalEmbedding,
    ) -> None:
        super().__init__(rule_embedding, pos_embedding)
        config = Config()
        self._linear_decoder = nn.Linear(
            config.GVAE.D_MU,
            config.GVAE.TRANSFORMER.D_MODEL * config.GVAE.MAX_LEN,
        )
        self._transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.GVAE.TRANSFORMER.D_MODEL,
                config.GVAE.TRANSFORMER.NHEAD,
                config.GVAE.TRANSFORMER.DIM_FEEDFORWARD,
                batch_first=True,
            ),
            config.GVAE.TRANSFORMER.NUM_DECODER_LAYERS,
        )
        self._fc = nn.Linear(config.GVAE.TRANSFORMER.D_MODEL, config.GVAE.NRULE)
        self._max_len = config.GVAE.MAX_LEN

    def forward(self, memory: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(
            memory.device
        )
        memory = self._linear_decoder(memory).reshape(memory.size(0), self._max_len, -1)
        tgt = self._rule_embedding(tgt)
        tgt = self._pos_embedding(tgt)
        out = self._transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self._fc(out)

    def decode(
        self, memory: torch.Tensor, data_util: Optional[DataUtils] = None, max_iter=100
    ) -> List[Optional[np.ndarray]]:
        memory = self._linear_decoder(memory).reshape(memory.size(0), self._max_len, -1)
        tgt = torch.zeros(memory.size(0), 1).long().to(memory.device)
        if data_util:
            data_util.initialize_stack(memory.size(0))
        for _ in range(self._max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size()[-1]
            ).to(tgt.device)
            tgt_ = self._rule_embedding(tgt)
            tgt_ = self._pos_embedding(tgt_)
            output = self._transformer_decoder(
                tgt_,
                memory,
                tgt_mask=tgt_mask,
            )
            output = self._fc(output[:, -1])
            if data_util:
                y = data_util.get_rule_from_prob(output)
            else:
                output = torch.softmax(output, dim=-1)
                y = torch.argmax(output, dim=1, keepdim=True)
            tgt = torch.cat([tgt, y], dim=1)
        return tgt


class GVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = Config()
        self._higher_bound = None
        self._lower_bound = None
        rule_embedding = nn.Embedding(
            config.GVAE.NRULE, config.GVAE.TRANSFORMER.D_MODEL
        )
        pos_embedding = PositionalEmbedding(
            config.GVAE.TRANSFORMER.D_MODEL, max_len=config.GVAE.MAX_LEN
        )
        self._encoder = TransformerEncoder(rule_embedding, pos_embedding)
        self._decoder = TransformerDecoder(rule_embedding, pos_embedding)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self._encoder(src)
        memory = sample(mu, log_var)
        out = self._decoder(memory, tgt)
        # out = self._decoder(mu, tgt)
        return out, mu, log_var

    def get_bound(self, src: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            src = src.to(torch.long)
            memory, _ = self._encoder(src)
            higher_bound = memory.max(dim=0).values.cpu().detach().numpy()
            lower_bound = memory.min(dim=0).values.cpu().detach().numpy()
        return higher_bound, lower_bound

    def encode(self, src: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        if type(src) == List:
            src = torch.tensor(src).long()
        elif type(src) == np.ndarray:
            src = torch.from_numpy(src).long()
        elif type(src) == torch.Tensor:
            src = src.long()
        if src.dim() == 1:
            src = src.unsqueeze(0)
        with torch.no_grad():
            memory = self._encoder.encode(src)
        return memory

    def decode(
        self, memory: torch.Tensor, data_util: Optional[DataUtils]
    ) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        self._decoder.eval()
        if memory.dim() == 1:
            memory = memory.unsqueeze(0)
        with torch.no_grad():
            out = self._decoder.decode(memory.to(torch.float32), data_util)
        if data_util:
            out = data_util.check_available(out)
        return out


class VAELoss(nn.Module):
    def __init__(self, raw_similarity: Union[np.ndarray, torch.Tensor, None]) -> None:
        super().__init__()
        config = Config()
        self._rec_factor = config.TRAIN_OF_GVAE.REC_FACTOR
        self._reconstruct_loss = CrossEntropyLoss()
        self._kld_loss = KLDLoss()
        self._similarity_loss = SimilarityLoss(raw_similarity)

    def forward(
        self,
        out: torch.Tensor,
        tgt: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        info = {}
        loss = 0
        reconstuct_loss = self._reconstruct_loss(out, tgt, mask)
        info["reconstruct_loss"] = reconstuct_loss.item()
        loss = loss + self._rec_factor * reconstuct_loss

        kld_loss = self._kld_loss(mu, log_var)
        info["kld_loss"] = kld_loss.item()
        loss = loss + kld_loss

        similarity_loss = self._similarity_loss(mu)
        info["similarity_loss"] = similarity_loss.item()
        loss = loss + similarity_loss

        info["total"] = loss.item()
        return loss, info
        # return entropy_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = Config()
        self._use_mask = config.TRAIN_OF_GVAE.USE_MASK

    def forward(
        self, out: torch.Tensor, tgt: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        tgt = tgt[:, 1:]
        if self._use_mask:
            mask = mask[:, 1:]
            out[mask == 0] = -torch.inf

        if tgt.dim() == 3:
            tgt = tgt.argmax(dim=-1)
        out = out.reshape(-1, out.size(-1))
        tgt = tgt.reshape(-1)
        return F.cross_entropy(out, tgt, reduction='sum') / out.size(0) * out.size(1)


class KLDLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()), dim=1
        ).mean()


class SimilarityLoss(nn.Module):
    def __init__(self, raw_similarity: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__()
        config = Config()
        self._raw_similarity = torch.as_tensor(raw_similarity, dtype=torch.float32)
        self._raw_similarity = (self._raw_similarity - self._raw_similarity.min()) / (
            self._raw_similarity.max() - self._raw_similarity.min()
        )
        self._raw_similarity = self._raw_similarity.to(config.DEVICE)
        self._loss = nn.MSELoss(reduction="mean")

    def forward(self, memory: torch.Tensor):
        similarity = torch.cdist(memory, memory, p=2)
        similarity = (similarity - similarity.min()) / (
            similarity.max() - similarity.min()
        )
        return self._loss(similarity, self._raw_similarity) * 2000


def train():
    def decode_eval():
        gvae.eval()
        with torch.no_grad():
            correct = 0
            src = eval_loader.to(config.DEVICE)
            tgt_ = src.clone()
            memory = gvae.encode(src)
            out = gvae.decode(memory, None)
            index = tgt_ == config.GVAE.NRULE - 1
            out[index] = config.GVAE.NRULE - 1
            result = ((out == tgt_)).sum(dim=1) == config.GVAE.MAX_LEN
            correct += result.sum()
        return correct / eval_loader.size(0)

    def encode_eval():
        gvae.eval()
        with torch.no_grad():
            eval_src = eval_loader.to(config.DEVICE)
            eval_mask_ = eval_mask.to(config.DEVICE)
            eval_tgt = eval_src.clone()
            eval_tgt_y = eval_src.clone()
            out, mu, log_var = gvae(eval_src, eval_tgt)
            loss, info = eval_loss_fn(out, eval_tgt_y, mu, log_var, eval_mask_)
            return loss.item(), info

    config = Config()
    writer = SummaryWriter(log_dir=Logger.ROOT)
    gvae = GVAE()
    gvae.to(config.DEVICE)
    optimizer = torch.optim.Adam(gvae.parameters(), lr=config.TRAIN_OF_GVAE.LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 1 - e / config.TRAIN_OF_GVAE.EPOCHS
    )
    (
        train_loader,
        mask_loader,
        train_similarity,
        eval_loader,
        eval_mask,
        eval_similarity,
    ) = get_data(False)
    train_loss_fn = VAELoss(train_similarity)
    eval_loss_fn = VAELoss(eval_similarity)

    min_loss = torch.inf
    for epoch in range(config.TRAIN_OF_GVAE.EPOCHS):
        gvae.train()
        loss_val = []
        out_list = []
        mu_list = []
        log_var_list = []
        tgt_list = []
        mask_list = []
        for src, mask in zip(train_loader, mask_loader):
            src = src.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            tgt = src.clone()
            out, mu, log_var = gvae(src, tgt)
            out_list.append(out)
            mu_list.append(mu)
            log_var_list.append(log_var)
            tgt_list.append(src.clone())
            mask_list.append(mask)
        out = torch.cat(out_list, dim=0)
        mu = torch.cat(mu_list, dim=0)
        log_var = torch.cat(log_var_list, dim=0)
        tgt_y = torch.cat(tgt_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        loss, info = train_loss_fn(out, tgt_y, mu, log_var, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val.append(loss.item())
        scheduler.step()
        loss_val = sum(loss_val) / len(loss_val)
        Logger.info(f"Epoch: {epoch}, Train Loss: {loss_val}")
        Logger.tb_add_scalars(writer, "train loss", info, epoch)

        if epoch % 20 == 0 and epoch != 0:
            loss_val, info = encode_eval()
            Logger.info(f"Epoch: {epoch}, Eval Loss: {loss_val}")
            Logger.tb_add_scalars(writer, "eval loss", info, epoch)
            if loss_val < min_loss:
                Logger.save_data("gvae", "pth", gvae.state_dict())
                min_loss = loss_val

            acc = decode_eval()
            Logger.info(f"Epoch: {epoch}, Acc: {acc}")
            Logger.tb_add_scalar(writer, "scalars/acc", acc, epoch)
