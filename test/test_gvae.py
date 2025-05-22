import os
import sys
import unittest

sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from src.gvae import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from utils.config import Config

cfg = Config()
cfg.load("config/default.json")


class TestGVAEComponents(unittest.TestCase):
    def setUp(self):
        self.rule_embedding = nn.Embedding(
            cfg.GVAE.NRULE, cfg.GVAE.TRANSFORMER.D_MODEL
        )
        self.pos_embedding = PositionalEmbedding(
            cfg.GVAE.TRANSFORMER.D_MODEL, max_len=cfg.GVAE.MAX_LEN
        )

    def test_transformer_encoder_output_shape(self):
        encoder = TransformerEncoder(self.rule_embedding, self.pos_embedding)
        fake_data = torch.tensor([[i + 1 for i in range(cfg.GVAE.MAX_LEN)]])
        self.assertEqual(fake_data.shape, (1, cfg.GVAE.MAX_LEN))
        mu, var = encoder(fake_data)
        self.assertEqual(mu.shape, (1, cfg.GVAE.D_MU))
        self.assertEqual(var.shape, (1, cfg.GVAE.D_MU))

    def test_transformer_encoder_batch(self):
        encoder = TransformerEncoder(self.rule_embedding, self.pos_embedding)
        batch_size = 4
        fake_data = torch.randint(0, cfg.GVAE.NRULE, (batch_size, cfg.GVAE.MAX_LEN))
        mu, var = encoder(fake_data)
        self.assertEqual(mu.shape, (batch_size, cfg.GVAE.D_MU))
        self.assertEqual(var.shape, (batch_size, cfg.GVAE.D_MU))

    def test_transformer_decoder_output_shape(self):
        decoder = TransformerDecoder(self.rule_embedding, self.pos_embedding)
        fake_data = torch.rand((1, cfg.GVAE.D_MU))
        tgt = torch.tensor([[i for i in range(cfg.GVAE.MAX_LEN)]])
        out = decoder(fake_data, tgt)
        self.assertEqual(out.shape, (1, cfg.GVAE.MAX_LEN - 1, cfg.GVAE.NRULE))

    def test_transformer_decoder_batch(self):
        decoder = TransformerDecoder(self.rule_embedding, self.pos_embedding)
        batch_size = 3
        fake_data = torch.rand((batch_size, cfg.GVAE.D_MU))
        tgt = torch.randint(0, cfg.GVAE.NRULE, (batch_size, cfg.GVAE.MAX_LEN))
        out = decoder(fake_data, tgt)
        self.assertEqual(out.shape, (batch_size, cfg.GVAE.MAX_LEN - 1, cfg.GVAE.NRULE))