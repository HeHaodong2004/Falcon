"""Shared track GRU, scene/query attention, and explicit query-to-scene fusion."""
from dataclasses import dataclass

import torch
from torch import nn

from .inputs import HUMAN_FEATURES, ROBOT_FEATURES, QUERY_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    size: str = 'M'
    width: int = 128
    heads: int = 4
    temporal_layers: int = 2
    scene_layers: int = 2
    query_layers: int = 2
    fusion_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.

    @classmethod
    def family(cls, size='M', dropout=0.):
        return cls(size=size, width={'S': 64, 'M': 128, 'L': 224}[size], dropout=dropout)


class FusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, ff = config.width, config.width*config.ffn_multiplier
        self.query_norm = nn.LayerNorm(d)
        self.scene_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(ff, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, scene, padding):
        memory = self.scene_norm(scene)
        update, _ = self.attention(self.query_norm(query), memory, memory,
                                   key_padding_mask=padding, need_weights=False)
        query = query+self.dropout(update)
        return query+self.dropout(self.ff(self.ff_norm(query)))


class EventQueryWM(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.human_gru = nn.GRU(len(HUMAN_FEATURES), d, config.temporal_layers,
                                batch_first=True, dropout=config.dropout)
        self.robot_step = nn.Sequential(nn.Linear(len(ROBOT_FEATURES), d), nn.GELU(), nn.Linear(d, d))
        self.robot_pool = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.LayerNorm(d))
        self.human_norm = nn.LayerNorm(d)

        def encoder(layers):
            # Separate construction avoids cloned identical layer initialization.
            return nn.ModuleList([nn.TransformerEncoderLayer(
                d, config.heads, d*config.ffn_multiplier, config.dropout,
                activation='gelu', batch_first=True, norm_first=True) for _ in range(layers)])

        self.scene = encoder(config.scene_layers)
        self.query_projection = nn.Linear(len(QUERY_FEATURES), d)
        self.query = encoder(config.query_layers)
        self.fusion = nn.ModuleList([FusionLayer(config) for _ in range(config.fusion_layers)])
        self.output_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))

    def project_query(self, value):
        return self.query_projection(value.query)

    def encode_scene(self, value):
        """Shared, query-independent arrived-history encoder; original parameters."""
        humans = value.humans.masked_fill(value.human_padding[:, :, None, None], 0.)
        humans = humans.masked_fill(value.history_padding[:, None, :, None], 0.)
        b, n, t, features = humans.shape
        lengths = (~value.history_padding).sum(1)
        if (lengths == 0).any():
            raise ValueError('Each example needs arrived history')
        # Select final real timestep, not the last padded recurrent state.
        encoded, _ = self.human_gru(humans.reshape(b*n, t, features))
        final = (lengths-1).repeat_interleave(n)
        human_tokens = encoded[torch.arange(b*n, device=humans.device), final].reshape(b, n, -1)
        human_tokens = self.human_norm(human_tokens)
        robot = value.robot.masked_fill(value.history_padding[:, :, None], 0.)
        steps = self.robot_step(robot).masked_fill(value.history_padding[:, :, None], 0.)
        mean = steps.sum(1)/lengths[:, None]
        last = steps[torch.arange(b, device=humans.device), lengths-1]
        robot_token = self.robot_pool(torch.cat([mean, last], -1))
        scene = torch.cat([robot_token[:, None], human_tokens], 1)
        # The robot is always valid: even zero humans never produce all-masked attention.
        padding = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=humans.device), value.human_padding], 1)
        for layer in self.scene:
            scene = layer(scene, src_key_padding_mask=padding)
        return scene, padding

    def latent_from_scene(self, value, scene, padding):
        query = self.project_query(value)
        # Physical time and time_fraction are deterministic positional information.
        for layer in self.query:
            query = layer(query)
        for layer in self.fusion:
            query = layer(query, scene, padding)
        query = self.output_norm(query)
        return torch.cat([query.mean(1), query.max(1).values], -1)

    def sr_from_scene(self, value, scene, padding):
        return self.head(self.latent_from_scene(value, scene, padding)).squeeze(-1)

    def forward(self, value):
        return self.sr_from_scene(value, *self.encode_scene(value))
