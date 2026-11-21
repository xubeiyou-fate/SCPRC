"""Sparse phase/lag conditioned exchange of PDM seasonal innovations.

This is an experimental research operator, not part of upstream TimeMixer.
Input order is TimeMixer channel-independent order: [batch * nodes, time, dim].
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PhaseLagInnovationExchange(nn.Module):
    def __init__(self, adjacency: np.ndarray, d_model: int, scales: int,
                 topk: int = 4, rank: int = 8, lags: tuple[int, ...] = (0, 1, 2),
                 down_sampling_window: int = 2):
        super().__init__()
        a = np.maximum(np.asarray(adjacency, dtype=np.float32), 0.0).copy()
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("adjacency must be square")
        np.fill_diagonal(a, 0.0)
        nodes = int(a.shape[0])
        if topk < 1 or rank < 1 or down_sampling_window < 2 or not lags or 0 not in lags or any(l < 0 for l in lags):
            raise ValueError("invalid graph rank, topk, or lags")
        k = min(int(topk), nodes - 1)
        idx = np.argsort(-a, axis=1)[:, :k].copy()
        edge = np.take_along_axis(a, idx, axis=1)
        isolated = np.flatnonzero(edge.sum(1) <= 0)
        # A self edge on an isolated node contributes equal incoming/outgoing
        # flux and therefore leaves that node unchanged.
        idx[isolated, 0] = isolated
        edge[isolated, 0] = 1.0
        self.register_buffer("neighbor_index", torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer("edge_log_weight", torch.as_tensor(
            np.log(np.maximum(edge, 1e-8)), dtype=torch.float32))
        self.register_buffer("edge_valid", torch.as_tensor(edge > 0))
        self.nodes = nodes
        self.rank = int(rank)
        self.scales = int(scales)
        self.lags = tuple(int(l) for l in lags)
        self.down_sampling_window = int(down_sampling_window)
        self.query = nn.Linear(d_model, rank, bias=False)
        self.key = nn.Linear(d_model, rank, bias=False)
        self.value = nn.Linear(d_model, rank, bias=False)
        self.phase_embedding = nn.Parameter(torch.zeros(24, rank))
        self.phase_lag_bias = nn.Parameter(torch.zeros(scales, 24, len(lags)))
        self.out = nn.Linear(rank, d_model, bias=False)

    def forward(self, season: torch.Tensor, scale: int) -> torch.Tensor:
        if season.ndim != 3 or season.shape[0] % self.nodes:
            raise ValueError("season must have shape [batch * nodes, time, dim]")
        if not 0 <= scale < self.scales:
            raise ValueError("invalid scale")
        bn, steps, dim = season.shape
        if max(self.lags) >= steps:
            raise ValueError("all lags must be shorter than the current scale")
        batch = bn // self.nodes
        z = season.reshape(batch, self.nodes, steps, dim)
        # Only regional deviations travel; the citywide seasonal component is preserved.
        innovation = z - z.mean(dim=1, keepdim=True)
        phase = (torch.arange(steps, device=season.device) *
                 (self.down_sampling_window ** scale)) % 24
        q_base = self.query(innovation)
        q = q_base + self.phase_embedding[phase].to(q_base.dtype).view(1, 1, steps, self.rank)
        keys = self.key(innovation)
        values = self.value(innovation)
        logits = []
        candidates = []
        for lag_index, lag in enumerate(self.lags):
            if lag:
                source_key = F.pad(keys[:, :, :-lag, :], (0, 0, lag, 0))
                source_value = F.pad(values[:, :, :-lag, :], (0, 0, lag, 0))
            else:
                source_key, source_value = keys, values
            neighbor_key = source_key[:, self.neighbor_index]
            neighbor_value = source_value[:, self.neighbor_index]
            score = (q.unsqueeze(2) * neighbor_key).sum(-1) / math.sqrt(self.rank)
            score = score + self.edge_log_weight.view(1, self.nodes, -1, 1)
            score = score + self.phase_lag_bias[scale, phase, lag_index].view(1, 1, 1, steps)
            valid = self.edge_valid.view(1, self.nodes, -1, 1)
            if lag:
                valid = valid & (torch.arange(steps, device=season.device) >= lag).view(1, 1, 1, steps)
            score = score.masked_fill(~valid, -1e4)
            logits.append(score)
            candidates.append(neighbor_value)
        # [B,N,T,K*L] attention over actual neighbors and available lags.
        logit = torch.stack(logits, dim=-1).permute(0, 1, 3, 2, 4).flatten(-2)
        candidate = torch.stack(candidates, dim=-2).permute(0, 1, 3, 2, 4, 5)
        candidate = candidate.reshape(batch, self.nodes, steps, -1, self.rank)
        attention = logit.softmax(dim=-1).to(candidate.dtype)
        flux = attention.unsqueeze(-1) * (candidate - values.unsqueeze(-2))
        incoming = flux.sum(dim=-2)
        # Every directed transfer adds to its destination and subtracts the
        # identical latent amount from its source at the current time index.
        # This keeps the citywide latent component unchanged without a dense
        # global correction that would connect otherwise unrelated nodes.
        edge_flux = flux.permute(0, 1, 3, 2, 4).reshape(batch, -1, steps, self.rank)
        source_nodes = self.neighbor_index.repeat_interleave(len(self.lags), dim=1).reshape(-1)
        scatter_index = source_nodes.view(1, -1, 1, 1).expand(batch, -1, steps, self.rank)
        outgoing = torch.zeros_like(values).scatter_add(1, scatter_index, edge_flux)
        delta = incoming - outgoing
        changed = z + 0.1 * self.out(delta)
        return changed.reshape(bn, steps, dim)
