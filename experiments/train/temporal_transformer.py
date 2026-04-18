from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TemporalResidualTransformer(nn.Module):
    """Stateful temporal residual head shared by training and evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str,
        window_override: int | None = None,
    ):
        super().__init__()

        model_cfg = cfg.model if hasattr(cfg, "model") else None
        d_model = model_cfg.get("transformer_d_model", 64) if model_cfg is not None else 64
        nhead = model_cfg.get("transformer_nhead", 4) if model_cfg is not None else 4
        num_layers = (
            model_cfg.get("transformer_layers", 2) if model_cfg is not None else 2
        )
        dim_ff = (
            model_cfg.get("transformer_dim_ff", d_model * 2)
            if model_cfg is not None
            else d_model * 2
        )

        self.window_size = (
            window_override
            if window_override is not None
            else (
                model_cfg.get("transformer_window", None)
                if model_cfg is not None
                else None
            )
        )

        self.tf_in = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=False,
        )
        self.temporal_tf = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tf_delta = nn.Linear(d_model, 3)
        self.tf_gate = nn.Linear(d_model, 3)

        self._feat_window: list[torch.Tensor] = []
        self.to(device)

    def reset_window(self) -> None:
        self._feat_window = []

    def load_component_state_dicts(self, state_dict: dict[str, Any] | None) -> None:
        if not isinstance(state_dict, dict):
            return

        self.tf_in.load_state_dict(state_dict.get("tf_in", {}), strict=False)
        self.temporal_tf.load_state_dict(
            state_dict.get("temporal_tf", {}), strict=False
        )
        self.tf_delta.load_state_dict(state_dict.get("tf_delta", {}), strict=False)
        self.tf_gate.load_state_dict(state_dict.get("tf_gate", {}), strict=False)

    def export_component_state_dicts(self) -> dict[str, dict[str, Any]]:
        return {
            "tf_in": self.tf_in.state_dict(),
            "temporal_tf": self.temporal_tf.state_dict(),
            "tf_delta": self.tf_delta.state_dict(),
            "tf_gate": self.tf_gate.state_dict(),
        }

    @staticmethod
    def _build_sinusoidal_pe(
        length: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, 1, d_model, device=device, dtype=dtype)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        points_feats: torch.Tensor,
        *,
        rollout_window_size: int | None = None,
        residual_scale: float = 0.1,
    ) -> torch.Tensor:
        bn = points_feats.shape[0] * points_feats.shape[1]
        curr = points_feats.reshape(bn, -1)
        self._feat_window.append(curr)

        window_size = (
            self.window_size if self.window_size is not None else rollout_window_size
        )
        if window_size is None:
            window_size = len(self._feat_window)
        window_size = max(1, int(window_size))

        if len(self._feat_window) > window_size:
            self._feat_window.pop(0)

        if len(self._feat_window) < window_size:
            pad_count = window_size - len(self._feat_window)
            pad_frame = self._feat_window[-1]
            seq_list = [pad_frame for _ in range(pad_count)] + self._feat_window
        else:
            seq_list = self._feat_window

        seq = torch.stack(seq_list, dim=0)
        seq_emb = self.tf_in(seq)
        seq_emb = seq_emb + self._build_sinusoidal_pe(
            seq_emb.shape[0],
            seq_emb.shape[2],
            seq_emb.device,
            seq_emb.dtype,
        )
        tf_out = self.temporal_tf(seq_emb)
        out_last = tf_out[-1]
        delta = torch.tanh(self.tf_delta(out_last)).reshape_as(points_feats)
        gate = torch.sigmoid(self.tf_gate(out_last)).reshape_as(points_feats)
        return ((1.0 - gate) * points_feats + gate * delta) * residual_scale
