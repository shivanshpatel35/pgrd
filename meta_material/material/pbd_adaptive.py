from typing import Optional
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .abstract import NP2GMaterial
from .network.pointnet import PointNetEncoder
from .network.nerf import CondNeRFModel
from .network.ptv3 import PTv3Encoder


class PointNetPBDAdaptiveMetaNP2G(NP2GMaterial):
    def __init__(self, cfg: DictConfig, n_history: int) -> None:
        super().__init__(cfg)

        self.feature_dim = 64
        self.feat_to_6d = nn.Conv1d(self.feature_dim, 6, 1)
        self.radius = cfg.radius
        self.n_history = n_history

        self.his_type = 0
        if self.his_type == 0:
            # self.encoder = PointNetEncoder(
            #     global_feat=(cfg.radius <= 0),
            #     feature_transform=False,
            #     feature_dim=self.feature_dim,
            #     channel=6 * (2 + self.n_history),
            # )
            self.encoder = PTv3Encoder(
                global_feat=(cfg.radius <= 0),
                feature_transform=False,
                feature_dim=self.feature_dim,
                channel=6 * (2 + self.n_history),
            )
        elif self.his_type == 1:
            self.encoder = PointNetEncoder(
                global_feat=(cfg.radius <= 0),
                feature_transform=False,
                feature_dim=self.feature_dim,
                channel=6,
            )
            self.history_encoder = PointNetEncoder(
                global_feat=True,
                feature_transform=False,
                feature_dim=16,
                channel=6,
            )
        else:
            raise NotImplementedError
        self.bottleneck = nn.Sequential(
            nn.Identity(),
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
        )
        self.decoder = None
        self.output_scale = cfg.output_scale
        self.input_scale = cfg.input_scale
        self.absolute_y = cfg.absolute_y if hasattr(cfg, "absolute_y") else False
        self.pe_num_func_res = (
            cfg.pe_num_func_res if hasattr(cfg, "pe_num_func_res") else 0
        )

    def set_params(self, num_grids, num_grids_flexible=None, requires_grad=True):
        if num_grids_flexible is None:
            self.num_grids_list = [num_grids, num_grids, num_grids]
            self.dx = 1 / num_grids
            self.inv_dx = float(num_grids)
        else:
            self.num_grids_list = num_grids_flexible[:3]
            self.dx = num_grids_flexible[3]
            self.inv_dx = 1 / self.dx
        self.requires_grad = requires_grad
        self.pe_num_func = int(np.log2(self.inv_dx)) + self.pe_num_func_res
        self.pe_include_input = True
        self.pe_log_sampling = True
        self.pe_dim = (
            3 + self.pe_num_func * 6 if self.pe_include_input else self.pe_num_func * 6
        )
        if self.his_type == 0:
            self.decoder = CondNeRFModel(
                xyz_dim=self.pe_dim,
                condition_dim=self.feature_dim,
                out_channel=3,
                num_layers=2,
                hidden_size=64,
                skip_connect_every=4,
            )
        elif self.his_type == 1:
            self.decoder = CondNeRFModel(
                xyz_dim=self.pe_dim,
                condition_dim=self.feature_dim + 16 * self.n_history,
                out_channel=3,
                num_layers=2,
                hidden_size=64,
                skip_connect_every=4,
            )

    # def positional_encoding(self, tensor):
    #     num_encoding_functions = self.pe_num_func  # 5
    #     include_input = self.pe_include_input  # True
    #     log_sampling = self.pe_log_sampling  # True
    #     if num_encoding_functions == 0:
    #         assert include_input
    #         return tensor

    #     encoding = [tensor] if include_input else []
    #     frequency_bands = None
    #     if log_sampling:
    #         frequency_bands = 2.0 ** torch.linspace(
    #             0.0,
    #             num_encoding_functions - 1,
    #             num_encoding_functions,
    #             dtype=tensor.dtype,
    #             device=tensor.device,
    #         )  # (5,)
    #     else:
    #         frequency_bands = torch.linspace(
    #             2.0**0.0,
    #             2.0 ** (num_encoding_functions - 1),
    #             num_encoding_functions,
    #             dtype=tensor.dtype,
    #             device=tensor.device,
    #         )

    #     for freq in frequency_bands:
    #         for func in [torch.sin, torch.cos]:
    #             encoding.append(func(tensor * freq))

    #     # Special case, for no positional encoding
    #     if len(encoding) == 1:
    #         return encoding[0]
    #     else:
    #         return torch.cat(encoding, dim=-1)

    def positional_encoding(self, tensor):
        # tensor: [B, 3]
        if self.pe_num_func == 0:
            return tensor

        # Create frequency bands, shape [1, num_encoding_functions]
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            self.pe_num_func - 1,
            self.pe_num_func,
            dtype=tensor.dtype,
            device=tensor.device,
        ).unsqueeze(0)

        # Use broadcasting to apply frequencies to input tensor
        # (B, 1, 3) * (1, F, 1) -> (B, F, 3)
        inputs = tensor.unsqueeze(1) * frequency_bands.unsqueeze(2)

        # Apply sin/cos and reshape
        # (B, F, 3) -> (B, F*3)
        encoded = torch.cat([torch.sin(inputs), torch.cos(inputs)], dim=1)
        encoded = encoded.reshape(tensor.shape[0], -1)  # (B, F*3*2)

        # Concatenate original input if needed
        if self.pe_include_input:
            return torch.cat([tensor, encoded], dim=-1)
        else:
            return encoded

    def forward(
        self,
        x: Tensor,
        v: Tensor,
        x_his: Tensor,
        v_his: Tensor,
        enabled: Tensor,
        x_sim: Tensor,
        v_sim: Tensor,
        rot: Optional[Tensor] = None,
    ) -> Tensor:
        # x: (bsz, num_particles, 3)
        # v: (bsz, num_particles, 3)
        # F: (bsz, num_particles, 3, 3)
        bsz = x.shape[0]
        num_particles = x.shape[1]
        v = v * self.input_scale
        v_his = v_his * self.input_scale
        v_sim = v_sim * self.input_scale

        x_his = x_his.reshape(bsz, num_particles, self.n_history, 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history, 3)
        x_his = x_his.detach()
        v_his = v_his.detach()

        # centering
        x_center = x.mean(1, keepdim=True)  # (bsz, 1, 3)
        if self.absolute_y:
            x_center[:, :, 1] = 0  # only centering x and z
        x_rotated = x - x_center
        x_his_rotated = x_his - x_center[:, :, None]
        x_sim_rotated = x_sim - x_center

        # Apply provided rotation if given, otherwise use identity per batch element
        if rot is None:
            rot = (
                torch.eye(3, dtype=x.dtype, device=x.device)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
            )  # (bsz, 3, 3)
        else:
            # Ensure expected shape/dtype/device
            assert rot.shape == (bsz, 3, 3), "rot must have shape (bsz, 3, 3)"
            rot = rot.to(device=x.device, dtype=x.dtype)
        inv_rot = rot.transpose(1, 2)  # (bsz, 3, 3)
        x_rotated = torch.einsum(
            "bij,bjk->bik", x_rotated, rot
        )  # (bsz, num_particles, 3)
        x_his_rotated = torch.einsum(
            "bcij,bjk->bcik", x_his_rotated, rot
        )  # (bsz, num_particles, n_history, 3)
        v_rotated = torch.einsum("bij,bjk->bik", v, rot)
        v_his_rotated = torch.einsum(
            "bcij,bjk->bcik", v_his, rot
        )  # (bsz, num_particles, n_history, 3)
        x_sim_rotated = torch.einsum("bij,bjk->bik", x_sim_rotated, rot)
        v_sim_rotated = torch.einsum("bij,bjk->bik", v_sim, rot)

        if self.his_type == 0:
            x_his_rotated = x_his_rotated.reshape(
                bsz, num_particles, self.n_history * 3
            )  # (bsz, num_particles, n_history * 3)
            v_his_rotated = v_his_rotated.reshape(
                bsz, num_particles, self.n_history * 3
            )  # (bsz, num_particles, n_history * 3)
        elif self.his_type == 1:
            if self.n_history > 0:
                x_his_permuted = x_his_rotated.permute(
                    0, 2, 1, 3
                )  # (bsz, n_history, num_particles, 3)
                v_his_permuted = v_his_rotated.permute(
                    0, 2, 1, 3
                )  # (bsz, n_history, num_particles, 3)
                x_his_permuted = x_his_permuted.reshape(
                    bsz * self.n_history, num_particles, 3
                )
                v_his_permuted = v_his_permuted.reshape(
                    bsz * self.n_history, num_particles, 3
                )
                feat_his = torch.cat(
                    [x_his_permuted, v_his_permuted], dim=-1
                )  # (bsz * n_history, num_particles, 6)
                feat_his = feat_his.permute(
                    0, 2, 1
                )  # (bsz * n_history, 6, num_particles)
                enabled_his = (
                    enabled[:, None, :]
                    .repeat(1, self.n_history, 1)
                    .reshape(-1, num_particles)
                )
                feat_his, _, _ = self.history_encoder(
                    feat_his, enabled_his
                )  # feat: (bsz * n_history, 16)
                feat_his = feat_his.reshape(
                    bsz, self.n_history * 16
                )  # (bsz, n_history * 16)
                feat_his = feat_his[:, None, :].repeat(
                    1, num_particles, 1
                )  # (bsz, num_particles, n_history * 16)
            else:
                feat_his = torch.zeros(bsz, num_particles, 0, device=x.device)

        if self.his_type == 0:
            feat = torch.cat(
                [
                    x_rotated,
                    v_rotated,
                    x_his_rotated,
                    v_his_rotated,
                    x_sim_rotated,
                    v_sim_rotated,
                ],
                dim=-1,
            )  # (bsz, num_particles, 6 * (2 + n_history))
            feat = feat.permute(0, 2, 1)  # (bsz, 6 * (2 + n_history), num_particles)
        elif self.his_type == 1:
            feat = torch.cat([x_rotated, v_rotated], dim=-1)  # (bsz, num_particles, 6)
            feat = feat.permute(0, 2, 1)  # (bsz, 6, num_particles)

        feat, trans, trans_feat = self.encoder(
            feat, enabled
        )  # feat: (bsz, feature_dim, num_particles)
        feat = self.bottleneck(feat)  # (bsz, feature_dim, num_particles)
        feat = feat.permute(0, 2, 1)  # (bsz, num_particles, feature_dim)

        if self.his_type == 1:
            feat = torch.cat(
                [feat, feat_his], dim=-1
            )  # (bsz, num_particles, feature_dim + n_history * 16)

        feat = feat.reshape(-1, feat.shape[-1])  # (bsz * num_particles, feature_dim)

        pos_in = self.positional_encoding(
            x_rotated.reshape(-1, 3)
        )  # (bsz * num_particles, 33)
        feat_points = self.decoder(pos_in, feat)
        feat_points = feat_points * self.output_scale
        feat_points = feat_points.reshape(bsz, num_particles, feat_points.shape[-1])

        # v1 = feat_points[..., :3]  # (B, P, 3)
        # v2 = feat_points[..., 3:]  # (B, P, 3)

        # # rotate both
        # r1 = torch.bmm(v1, inv_rot)  # (B, P, 3)
        # r2 = torch.bmm(v2, inv_rot)  # (B, P, 3)

        # feat_per_point = torch.cat([r1, r2], dim=-1)  # (B, P, 6)

        feat_per_point = torch.bmm(feat_points, inv_rot)  # (B, P, 3)

        return feat_per_point
