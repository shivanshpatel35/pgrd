from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .abstract import NP2GMaterial
from .network.pointnet import PointNetEncoder
from .network.nerf import CondNeRFModel


def get_grid_locations(x, num_grids_list, dx):
    bsz = x.shape[0]
    x_grid = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0, (num_grids_list[0] - 1) * dx, num_grids_list[0], device=x.device
            ),
            torch.linspace(
                0, (num_grids_list[1] - 1) * dx, num_grids_list[1], device=x.device
            ),
            torch.linspace(
                0, (num_grids_list[2] - 1) * dx, num_grids_list[2], device=x.device
            ),
        ),
        dim=-1,
    ).reshape(-1, 3)
    grid_idxs = torch.stack(
        torch.meshgrid(
            torch.linspace(
                0, (num_grids_list[0] - 1), num_grids_list[0], device=x.device
            ),
            torch.linspace(
                0, (num_grids_list[1] - 1), num_grids_list[1], device=x.device
            ),
            torch.linspace(
                0, (num_grids_list[2] - 1), num_grids_list[2], device=x.device
            ),
        ),
        dim=-1,
    ).reshape(-1, 3)

    grid_hits = torch.zeros(
        bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], device=x.device
    )
    for i in range(3):
        for j in range(3):
            for k in range(3):
                grid_hits[
                    :,
                    ((x[:, :, 0] / dx - 0.5).int() + i).clamp(0, num_grids_list[0] - 1),
                    ((x[:, :, 1] / dx - 0.5).int() + j).clamp(0, num_grids_list[1] - 1),
                    ((x[:, :, 2] / dx - 0.5).int() + k).clamp(0, num_grids_list[2] - 1),
                ] = 1
    grid_hits = grid_hits.sum(0) > 0

    x_grid = x_grid[grid_hits.reshape(-1)].reshape(1, -1, 3).repeat(bsz, 1, 1)
    grid_idxs = grid_idxs[grid_hits.reshape(-1)]
    return x_grid, grid_idxs


def fill_grid_locations(feat, grid_idxs, num_grids_list):
    # feat: (bsz, num_active_grids, feature_dim)
    # grid_idxs: (num_active_grids, 3)
    bsz = feat.shape[0]
    feat_filled = torch.zeros(
        bsz,
        num_grids_list[0],
        num_grids_list[1],
        num_grids_list[2],
        3,
        device=feat.device,
    )
    grid_idxs_1d = (
        grid_idxs[:, 0] * num_grids_list[1] * num_grids_list[2]
        + grid_idxs[:, 1] * num_grids_list[2]
        + grid_idxs[:, 2]
    )
    feat_filled = feat_filled.reshape(bsz, -1, 3)
    feat_filled[:, grid_idxs_1d.long()] = feat.clone()
    feat_filled = feat_filled.reshape(
        bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], 3
    )
    return feat_filled


class PointNetNeRFRadiusAdaptiveMetaNP2G(NP2GMaterial):
    def __init__(self, cfg: DictConfig, n_history: int) -> None:
        super().__init__(cfg)

        self.feature_dim = 64
        self.radius = cfg.radius
        self.n_history = n_history

        self.his_type = 0
        if self.his_type == 0:
            self.encoder = PointNetEncoder(
                global_feat=(cfg.radius <= 0),
                feature_transform=False,
                feature_dim=self.feature_dim,
                channel=6 * (1 + self.n_history),
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

    def positional_encoding(self, tensor):
        num_encoding_functions = self.pe_num_func
        include_input = self.pe_include_input
        log_sampling = self.pe_log_sampling
        if num_encoding_functions == 0:
            assert include_input
            return tensor

        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0**0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    def forward(
        self, x: Tensor, v: Tensor, x_his: Tensor, v_his: Tensor, enabled: Tensor
    ) -> Tensor:
        # x: (bsz, num_particles, 3)
        # v: (bsz, num_particles, 3)
        # F: (bsz, num_particles, 3, 3)
        bsz = x.shape[0]
        num_particles = x.shape[1]
        v = v * self.input_scale
        v_his = v_his * self.input_scale

        x_his = x_his.reshape(bsz, num_particles, self.n_history, 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history, 3)
        x_his = x_his.detach()
        v_his = v_his.detach()

        x_grid, grid_idxs = get_grid_locations(x, self.num_grids_list, self.dx)
        x_grid = x_grid.detach()
        grid_idxs = grid_idxs.detach()

        # centering
        x_center = x.mean(1, keepdim=True)
        if self.absolute_y:
            x_center[:, :, 1] = 0  # only centering x and z
        x = x - x_center
        x_his = x_his - x_center[:, :, None]
        if self.training:
            # random azimuth
            theta = torch.rand(bsz, 1, device=x.device) * 2 * np.pi
            rot = torch.stack(
                [
                    torch.cos(theta),
                    torch.zeros_like(theta),
                    torch.sin(theta),
                    torch.zeros_like(theta),
                    torch.ones_like(theta),
                    torch.zeros_like(theta),
                    -torch.sin(theta),
                    torch.zeros_like(theta),
                    torch.cos(theta),
                ],
                dim=-1,
            ).reshape(bsz, 3, 3)
            inv_rot = rot.transpose(1, 2)
        else:
            rot = (
                torch.eye(3, dtype=x.dtype, device=x.device)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
            )
            inv_rot = rot.transpose(1, 2)
        x = torch.einsum("bij,bjk->bik", x, rot)
        x_his = torch.einsum(
            "bcij,bjk->bcik", x_his, rot
        )  # (bsz, num_particles, n_history, 3)
        v = torch.einsum("bij,bjk->bik", v, rot)
        v_his = torch.einsum(
            "bcij,bjk->bcik", v_his, rot
        )  # (bsz, num_particles, n_history, 3)

        # x_his = x_his.reshape(bsz, num_particles, self.n_history * 3)
        # v_his = v_his.reshape(bsz, num_particles, self.n_history * 3)

        # assert 'x_grid' in kwargs
        # x_grid = kwargs['x_grid']  # (bsz, num_grids, num_grids, num_grids, 3)
        # num_grids_x, num_grids_y, num_grids_z = x_grid.shape[1], x_grid.shape[2], x_grid.shape[3]

        x_grid = x_grid - x_center
        x_grid = x_grid @ rot

        if self.his_type == 0:
            x_his = x_his.reshape(bsz, num_particles, self.n_history * 3)
            v_his = v_his.reshape(bsz, num_particles, self.n_history * 3)
        elif self.his_type == 1:
            if self.n_history > 0:
                x_his = x_his.permute(0, 2, 1, 3)  # (bsz, n_history, num_particles, 3)
                v_his = v_his.permute(0, 2, 1, 3)  # (bsz, n_history, num_particles, 3)
                x_his = x_his.reshape(bsz * self.n_history, num_particles, 3)
                v_his = v_his.reshape(bsz * self.n_history, num_particles, 3)
                feat_his = torch.cat(
                    [x_his, v_his], dim=-1
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
                    1, x_grid.shape[1], 1
                )  # (bsz, num_grids_total, n_history * 16)
            else:
                feat_his = torch.zeros(bsz, x_grid.shape[1], 0, device=x.device)

        # F = F.reshape(bsz, num_particles, -1)
        # feat = torch.cat([x, v, F], dim=-1)  # (bsz, num_particles, 15)

        if self.his_type == 0:
            feat = torch.cat(
                [x, v, x_his, v_his], dim=-1
            )  # (bsz, num_particles, 6 * (1 + n_history))
            feat = feat.permute(0, 2, 1)  # (bsz, 6 * (1 + n_history), num_particles)
        elif self.his_type == 1:
            feat = torch.cat([x, v], dim=-1)  # (bsz, num_particles, 6)
            feat = feat.permute(0, 2, 1)  # (bsz, 6, num_particles)

        feat, trans, trans_feat = self.encoder(
            feat, enabled
        )  # feat: (bsz, feature_dim, num_particles)
        feat = self.bottleneck(feat)  # (bsz, feature_dim, num_particles)

        if self.radius > 0:
            # aggregate neighborhood
            # x.shape: (bsz, num_particles, 3)
            # x_grid.shape: (bsz, num_grids_total, 3)
            dist_pt_grid = torch.cdist(x_grid, x, p=2)
            mask = dist_pt_grid < self.radius  # (bsz, num_grids_total, num_particles)
            mask_normed = mask / (
                mask.sum(dim=-1, keepdim=True) + 1e-5
            )  # for each grid, normalize the weights
            mask_normed = mask_normed.detach()
            feat = mask_normed @ feat.permute(
                0, 2, 1
            )  # (bsz, num_grids_total, feature_dim)
        else:
            # global max pooling
            feat = feat[:, None, :].repeat(
                1, x_grid.shape[1], 1
            )  # (bsz, num_grids_total, feature_dim)

        # deprecated: just using the global feature
        # feat = feat[:, None, None, None, :].repeat(1, self.num_grids_list[0], self.num_grids_list[1], self.num_grids_list[2], 1)

        if self.his_type == 0:
            feat = feat.reshape(-1, self.feature_dim)
        elif self.his_type == 1:
            feat = torch.cat(
                [feat, feat_his], dim=-1
            )  # (bsz, num_grids_total, feature_dim + n_history * 16)
            feat = feat.reshape(-1, self.feature_dim + self.n_history * 16)

        x_grid = x_grid.reshape(-1, 3)
        x_grid = self.positional_encoding(x_grid)
        feat = self.decoder(x_grid, feat)
        feat = feat * self.output_scale
        feat = feat.reshape(bsz, -1, feat.shape[-1])
        feat = torch.bmm(feat, inv_rot)

        feat = fill_grid_locations(feat, grid_idxs, self.num_grids_list)
        return feat
