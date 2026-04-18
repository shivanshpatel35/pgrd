import torch
import torch.nn as nn


class CondNeRFModel(torch.nn.Module):
    def __init__(
        self,
        xyz_dim=3,
        condition_dim=64,
        out_channel=3,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
    ):
        super(CondNeRFModel, self).__init__()

        self.dim_xyz = xyz_dim
        self.dim_cond = condition_dim
        self.skip_connect_every = skip_connect_every

        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_cond, hidden_size)

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if (
                self.skip_connect_every is not None
                and i % self.skip_connect_every == 0
                and i > 0
                and i != num_layers - 1
            ):
                self.layers.append(
                    torch.nn.Linear(
                        self.dim_xyz + self.dim_cond + hidden_size, hidden_size
                    )
                )
            else:
                self.layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.layers_out = torch.nn.ModuleList(
            [
                # torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.Linear(hidden_size // 2, out_channel),
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, xyz, cond):
        assert xyz.shape[1] == self.dim_xyz
        xyz = xyz[..., : self.dim_xyz]  # (bsz * num_particles, 33)
        xyz = torch.cat([xyz, cond], 1)
        x = self.layer1(xyz)
        for i in range(len(self.layers)):
            if (
                self.skip_connect_every is not None
                and i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers[i](x))
        for i in range(len(self.layers_out) - 1):
            x = self.relu(self.layers_out[i](x))
        x = self.layers_out[-1](x)
        return torch.tanh(x)
