from pathlib import Path
import random
import time
import os
import json
from collections import defaultdict
from tqdm import trange
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import warp as wp
from warp import build
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import kornia
import traceback
import open3d as o3d

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import meta_material
from meta_material.data import RealTeleopBatchDataset, RealGripperDataset
from meta_material.utils import get_root, mkdir
from meta_material.wandb import Logger
from meta_material.sim import create_simulator

from experiments.train.eval_during_training import eval_parallel
from experiments.train.temporal_transformer import TemporalResidualTransformer

root: Path = get_root(__file__)


SIM_PARAM_FIELDS = (
    "threshold",
    "stiffness",
    "damping",
    "max_springs_per_node",
    "ground_friction",
)


def infer_object_name(cfg: DictConfig) -> str:
    explicit_object_name = cfg.train.get("object_name")
    if explicit_object_name:
        return str(explicit_object_name)

    dataset_name = cfg.train.get("dataset_name")
    if dataset_name:
        return Path(str(dataset_name)).parts[0]

    train_name = cfg.train.get("name")
    if train_name:
        return Path(str(train_name)).parts[0]

    raise ValueError(
        "Could not infer object name. Set train.object_name or train.dataset_name."
    )


def apply_object_sim_params(cfg: DictConfig) -> str:
    object_name = infer_object_name(cfg)
    object_sim_params = OmegaConf.select(cfg, f"sim_params.objects.{object_name}")
    if object_sim_params is None:
        raise KeyError(
            f"No simulator parameters configured for object '{object_name}' in sim_params."
        )

    missing_fields = [
        field for field in SIM_PARAM_FIELDS if OmegaConf.select(object_sim_params, field) is None
    ]
    if missing_fields:
        missing_fields_str = ", ".join(missing_fields)
        raise KeyError(
            f"Simulator parameters for object '{object_name}' are missing: {missing_fields_str}"
        )

    with open_dict(cfg):
        cfg.train.object_name = object_name
        for field in SIM_PARAM_FIELDS:
            cfg.sim[field] = OmegaConf.select(object_sim_params, field)

    return object_name


def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        cnt += 1
        for data in dataloader:
            yield data

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg, resolve=True))

        wp.init()
        wp.ScopedTimer.enabled = False
        wp.set_module_options({"fast_math": False})
        wp.config.verify_autograd_array_access = True

        gpus = [int(gpu) for gpu in cfg.gpus]
        wp_devices = [wp.get_device(f"cuda:{gpu}") for gpu in gpus]
        torch_devices = [torch.device(f"cuda:{gpu}") for gpu in gpus]
        device_count = len(torch_devices)

        assert device_count == 1
        self.wp_device = wp_devices[0]
        self.torch_device = torch_devices[0]

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.autograd.set_detect_anomaly(True)

        torch.backends.cudnn.benchmark = True

        log_root: Path = root / "log"
        exp_root: Path = log_root / cfg.train.name
        ckpt_root: Path = exp_root / "ckpt"

        if not (cfg.resume and cfg.train.resume_iteration > 0):
            mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
            OmegaConf.save(cfg, exp_root / "hydra.yaml", resolve=True)
            ckpt_root.mkdir(parents=True, exist_ok=True)

        self.log_root = log_root
        self.ckpt_root = ckpt_root

        if str(root).startswith("/burg") or str(root).startswith("/local"):
            self.use_pv = False
            self.dataset_non_overwrite = True
        else:
            self.use_pv = not cfg.train.disable_pv
            self.dataset_non_overwrite = False
        if not self.use_pv:
            print("not using pv rendering...")

        assert self.cfg.train.source_dataset_name is not None
        source_dataset_path = log_root / str(cfg.train.source_dataset_name)
        if os.path.exists(source_dataset_path / f'episode_0000' / 'meta.txt'):
            meta = np.loadtxt(source_dataset_path / f'episode_0000' / 'meta.txt')
            metadata_file = source_dataset_path / 'metadata.json'
            if os.path.exists(metadata_file):
                with open(metadata_file) as f:
                    datadir_list = json.load(f)
                datadir = datadir_list[0]
                source_data_dir = datadir['path']
                source_episode_id = int(meta[0])
                source_frame_start = int(meta[1]) + int(cfg.sim.n_history) * int(cfg.train.dataset_load_skip_frame) * int(cfg.train.dataset_skip_frame)
                self.use_gs = os.path.exists((log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' / 'gs' / f'{source_frame_start:06d}.splat')
            else:
                # Fallback for non-merged datasets
                self.use_gs = os.path.exists(source_dataset_path.parent / "episode_0000" / "gs")
        else:
            # Fallback for datasets without meta.txt
            self.use_gs = os.path.exists((log_root / str(cfg.train.source_dataset_name)).parent / "episode_0000" / "gs")

        # logging
        self.verbose = False
        if not cfg.debug:
            logger = Logger(cfg, project="meta-material-np2g-teleop-eval")
            self.logger = logger

    def load_train_dataset(self):
        cfg = self.cfg
        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / "dataset"

        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        if not os.path.exists(source_dataset_root):
            source_dataset_root = (
                Path("./data/meta-material") / cfg.train.source_dataset_name
            )
            assert os.path.exists(source_dataset_root)

        dataset = RealTeleopBatchDataset(
            cfg,
            dataset_root=self.log_root / cfg.train.dataset_name / "state",
            source_data_root=source_dataset_root,
            device=self.torch_device,
            num_steps=cfg.sim.num_steps_train,
            train=True,
            # dataset_non_overwrite=self.dataset_non_overwrite,
            dataset_non_overwrite=False,
            lazy_load=cfg.train.lazy_load,
        )
        self.dataset = dataset

    def init_train(self):
        cfg = self.cfg

        dataloader = dataloader_wrapper(
            DataLoader(
                self.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            "dataset",
        )
        self.dataloader = dataloader

        residualnet_requires_grad = cfg.model.residualnet.requires_grad

        n_history = cfg.sim.n_history if hasattr(cfg.sim, "n_history") else 0
        residualnet: nn.Module = getattr(
            meta_material.material, cfg.model.residualnet.cls
        )(cfg.model.residualnet, n_history)
        residualnet.set_params(
            cfg.sim.num_grids, num_grids_flexible=cfg.sim.num_grids_flexible
        )
        residualnet.to(self.torch_device)
        if len(list(residualnet.parameters())) == 0:
            residualnet_requires_grad = False
        residualnet.requires_grad_(residualnet_requires_grad)

        self.temporal_model = TemporalResidualTransformer(
            cfg,
            device=self.torch_device,
        )
        residualnet.train(True)

        if cfg.resume and cfg.train.resume_iteration > 0:
            print(f"\033[91mLoading checkpoint from {self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt'}\033[0m")
       
            assert (self.ckpt_root / f"{cfg.train.resume_iteration:06d}.pt").exists()
            ckpt = torch.load(
                self.ckpt_root / f"{cfg.train.resume_iteration:06d}.pt",
                map_location=self.torch_device,
            )
            residualnet.load_state_dict(ckpt["residualnet"])
            # Load transformer if present; keep backward-compat for older RNN ckpts
            if "transformer" in ckpt:
                self.temporal_model.load_component_state_dicts(ckpt["transformer"])

        elif cfg.model.ckpt:
            ckpt = torch.load(
                self.log_root / cfg.model.ckpt, map_location=self.torch_device
            )
            residualnet.load_state_dict(ckpt["residualnet"])
            if "transformer" in ckpt:
                self.temporal_model.load_component_state_dicts(ckpt["transformer"])

        if not (cfg.resume and cfg.train.resume_iteration > 0):
            torch.save(
                {
                    "residualnet": residualnet.state_dict(),
                    "transformer": self.temporal_model.export_component_state_dicts(),
                },
                self.ckpt_root / f"{cfg.train.resume_iteration:06d}.pt",
            )

        params_to_optimize = list(self.temporal_model.parameters())
        if residualnet_requires_grad:
            params_to_optimize += list(residualnet.parameters())

        residualnet_optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=cfg.train.residualnet_lr,
            weight_decay=cfg.train.residualnet_wd,
        )
        self.residualnet_lr_T_max = 2000
        residualnet_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            residualnet_optimizer,
            T_max=self.residualnet_lr_T_max,
            eta_min=0.01 * cfg.train.residualnet_lr,
        )
        if cfg.train.resume_iteration > 0:
            residualnet_lr_scheduler.last_epoch = cfg.train.resume_iteration - 1
            residualnet_lr_scheduler.step()

        criterion = nn.MSELoss(reduction="mean")
        criterion.to(self.torch_device)
        criterion_l1 = nn.L1Loss(reduction="mean")
        criterion_l1.to(self.torch_device)

        total_step_count = 0
        if cfg.resume and cfg.train.resume_iteration > 0:
            total_step_count = cfg.train.resume_iteration * cfg.sim.num_steps_train
        losses_log = defaultdict(int)
        loss_factor_v = (
            cfg.train.loss_factor_v
            if hasattr(cfg.train, "loss_factor_v")
            else cfg.train.loss_factor
        )
        loss_factor_x = (
            cfg.train.loss_factor_x
            if hasattr(cfg.train, "loss_factor_x")
            else cfg.train.loss_factor * 100.0
        )

        self.loss_factor_v = loss_factor_v
        self.loss_factor_x = loss_factor_x
        self.residualnet_requires_grad = residualnet_requires_grad
        self.residualnet = residualnet
        self.residualnet_optimizer = residualnet_optimizer
        self.residualnet_lr_scheduler = residualnet_lr_scheduler
        self.criterion = criterion
        # self.criterion_l1 = criterion_l1
        self.total_step_count = total_step_count
        self.losses_log = losses_log

        self.start_time = time.time()

        # Clear kernel cache before training starts to avoid conflicts between eval and train
        build.clear_kernel_cache()

    def train(self, start_iteration, end_iteration, save=True):
        cfg = self.cfg
        self.residualnet.train(True)
        self.temporal_model.train(True)

        for iteration in trange(start_iteration, end_iteration, dynamic_ncols=True):
            if self.residualnet_requires_grad:
                self.residualnet_optimizer.zero_grad()

            losses = defaultdict(int)

            init_data, actions, gt_states = next(self.dataloader)

            if len(init_data) >= 8 and int(cfg.sim.num_fill_points) > 0:
                x, v, x_his, v_his, clip_bound, enabled, episode_vec, fill_points = (
                    init_data
                )
            else:
                x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_data
                fill_points = None

            x = x.to(self.torch_device)
            v = v.to(self.torch_device)
            x_his = x_his.to(self.torch_device)
            v_his = v_his.to(self.torch_device)
            if fill_points is not None:
                fill_points = fill_points.to(self.torch_device)

            _, grippers = actions
            grippers = grippers.to(self.torch_device)

            if fill_points is not None and int(cfg.sim.num_fill_points) > 0:
                target_fill = int(cfg.sim.num_fill_points)
                if fill_points.dim() == 2:
                    fill_points = fill_points.unsqueeze(0)
                cur_k = fill_points.shape[1]
                if cur_k < target_fill:
                    pad = torch.zeros(
                        (fill_points.shape[0], target_fill - cur_k, 3),
                        device=fill_points.device,
                        dtype=fill_points.dtype,
                    )
                    fill_points = torch.cat([fill_points, pad], dim=1)
                elif cur_k > target_fill:
                    fill_points = fill_points[:, :target_fill]

                zeros_vel = torch.zeros_like(fill_points)
                x = torch.cat([x, fill_points], dim=1)
                v = torch.cat([v, zeros_vel], dim=1)
                enabled = torch.cat(
                    [
                        enabled,
                        torch.ones(
                            (enabled.shape[0], target_fill),
                            device=enabled.device,
                            dtype=enabled.dtype,
                        ),
                    ],
                    dim=1,
                )
                appended_fill_points = True
            else:
                appended_fill_points = False

            dataset_context = " ".join(
                (
                    str(cfg.train.dataset_name).lower(),
                    str(cfg.train.source_dataset_name).lower(),
                    str(cfg.train.get("object_name", "")).lower(),
                )
            )
            use_flag_dataset_behavior = "flag" in dataset_context
            use_teddy_dataset_behavior = "teddy" in dataset_context

            # Keep dataset-specific training-time radii explicit while preserving defaults.
            train_gripper_radius = float(getattr(cfg.model, "gripper_radius", 0.04))
            for dataset_key, radius in (("teddy", 0.015), ("flag", 0.01), ("cleaner", 0.1)):
                if dataset_key in dataset_context:
                    train_gripper_radius = radius
                    break

            gt_x, gt_v = gt_states
            gt_x = gt_x.to(self.torch_device)
            gt_v = gt_v.to(self.torch_device)

            batch_size = gt_x.shape[0]
            num_steps_total = gt_x.shape[1]
            num_particles = gt_x.shape[2]

            num_particles_orig = num_particles

            enabled = enabled.to(self.torch_device)

            self.temporal_model.reset_window()
            enabled_mask = enabled.unsqueeze(-1).repeat(
                1, 1, 3
            )

            sim = create_simulator(
                backend=getattr(cfg.sim, "backend", "spring"),
                x=x.detach().clone().cpu().numpy(),
                v=v.detach().clone().cpu().numpy(),
                grippers=grippers.detach().clone().cpu().numpy(),
                points_per_env=(
                    int(x.shape[1])
                    if appended_fill_points
                    else int(cfg.sim.n_particles)
                ),
                batch_size=batch_size,
                threshold=cfg.sim.threshold,
                stiffness=cfg.sim.stiffness,
                damping=cfg.sim.damping,
                mass_per_point=cfg.sim.mass_per_point,
                sim_dt=cfg.sim.dt,
                sim_substeps=cfg.sim.sim_substeps,
                device=self.wp_device,
                visualize=cfg.visualize,
                use_flag_dataset_behavior=use_flag_dataset_behavior,
                is_teddy_dataset=use_teddy_dataset_behavior,
                poke_stiffness=float(getattr(cfg.sim, "poke_stiffness", 100.0)),
                gripper_radius=train_gripper_radius,
                max_springs_per_node=cfg.sim.max_springs_per_node,
                ground_friction=cfg.sim.ground_friction,
            )

            x_full, v_full = sim.get_initial_state()
            x = x_full.to(self.torch_device)
            v = v_full.to(self.torch_device)

            num_particles = x.shape[1]

            if cfg.sim.n_history > 0:
                num_internal_particles = num_particles - num_particles_orig
                if num_internal_particles > 0:
                    x_internal = x[:, num_particles_orig:]
                    v_internal = v[:, num_particles_orig:]

                    n_history = cfg.sim.n_history
                    x_his_padding = (
                        x_internal.unsqueeze(2)
                        .repeat(1, 1, n_history, 1)
                        .reshape(x.shape[0], num_internal_particles, -1)
                    )
                    v_his_padding = (
                        v_internal.unsqueeze(2)
                        .repeat(1, 1, n_history, 1)
                        .reshape(v.shape[0], num_internal_particles, -1)
                    )

                    x_his = torch.cat([x_his, x_his_padding], dim=1)
                    v_his = torch.cat([v_his, v_his_padding], dim=1)

            enabled_padding_shape = (
                enabled.shape[0],
                num_particles - num_particles_orig,
            )
            enabled_padding = torch.ones(
                enabled_padding_shape, device=enabled.device, dtype=enabled.dtype
            )
            enabled_full = torch.cat([enabled, enabled_padding], dim=1)

            held_mask: torch.Tensor | None = None
            if (
                grippers is not None
                and grippers.shape[2] > 0
                and cfg.sim.gripper_forcing
                and not cfg.sim.gripper_points
            ):
                g_xyz0 = grippers[:, 0, :, :3].to(x.device)  # (B, G, 3)
                g_closed0 = (grippers[:, 0, :, -1] < 0.5).to(x.device)  # (B, G)

                dists0 = torch.norm(
                    x.unsqueeze(2) - g_xyz0.unsqueeze(1), dim=-1
                )  # (B, N, G)
                is_held0 = (dists0 < sim.gripper_radius) & g_closed0.unsqueeze(1)
                held_mask = is_held0.any(dim=-1)  # (B, N)
                if use_flag_dataset_behavior and hasattr(sim, "_flag_y_thresholds"):
                    y_thresh_env = wp.to_torch(sim._flag_y_thresholds).to(x.device).view(
                        -1, 1
                    )
                    y_thresh = y_thresh_env - 1.0e-6
                    top_band = x[..., 1] > y_thresh
                    any_closed = g_closed0.any(dim=1, keepdim=True)
                    held_mask = held_mask | (top_band & any_closed)
                if use_teddy_dataset_behavior:
                    held_mask = None

            theta = 2 * np.pi * torch.rand(batch_size, 1, device=self.torch_device)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            rot = (
                torch.stack(
                    [
                        cos_t,
                        torch.zeros_like(theta),
                        sin_t,
                        torch.zeros_like(theta),
                        torch.ones_like(theta),
                        torch.zeros_like(theta),
                        -sin_t,
                        torch.zeros_like(theta),
                        cos_t,
                    ],
                    dim=-1,
                )
                .reshape(batch_size, 3, 3)
                .to(dtype=x.dtype)
            )
            for step in range(num_steps_total):
                x_in = x.clone()
                if step == 0:
                    x_in_gt = x.clone()
                    v_in_gt = v.clone()
                else:
                    x_in_gt = x_in_gt + v_in_gt * cfg.sim.dt * cfg.sim.interval

                gripper_data_to_pass = (
                    grippers
                    if (
                        cfg.sim.gripper_forcing
                        and not cfg.sim.gripper_points
                        and grippers is not None
                        and grippers.shape[2] > 0
                    )
                    else None
                )

                x_sim, v_sim = sim(
                    step,
                    x.detach().clone(),
                    v.detach().clone(),
                    None,
                    gripper_data_to_pass,
                )

                x_sim = x_sim.detach()
                v_sim = v_sim.detach()

                points_feats = self.residualnet(
                    x,
                    v,
                    x_his,
                    v_his,
                    enabled_full,
                    x_sim,
                    v_sim,
                    rot,
                )

                residual_v = self.temporal_model(
                    points_feats,
                    rollout_window_size=num_steps_total,
                )

                mask_to_apply = held_mask
                if (
                    use_flag_dataset_behavior
                    and grippers is not None
                    and grippers.shape[2] > 0
                    and cfg.sim.gripper_forcing
                    and not cfg.sim.gripper_points
                    and hasattr(sim, "_flag_y_thresholds")
                ):
                    g_xyz_step = grippers[:, step, :, :3].to(x.device)
                    g_closed_step = (grippers[:, step, :, -1] < 0.5).to(x.device)
                    dists_step = torch.norm(x.unsqueeze(2) - g_xyz_step.unsqueeze(1), dim=-1)
                    step_mask = (
                        (dists_step < sim.gripper_radius) & g_closed_step.unsqueeze(1)
                    ).any(dim=-1)
                    y_thresh_env = wp.to_torch(sim._flag_y_thresholds).to(x.device).view(
                        -1, 1
                    )
                    y_thresh = y_thresh_env - 1.0e-6
                    top_band = x[..., 1] > y_thresh
                    any_closed_step = g_closed_step.any(dim=1, keepdim=True)
                    mask_to_apply = step_mask | (top_band & any_closed_step)

                if mask_to_apply is not None:
                    residual_v = torch.where(
                        mask_to_apply.unsqueeze(-1),
                        torch.zeros_like(residual_v),
                        residual_v,
                    )

                v = v_sim + residual_v

                residual_x = residual_v * sim.sim_dt
                x = x_sim + residual_x
                x_y_clamped = x[..., 1].clamp(min=sim.ground_y)
                x = torch.stack([x[..., 0], x_y_clamped, x[..., 2]], dim=-1)

                if step == 0:
                    with torch.no_grad():
                        loss_x_baseline = (
                            self.criterion(
                                x_sim[:, :num_particles_orig][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                                gt_x[:, step][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                            )
                            * self.loss_factor_x
                        )
                        self.losses_log["loss_x_baseline"] += loss_x_baseline.item()

                        loss_x_final = (
                            self.criterion(
                                x[:, :num_particles_orig][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                                gt_x[:, step][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                            )
                            * self.loss_factor_x
                        )
                        loss_x_improvement = (
                            loss_x_baseline.sqrt() - loss_x_final.sqrt()
                        )
                        self.losses_log["loss_x_improvement"] += (
                            loss_x_improvement.item()
                        )

                if cfg.sim.n_history > 0:
                    x_his_new = torch.cat(
                        [
                            x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                            x[:, :, None].detach(),
                        ],
                        dim=2,
                    )
                    v_his_new = torch.cat(
                        [
                            v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                            v[:, :, None].detach(),
                        ],
                        dim=2,
                    )
                    x_his = x_his_new.reshape(batch_size, num_particles, -1)
                    v_his = v_his_new.reshape(batch_size, num_particles, -1)

                if self.loss_factor_x > 0:
                    # Mask both prediction and target to keep shape [N, 3]
                    points_scaling_factor = 1
                    pred_residual = residual_x
                    target_residual = (
                        gt_x[:, step, :num_particles_orig, :3]
                        - x_sim[:, :num_particles_orig, :3]
                    )
                    mask = enabled_mask[:, :num_particles_orig, :3] > 0
                    pred_residual_masked = pred_residual[:, :num_particles_orig, :3][
                        mask
                    ].view(-1, 3)
                    target_residual_masked = target_residual[:, :num_particles_orig][
                        mask
                    ].view(-1, 3)
                    loss_x = self.criterion(
                        pred_residual_masked * points_scaling_factor,
                        target_residual_masked * points_scaling_factor,
                    )
                    losses["loss_x"] += loss_x
                    self.losses_log["loss_x"] += loss_x.item()

                with torch.no_grad():
                    # loss_x_trivial: Baseline error if we just use previous position + velocity * dt (Euler step) to predict next position. Useful to check if the model is learning anything beyond trivial integration.
                    if self.loss_factor_x > 0:
                        loss_x_trivial = (
                            self.criterion(
                                (
                                    x_in_gt[:, :num_particles_orig]
                                    + v_in_gt[:, :num_particles_orig]
                                    * cfg.sim.dt
                                    * cfg.sim.interval
                                )[enabled_mask[:, :num_particles_orig, :3] > 0],
                                gt_x[:, step][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                            )
                            * self.loss_factor_x
                        )
                        self.losses_log["loss_x_trivial"] += loss_x_trivial.item()

                    # loss_v_trivial: Baseline error for velocity (not position), using previous velocity as prediction. Similar use as above for velocity.
                    if self.loss_factor_v > 0:
                        loss_v_trivial = (
                            self.criterion(
                                v_in_gt[:, :num_particles_orig][enabled_mask > 0],
                                gt_v[:, step][enabled_mask > 0],
                            )
                            * self.loss_factor_v
                        )
                        self.losses_log["loss_v_trivial"] += loss_v_trivial.item()

                    # loss_x_sanity: Checks if integrating backward (x - v*dt) returns to previous position. Useful for catching integration bugs or unexpected state changes (e.g., clipping).
                    loss_x_sanity = (
                        self.criterion(
                            x_in[:, :num_particles_orig][
                                enabled_mask[:, :num_particles_orig, :3] > 0
                            ],
                            (
                                x[:, :num_particles_orig]
                                - v[:, :num_particles_orig]
                                * cfg.sim.dt
                                * cfg.sim.interval
                            )[enabled_mask[:, :num_particles_orig, :3] > 0],
                        )
                        * self.loss_factor_x
                    )
                    self.losses_log["loss_x_sanity"] += (
                        loss_x_sanity.item()
                    )  # if > 0 then clipping issue

                    # loss_x_gt_sanity: Checks if ground-truth velocity can accurately predict next ground-truth position (Euler step). Useful for diagnosing inconsistencies or noise in ground-truth data.
                    if step > 0:
                        loss_x_gt_sanity = (
                            self.criterion(
                                (
                                    gt_x[:, step - 1]
                                    + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval
                                )[enabled_mask[:, :num_particles_orig, :3] > 0],
                                gt_x[:, step][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                            )
                            * self.loss_factor_x
                        )
                        self.losses_log["loss_x_gt_sanity"] += loss_x_gt_sanity.item()
                    else:
                        loss_x_gt_sanity = (
                            self.criterion(
                                (
                                    x_in[:, :num_particles_orig]
                                    + gt_v[:, step] * cfg.sim.dt * cfg.sim.interval
                                )[enabled_mask[:, :num_particles_orig, :3] > 0],
                                gt_x[:, step][
                                    enabled_mask[:, :num_particles_orig, :3] > 0
                                ],
                            )
                            * self.loss_factor_x
                        )
                        self.losses_log["loss_x_gt_sanity"] += loss_x_gt_sanity.item()

                if save and not cfg.debug:
                    self.logger.add_scalar(
                        "stat/iteration", iteration, step=self.total_step_count
                    )
                    if self.loss_factor_x > 0:
                        self.logger.add_scalar(
                            "main/loss_x", loss_x.item(), step=self.total_step_count
                        )

                    if self.loss_factor_x > 0:
                        loss_dist = torch.norm(
                            pred_residual_masked - target_residual_masked, dim=-1
                        ).mean()
                        self.logger.add_scalar(
                            "main/dist_x",
                            loss_dist.item(),
                            step=self.total_step_count,
                        )
                        if step == 0:
                            with torch.no_grad():
                                gt_pos = gt_x[:, step, :num_particles_orig, :3]
                                sim_pos = x_sim[:, :num_particles_orig, :3]
                                mask = enabled_mask[:, :num_particles_orig, :3] > 0
                                dist_gt_sim = torch.norm(
                                    gt_pos[mask].view(-1, 3)
                                    - sim_pos[mask].view(-1, 3),
                                    dim=-1,
                                ).mean()
                                self.logger.add_scalar(
                                    "main/dist_gt_sim_x",
                                    dist_gt_sim.item(),
                                    step=self.total_step_count,
                                )
                self.total_step_count += 1

            loss = sum(losses.values())
            try:
                loss.backward()
            except Exception as e:
                print(traceback.format_exc())
                print(f"loss.backward() failed: {e.with_traceback()}")
                continue

            if self.residualnet_requires_grad:
                residualnet_grad_norm = clip_grad_norm_(
                    self.residualnet.parameters(),
                    max_norm=cfg.train.residualnet_grad_max_norm,
                    error_if_nonfinite=True,
                )
                self.residualnet_optimizer.step()

            if (iteration + 1) % cfg.train.iteration_log_interval == 0:
                msgs = [
                    cfg.train.name,
                    time.strftime("%H:%M:%S"),
                    "iteration {:{width}d}/{}".format(
                        iteration + 1,
                        cfg.train.num_iterations,
                        width=len(str(cfg.train.num_iterations)),
                    ),
                ]

                if self.residualnet_requires_grad:
                    residualnet_lr = self.residualnet_optimizer.param_groups[0]["lr"]
                    msgs.extend(
                        [
                            "e-lr {:.2e}".format(residualnet_lr),
                            "e-|grad| {:.4f}".format(residualnet_grad_norm),
                        ]
                    )

                elapsed_time_minutes = (time.time() - self.start_time) / 60
                msgs.append(f"time {elapsed_time_minutes:.2f}m")
                if save and not cfg.debug:
                    self.logger.add_scalar(
                        "stat/time_minutes",
                        elapsed_time_minutes,
                        step=self.total_step_count,
                    )

                for loss_k, loss_v in self.losses_log.items():
                    msgs.append(
                        "{} {:.8f}".format(
                            loss_k, loss_v / cfg.train.iteration_log_interval
                        )
                    )
                    if save and not cfg.debug:
                        self.logger.add_scalar(
                            "stat/mean_{}".format(loss_k),
                            loss_v / cfg.train.iteration_log_interval,
                            step=self.total_step_count,
                        )

                msg = ",".join(msgs)
                print("[{}]".format(msg))
                self.losses_log = defaultdict(int)

            if self.residualnet_requires_grad:
                residualnet_lr = self.residualnet_optimizer.param_groups[0]["lr"]
                if save and not cfg.debug:
                    self.logger.add_scalar(
                        "stat/residualnet_lr",
                        residualnet_lr,
                        step=self.total_step_count,
                    )
                    self.logger.add_scalar(
                        "stat/residualnet_grad_norm",
                        residualnet_grad_norm,
                        step=self.total_step_count,
                    )

            if save and (iteration + 1) % cfg.train.iteration_save_interval == 0:
                state = {
                    "residualnet": self.residualnet.state_dict(),
                    "transformer": self.temporal_model.export_component_state_dicts(),
                }
                torch.save(state, self.ckpt_root / "{:06d}.pt".format(iteration + 1))

            if self.residualnet_requires_grad:
                # Clamp LR at the cosine minimum after T_max iterations
                if (iteration + 1) <= self.residualnet_lr_T_max:
                    self.residualnet_lr_scheduler.step()


@hydra.main(version_base="1.2", config_path=str(root / "cfg"), config_name="default")
def main(cfg: DictConfig):
    object_name = apply_object_sim_params(cfg)
    print(f"Using simulator parameters for object '{object_name}'")
    trainer = Trainer(cfg)
    trainer.load_train_dataset()
    trainer.init_train()

    for iteration in range(
        cfg.train.resume_iteration,
        cfg.train.num_iterations,
        cfg.train.iteration_eval_interval,
    ):
        start_iteration = iteration
        end_iteration = min(
            iteration + cfg.train.iteration_eval_interval, cfg.train.num_iterations
        )
        trainer.train(start_iteration, end_iteration)
        eval_parallel(
            trainer,
            trainer.total_step_count,
            save=True,
            num_workers=cfg.train.get("num_eval_workers", getattr(cfg.train, "eval_num_workers", 1)),
        )


if __name__ == "__main__":
    main()
