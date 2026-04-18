from pathlib import Path
import random
import time
import os
import json
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image
import warp as wp
from warp import build
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import kornia
import traceback
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Suppress the timm deprecation warning that appears in child processes
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
    module="timm.models.layers",
)


# -----------------------------------------------------------------------------
# Pool helper: make workers non-daemon by resetting the flag in an initializer.
# This avoids the “daemonic processes are not allowed to have children” error
# while keeping the standard `multiprocessing.Pool` implementation unchanged.
# -----------------------------------------------------------------------------


def _pool_worker_init():
    """Executed in every Pool worker – clear the daemon flag."""
    mp.current_process().daemon = False


# -------------------------------------------------------------
# Helper function for running rendering tasks in separate child
# processes.  Defined at module level so it can be pickled.
# -------------------------------------------------------------


def _execute_task(func, args, kwargs):  # noqa: D401, non-camel-case intended for pickling
    """Wrapper used by multiprocessing to execute a callable.

    Any exception stack-trace is printed so that it is not swallowed by
    the process boundary.
    """
    try:
        func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover – debugging helper
        import traceback, sys

        print(f"[child-proc] task {func.__name__} failed: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()


import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_eval import (
    Trainer,
    apply_object_sim_params,
    dataloader_wrapper,
    root,
    # transform_gripper_points,
)
from train.metric_eval import do_metric
from meta_material.utils import mkdir
from meta_material.data import RealTeleopBatchDataset, RealGripperDataset
from meta_material.sim import create_simulator
from experiments.train.temporal_transformer import TemporalResidualTransformer
import torch.multiprocessing as mp
from train.pv_train import do_train_pv
from train.pv_dataset import do_dataset_pv
from train.pv_combined import do_combined_pv
import meta_material


def eval_episode(
    cfg: DictConfig,
    residualnet_state_dict: dict,
    transformer_state_dict: dict | None,
    iteration: int,
    episode: int,
    save: bool = True,
    gpu_id: int = 0,
):
    wp.init()
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        wp_device = wp.get_device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        wp_device = wp.get_device("cpu")

    log_root: Path = root / "log"

    # Re-calculate use_pv and use_gs as in the Trainer
    if str(root).startswith("/burg") or str(root).startswith("/local"):
        use_pv = False
    else:
        use_pv = not cfg.train.disable_pv

    visualize_residuals = cfg.train.get("visualize_residuals", False)
    if visualize_residuals:
        use_pv = True

    # Re-calculate use_gs as in the Trainer
    use_gs = False
    source_dataset_path = log_root / str(cfg.train.source_dataset_name)
    if os.path.exists(source_dataset_path / f'episode_{episode:04d}' / 'meta.txt'):
        meta = np.loadtxt(source_dataset_path / f'episode_{episode:04d}' / 'meta.txt')
        metadata_file = source_dataset_path / 'metadata.json'
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                datadir_list = json.load(f)
            datadir = datadir_list[episode]
            source_data_dir = datadir['path']
            source_episode_id = int(meta[0])
            source_frame_start = int(meta[1]) + int(cfg.sim.n_history) * int(cfg.train.dataset_load_skip_frame) * int(cfg.train.dataset_skip_frame)
            use_gs = os.path.exists((log_root.parent.parent / source_data_dir).parent / f'episode_{source_episode_id:04d}' / 'gs' / f'{source_frame_start:06d}.splat')
        else:
            # Fallback for non-merged datasets
            use_gs = os.path.exists(source_dataset_path.parent / "episode_0000" / "gs")
    else:
        # Fallback for datasets without meta.txt
        use_gs = os.path.exists((log_root / str(cfg.train.source_dataset_name)).parent / "episode_0000" / "gs")
    use_gs = True
    eval_name = f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}"
    eval_name_sim_only = f"{cfg.train.name}/eval-val-sim-only/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}"
    exp_root: Path = log_root / eval_name
    exp_root_sim_only: Path = log_root / eval_name_sim_only
    if save:
        state_root: Path = exp_root / "state"
        episode_state_root = state_root / f"episode_{episode:04d}"
        state_root_sim_only: Path = exp_root_sim_only / "state"
        episode_state_root_sim_only = state_root_sim_only / f"episode_{episode:04d}"
        # Use the safe, atomic mkdir to avoid race conditions in multiprocessing
        episode_state_root.mkdir(parents=True, exist_ok=True)
        episode_state_root_sim_only.mkdir(parents=True, exist_ok=True)

    if cfg.train.dataset_name is None:
        cfg.train.dataset_name = Path(cfg.train.name).parent / "dataset"
    assert cfg.train.source_dataset_name is not None

    source_dataset_root = log_root / str(cfg.train.source_dataset_name)
    if not os.path.exists(source_dataset_root):
        source_dataset_root = (
            Path("./data/meta-material") / cfg.train.source_dataset_name
        )
        assert os.path.exists(source_dataset_root)

    eval_dataset = RealTeleopBatchDataset(
        cfg,
        dataset_root=log_root / cfg.train.dataset_name / "state",
        source_data_root=source_dataset_root,
        device=device,
        num_steps=cfg.sim.num_steps,
        eval_episode_name=f"episode_{episode:04d}",
    )
    eval_dataloader = dataloader_wrapper(
        DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Must be 0 when running in a subprocess
            pin_memory=True,
        ),
        "dataset",
    )
    if cfg.sim.gripper_points:
        eval_gripper_dataset = RealGripperDataset(
            cfg,
            device=device,
        )
        eval_gripper_dataloader = dataloader_wrapper(
            DataLoader(
                eval_gripper_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,  # Must be 0 when running in a subprocess
                pin_memory=True,
            ),
            "gripper_dataset",
        )
    init_data, actions, gt_states, downsample_indices = next(eval_dataloader)

    # Unpack optional interior/fill points as 8th field
    if len(init_data) >= 8 and int(cfg.sim.num_fill_points) > 0:
        x, v, x_his, v_his, clip_bound, enabled, episode_vec, fill_points = init_data
    else:
        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_data
        fill_points = None
    x = x.to(device)
    v = v.to(device)
    x_his = x_his.to(device)
    v_his = v_his.to(device)
    if fill_points is not None:
        fill_points = fill_points.to(device)

    cylinders, grippers = actions
    cylinders = cylinders.to(device)
    grippers = grippers.to(device)

    if cfg.sim.gripper_points:
        gripper_points, _ = next(eval_gripper_dataloader)
        gripper_points = gripper_points.to(device)
        gripper_x, gripper_v, gripper_mask = transform_gripper_points(
            cfg, gripper_points, grippers
        )  # (bsz, num_steps, num_grippers, 3)

    gt_x, gt_v = gt_states
    gt_x = gt_x.to(device)
    gt_v = gt_v.to(device)

    batch_size = gt_x.shape[0]
    num_steps_total = gt_x.shape[1]
    num_particles = gt_x.shape[2]
    assert batch_size == 1

    if cfg.sim.gripper_points:
        num_gripper_particles = gripper_x.shape[2]
        num_particles_orig = num_particles
        num_particles = num_particles + num_gripper_particles
    else:
        num_particles_orig = num_particles

    if cfg.sim.gripper_points:
        assert not cfg.sim.gripper_forcing
        num_grippers = 0
    else:
        num_grippers = cfg.sim.num_grippers

    enabled = enabled.to(device)  # (bsz, num_particles)
    # Append interior fill points to x and zero-velocity to v prior to simulation
    if fill_points is not None and int(cfg.sim.num_fill_points) > 0:
        target_fill = int(cfg.sim.num_fill_points)
        # Ensure (B, K, 3)
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
        # Extend enabled mask for new internal points (all enabled)
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

    # (enabled mask over all points will be recomputed where needed later)
    points_per_env_to_use = (
        int(x.shape[1]) if appended_fill_points else int(cfg.sim.n_particles)
    )

    dataset_context = " ".join(
        (
            str(cfg.train.dataset_name).lower(),
            str(cfg.train.source_dataset_name).lower(),
            str(cfg.train.get("object_name", "")).lower(),
        )
    )
    use_flag_dataset_behavior = "flag" in dataset_context
    use_teddy_dataset_behavior = "teddy" in dataset_context

    # Keep dataset-specific eval radii explicit while preserving default behavior.
    eval_gripper_radius = float(getattr(cfg.model, "gripper_radius", 0.04))
    for dataset_key, radius in (("teddy", 0.015), ("flag", 0.01), ("cleaner", 0.1)):
        if dataset_key in dataset_context:
            eval_gripper_radius = radius
            break

    sim = create_simulator(
        backend=getattr(cfg.sim, "backend", "spring"),
        x=x.detach().clone().cpu().numpy(),
        v=v.detach().clone().cpu().numpy(),
        grippers=grippers.detach().clone().cpu().numpy(),
        points_per_env=points_per_env_to_use,
        batch_size=1,
        threshold=cfg.sim.threshold,
        stiffness=cfg.sim.stiffness,
        damping=cfg.sim.damping,
        mass_per_point=cfg.sim.mass_per_point,
        sim_dt=cfg.sim.dt,
        sim_substeps=cfg.sim.sim_substeps,
        device=wp_device,
        visualize=cfg.visualize,
        use_flag_dataset_behavior=use_flag_dataset_behavior,
        is_teddy_dataset=use_teddy_dataset_behavior,
        poke_stiffness=float(getattr(cfg.sim, "poke_stiffness", 100.0)),
        gripper_radius=eval_gripper_radius,
        max_springs_per_node=cfg.sim.max_springs_per_node,
        ground_friction=cfg.sim.ground_friction,
    )

    x_full, v_full = sim.get_initial_state()
    x = x_full.to(device)
    v = v_full.to(device)

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

    if visualize_residuals:
        x_no_res, v_no_res = x.clone(), v.clone()
        if cfg.sim.n_history > 0:
            x_his_no_res, v_his_no_res = x_his.clone(), v_his.clone()

    enabled_padding_shape = (
        enabled.shape[0],
        num_particles - num_particles_orig,
    )
    enabled_padding = torch.ones(
        enabled_padding_shape, device=enabled.device, dtype=enabled.dtype
    )
    enabled_full = torch.cat([enabled, enabled_padding], dim=1)

    n_history = cfg.sim.n_history if hasattr(cfg.sim, "n_history") else 0
    residualnet: nn.Module = getattr(meta_material.material, cfg.model.residualnet.cls)(
        cfg.model.residualnet, n_history
    )
    residualnet.set_params(
        cfg.sim.num_grids, num_grids_flexible=cfg.sim.num_grids_flexible
    )
    residualnet.load_state_dict(residualnet_state_dict)
    residualnet.to(device)
    residualnet.eval()

    temporal_model = TemporalResidualTransformer(
        cfg,
        device=device,
        window_override=5,
    )
    temporal_model.load_component_state_dicts(transformer_state_dict)
    temporal_model.eval()

    ckpt = dict(x=x[0], v=v[0])

    try:
        print("grippers: ", grippers.shape)
        if grippers is not None and grippers.shape[2] > 0:
            print("inside grippers if condition")
            ckpt["grippers"] = grippers[0, 0]
    except Exception:
        pass

    if save:
        torch.save(ckpt, episode_state_root / f"{0:04d}.pt")
        if visualize_residuals:
            ckpt_sim_only = dict(x=x_no_res[0], v=v_no_res[0])
            torch.save(ckpt_sim_only, episode_state_root_sim_only / f"{0:04d}.pt")

    losses = {}
    # --------------------------------------------------------------
    #  Pre-compute mask of particles that are inside a closed gripper
    #  at the initial timestep. These particles are considered rigidly
    #  attached to the gripper for the entire rollout, therefore no
    #  residual correction will be applied to them.
    # --------------------------------------------------------------
    held_mask: torch.Tensor | None = None
    if (
        grippers is not None
        and grippers.shape[2] > 0
        and cfg.sim.gripper_forcing
        and not cfg.sim.gripper_points
    ):
        g_xyz0 = grippers[:, 0, :, :3].to(x.device)  # (B, G, 3)
        g_closed0 = (grippers[:, 0, :, -1] < 0.5).to(x.device)  # (B, G)
        dists0 = torch.norm(x.unsqueeze(2) - g_xyz0.unsqueeze(1), dim=-1)  # (B, N, G)
        is_held0 = (dists0 < sim.gripper_radius) & g_closed0.unsqueeze(1)
        held_mask = is_held0.any(dim=-1)  # (B, N)
        if (
            use_flag_dataset_behavior
            and hasattr(sim, "_flag_y_thresholds")
        ):
            y_thresh_env = wp.to_torch(sim._flag_y_thresholds).to(x.device).view(-1, 1)
            y_thresh = y_thresh_env - 1.0e-6
            top_band = x[..., 1] > y_thresh
            any_closed = g_closed0.any(dim=1, keepdim=True)
            held_mask = held_mask | (top_band & any_closed)
        if use_teddy_dataset_behavior:
            held_mask = None

    with torch.no_grad():
        temporal_model.reset_window()
        for step in trange(num_steps_total, desc=f"Eval ep {episode} on GPU {gpu_id}"):
            if cfg.sim.gripper_points:
                x_gripper_step = gripper_x[:, step]
                v_gripper_step = gripper_v[:, step]
                x = torch.cat([x, x_gripper_step], dim=1)
                v = torch.cat([v, v_gripper_step], dim=1)

                x_his_gripper_padding = torch.zeros(
                    (gripper_x.shape[0], gripper_x.shape[2], n_history * 3),
                    device=x_his.device,
                    dtype=x_his.dtype,
                )
                v_his_gripper_padding = torch.zeros(
                    (gripper_x.shape[0], gripper_x.shape[2], n_history * 3),
                    device=v_his.device,
                    dtype=v_his.dtype,
                )
                x_his = torch.cat([x_his, x_his_gripper_padding], dim=1)
                v_his = torch.cat([v_his, v_his_gripper_padding], dim=1)

                if visualize_residuals:
                    x_no_res = torch.cat([x_no_res, x_gripper_step], dim=1)
                    v_no_res = torch.cat([v_no_res, v_gripper_step], dim=1)
                    if n_history > 0:
                        x_his_no_res = torch.cat(
                            [x_his_no_res, x_his_gripper_padding], dim=1
                        )
                        v_his_no_res = torch.cat(
                            [v_his_no_res, v_his_gripper_padding], dim=1
                        )

                if enabled.shape[1] < num_particles:
                    enabled = torch.cat([enabled, gripper_mask[:, step]], dim=1)

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

            # Sim+Residual Branch
            x_sim, v_sim = sim(
                step,
                x.detach().clone(),
                v.detach().clone(),
                None,
                gripper_data_to_pass,
            )

            points_feats = residualnet(
                x,
                v,
                x_his,
                v_his,
                enabled_full,
                x_sim,
                v_sim,
            )

            if points_feats.isnan().any() or points_feats.isinf().any():
                print("points_feats has nan/inf")
                break

            residual_v = temporal_model(points_feats)

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
                dists_step = torch.norm(
                    x.unsqueeze(2) - g_xyz_step.unsqueeze(1), dim=-1
                )
                step_mask = ((dists_step < sim.gripper_radius) & g_closed_step.unsqueeze(1)).any(
                    dim=-1
                )
                y_thresh_env = wp.to_torch(sim._flag_y_thresholds).to(x.device).view(-1, 1)
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

            # v = v_sim
            # x = x_sim

            x_y_clamped = x[..., 1].clamp(min=sim.ground_y)
            x = torch.stack([x[..., 0], x_y_clamped, x[..., 2]], dim=-1)

            # Sim-Only Branch
            ckpt_extra = {}
            if visualize_residuals:
                x_sim_no_res, v_sim_no_res = sim(
                    step,
                    x_no_res.detach().clone(),
                    v_no_res.detach().clone(),
                    None,
                    gripper_data_to_pass,
                )
                x_no_res = x_sim_no_res
                v_no_res = v_sim_no_res
                ckpt_extra["x_no_res"] = x_no_res[:, :num_particles_orig][0].clone()

            if cfg.sim.n_history > 0:
                x_his_new = torch.cat(
                    [
                        x_his.reshape(batch_size, x.shape[1], -1, 3)[:, :, 1:],
                        x[:, :, None].detach(),
                    ],
                    dim=2,
                )
                v_his_new = torch.cat(
                    [
                        v_his.reshape(batch_size, v.shape[1], -1, 3)[:, :, 1:],
                        v[:, :, None].detach(),
                    ],
                    dim=2,
                )
                x_his = x_his_new.reshape(batch_size, x.shape[1], -1)
                v_his = v_his_new.reshape(batch_size, v.shape[1], -1)

                if visualize_residuals:
                    x_his_no_res_new = torch.cat(
                        [
                            x_his_no_res.reshape(batch_size, x_no_res.shape[1], -1, 3)[
                                :, :, 1:
                            ],
                            x_no_res[:, :, None].detach(),
                        ],
                        dim=2,
                    )
                    v_his_no_res_new = torch.cat(
                        [
                            v_his_no_res.reshape(batch_size, v_no_res.shape[1], -1, 3)[
                                :, :, 1:
                            ],
                            v_no_res[:, :, None].detach(),
                        ],
                        dim=2,
                    )
                    x_his_no_res = x_his_no_res_new.reshape(
                        batch_size, x_no_res.shape[1], -1
                    )
                    v_his_no_res = v_his_no_res_new.reshape(
                        batch_size, v_no_res.shape[1], -1
                    )

            if cfg.sim.gripper_points:
                extra_save = {
                    "gripper_x": gripper_x[0, step],
                    "gripper_v": gripper_v[0, step],
                    "grippers": grippers[0, step],
                }
                x = x[:, :num_particles_orig]
                v = v[:, :num_particles_orig]
                x_his = x_his[:, :num_particles_orig]
                v_his = v_his[:, :num_particles_orig]
                enabled = enabled[:, :num_particles_orig]
                if visualize_residuals:
                    x_no_res = x_no_res[:, :num_particles_orig]
                    v_no_res = v_no_res[:, :num_particles_orig]
                    x_his_no_res = x_his_no_res[:, :num_particles_orig]
                    v_his_no_res = v_his_no_res[:, :num_particles_orig]
            else:
                extra_save = {}
                if grippers is not None and grippers.shape[2] > 0:
                    extra_save["grippers"] = grippers[0, step]

            # Recompute enabled mask restricted to original surface points for loss
            enabled_mask_orig = (
                enabled[:, :num_particles_orig].unsqueeze(-1).repeat(1, 1, 3)
            )
            loss_x = nn.functional.mse_loss(
                x[:, :num_particles_orig][enabled_mask_orig > 0],
                gt_x[:, step][enabled_mask_orig > 0],
            )
            loss_v = nn.functional.mse_loss(
                v[:, :num_particles_orig][enabled_mask_orig > 0],
                gt_v[:, step][enabled_mask_orig > 0],
            )
            losses[step] = dict(loss_x=loss_x.item(), loss_v=loss_v.item())

            ckpt = dict(x=x[0], v=v[0], **extra_save, **ckpt_extra)

            if save and step % cfg.sim.skip_frame == 0:
                torch.save(
                    ckpt,
                    episode_state_root / f"{int(step / cfg.sim.skip_frame):04d}.pt",
                )
                if visualize_residuals:
                    ckpt_no_res = dict(
                        x=x_no_res[0].clone(),
                        v=v_no_res[0].clone(),
                        **extra_save,
                    )
                    torch.save(
                        ckpt_no_res,
                        episode_state_root_sim_only
                        / f"{int(step / cfg.sim.skip_frame):04d}.pt",
                    )

    if hasattr(sim, "save_renderer"):
        sim.save_renderer()

    metrics = None
    if save:
        if not losses:
            print(
                f"[eval_episode] No rollout losses recorded for episode {episode:04d}; "
                "skipping plots/metrics for this episode."
            )
            return None

        first_step = next(iter(losses))
        for loss_k in losses[first_step].keys():
            plt.figure(figsize=(10, 5))
            loss_list = [losses[step][loss_k] for step in losses]
            plt.plot(loss_list)
            plt.title(loss_k)
            plt.grid()
            plt.savefig(state_root / f"episode_{episode:04d}_{loss_k}.png", dpi=300)
            plt.close()

        # ------------------------------------------------------------------
        # Launch rendering/video-generation tasks in parallel.
        # ------------------------------------------------------------------
        parallel_tasks = []

        if use_pv:
            from train.pv_dataset import do_dataset_pv

            parallel_tasks.append(
                (
                    do_train_pv,
                    (
                        cfg,
                        log_root,
                        iteration,
                        [f"episode_{episode:04d}"],
                    ),
                    dict(
                        eval_dirname="eval-val",
                        dataset_name=cfg.train.dataset_name.split("/")[-1],
                        eval_postfix="",
                    ),
                )
            )

            parallel_tasks.append(
                (
                    do_dataset_pv,
                    (
                        cfg,
                        log_root / str(cfg.train.dataset_name),
                        [f"episode_{episode:04d}"],
                    ),
                    dict(
                        save_dir=log_root
                        / f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}/pv",
                        downsample_indices=downsample_indices,
                    ),
                )
            )

            if not visualize_residuals:
                parallel_tasks.append(
                    (
                        do_combined_pv,
                        (cfg,),
                        dict(
                            sim_state_root=exp_root / "state",
                            gt_state_root=log_root
                            / str(cfg.train.dataset_name)
                            / "state",
                            episode_names=[f"episode_{episode:04d}"],
                            save_dir=log_root
                            / f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}/pv_combined",
                            downsample_indices=downsample_indices,
                            eval_postfix="_combined",
                        ),
                    )
                )

        if visualize_residuals:
            from train.pv_residual import (
                do_residual_pv_single,
                do_residual_pv_combined,
            )

            parallel_tasks.extend(
                [
                    (
                        do_residual_pv_single,
                        (cfg,),
                        dict(
                            state_root=exp_root / "state",
                            episode_names=[f"episode_{episode:04d}"],
                            save_dir=log_root
                            / f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}/pv_residual",
                            state_key="x_no_res",
                            color="red",
                            eval_postfix="_sim_only",
                        ),
                    ),
                    (
                        do_residual_pv_single,
                        (cfg,),
                        dict(
                            state_root=exp_root / "state",
                            episode_names=[f"episode_{episode:04d}"],
                            save_dir=log_root
                            / f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}/pv_residual",
                            state_key="x",
                            color="purple",
                            eval_postfix="_sim_plus_residual",
                        ),
                    ),
                    (
                        do_residual_pv_combined,
                        (cfg,),
                        dict(
                            sim_state_root=exp_root / "state",
                            gt_state_root=log_root
                            / str(cfg.train.dataset_name)
                            / "state",
                            episode_names=[f"episode_{episode:04d}"],
                            save_dir=log_root
                            / f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}/pv_residual_combined",
                            downsample_indices=downsample_indices,
                            eval_postfix="_combined",
                        ),
                    ),
                ]
            )

        if use_gs:
            from gs import do_gs

            parallel_tasks.append(
                (
                    do_gs,
                    (
                        cfg,
                        log_root,
                        iteration,
                        [f"episode_{episode:04d}"],
                    ),
                    dict(
                        eval_dirname="eval-val",
                        dataset_name=cfg.train.dataset_name.split("/")[-1],
                        eval_postfix="",
                        camera_id=1,
                        with_mask=True,
                        with_bg=False,
                    ),
                )
            )
            # Also render GS for the sim-only branch so metrics can read pv_gs/mask there.
            if visualize_residuals:
                parallel_tasks.append(
                    (
                        do_gs,
                        (
                            cfg,
                            log_root,
                            iteration,
                            [f"episode_{episode:04d}"],
                        ),
                        dict(
                            eval_dirname="eval-val-sim-only",
                            dataset_name=cfg.train.dataset_name.split("/")[-1],
                            eval_postfix="",
                            camera_id=1,
                            with_mask=True,
                            with_bg=False,
                        ),
                    )
                )

        # Execute tasks concurrently using threads to avoid nested process issues.
        if parallel_tasks:
            # Tasks that rely on Xvfb / pyvista (Plotter) are NOT thread-safe.
            # Running them in parallel leads to `pyvista` partially initialising,
            # DISPLAY becoming invalid, or X11 errors (as observed).  We therefore
            # execute those tasks sequentially and only parallelise the safe ones
            # (currently `do_gs`).

            xvfb_funcs = {
                "do_train_pv",
                "do_dataset_pv",
                "do_combined_pv",
                "do_residual_pv_single",
                "do_residual_pv_combined",
            }

            sequential_tasks = []
            concurrent_tasks = []

            for func, f_args, f_kwargs in parallel_tasks:
                if func.__name__ in xvfb_funcs:
                    sequential_tasks.append((func, f_args, f_kwargs))
                else:
                    concurrent_tasks.append((func, f_args, f_kwargs))

            # First run the heavy Xvfb/pyvista tasks one-by-one to avoid
            # conflicts with the global DISPLAY environment variable.
            # Launch all Xvfb/PyVista tasks concurrently (one child process each)
            ctx = mp.get_context("spawn")
            procs = []
            for func, f_args, f_kwargs in sequential_tasks:
                p = ctx.Process(target=_execute_task, args=(func, f_args, f_kwargs))
                p.start()
                procs.append((func.__name__, p))

            seq_err = False
            for name, p in procs:
                p.join()
                if p.exitcode != 0:
                    print(
                        f"[eval_episode] Xvfb task {name} exited with code {p.exitcode}"
                    )
                    seq_err = True

            if seq_err:
                print(
                    "[eval_episode] One or more Xvfb tasks failed – check logs above."
                )

            # Now run the remaining safe tasks in parallel (if any).
            if concurrent_tasks:
                with ThreadPoolExecutor(
                    max_workers=min(len(concurrent_tasks), 4)
                ) as _executor:
                    futures = [
                        _executor.submit(fn, *f_args, **f_kwargs)
                        for fn, f_args, f_kwargs in concurrent_tasks
                    ]
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as exc:
                            print(f"[eval_episode] parallel task failed: {exc}")

        # After all parallel rendering tasks are done, proceed to compute metrics.

        metrics_residual = do_metric(
            cfg,
            log_root,
            iteration,
            [f"episode_{episode:04d}"],
            downsample_indices,
            eval_dirname="eval-val",
            dataset_name=cfg.train.dataset_name.split("/")[-1],
            eval_postfix="",
            camera_id=1,
            use_gs=use_gs,
        )

        metrics_sim_only = None
        if visualize_residuals:
            metrics_sim_only = do_metric(
                cfg,
                log_root,
                iteration,
                [f"episode_{episode:04d}"],
                downsample_indices,
                eval_dirname="eval-val-sim-only",
                dataset_name=cfg.train.dataset_name.split("/")[-1],
                eval_postfix="",
                camera_id=1,
                use_gs=use_gs,
            )

        return (metrics_residual, metrics_sim_only)


def eval_parallel(trainer, eval_iteration, save=True, num_workers: int = 5):
    cfg = trainer.cfg

    start_episode = cfg.train.eval_start_episode
    end_episode = (
        cfg.train.eval_end_episode if save else cfg.train.eval_start_episode + 2
    )
    episodes_to_run = [
        e
        for e in range(start_episode, end_episode)
        if e not in cfg.train.get("eval_skip_episode", [])
    ]

    log_root: Path = root / "log"
    eval_name = f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{eval_iteration:06d}"
    exp_root: Path = log_root / eval_name

    if save:
        # Create the shared output directories once in the main process to avoid race conditions.
        state_root: Path = exp_root / "state"
        mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
        OmegaConf.save(cfg, exp_root / "hydra.yaml", resolve=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices found. Running evaluation sequentially on CPU.")
        num_workers = 5

    # Simple parallelism: divide episodes among GPUs
    # For more advanced parallelism, you could use a queue
    residualnet_state_dict = trainer.residualnet.state_dict()
    transformer_state_dict = trainer.temporal_model.export_component_state_dicts()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_pool_worker_init) as pool:
        args = [
            (
                trainer.cfg,
                residualnet_state_dict,
                transformer_state_dict,
                eval_iteration,
                episode,
                save,
                i % num_gpus if num_gpus > 0 else -1,
            )
            for i, episode in enumerate(episodes_to_run)
        ]
        metrics_list = pool.starmap(eval_episode, args)

    # ---------------- Aggregate metrics lists (residual vs sim-only) ----------------
    metrics_list_residual = []
    metrics_list_sim_only = []
    for m in metrics_list:
        if m is None:
            continue
        if isinstance(m, tuple):
            res, sim_only = m
            if res is not None:
                metrics_list_residual.append(res)
            if sim_only is not None:
                metrics_list_sim_only.append(sim_only)
        else:
            metrics_list_residual.append(m)

    metrics_list = metrics_list_residual

    if not save:
        return

    # Aggregate and save metrics
    log_root: Path = root / "log"
    eval_dirname = "eval-val"
    dataset_name = cfg.train.dataset_name.split("/")[-1]
    save_dir = (
        log_root
        / cfg.train.name
        / eval_dirname
        / dataset_name
        / f"{eval_iteration:06d}"
        / "metric"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        print("No metrics were generated. Skipping metric aggregation.")
        return
    metrics_list = np.array(metrics_list)[:, 0]

    if trainer.use_gs:
        metric_names = [
            "mse",
            "avg_d",
            "chamfer",
            "emd",
            "jscore",
            "fscore",
            "jfscore",
            "perception",
            "psnr",
            "ssim",
            "iou",
        ]
    else:
        metric_names = ["mse", "avg_d", "chamfer", "emd"]

    median_metric = np.median(metrics_list, axis=0)
    step_75_metric = np.percentile(metrics_list, 75, axis=0)
    step_25_metric = np.percentile(metrics_list, 25, axis=0)
    mean_metric = np.mean(metrics_list, axis=0)
    std_metric = np.std(metrics_list, axis=0)

    # for i, metric_name in enumerate(metric_names):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(mean_metric[:, i], label="mean")
    #     plt.plot(median_metric[:, i], label="median")
    #     plt.fill_between(
    #         range(len(mean_metric[:, i])),
    #         step_25_metric[:, i],
    #         step_75_metric[:, i],
    #         alpha=0.2,
    #     )
    #     plt.title(metric_name)
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(save_dir / f"{i:02d}-{metric_name}.png", dpi=300)
    #     plt.close()

    total_metrics_path = save_dir / "total_metrics.txt"
    with open(total_metrics_path, "w") as f:
        header = "metric," + ",".join(metric_names) + "\n"
        f.write(header)
        mean_over_steps = np.mean(mean_metric, axis=0)
        median_over_steps = np.median(median_metric, axis=0)
        std_over_steps = np.mean(std_metric, axis=0)
        f.write("mean," + ",".join(map(str, mean_over_steps)) + "\n")
        f.write("median," + ",".join(map(str, median_over_steps)) + "\n")
        f.write("std," + ",".join(map(str, std_over_steps)) + "\n")
    print(f"Total metrics saved to {total_metrics_path}")

    if not cfg.debug and hasattr(trainer, "logger"):
        for i, metric_name in enumerate(metric_names):
            trainer.logger.add_scalar(
                f"metric/{metric_name}-mean",
                mean_metric[:, i].mean(),
                step=eval_iteration,
            )
            trainer.logger.add_scalar(
                f"metric/{metric_name}-std",
                std_metric[:, i].mean(),
                step=eval_iteration,
            )
            img = np.array(
                Image.open(save_dir / f"{i:02d}-{metric_name}.png").convert("RGB")
            )
            trainer.logger.add_image(
                f"metric_curve/{metric_name}", img, step=eval_iteration
            )

    # ---------------- Aggregate metrics for sim-only branch ----------------
    if metrics_list_sim_only:
        metrics_sim = np.array(metrics_list_sim_only)[:, 0]

        median_metric = np.median(metrics_sim, axis=0)
        step_75_metric = np.percentile(metrics_sim, 75, axis=0)
        step_25_metric = np.percentile(metrics_sim, 25, axis=0)
        mean_metric = np.mean(metrics_sim, axis=0)
        std_metric = np.std(metrics_sim, axis=0)

        save_dir_sim = (
            log_root
            / cfg.train.name
            / "eval-val-sim-only"
            / dataset_name
            / f"{eval_iteration:06d}"
            / "metric"
        )
        save_dir_sim.mkdir(parents=True, exist_ok=True)

        # for i, metric_name in enumerate(metric_names):
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(mean_metric[:, i], label="mean")
        #     plt.plot(median_metric[:, i], label="median")
        #     plt.fill_between(
        #         range(len(mean_metric[:, i])),
        #         step_25_metric[:, i],
        #         step_75_metric[:, i],
        #         alpha=0.2,
        #     )
        #     plt.title(metric_name + " (Sim Only)")
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig(save_dir_sim / f"{i:02d}-{metric_name}.png", dpi=300)
        #     plt.close()

        total_metrics_path_sim = save_dir_sim / "total_metrics.txt"
        with open(total_metrics_path_sim, "w") as f:
            header = "metric," + ",".join(metric_names) + "\n"
            f.write(header)
            mean_over_steps = np.mean(mean_metric, axis=0)
            median_over_steps = np.median(median_metric, axis=0)
            std_over_steps = np.mean(std_metric, axis=0)
            f.write("mean," + ",".join(map(str, mean_over_steps)) + "\n")
            f.write("median," + ",".join(map(str, median_over_steps)) + "\n")
            f.write("std," + ",".join(map(str, std_over_steps)) + "\n")
        print(f"Total sim-only metrics saved to {total_metrics_path_sim}")

        if not cfg.debug and hasattr(trainer, "logger"):
            for i, metric_name in enumerate(metric_names):
                trainer.logger.add_scalar(
                    f"metric_sim_only/{metric_name}-mean",
                    mean_metric[:, i].mean(),
                    step=eval_iteration,
                )
                trainer.logger.add_scalar(
                    f"metric_sim_only/{metric_name}-std",
                    std_metric[:, i].mean(),
                    step=eval_iteration,
                )
                img = np.array(
                    Image.open(save_dir_sim / f"{i:02d}-{metric_name}.png").convert(
                        "RGB"
                    )
                )
                trainer.logger.add_image(
                    f"metric_curve_sim_only/{metric_name}", img, step=eval_iteration
                )


@hydra.main(version_base="1.2", config_path=str(root / "cfg"), config_name="default")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    object_name = apply_object_sim_params(cfg)
    print(f"Using simulator parameters for object '{object_name}'")

    eval_trainer = Trainer(cfg)
    eval_trainer.load_train_dataset()
    eval_trainer.init_train()

    eval_iteration = cfg.train.resume_iteration
    num_workers = cfg.train.get("num_eval_workers", 2)

    eval_parallel(eval_trainer, eval_iteration, save=True, num_workers=num_workers)
    # eval_open_loop_parallel(
    #     eval_trainer, eval_iteration, save=True, num_workers=num_workers
    # )


if __name__ == "__main__":
    main()
