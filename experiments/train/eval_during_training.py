"""Lightweight evaluation utilities (sim + residual only).

This module avoids any heavy qualitative rendering (PV / GS) and focuses on
computing quantitative metrics quickly in parallel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import multiprocessing as mp

import numpy as np
import torch
from tqdm import trange
import warp as wp
from omegaconf import DictConfig, OmegaConf

import meta_material
from meta_material.utils import get_root, mkdir
from meta_material.data import RealTeleopBatchDataset
from meta_material.sim import create_simulator
from experiments.train.temporal_transformer import TemporalResidualTransformer

from train.metric_eval import do_metric

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
root: Path = get_root(__file__)


# -----------------------------------------------------------------------------
# Multiprocessing helpers
# -----------------------------------------------------------------------------


def _pool_worker_init() -> None:  # pragma: no cover – helper for mp.Pool
    """Ensure Pool workers are non-daemon so they can spawn children."""

    mp.current_process().daemon = False  # pylint: disable=protected-access


# -----------------------------------------------------------------------------
# Per-episode evaluation (single process / single GPU)
# -----------------------------------------------------------------------------


def _build_dataloader(dataset: torch.utils.data.Dataset):
    """Yield data indefinitely (avoids StopIteration inside subprocesses)."""

    from torch.utils.data import DataLoader  # local import to avoid overhead

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    while True:
        for item in loader:  # pragma: no cover – simple generator
            yield item


def eval_episode(
    cfg: DictConfig,
    residualnet_state_dict: dict[str, Any],
    transformer_state_dict: dict[str, Any] | None,
    iteration: int,
    episode: int,
    save: bool = True,
    gpu_id: int = 0,
):
    """Run a single episode evaluation on one GPU / CPU.

    Returns the list of frame-wise metrics produced by :pyfunc:`train.metric_eval.do_metric`.
    """

    # ------------------ Device / Warp initialisation -------------------------
    wp.init()
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        wp_device = wp.get_device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        wp_device = wp.get_device("cpu")

    # ------------------ Paths & dirs -----------------------------------------
    log_root: Path = root / "log"
    eval_name = f"{cfg.train.name}/eval-val/{cfg.train.dataset_name.split('/')[-1]}/{iteration:06d}"
    exp_root: Path = log_root / eval_name
    if save:
        episode_state_root = exp_root / "state" / f"episode_{episode:04d}"
        mkdir(episode_state_root, overwrite=cfg.overwrite, resume=cfg.resume)
        OmegaConf.save(cfg, exp_root / "hydra.yaml", resolve=True)

    # ------------------ Load dataset sample ----------------------------------
    if cfg.train.dataset_name is None:
        cfg.train.dataset_name = Path(cfg.train.name).parent / "dataset"

    source_dataset_root = log_root / str(cfg.train.source_dataset_name)
    if not source_dataset_root.exists():
        source_dataset_root = (
            Path("./data/meta-material") / cfg.train.source_dataset_name
        )
        assert source_dataset_root.exists()

    eval_dataset = RealTeleopBatchDataset(
        cfg,
        dataset_root=log_root / cfg.train.dataset_name / "state",
        source_data_root=source_dataset_root,
        device=device,
        num_steps=cfg.sim.num_steps,
        eval_episode_name=f"episode_{episode:04d}",
    )

    dl_iter = _build_dataloader(eval_dataset)
    init_data, actions, gt_states, downsample_indices = next(dl_iter)

    # Initial particle data (B=1 always for evaluation)
    # Unpack optional interior/fill points as 8th field
    if len(init_data) >= 8 and int(cfg.sim.num_fill_points) > 0:
        x, v, x_his, v_his, clip_bound, enabled, _, fill_points = init_data
    else:
        x, v, x_his, v_his, clip_bound, enabled, _ = init_data
        fill_points = None

    x, v, x_his, v_his, enabled = [t.to(device) for t in (x, v, x_his, v_his, enabled)]
    if fill_points is not None:
        fill_points = fill_points.to(device)
    cylinders, grippers = [t.to(device) for t in actions]
    gt_x, gt_v = [t.to(device) for t in gt_states]

    batch_size = 1
    num_steps_total = gt_x.shape[1]
    num_particles_orig = x.shape[1]

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
        appended_fill_points = True
    else:
        appended_fill_points = False

    # ------------------ Simulator -------------------------------------------
    points_per_env_to_use = (
        int(x.shape[1]) if appended_fill_points else int(cfg.sim.n_particles)
    )

    sim = create_simulator(
        backend=getattr(cfg.sim, "backend", "spring"),
        x=x.detach().clone().cpu().numpy(),
        v=v.detach().clone().cpu().numpy(),
        grippers=grippers.detach().clone().cpu().numpy(),
        points_per_env=points_per_env_to_use,
        batch_size=batch_size,
        threshold=cfg.sim.threshold,
        stiffness=cfg.sim.stiffness,
        damping=cfg.sim.damping,
        mass_per_point=cfg.sim.mass_per_point,
        sim_dt=cfg.sim.dt,
        sim_substeps=cfg.sim.sim_substeps,
        device=wp_device,
        visualize=False,
        gripper_radius=cfg.model.gripper_radius,
        max_springs_per_node=cfg.sim.max_springs_per_node,
    )

    # Overwrite x, v with simulator-consistent tensors (incl. internal particles)
    x_full, v_full = sim.get_initial_state()
    x, v = x_full.to(device), v_full.to(device)
    num_particles = x.shape[1]

    # If we added internal points, extend history tensors to match
    if getattr(cfg.sim, "n_history", 0) > 0:
        num_internal_particles = num_particles - num_particles_orig
        if num_internal_particles > 0:
            x_internal = x[:, num_particles_orig:]
            v_internal = v[:, num_particles_orig:]
            n_history_local = cfg.sim.n_history
            x_his_padding = (
                x_internal.unsqueeze(2)
                .repeat(1, 1, n_history_local, 1)
                .reshape(x.shape[0], num_internal_particles, -1)
            )
            v_his_padding = (
                v_internal.unsqueeze(2)
                .repeat(1, 1, n_history_local, 1)
                .reshape(v.shape[0], num_internal_particles, -1)
            )
            x_his = torch.cat([x_his, x_his_padding], dim=1)
            v_his = torch.cat([v_his, v_his_padding], dim=1)

    # Extend enabled mask to internal particles (internal always enabled=1)
    enabled_padding = torch.ones(
        (batch_size, num_particles - num_particles_orig),
        dtype=enabled.dtype,
        device=device,
    )
    enabled_full = torch.cat([enabled, enabled_padding], dim=1)

    # ------------------ Networks --------------------------------------------
    n_history: int = getattr(cfg.sim, "n_history", 0)
    residualnet = getattr(meta_material.material, cfg.model.residualnet.cls)(
        cfg.model.residualnet, n_history
    )
    residualnet.set_params(
        cfg.sim.num_grids, num_grids_flexible=cfg.sim.num_grids_flexible
    )
    residualnet.load_state_dict(residualnet_state_dict)
    residualnet.to(device)
    residualnet.eval()

    temporal_model = TemporalResidualTransformer(cfg, device=device)
    temporal_model.load_component_state_dicts(transformer_state_dict)
    temporal_model.eval()

    # ------------------ Rollout ---------------------------------------------
    # Sliding window buffer
    temporal_model.reset_window()
    skip_frame = cfg.sim.skip_frame

    for step in trange(num_steps_total, desc=f"Eval ep{episode}", leave=False):
        # Warp sim one step
        x_sim, v_sim = sim(step, x.detach().clone(), v.detach().clone(), None, grippers)
        x_sim = x_sim.to(device)
        v_sim = v_sim.to(device)

        # Ensure all tensors are on the same device before the network call
        x = x.to(device)
        v = v.to(device)
        x_his = x_his.to(device)
        v_his = v_his.to(device)
        enabled_full = enabled_full.to(device)

        # Elasticity residuals -------------------------------------------------
        pts_feat = residualnet(
            x, v, x_his, v_his, enabled_full.unsqueeze(-1).repeat(1, 1, 3), x_sim, v_sim
        )

        residual_v = temporal_model(
            pts_feat,
            rollout_window_size=num_steps_total,
        )

        v = v_sim + residual_v
        residual_x = residual_v * cfg.sim.dt
        x = x_sim + residual_x
        x = torch.stack(
            [x[..., 0], x[..., 1].clamp(min=sim.ground_y), x[..., 2]], dim=-1
        )

        # Update history tensors ---------------------------------------------
        if n_history > 0:
            x_his = torch.cat(
                [
                    x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                    x[:, :, None],
                ],
                dim=2,
            ).reshape(batch_size, num_particles, -1)
            v_his = torch.cat(
                [
                    v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                    v[:, :, None],
                ],
                dim=2,
            ).reshape(batch_size, num_particles, -1)

        # Save state for metric computation ----------------------------------
        if save and step % skip_frame == 0:
            torch.save(
                {"x": x[0]}, episode_state_root / f"{int(step / skip_frame):04d}.pt"
            )

    # ------------------ Metrics ---------------------------------------------
    metrics = None
    if save:
        metrics = do_metric(
            cfg,
            log_root,
            iteration,
            [f"episode_{episode:04d}"],
            downsample_indices,
            eval_dirname="eval-val",
            dataset_name=cfg.train.dataset_name.split("/")[-1],
            eval_postfix="",
            camera_id=1,
            use_gs=False,  # Disable GS-specific metrics
        )
    return metrics


# -----------------------------------------------------------------------------
# Parallel helper – launches multiple episodes at once
# -----------------------------------------------------------------------------


def eval_parallel(
    trainer: "meta_material",  # type: ignore [valid-type] – forward ref
    eval_iteration: int,
    *,
    save: bool = True,
    num_workers: int = 5,
):
    """Parallel evaluation over episodes.

    Each worker process reconstructs the networks from *state_dict*s so that GPU
    memory is not shared between processes.
    """

    cfg = trainer.cfg
    episodes = [
        e
        for e in range(cfg.train.eval_start_episode, cfg.train.eval_end_episode)
        if e not in cfg.train.get("eval_skip_episode", [])
    ]

    if not episodes:
        return

    # Collect state-dicts once and broadcast to workers
    residualnet_sd = trainer.residualnet.state_dict()
    transformer_sd = trainer.temporal_model.export_component_state_dicts()

    num_gpus = torch.cuda.device_count()
    num_workers = min(num_workers, len(episodes))
    if num_gpus == 0:
        print("[eval] No CUDA devices found – running on CPU.")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_pool_worker_init) as pool:
        args = [
            (
                cfg,
                residualnet_sd,
                transformer_sd,
                eval_iteration,
                ep,
                save,
                i % num_gpus if num_gpus > 0 else -1,
            )
            for i, ep in enumerate(episodes)
        ]
        metrics_list = pool.starmap(eval_episode, args)

    # Aggregate and save metrics
    if not save:
        return metrics_list

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

    # Filter out None results and stack
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        print("No metrics were generated. Skipping metric aggregation.")
        return
    metrics_list = np.array(metrics_list)[:, 0]

    # We do not use GS here; keep the classic three metrics
    metric_names = ["mse", "avg_d", "chamfer", "emd"]

    median_metric = np.median(metrics_list, axis=0)
    mean_metric = np.mean(metrics_list, axis=0)
    std_metric = np.std(metrics_list, axis=0)

    # Save summary to disk
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

    # Optional: log to wandb if enabled and logger exists
    if not cfg.debug and hasattr(trainer, "logger"):
        for i, metric_name in enumerate(metric_names):
            trainer.logger.add_scalar(
                f"eval/{metric_name}-mean",
                mean_metric[:, i].mean(),
                step=eval_iteration,
            )

    return metrics_list
