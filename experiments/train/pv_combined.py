from pathlib import Path
import random
from tqdm import tqdm, trange

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
import math
import pyvista as pv
from dgl.geometry import farthest_point_sampler

import meta_material
from meta_material.utils import get_root, mkdir
from meta_material.ffmpeg import make_video

from train.pv_utils import Xvfb, get_camera_custom


def fps(x, n, random_start=False):
    start_idx = random.randint(0, x.shape[0] - 1) if random_start else 0
    fps_idx = farthest_point_sampler(x[None], n, start_idx=start_idx)[0]
    fps_idx = fps_idx.to(x.device)
    return fps_idx


@torch.no_grad()
def render(
    cfg,
    sim_state_root,
    gt_state_root,
    episode_names,
    save_dir,
    eval_postfix="",
    downsample_indices=None,
    clean_bg=False,
):
    clean_bg = False
    render_type = "pv_combined"
    image_root: Path = save_dir
    image_root.mkdir(parents=True, exist_ok=True)

    video_path_list = []
    for episode_idx, episode in enumerate(episode_names):
        plotter = pv.Plotter(
            lighting="three lights",
            off_screen=True,
            window_size=(cfg.render.width, cfg.render.height),
        )
        plotter.set_background("white")
        plotter.camera_position = get_camera_custom(
            cfg.render.center,
            cfg.render.distance,
            cfg.render.azimuth,
            cfg.render.elevation,
        )
        plotter.enable_shadows()

        # add bounding box
        if cfg.sim.num_grids_flexible is None:
            scale = cfg.sim.num_grids / (cfg.sim.num_grids - 2 * cfg.render.bound)
            scale_mean = scale
            bbox = pv.Box(bounds=[0, 1, 0, 1, 0, 1])
        else:
            scale_x = cfg.sim.num_grids_flexible[0] / (
                cfg.sim.num_grids_flexible[0] - 2 * cfg.render.bound
            )
            scale_y = cfg.sim.num_grids_flexible[1] / (
                cfg.sim.num_grids_flexible[1] - 2 * cfg.render.bound
            )
            scale_z = cfg.sim.num_grids_flexible[2] / (
                cfg.sim.num_grids_flexible[2] - 2 * cfg.render.bound
            )
            scale = np.array([scale_x, scale_y, scale_z])
            scale_mean = np.power(np.prod(scale), 1 / 3)
            bbox = pv.Box(bounds=[0, 1, 0, 1, 0, 1])
        if not clean_bg:
            plotter.add_mesh(bbox, style="wireframe", color="black")

        # add axis
        if not clean_bg:
            for axis, color in enumerate(["r", "g", "b"]):
                mesh = pv.Arrow(start=[0, 0, 0], direction=np.eye(3)[axis], scale=0.2)
                plotter.add_mesh(mesh, color=color, show_scalar_bar=False)

        sim_episode_state_root = sim_state_root / episode
        gt_episode_state_root = gt_state_root / episode
        episode_image_root = image_root / f"{episode}{eval_postfix}"
        mkdir(episode_image_root, overwrite=True, resume=True)

        sim_ckpt_paths = list(
            sorted(sim_episode_state_root.glob("*.pt"), key=lambda x: int(x.stem))
        )

        gt_ckpt_paths = list(
            sorted(gt_episode_state_root.glob("*.pt"), key=lambda x: int(x.stem))
        )

        # Match GT frames to sim frames
        gt_ckpt_paths_map = {int(p.stem): p for p in gt_ckpt_paths}

        num_frames = len(sim_ckpt_paths)

        # The GT video may have a different frame rate. We need to find the right mapping.
        # In eval_custom_faster, the gt_x is indexed by `step` from the simulation loop.
        # gt_x[step] comes from original frame `start_frame + step * total_skip_frame`.
        # However, `start_frame` is not known here.
        # Let's assume for evaluation `start_frame` is 0.
        total_skip_frame = (
            cfg.train.dataset_skip_frame * cfg.train.dataset_load_skip_frame
        )

        n_history = cfg.sim.get("n_history", 0)

        for i in trange(num_frames, desc=render_type):
            sim_path = sim_ckpt_paths[i]
            # frame_id from sim checkpoint is `step // cfg.sim.skip_frame`
            sim_frame_id = int(sim_path.stem)

            # The simulation step corresponding to this frame
            sim_step = sim_frame_id * cfg.sim.skip_frame

            # The corresponding GT frame should be offset by n_history and the simulation step.
            # The +1 is because gt_xs starts from frame + skip_frame in the dataloader.
            gt_frame_id = (sim_step + n_history + 1) * total_skip_frame

            if gt_frame_id not in gt_ckpt_paths_map:
                # This can happen if the simulation is longer than the available GT data.
                continue

            sim_ckpt = torch.load(sim_path, map_location="cpu")

            # Render GT points (green)
            gt_ckpt = torch.load(gt_ckpt_paths_map[gt_frame_id], map_location="cpu")
            gt_p_x = gt_ckpt["x"].cpu().detach().numpy()

            sim_x = sim_ckpt["x"].cpu().detach().numpy()

            if downsample_indices is not None:
                gt_p_x = gt_p_x[downsample_indices[0]]
            else:
                # This might be an issue if we need consistent downsampling across frames
                temp_downsample_indices = fps(
                    torch.from_numpy(gt_p_x), sim_x.shape[0], random_start=True
                )[None]
                gt_p_x = gt_p_x[temp_downsample_indices[0]]

            gt_x = (gt_p_x - 0.5) * scale + 0.5
            radius = 0.5 * np.power((0.5**3) / gt_x.shape[0], 1 / 3) * scale_mean
            gt_x_clipped = np.clip(gt_x, radius, 1 - radius)
            polydata_gt = pv.PolyData(gt_x_clipped)
            plotter.add_mesh(
                polydata_gt,
                style="points",
                name="gt_object",
                render_points_as_spheres=True,
                point_size=radius * cfg.render.radius_scale,
                color="green",
            )

            # Render sim points (red)
            sim_x = (sim_x - 0.5) * scale + 0.5
            sim_x_clipped = np.clip(sim_x, radius, 1 - radius)
            polydata_sim = pv.PolyData(sim_x_clipped)
            plotter.add_mesh(
                polydata_sim,
                style="points",
                name="sim_object",
                render_points_as_spheres=True,
                point_size=radius * cfg.render.radius_scale,
                color="red",
            )

            # Common elements: grippers
            use_grippers = "grippers" in sim_ckpt
            use_gripper_points = "gripper_x" in sim_ckpt
            n_eef = 0
            if use_grippers:
                grippers = sim_ckpt["grippers"].cpu().detach().numpy()
                n_eef = grippers.shape[0]
                for j in range(n_eef):
                    gripper = pv.Sphere(
                        center=grippers[j, :3], radius=0.04
                    )  # from pv_train.py
                    plotter.add_mesh(gripper, color="blue", name=f"gripper_{j}")

            if use_gripper_points:
                gripper_points = sim_ckpt["gripper_x"].cpu().detach().numpy()
                gripper_points = (gripper_points - 0.5) * scale + 0.5
                gripper_points_clipped = np.clip(gripper_points, radius, 1 - radius)
                gripper_polydata = pv.PolyData(gripper_points_clipped)
                plotter.add_mesh(
                    gripper_polydata,
                    style="points",
                    name=f"gripper_points",
                    render_points_as_spheres=True,
                    point_size=radius * cfg.render.radius_scale,
                    color="blue",
                )

            plotter.show(
                auto_close=False,
                screenshot=str(episode_image_root / f"{sim_frame_id:04d}.png"),
            )
            # Remove objects for next frame to avoid overlap
            plotter.remove_actor("sim_object")
            plotter.remove_actor("gt_object")
            if use_grippers:
                for j in range(n_eef):
                    plotter.remove_actor(f"gripper_{j}")
            if use_gripper_points:
                plotter.remove_actor("gripper_points")

        plotter.close()
        make_video(
            episode_image_root,
            image_root / f"{episode}{eval_postfix}.mp4",
            "%04d.png",
            cfg.render.fps,
        )
        video_path_list.append(image_root / f"{episode}{eval_postfix}.mp4")
    return video_path_list


@torch.no_grad()
def do_combined_pv(*args, **kwargs):
    with Xvfb():
        ret = render(*args, **kwargs)
    return ret
