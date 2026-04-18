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
def render_single_state(
    cfg,
    state_root,
    episode_names,
    save_dir,
    state_key,
    color,
    eval_postfix="",
    clean_bg=False,
):
    clean_bg = False
    render_type = f"pv_{state_key}"
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

        if not clean_bg:
            for axis, a_color in enumerate(["r", "g", "b"]):
                mesh = pv.Arrow(start=[0, 0, 0], direction=np.eye(3)[axis], scale=0.2)
                plotter.add_mesh(mesh, color=a_color, show_scalar_bar=False)

        episode_state_root = state_root / episode
        episode_image_root = image_root / f"{episode}{eval_postfix}"
        mkdir(episode_image_root, overwrite=True, resume=True)

        ckpt_paths = list(
            sorted(episode_state_root.glob("*.pt"), key=lambda x: int(x.stem))
        )
        for i, path in enumerate(tqdm(ckpt_paths, desc=render_type)):
            if i % cfg.render.skip_frame != 0:
                continue

            ckpt = torch.load(path, map_location="cpu")
            if state_key not in ckpt:
                print(
                    f"State key {state_key} not in checkpoint {path}. Skipping frame."
                )
                continue
            p_x = ckpt[state_key].cpu().detach().numpy()
            x = (p_x - 0.5) * scale + 0.5

            use_grippers = "grippers" in ckpt
            use_gripper_points = "gripper_x" in ckpt
            n_eef = 0
            if use_grippers:
                grippers = ckpt["grippers"].cpu().detach().numpy()
                n_eef = grippers.shape[0]

            radius = 0.5 * np.power((0.5**3) / x.shape[0], 1 / 3) * scale_mean
            x = np.clip(x, radius, 1 - radius)

            polydata = pv.PolyData(x)
            plotter.add_mesh(
                polydata,
                style="points",
                name="object",
                render_points_as_spheres=True,
                point_size=radius * cfg.render.radius_scale,
                color=color,
            )
            for j in range(n_eef):
                if use_grippers:
                    gripper = pv.Sphere(center=grippers[j, :3], radius=0.04)
                    plotter.add_mesh(gripper, color="blue", name=f"gripper_{j}")
            if use_gripper_points:
                gripper_points = ckpt["gripper_x"].cpu().detach().numpy()
                gripper_points = (gripper_points - 0.5) * scale + 0.5
                gripper_points = np.clip(gripper_points, radius, 1 - radius)
                gripper_polydata = pv.PolyData(gripper_points)
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
                screenshot=str(
                    episode_image_root / f"{i // cfg.render.skip_frame:04d}.png"
                ),
            )
            plotter.remove_actor("object")
            if use_grippers:
                for j in range(n_eef):
                    plotter.remove_actor(f"gripper_{j}")
            if use_gripper_points:
                plotter.remove_actor(f"gripper_points")

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
def render_residual_combined(
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
    render_type = "pv_residual_combined"
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
        gt_ckpt_paths_map = {int(p.stem): p for p in gt_ckpt_paths}
        total_skip_frame = (
            cfg.train.dataset_skip_frame * cfg.train.dataset_load_skip_frame
        )

        n_history = cfg.sim.get("n_history", 0)

        for i in trange(len(sim_ckpt_paths), desc=render_type):
            sim_path = sim_ckpt_paths[i]
            sim_frame_id = int(sim_path.stem)
            sim_step = sim_frame_id * cfg.sim.skip_frame
            gt_frame_id = (sim_step + n_history) * total_skip_frame

            if gt_frame_id not in gt_ckpt_paths_map:
                continue

            sim_ckpt = torch.load(sim_path, map_location="cpu")
            if "x_no_res" not in sim_ckpt:
                print(
                    f"Key 'x_no_res' not in {sim_path}, skipping combined residual visualization."
                )
                break

            # Render GT points (green)
            gt_ckpt = torch.load(gt_ckpt_paths_map[gt_frame_id], map_location="cpu")
            gt_p_x = gt_ckpt["x"].cpu().detach().numpy()
            sim_x_res = sim_ckpt["x"].cpu().detach().numpy()

            if downsample_indices is not None:
                gt_p_x = gt_p_x[downsample_indices[0]]
            else:
                temp_downsample_indices = fps(
                    torch.from_numpy(gt_p_x), sim_x_res.shape[0], random_start=True
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

            # Render sim+res points (purple)
            sim_x_res = (sim_x_res - 0.5) * scale + 0.5
            sim_x_res_clipped = np.clip(sim_x_res, radius, 1 - radius)
            polydata_sim_res = pv.PolyData(sim_x_res_clipped)
            plotter.add_mesh(
                polydata_sim_res,
                style="points",
                name="sim_res_object",
                render_points_as_spheres=True,
                point_size=radius * cfg.render.radius_scale,
                color="purple",
            )

            # Render sim-only points (red)
            sim_x_no_res = sim_ckpt["x_no_res"].cpu().detach().numpy()
            sim_x_no_res = (sim_x_no_res - 0.5) * scale + 0.5
            sim_x_no_res_clipped = np.clip(sim_x_no_res, radius, 1 - radius)
            polydata_sim_no_res = pv.PolyData(sim_x_no_res_clipped)
            plotter.add_mesh(
                polydata_sim_no_res,
                style="points",
                name="sim_no_res_object",
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
                    gripper = pv.Sphere(center=grippers[j, :3], radius=0.04)
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
            plotter.remove_actor("gt_object")
            plotter.remove_actor("sim_res_object")
            plotter.remove_actor("sim_no_res_object")
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
def do_residual_pv_single(
    cfg, state_root, episode_names, save_dir, state_key, color, eval_postfix=""
):
    with Xvfb():
        ret = render_single_state(
            cfg, state_root, episode_names, save_dir, state_key, color, eval_postfix
        )
    return ret


@torch.no_grad()
def do_residual_pv_combined(
    cfg,
    sim_state_root,
    gt_state_root,
    episode_names,
    save_dir,
    downsample_indices,
    eval_postfix="",
):
    with Xvfb():
        ret = render_residual_combined(
            cfg,
            sim_state_root,
            gt_state_root,
            episode_names,
            save_dir,
            eval_postfix,
            downsample_indices,
        )
    return ret
