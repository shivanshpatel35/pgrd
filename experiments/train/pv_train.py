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

import meta_material
from meta_material.utils import get_root, mkdir
from meta_material.ffmpeg import make_video

from train.pv_utils import Xvfb, get_camera_custom


@torch.no_grad()
def render(
    cfg,
    log_root,
    iteration,
    episode_names,
    eval_dirname="eval",
    eval_postfix="",
    dataset_name="",
    start_step=None,
    end_step=None,
    clean_bg=False,
):
    clean_bg = False

    if dataset_name == "":
        eval_name = f"{cfg.train.name}/{eval_dirname}/{iteration:06d}"
    else:
        eval_name = f"{cfg.train.name}/{eval_dirname}/{dataset_name}/{iteration:06d}"
    render_type = "pv"

    exp_root: Path = log_root / eval_name
    state_root: Path = exp_root / "state"
    image_root: Path = exp_root / render_type
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

        episode_state_root = state_root / episode
        episode_image_root = image_root / episode
        mkdir(episode_image_root, overwrite=cfg.overwrite, resume=cfg.resume)

        # global_translation = np.array([0.0, 0.0, 0.0])

        ckpt_paths = list(
            sorted(episode_state_root.glob("*.pt"), key=lambda x: int(x.stem))
        )
        for i, path in enumerate(tqdm(ckpt_paths, desc=render_type)):
            if i % cfg.render.skip_frame != 0:
                continue

            ckpt = torch.load(path, map_location="cpu")
            p_x = ckpt["x"].cpu().detach().numpy()
            # if p_x[:, 1].min() < 0.04 and global_translation.sum() == 0.0:
            #     global_translation = np.array([0.0, 0.04 - p_x[:, 1].min(), 0.0])
            # p_x += global_translation
            # x_sections = [(p_x - 0.5) * scale + 0.5, []]
            # x_sections = np.split((ckpt['x'].cpu().detach().numpy() - 0.5) * scale + 0.5, np.cumsum(ckpt['sections']), axis=0)
            x = (p_x - 0.5) * scale + 0.5

            use_cylinders = "cylinders" in ckpt
            use_grippers = "grippers" in ckpt
            use_gripper_points = "gripper_x" in ckpt
            if use_cylinders:
                cylinders = ckpt["cylinders"].cpu().detach().numpy()
                n_eef = cylinders.shape[0]
            if use_grippers:
                grippers = ckpt["grippers"].cpu().detach().numpy()
                n_eef = grippers.shape[0]
            else:
                n_eef = 0

            radius = 0.5 * np.power((0.5**3) / x.shape[0], 1 / 3) * scale_mean
            x = np.clip(x, radius, 1 - radius)

            polydata = pv.PolyData(x)
            plotter.add_mesh(
                polydata,
                style="points",
                name="object",
                render_points_as_spheres=True,
                point_size=radius * cfg.render.radius_scale,
                color=list(cfg.render.reflectance),
            )
            for j in range(n_eef):
                if use_cylinders:
                    cylinder = pv.Cylinder(
                        center=cylinders[j, :3],
                        direction=cylinders[j, 8:11],
                        height=cylinders[j, 6],
                        radius=cylinders[j, 7],
                    )
                    plotter.add_mesh(cylinder, color="blue", name=f"cylinder_{j}")
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
def do_train_pv(*args, **kwargs):
    with Xvfb():
        ret = render(*args, **kwargs)
    return ret
