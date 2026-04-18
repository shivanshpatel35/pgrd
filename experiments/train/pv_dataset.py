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
    dataset_root,
    episode_names,
    iteration=None,
    start_step=None,
    end_step=None,
    save_dir=None,
    downsample_indices=None,
    clean_bg=False,
):
    render_type = "pv"

    exp_root: Path = dataset_root
    state_root: Path = exp_root / "state"
    # image_root: Path = exp_root / render_type
    # mkdir(image_root, overwrite=False, resume=True)

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
        # episode_image_root = image_root / episode
        # mkdir(episode_image_root, overwrite=True, resume=True)

        # global_translation = np.array([0.0, 0.0, 0.0])

        episode_image_root = save_dir / f"{episode}_gt"
        mkdir(episode_image_root, overwrite=True, resume=True)

        ckpt_paths = list(
            sorted(episode_state_root.glob("*.pt"), key=lambda x: int(x.stem))
        )
        if start_step is not None and end_step is not None:
            ckpt_paths = ckpt_paths[start_step:end_step]
        skip_frame = cfg.train.dataset_skip_frame * cfg.train.dataset_load_skip_frame
        ckpt_paths = ckpt_paths[skip_frame::skip_frame]
        for i, path in enumerate(tqdm(ckpt_paths, desc=render_type)):
            if i % cfg.render.skip_frame != 0:
                continue

            ckpt = torch.load(path, map_location="cpu")
            p_x = ckpt["x"].cpu().detach().numpy()
            if downsample_indices is not None:
                p_x = p_x[downsample_indices[0]]
            else:
                downsample_indices = fps(
                    torch.from_numpy(p_x), cfg.sim.n_particles, random_start=True
                )[None]
                p_x = p_x[downsample_indices[0]]
            # if p_x[:, 1].min() < 0.04 and global_translation.sum() == 0.0:
            #     global_translation = np.array([0.0, 0.04 - p_x[:, 1].min(), 0.0])
            # p_x += global_translation
            # x_sections = [(p_x - 0.5) * scale + 0.5, []]
            # x_sections = np.split((ckpt['x'].cpu().detach().numpy() - 0.5) * scale + 0.5, np.cumsum(ckpt['sections']), axis=0)
            x = (p_x - 0.5) * scale + 0.5

            use_cylinders = "cylinders" in ckpt
            use_grippers = "grippers" in ckpt
            if use_cylinders:
                cylinders = ckpt["cylinders"].cpu().detach().numpy()
                n_eef = cylinders.shape[0]
            if use_grippers:
                use_grippers = True
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
                    gripper = pv.Sphere(center=grippers[j, :3], radius=grippers[j, -2])
                    plotter.add_mesh(gripper, color="blue", name=f"gripper_{j}")

            plotter.show(
                auto_close=False,
                screenshot=str(
                    episode_image_root / f"{i // cfg.render.skip_frame:04d}.png"
                ),
            )

        plotter.close()
        if save_dir is not None:
            make_video(
                episode_image_root,
                save_dir / f"{episode}_gt.mp4",
                "%04d.png",
                cfg.render.fps,
            )
            video_path_list.append(save_dir / f"{episode}_gt.mp4")
        # if iteration is not None:
        #     make_video(episode_image_root, image_root / f'{episode}_iteration_{iteration:04d}.mp4', '%04d.png', cfg.render.fps)
        #     video_path_list.append(image_root / f'{episode}_iteration_{iteration:04d}.mp4')
        # else:
        #     make_video(episode_image_root, image_root / f'{episode}.mp4', '%04d.png', cfg.render.fps)
        #     video_path_list.append(image_root / f'{episode}.mp4')
    return video_path_list


@torch.no_grad()
def do_dataset_pv(*args, **kwargs):
    with Xvfb():
        ret = render(*args, **kwargs)
    return ret
