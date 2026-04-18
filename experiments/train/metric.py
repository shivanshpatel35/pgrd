from pathlib import Path
import random
import time
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import os
import glob
from PIL import Image
import argparse
import yaml
import scipy
import math
import lpips
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_sk

from typing import Union


def get_root(path: Union[str, Path], name: str = ".root") -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root


root: Path = get_root(__file__)


def calc_psnr(img1, img2, mask):
    # img1: (H, W, 3)
    # img2: (H, W, 3)
    # mask: (H, W)
    mse = np.sum((img1 - img2) ** 2 * mask[..., None]) / np.sum(mask)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calc_ssim(img1, img2, mask):
    # img1: (H, W, 3)
    # img2: (H, W, 3)
    # mask: (H, W)
    return ssim_sk(
        img1, img2, data_range=255, mask=mask, multichannel=True, channel_axis=2
    )


def mse_dist(xyz, xyz_gt):
    # xyz: (N, 3)
    # xyz_gt: (N, 3)
    return torch.mean((xyz - xyz_gt) ** 2).item()


def chamfer_dist(xyz, xyz_gt):
    # xyz: (N, 3)
    # xyz_gt: (M, 3)
    dist = torch.sqrt(torch.sum((xyz[:, None] - xyz_gt[None]) ** 2, dim=2))  # (N, M)
    chamfer = torch.mean(torch.min(dist, dim=1).values)
    return chamfer.item()


def em_distance(x, y):
    # x: [N, D]
    # y: [M, D]
    cost_matrix = scipy.spatial.distance.cdist(x.cpu(), y.cpu())
    try:
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
    except:
        print("Error in linear sum assignment!")
    ind1 = torch.tensor(ind1).to(x.device)
    ind2 = torch.tensor(ind2).to(y.device)
    x_new = x[ind1]
    y_new = y[ind2]

    emd = torch.mean(torch.norm(x_new - y_new, 2, dim=1))
    return emd.item()


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), (
        "Cant convert %dx%d seg to %dx%d bmap." % (w, h, width, height)
    )

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def compute_f(mask, mask_gt):
    # Only loaded when run to reduce minimum requirements
    # from pycocotools import mask as mask_utils
    from skimage.morphology import disk
    import cv2

    bound_th = 0.008

    bound_pix = (
        bound_th
        if bound_th >= 1 - np.finfo("float").eps
        else np.ceil(bound_th * np.linalg.norm(mask.shape))
    )

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(mask)
    gt_boundary = seg2bmap(mask_gt)

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        f_val = 0
    else:
        f_val = 2 * precision * recall / (precision + recall)

    return f_val


def compute_j(mask, mask_gt):
    iou = np.sum(mask & mask_gt) / np.sum(mask | mask_gt)
    return iou


def compute_lpips(fn, im, im_gt):
    im = torch.tensor(im).permute(2, 0, 1).unsqueeze(0).float()
    im_gt = torch.tensor(im_gt).permute(2, 0, 1).unsqueeze(0).float()
    # im = (im / 255.0 - 0.5) / 0.5
    # im_gt = (im_gt / 255.0 - 0.5) / 0.5
    im = im / 255.0
    im_gt = im_gt / 255.0
    perception = fn.forward(im, im_gt)
    return perception.item()


def inverse_preprocess(cfg, p_x, xyz):
    if cfg.sim.num_grids_flexible is not None:
        dx = cfg.sim.num_grids_flexible[-1]
    else:
        dx = 1 / cfg.sim.num_grids

    R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to(xyz.device).to(xyz.dtype)
    xyz = torch.einsum("nij,jk->nik", xyz, R.T)

    scale = cfg.sim.preprocess_scale if hasattr(cfg.sim, "preprocess_scale") else 1.2
    xyz = xyz * scale

    if cfg.sim.preprocess_with_table:
        global_translation = torch.tensor(
            [
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                dx * (cfg.model.clip_bound + 0.5) + 1e-5 - xyz[:, :, 1].min(),
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ],
            dtype=xyz.dtype,
        )
    else:
        global_translation = torch.tensor(
            [
                0.5 - (xyz[:, :, 0].max() + xyz[:, :, 0].min()) / 2,
                0.5 - (xyz[:, :, 1].max() + xyz[:, :, 1].min()) / 2,
                0.5 - (xyz[:, :, 2].max() + xyz[:, :, 2].min()) / 2,
            ],
            dtype=xyz.dtype,
        )

    xyz += global_translation
    if not (
        xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
        and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
        and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
        and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
        and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
        and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
    ):
        print("inverse_preprocess out of bound")
        xyz_max = xyz.max(dim=0).values
        xyz_max_mask = (
            (xyz_max[:, 0] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
            | (xyz_max[:, 1] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
            | (xyz_max[:, 2] > 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6)
        )
        xyz_min = xyz.min(dim=0).values
        xyz_min_mask = (
            (xyz_min[:, 0] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
            | (xyz_min[:, 1] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
            | (xyz_min[:, 2] < dx * (cfg.model.clip_bound + 0.5) - 1e-6)
        )
        xyz_mask = xyz_max_mask | xyz_min_mask
        # if xyz_mask.sum() > 0.05 * xyz.shape[1]:
        #     import ipdb; ipdb.set_trace()
        xyz = xyz[:, ~xyz_mask]

        assert (
            xyz[:, :, 0].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
            and xyz[:, :, 0].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
            and xyz[:, :, 1].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
            and xyz[:, :, 1].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
            and xyz[:, :, 2].min() >= dx * (cfg.model.clip_bound + 0.5) - 1e-6
            and xyz[:, :, 2].max() <= 1 - dx * (cfg.model.clip_bound + 0.5) + 1e-6
        )
    xyz -= global_translation

    p_x -= global_translation
    p_x = p_x / scale
    p_x = torch.einsum("nij,jk->nik", p_x, torch.linalg.inv(R).T)

    # optional: recover xyz
    xyz = xyz / scale
    xyz = torch.einsum("nij,jk->nik", xyz, torch.linalg.inv(R).T)
    return p_x, xyz


def main(cfg, config_extra):
    eval_dir = f"eval-val"
    dataset_base_name = cfg.train.ataset_name.split("/")[-1]
    state_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "state"
    )
    pv_gs_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "pv_gs"
    )
    mask_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "mask"
    )

    pv_gs_gt_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "pv_gs_gt"
    )
    mask_gt_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "mask_gt"
    )

    save_dir = (
        root
        / "log"
        / cfg.train.name
        / eval_dir
        / dataset_base_name
        / f"{config_extra['iteration']:06d}"
        / "metric"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_fn_vgg = lpips.LPIPS(net="alex")

    metric_list_list = []

    for episode_id in range(config_extra["start_episode"], config_extra["end_episode"]):
        state_dir_episode = state_dir / f"episode_{episode_id:04d}"
        pv_gs_dir_episode = pv_gs_dir / f"episode_{episode_id:04d}"
        mask_dir_episode = mask_dir / f"episode_{episode_id:04d}"
        save_dir_episode = save_dir / f"episode_{episode_id:04d}"
        save_dir_episode.mkdir(parents=True, exist_ok=True)

        pv_gs_gt_dir_episode = pv_gs_gt_dir / f"episode_{episode_id:04d}"
        mask_gt_dir_episode = mask_gt_dir / f"episode_{episode_id:04d}"
        pv_gs_gt_dir_episode.mkdir(parents=True, exist_ok=True)
        mask_gt_dir_episode.mkdir(parents=True, exist_ok=True)

        episode_meta = np.loadtxt(
            root
            / "log"
            / cfg.train.source_dataset_name
            / f"episode_{episode_id:04d}"
            / "meta.txt"
        )
        meta_episode_id, meta_frame_start, meta_frame_end = episode_meta
        skip_frame = cfg.train.dataset_load_skip_frame * cfg.train.dataset_skip_frame
        frame_ids = np.arange(
            meta_frame_start + (cfg.sim.n_history + 1) * skip_frame,
            meta_frame_end,
            skip_frame,
        )
        n_frames = len(frame_ids)

        # load xyz_orig for inverse preprocess
        xyz_orig = np.load(
            root
            / "log"
            / cfg.train.source_dataset_name
            / f"episode_{episode_id:04d}"
            / "traj.npz"
        )["xyz"]
        xyz_orig = torch.tensor(xyz_orig, dtype=torch.float32)

        traj = []
        imgs = []
        masks = []
        # gt_traj = []
        gt_imgs = []
        gt_masks = []
        for frame_id in range(n_frames):
            frame_id_gt = frame_ids[frame_id]

            state = torch.load(state_dir_episode / f"{frame_id:04d}.pt")
            pv_gs = cv2.imread(pv_gs_dir_episode / f"{frame_id:04d}.png")
            pv_gs = cv2.cvtColor(pv_gs, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_dir_episode / f"{frame_id:04d}.png")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            x = state["x"].cpu()
            traj.append(x)  # (n, 3)

            imgs.append(pv_gs)  # (H, W, 3)

            gt_mask = cv2.imread(
                (root / "log" / cfg.train.source_dataset_name).parent
                / f"episode_{int(meta_episode_id):04d}"
                / f"camera_{config_extra['camera_id']}"
                / "mask"
                / f"{int(frame_id_gt):06d}.png"
            )
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_masks.append(gt_mask)

            gt_img = cv2.imread(
                (root / "log" / cfg.train.source_dataset_name).parent
                / f"episode_{int(meta_episode_id):04d}"
                / f"camera_{config_extra['camera_id']}"
                / "rgb"
                / f"{int(frame_id_gt):06d}.jpg"
            )
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = gt_img * (gt_mask > 0)[..., None]
            gt_imgs.append(gt_img)

            # save
            pv_gs_gt = Image.fromarray(gt_img)
            pv_gs_gt.save(pv_gs_gt_dir_episode / f"{frame_id:04d}.png")
            mask_gt = Image.fromarray(gt_mask)
            mask_gt.save(mask_gt_dir_episode / f"{frame_id:04d}.png")

            # gt_pcd = np.load((root / 'log' / cfg.train.source_dataset_name).parent / f'episode_{int(meta_episode_id):04d}' \
            #     / 'pcd_clean' / f'{int(frame_id_gt):06d}.npz')
            # gt_state = torch.tensor(gt_pcd['pts'], dtype=torch.float32)
            # gt_traj.append(gt_state)

        traj = torch.stack(traj, dim=0)
        traj, xyz_orig = inverse_preprocess(cfg, traj, xyz_orig)
        gt_traj = xyz_orig[(cfg.sim.n_history + 1) * skip_frame :: skip_frame]

        assert len(imgs) == len(gt_imgs) == len(gt_masks) == len(traj) == len(gt_traj)

        metric_list = []
        for i in range(len(imgs)):
            # if i % 10 != 0:
            #     continue
            im = imgs[i]
            mask = masks[i]
            im_gt = gt_imgs[i]
            mask_gt = gt_masks[i]

            xyz = traj[i].cuda()
            xyz_gt = gt_traj[i].cuda()

            mask = mask > 0
            mask_gt = mask_gt > 0

            if os.path.exists(save_dir_episode / f"downsample_indices.npy"):
                downsample_indices = np.load(
                    save_dir_episode / f"downsample_indices.npy"
                )
                downsample_indices = torch.from_numpy(downsample_indices).cuda()
                xyz_gt_downsampled = xyz_gt[downsample_indices]
                mse = mse_dist(xyz, xyz_gt_downsampled)
            else:
                mse = -1.0
            chamfer = chamfer_dist(xyz, xyz_gt)
            emd = em_distance(xyz, xyz_gt)

            jscore = compute_j(mask, mask_gt)
            fscore = compute_f(mask, mask_gt)
            jfscore = (jscore + fscore) / 2

            im = im.astype(np.uint8)
            perception = compute_lpips(loss_fn_vgg, im, im_gt)
            psnr = calc_psnr(im, im_gt, mask_gt)
            ssim = calc_ssim(im, im_gt, mask_gt)
            iou = np.sum(mask & mask_gt) / np.sum(mask | mask_gt)

            # visualize
            # cv2.imwrite('test.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            # cv2.imwrite('test_gt.png', cv2.cvtColor(im_gt, cv2.COLOR_RGB2BGR))
            # cv2.imwrite('test_mask.png', mask_gt.astype(np.uint8) * 255)

            metric_list.append(
                [
                    mse,
                    chamfer,
                    emd,
                    jscore,
                    fscore,
                    jfscore,
                    perception,
                    psnr,
                    ssim,
                    iou,
                ]
            )
            print(
                f"episode: {episode_id}, image: {i}, cam: {config_extra['camera_id']}",
                end=" ",
            )
            print(
                f"3D MSE: {mse:.4f}, 3D CD: {chamfer:.4f}, 3D EMD: {emd:.4f}", end=" "
            )
            print(
                f"J-Score: {jscore:.4f}, F-Score: {fscore:.4f}, JF-Score: {jfscore:.4f}",
                end=" ",
            )
            print(
                f"perception: {perception:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, IoU: {iou:.4f}"
            )

        metric_list_list.append(metric_list)

    metric_list_list = np.array(metric_list_list)  # (n_episodes, n_frames, 9)

    metric_names = [
        "mse",
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
    metric_mean = np.mean(metric_list_list, axis=0)  # (n_frames, 9)
    metric_std = np.std(metric_list_list, axis=0)  # (n_frames, 9)
    metric_lower = metric_mean - metric_std
    metric_upper = metric_mean + metric_std

    # median_metric = np.median(metric_list_list, axis=0)
    # step_75_metric = np.percentile(metric_list_list, 75, axis=0)
    # step_25_metric = np.percentile(metric_list_list, 25, axis=0)

    for i, metric_name in enumerate(metric_names):
        # plot error
        x = np.arange(1, len(metric_mean) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(x, metric_mean[:, i])
        plt.xlabel(f"prediction steps, dt={cfg.sim.dt}")
        plt.ylabel(metric_name)
        plt.grid()

        ax = plt.gca()
        x = np.arange(1, len(metric_mean) + 1)
        ax.fill_between(x, metric_lower[:, i], metric_upper[:, i], alpha=0.2)

        plt.savefig(os.path.join(save_dir, f"{i:02d}-{metric_name}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="1103_paperbag")
    args = parser.parse_args()

    if args.name == "1103_paperbag":
        cfg_name = "experiments/log/1103_paperbag/train-debug-2-v-2x-radius-adaptive-dx0.02/hydra.yaml"
        with open(cfg_name, "r") as f:
            cfg = yaml.load(f, Loader=yaml.CLoader)
            cfg = OmegaConf.create(cfg)

        config_extra = {}
        config_extra["iteration"] = 100000
        config_extra["start_episode"] = 120
        config_extra["end_episode"] = 133
        config_extra["camera_id"] = 1

    main(cfg, config_extra)
