import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
import glob
import json

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import meta_material
from meta_material.utils import get_root, mkdir
from meta_material.ffmpeg import make_video

from train.droid.visualize_zed_depth import get_zed_depth


class DroidDataset(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = list(tf_dataset.as_numpy_iterator())  # Convert to list

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        x, y = self.tf_dataset[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def preprocess_droid_rlds(dataset_name, dataset_dir, dataset_save_name):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    # builder = tfds.builder_from_directory(builder_dir=f"{dataset_dir}/{dataset_name}/1.0.0")
    # builder.info.features
    # FeaturesDict({
    #     'episode_metadata': FeaturesDict({
    #         'file_path': string,
    #         'recording_folderpath': string,
    #     }),
    #     'steps': Dataset({
    #         'action': Tensor(shape=(7,), dtype=float64),
    #         'action_dict': FeaturesDict({
    #             'cartesian_position': Tensor(shape=(6,), dtype=float64),
    #             'cartesian_velocity': Tensor(shape=(6,), dtype=float64),
    #             'gripper_position': Tensor(shape=(1,), dtype=float64),
    #             'gripper_velocity': Tensor(shape=(1,), dtype=float64),
    #             'joint_position': Tensor(shape=(7,), dtype=float64),
    #             'joint_velocity': Tensor(shape=(7,), dtype=float64),
    #         }),
    #         'discount': Scalar(shape=(), dtype=float32),
    #         'is_first': bool,
    #         'is_last': bool,
    #         'is_terminal': bool,
    #         'language_instruction': string,
    #         'language_instruction_2': string,
    #         'language_instruction_3': string,
    #         'observation': FeaturesDict({
    #             'cartesian_position': Tensor(shape=(6,), dtype=float64),
    #             'exterior_image_1_left': Image(shape=(180, 320, 3), dtype=uint8),
    #             'exterior_image_2_left': Image(shape=(180, 320, 3), dtype=uint8),
    #             'gripper_position': Tensor(shape=(1,), dtype=float64),
    #             'joint_position': Tensor(shape=(7,), dtype=float64),
    #             'wrist_image_left': Image(shape=(180, 320, 3), dtype=uint8),
    #         }),
    #         'reward': Scalar(shape=(), dtype=float32),
    #     }),
    # })

    # create dataset dir
    mkdir(Path(dataset_dir) / dataset_save_name, resume=True, overwrite=False)

    # load dataset
    dataset = tfds.load(dataset_name, data_dir=dataset_dir, split="train")

    for episode_id, episode in enumerate(dataset):  # .take(len(dataset))
        # create episode dir
        episode_dir = (
            Path(dataset_dir) / dataset_save_name / f"episode_{episode_id:04d}"
        )
        mkdir(episode_dir, resume=True, overwrite=False)
        mkdir(episode_dir / "camera_0" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_1" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_2" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "robot", resume=True, overwrite=False)
        mkdir(episode_dir / "robot_extra", resume=True, overwrite=False)

        for step_id, step in enumerate(episode["steps"]):
            # print(episode_id, step_id)
            ext_img_1_left = step["observation"]["exterior_image_1_left"].numpy()
            ext_img_2_left = step["observation"]["exterior_image_2_left"].numpy()
            wrist_image = step["observation"]["wrist_image_left"].numpy()
            action = step["action"].numpy()
            action_dict = step["action_dict"]
            cartesian_position = action_dict["cartesian_position"].numpy()
            cartesian_velocity = action_dict["cartesian_velocity"].numpy()
            gripper_position = action_dict["gripper_position"].numpy()
            gripper_velocity = action_dict["gripper_velocity"].numpy()
            joint_position = action_dict["joint_position"].numpy()
            joint_velocity = action_dict["joint_velocity"].numpy()
            instruction = step["language_instruction"]

            # print(image.shape, wrist_image.shape)
            cv2.imwrite(
                str(episode_dir / "camera_0" / "rgb" / f"{step_id:06d}.png"),
                cv2.cvtColor(ext_img_1_left, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(episode_dir / "camera_1" / "rgb" / f"{step_id:06d}.png"),
                cv2.cvtColor(ext_img_2_left, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(episode_dir / "camera_2" / "rgb" / f"{step_id:06d}.png"),
                cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR),
            )

            robot = np.concatenate(
                [
                    cartesian_position,
                    cartesian_velocity,
                    gripper_position,
                    gripper_velocity,
                    joint_position,
                    joint_velocity,
                ]
            )
            np.savetxt(str(episode_dir / "robot" / f"{step_id:06d}.txt"), robot)


def vis_video(dataset_name, dataset_dir, dataset_save_name):
    video_path = Path(dataset_dir) / dataset_save_name / "video"
    mkdir(video_path, resume=True, overwrite=False)

    n_episodes = len(list((Path(dataset_dir) / dataset_save_name).glob("episode_*")))
    for episode_id in range(n_episodes):
        episode_dir = (
            Path(dataset_dir) / dataset_save_name / f"episode_{episode_id:04d}"
        )
        n_steps = len(list((episode_dir / "camera_0" / "rgb").glob("*.png")))
        make_video(
            episode_dir / "camera_0" / "rgb",
            video_path / f"episode_{episode_id:04d}_camera_0.mp4",
            "%06d.png",
            30,
        )
        make_video(
            episode_dir / "camera_1" / "rgb",
            video_path / f"episode_{episode_id:04d}_camera_1.mp4",
            "%06d.png",
            30,
        )
        make_video(
            episode_dir / "camera_2" / "rgb",
            video_path / f"episode_{episode_id:04d}_camera_2.mp4",
            "%06d.png",
            30,
        )


def ext2mat(ext):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("xyz", ext[3:]).as_matrix()
    extrinsic[:3, 3] = ext[:3]
    return extrinsic


def preprocess_droid_raw(dataset_name, dataset_dir, dataset_save_name, input_paths):
    dataset_path = Path(dataset_dir) / dataset_name
    dataset_path_save = Path(dataset_dir) / dataset_save_name
    mkdir(dataset_path_save, resume=True, overwrite=False)

    n_episodes = len(input_paths)
    for episode_id in range(n_episodes):
        input_path = dataset_path / input_paths[episode_id]

        # create episode dir
        episode_dir = dataset_path_save / f"episode_{episode_id:04d}"
        mkdir(episode_dir, resume=True, overwrite=False)
        mkdir(episode_dir / "camera_0" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_1" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_2" / "rgb", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_0" / "depth", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_1" / "depth", resume=True, overwrite=False)
        mkdir(episode_dir / "camera_2" / "depth", resume=True, overwrite=False)
        mkdir(episode_dir / "robot", resume=True, overwrite=False)

        # load data
        input_path_recording = input_path / "recordings"
        assert len(glob.glob(str(input_path / "metadata*"))) == 1
        input_path_metadata = glob.glob(str(input_path / "metadata*"))[0]

        with open(input_path_metadata, "r") as f:
            metadata = json.load(f)

        uuid = metadata["uuid"]
        n_steps = metadata.get("n_steps", None)
        trajectory_length = metadata["trajectory_length"]

        wrist_cam_serial = metadata["wrist_cam_serial"]
        ext1_cam_serial = metadata["ext1_cam_serial"]
        ext2_cam_serial = metadata["ext2_cam_serial"]
        wrist_cam_intrinsics = metadata.get("wrist_cam_intrinsics", None)
        ext1_cam_intrinsics = metadata.get("ext1_cam_intrinsics", None)
        ext2_cam_intrinsics = metadata.get("ext2_cam_intrinsics", None)
        wrist_svo_path = metadata["wrist_svo_path"]
        ext1_svo_path = metadata["ext1_svo_path"]
        ext2_svo_path = metadata["ext2_svo_path"]
        wrist_mp4_path = metadata["wrist_mp4_path"]
        ext1_mp4_path = metadata["ext1_mp4_path"]
        ext2_mp4_path = metadata["ext2_mp4_path"]

        get_zed_depth(str(input_path))
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    # processing the droid_100 RLDS dataset
    # dataset_dir = 'experiments/log/data'
    # dataset_name = 'droid_100'
    # dataset_save_name = 'droid_100_processed'
    # preprocess_droid_rlds(dataset_name, dataset_dir, dataset_save_name)
    # vis_video(dataset_name, dataset_dir, dataset_save_name)

    dataset_dir = "experiments/log"
    dataset_name = "droid_raw"
    dataset_save_name = "droid_raw_processed"

    input_paths = [
        "1.0.1/AUTOLab/success/2023-07-07/Fri_Jul__7_09:55:14_2023",
    ]
    preprocess_droid_raw(dataset_name, dataset_dir, dataset_save_name, input_paths)
