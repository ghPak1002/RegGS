import json
import math
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh
from roma import unitquat_to_rotmat


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.nopo_enable = dataset_config.get("nopo_enable", False)
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]
        self.old_intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Re10KDataset(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.scene_name = dataset_config["scene_name"]

        input_path = Path(self.dataset_path) / self.scene_name

        self.load_dataset(input_path)

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))

    def load_dataset(self, input_path: Path):
        image_dir = input_path / "images"
        image_paths = sorted(list(image_dir.glob("*.png")),
                             key=lambda x: x.stem)
        self.images_paths = image_paths

        intrinsics_path = input_path / "intrinsics.json"
        with intrinsics_path.open("r", encoding="utf-8") as f:
            intrinsics = json.load(f)

        fx = intrinsics["fx"] * self.width
        fy = intrinsics["fy"] * self.height
        cx = intrinsics["cx"] * self.width
        cy = intrinsics["cy"] * self.height

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy

        self.intrinsics = intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        cameras_path = input_path / "cameras.json"
        with cameras_path.open("r", encoding="utf-8") as f:
            cameras = json.load(f)

        cam_quat = [cam["cam_quat"] for cam in cameras]
        cam_trans = [cam["cam_trans"] for cam in cameras]

        cam_quat = torch.tensor(cam_quat, dtype=torch.float32)
        cam_trans = torch.tensor(cam_trans, dtype=torch.float32)
        n = cam_quat.shape[0]

        rot_mat = unitquat_to_rotmat(cam_quat)

        c2ws = torch.stack([torch.eye(4, dtype=torch.float32)] * n).clone()

        c2ws[..., :3, :3] = rot_mat
        c2ws[..., :3, 3] = cam_trans

        c2ws = torch.inverse(c2ws[0]) @ c2ws

        self.poses = c2ws.numpy()

    def _load_image(self, index):
        image_path = self.images_paths[index]
        color_data = cv2.imread(str(image_path))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        return color_data

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, index):
        color_data = self._load_image(index)

        c2w = self.poses[index]

        return index, color_data, None, c2w


def get_dataset(dataset_name: str) -> type[Re10KDataset]:
    if dataset_name == "re10k":
        return Re10KDataset
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
