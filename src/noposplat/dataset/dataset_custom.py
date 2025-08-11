import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetCustomCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetCustomCfgWrapper:
    custom: DatasetCustomCfg


class DatasetCustom(IterableDataset):
    cfg: DatasetCustomCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetCustomCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def convert_camera_intrinsics(self, image: Float[Tensor, "w h"], camera: dict) -> Float[Tensor, "3 3"]:
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[0, 0] = camera["focal_x"] / image.shape[1]
        intrinsics[1, 1] = camera["focal_y"] / image.shape[2]
        intrinsics[0, 2] = camera["center_x"] / image.shape[1]
        intrinsics[1, 2] = camera["center_y"] / image.shape[2]
        return intrinsics

    def convert_camera_extrinsics(self, pose: dict) -> Float[Tensor, "4 4"]:
        extrinsics = torch.eye(4, dtype=torch.float32)
        qvec = torch.tensor(pose["rotation"], dtype=torch.float32)
        tvec = torch.tensor(pose["translation"], dtype=torch.float32)
        extrinsics[:3, :3] = qvec_to_mat(qvec)
        extrinsics[:3, 3] = tvec
        return extrinsics.inverse()

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        for root in self.cfg.roots:
            params = None
            with open(root / "image_params.json", mode="r", encoding="utf-8") as f:
                params = json.load(f)
            image_array = []
            intrinsics_array = []
            extrinsics_array = []
            for k, v in params.items():
                image_tensor = None
                with open(root / k, mode="rb") as f:
                    image = Image.open(f)
                    image_tensor = self.to_tensor(image)

                intrinsics = self.convert_camera_intrinsics(
                    image_tensor, v["camera"])
                extrinsics = self.convert_camera_extrinsics(v["pose"])
                image_array.append(image_tensor)
                intrinsics_array.append(intrinsics)
                extrinsics_array.append(extrinsics)

            images = torch.stack(image_array, dim=0)
            intrinsics = torch.stack(intrinsics_array, dim=0)
            extrinsics = torch.stack(extrinsics_array, dim=0)
            # TODO more from re10k
            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                print("Fov exceed!")
            if self.cfg.relative_pose:
                extrinsics = camera_normalization(
                    extrinsics[0:1], extrinsics)
            scale = 1.0
            example = {
                "context": {
                    "image": images[0:2],
                    "intrinsics": intrinsics[0:2],
                    "extrinsics": extrinsics[0:2],
                    "near": self.get_bound("near", len(images)) / scale,
                    "far": self.get_bound("far", len(images)) / scale,
                    "index": torch.arange(2, dtype=torch.int64),
                    "overlap": torch.tensor([0.5], dtype=torch.float32),
                },
                "scene": root.stem
            }
            yield example

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage


def qvec_to_mat(qvec: Float[Tensor, "4"]) -> Float[Tensor, "3 3"]:
    qvec = qvec / qvec.norm()
    w, x, y, z = qvec
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.tensor([
        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]
    ], dtype=torch.float32)
