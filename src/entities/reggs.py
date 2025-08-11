from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision
from einops import repeat

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_aligner import GaussianAligner
from src.entities.gaussian_integrator import GaussianIntegrator
from src.entities.gaussian_model import GaussianModel, MiniGaussianModel
from src.noposplat.model.types import Gaussians as NoPoGaussians
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.nopo_utils import (compute_gaussians, compute_pose,
                                  get_decoder_loss, get_nopo_decoder,
                                  get_nopo_encoder,
                                  scale_gaussians_by_render_depth_max)
from src.utils.utils import setup_seed


class RegGS:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        self.nopo_encoder = get_nopo_encoder(
            self.device, config["nopo_checkpoint"])
        self.nopo_decoder = get_nopo_decoder(self.device)
        self.nopo_losses = get_decoder_loss(self.device)

        self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])(
            {**config["data"], **config["cam"]})

        # 初始化训练帧与测试帧
        sample_rate = config["sample_rate"]
        n_views = config["n_views"]
        n_frames = len(self.dataset)
        frame_ids = np.arange(n_frames)
        test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]
        remain_frame_ids = np.array(
            [i for i in frame_ids if i not in test_frame_ids])
        train_frame_ids = remain_frame_ids[np.linspace(
            0, remain_frame_ids.shape[0] - 1, n_views).astype(int)]

        self.test_frame_ids = test_frame_ids
        self.train_frame_ids = train_frame_ids

        print(f"Training Frames: {train_frame_ids.tolist()}")
        print(f"Eval Frames: {test_frame_ids.tolist()}")

        self.estimated_c2ws = torch.empty(train_frame_ids.shape[0], 4, 4)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.opt = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        new_submap_every = config["new_submap_every"]
        new_submap_frame_ids = train_frame_ids[::new_submap_every]
        self.new_submap_frame_ids = new_submap_frame_ids[1:]

        config["aligner"]["output_path"] = config["data"]["output_path"]
        self.integrator = GaussianIntegrator(config, self.dataset, self.device)
        self.aligner = GaussianAligner(
            config["aligner"], self.dataset, self.device,
            nopo_encoder=self.nopo_encoder,
            nopo_decoder=self.nopo_decoder,
            nopo_losses=self.nopo_losses,)

    def compute_gaussians_and_rel_c2w(self, frame_id_i, frame_id_j) -> tuple[NoPoGaussians, torch.Tensor]:
        """
        return
        NoPoGaussians
        torch.Tensor [4, 4]
        """
        device = self.device

        gt_color_i = self.dataset[frame_id_i][1]
        gt_color_j = self.dataset[frame_id_j][1]

        color_transform = torchvision.transforms.ToTensor()
        gt_color_i = color_transform(gt_color_i).to(device)
        gt_color_j = color_transform(gt_color_j).to(device)
        width = self.dataset.width
        height = self.dataset.height

        nopo_intrinsics = torch.tensor(
            self.dataset.intrinsics, dtype=torch.float32, device=device)
        assert len(nopo_intrinsics.shape) == 2
        nopo_intrinsics[0, 0] = nopo_intrinsics[0, 0] / width
        nopo_intrinsics[1, 1] = nopo_intrinsics[1, 1] / height
        nopo_intrinsics[0, 2] = nopo_intrinsics[0, 2] / width
        nopo_intrinsics[1, 2] = nopo_intrinsics[1, 2] / height

        images = torch.stack([
            gt_color_i,
            gt_color_j,
        ], dim=0).to(device)  # [2 3 h w]

        # nopo计算高斯
        nopo_gaussians = compute_gaussians(
            self.nopo_encoder, images.unsqueeze(0), repeat(nopo_intrinsics, "r c -> 1 v r c", v=2))

        # nopo计算第二张图片的位姿
        nopo_c2w_rel_j = compute_pose(self.nopo_decoder, self.nopo_losses, nopo_gaussians,
                                      images[1, ...].unsqueeze(0), nopo_intrinsics.unsqueeze(0))
        nopo_c2w_rel_j = nopo_c2w_rel_j.squeeze(0)

        # 调整生成高斯的尺度与位姿
        nopo_scales = scale_gaussians_by_render_depth_max(
            self.nopo_decoder, nopo_gaussians, nopo_intrinsics.unsqueeze(0), height, width)
        nopo_scales = nopo_scales[0]
        nopo_c2w_rel_j[0:3, 3] \
            = nopo_c2w_rel_j[0:3, 3] * nopo_scales
        print(
            f"Normalize NoPo Gaussians by Scale: {nopo_scales.item():.5f}")
        return nopo_gaussians, nopo_c2w_rel_j, nopo_scales

    def align_sub_gaussians(self, frame_id_i,
                            frame_id_j,
                            estimated_c2w_i,
                            rel_c2w_j,
                            gaussian_model,
                            mini_gaussians) -> tuple[MiniGaussianModel, torch.Tensor, torch.Tensor]:
        mini_gaussians, estimated_c2w_i, estimated_c2w_j = self.aligner.align_sub_gaussians(
            frame_id_i, frame_id_j, gaussian_model, mini_gaussians, estimated_c2w_i, rel_c2w_j)
        return mini_gaussians, estimated_c2w_i, estimated_c2w_j

    def save_current_submap(self, gaussian_model: GaussianModel):
        submap_dir = self.output_path / "submaps"
        submap_dir.mkdir(exist_ok=True, parents=True)
        submap_ckpt = gaussian_model.capture_dict()
        save_dict_to_ckpt(
            submap_ckpt, f"submap_{self.submap_id:04d}.ckpt", directory=submap_dir)

    def start_new_submap(self) -> None:
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id += 1
        return gaussian_model

    def run(self):
        device = self.device

        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0
        nopo_gaussians = None
        rel_c2w_j = None

        train_frame_ids = self.train_frame_ids
        for i, (frame_id_i, frame_id_j) in enumerate(zip(train_frame_ids[:-1], train_frame_ids[1:])):
            nopo_gaussians, rel_c2w_j, normalize_scale = self.compute_gaussians_and_rel_c2w(
                frame_id_i, frame_id_j)
            mini_gaussians = MiniGaussianModel.from_nopo_gaussians(
                nopo_gaussians)
            mini_gaussians.store_normalize_scale(normalize_scale)

            if i == 0:
                # first frame doesn't need to align
                estimated_c2w_i = torch.eye(
                    4, dtype=torch.float32, device=device)
                estimated_c2w_j = estimated_c2w_i @ rel_c2w_j

                self.estimated_c2ws[i] = estimated_c2w_i.detach(
                ).cpu()
                self.estimated_c2ws[i + 1] = estimated_c2w_j.detach(
                ).cpu()
            else:
                estimated_c2w_i = self.estimated_c2ws[i].clone().to(
                    device)
                mini_gaussians, estimated_c2w_i, estimated_c2w_j = \
                    self.align_sub_gaussians(frame_id_i, frame_id_j, estimated_c2w_i,
                                             rel_c2w_j, gaussian_model, mini_gaussians)
                self.estimated_c2ws[i] = estimated_c2w_i.detach(
                ).cpu()
                self.estimated_c2ws[i + 1] = estimated_c2w_j.detach(
                ).cpu()

            if frame_id_i in self.new_submap_frame_ids:
                self.save_current_submap(gaussian_model)
                gaussian_model = self.start_new_submap()

            print(
                f"Integrate sub gaussians ({frame_id_i:04d},{frame_id_j:04d}) to main gaussians")
            gaussian_model.training_setup(self.opt)
            self.integrator.integrator(frame_id_i,
                                       frame_id_j,
                                       estimated_c2w_i,
                                       estimated_c2w_j,
                                       mini_gaussians,
                                       gaussian_model)

        self.output_path.mkdir(exist_ok=True, parents=True)
        save_dict_to_ckpt(
            self.estimated_c2ws, "estimated_c2w.ckpt", directory=self.output_path)
