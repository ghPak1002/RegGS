from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision

from src.entities.datasets import BaseDataset
from src.entities.gaussian_model import GaussianModel, MiniGaussianModel
from src.utils.integrator_utils import calc_psnr
from src.utils.utils import get_render_settings, render_gaussian_model


class GaussianIntegrator(object):
    def __init__(self, config: dict,
                 dataset: BaseDataset,
                 device: torch.device) -> None:
        self.config = config
        self.dataset = dataset
        self.device = device

    def vis_integrate_result(self,
                             frame_id_i: int,
                             frame_id_j: int,
                             gt_color_i: torch.Tensor,
                             gt_color_j: torch.Tensor,
                             render_image_i: torch.Tensor,
                             render_image_j: torch.Tensor,
                             render_depth_i: torch.Tensor,
                             render_depth_j: torch.Tensor):

        output_dir = Path(self.config["data"]["output_path"])
        output_dir = output_dir / "vis_integrate"
        output_dir.mkdir(exist_ok=True, parents=True)
        n_rows = 2
        n_cols = 3
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        gt_color_i = gt_color_i.clamp(0, 1).permute(
            1, 2, 0).detach().cpu().numpy()
        render_image_i = render_image_i.clamp(
            0, 1).permute(1, 2, 0).detach().cpu().numpy()
        render_depth_i = render_depth_i[0].detach().cpu().numpy()
        gt_color_j = gt_color_j.clamp(0, 1).permute(
            1, 2, 0).detach().cpu().numpy()
        render_image_j = render_image_j.clamp(
            0, 1).permute(1, 2, 0).detach().cpu().numpy()
        render_depth_j = render_depth_j[0].detach().cpu().numpy()

        axs[0, 0].imshow(gt_color_i)
        axs[0, 0].set_title(f"GT {frame_id_i}", fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[0, 1].imshow(render_image_i)
        axs[0, 1].set_title(f"Render {frame_id_i}", fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        axs[0, 2].imshow(render_depth_i, cmap="jet", vmin=0, vmax=6)
        axs[0, 2].set_title(f"Render Depth {frame_id_i}", fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        axs[1, 0].imshow(gt_color_j)
        axs[1, 0].set_title(f"GT {frame_id_j}", fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])

        axs[1, 1].imshow(render_image_j)
        axs[1, 1].set_title(f"Render {frame_id_j}", fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        axs[1, 2].imshow(render_depth_j, cmap="jet", vmin=0, vmax=6)
        axs[1, 2].set_title(f"Render Depth {frame_id_j}", fontsize=16)
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        fig_title = f"Frame {frame_id_i:04d} and Frame {frame_id_j:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)

        plt.subplots_adjust(top=0.90)
        fig.tight_layout()
        plt.savefig(
            output_dir / f"{frame_id_i:04d}_{frame_id_j:04d}.png", dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()

    @torch.no_grad()
    def log_integrate_result(self,
                             frame_id_i: int,
                             frame_id_j: int,
                             c2w_i: torch.Tensor,
                             c2w_j: torch.Tensor,
                             model: GaussianModel):
        gt_color_i = self.dataset[frame_id_i][1]
        gt_color_j = self.dataset[frame_id_j][1]
        w2c_i = torch.inverse(c2w_i)
        w2c_j = torch.inverse(c2w_j)
        color_transform = torchvision.transforms.ToTensor()
        width = self.dataset.width
        height = self.dataset.height
        intrinsics = self.dataset.intrinsics

        gt_color_i = color_transform(gt_color_i).to(self.device)
        gt_color_j = color_transform(gt_color_j).to(self.device)

        render_settings_i = get_render_settings(
            width,
            height,
            intrinsics,
            w2c_i.detach().cpu().numpy())

        render_pkg_i = render_gaussian_model(
            model, render_settings_i)
        render_image_i = render_pkg_i["color"]
        render_depth_i = render_pkg_i["depth"]

        render_settings_j = get_render_settings(
            width,
            height,
            intrinsics,
            w2c_j.detach().cpu().numpy())

        render_pkg_j = render_gaussian_model(
            model, render_settings_j)
        render_image_j = render_pkg_j["color"]
        render_depth_j = render_pkg_j["depth"]

        psnr_i = calc_psnr(gt_color_i, render_image_i).item()
        psnr_j = calc_psnr(gt_color_j, render_image_j).item()

        print(
            f"PSNR {frame_id_i:04d}: {psnr_i:.2f}, {frame_id_j:04d}: {psnr_j:.2f}")

        self.vis_integrate_result(frame_id_i, frame_id_j, gt_color_i, gt_color_j,
                                  render_image_i, render_image_j, render_depth_i, render_depth_j)

    @torch.no_grad()
    def integrator(self,
                   frame_id_i: int,
                   frame_id_j: int,
                   c2w_i: torch.Tensor,
                   c2w_j: torch.Tensor,
                   mini: MiniGaussianModel,
                   model: GaussianModel):
        mini = mini.rigid_transform_gaussians(c2w_i)
        model.add_mini_gaussians(mini)
        self.log_integrate_result(frame_id_i, frame_id_j, c2w_i, c2w_j, model)
