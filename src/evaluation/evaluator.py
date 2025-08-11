import json
from pathlib import Path

import evo.core.geometry as geometry
import matplotlib.pyplot as plt
import numpy as np
import roma
import seaborn
import torch
import torch.nn.functional as F
import torchvision
from matplotlib.colors import Normalize
from torchvision.utils import save_image
from tqdm import tqdm

from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.losses import l1_loss
from src.utils.align_utils import multiply_quaternions
from src.utils.io_utils import load_config
from src.utils.metrics_utils import compute_lpips, compute_psnr, compute_ssim
from src.utils.utils import (get_render_settings, render_gaussian_model,
                             setup_seed)


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    rot, trans, scale = geometry.umeyama_alignment(
        t_pred.T,
        t_gt.T,
        with_scale=True
    )
    t_align = scale * (t_pred @ rot.T) + trans
    return t_align


def pose_error(t_pred: np.ndarray, t_gt: np.ndarray):
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": np.sqrt(np.dot(trans_error, trans_error) / n),
        "mean": np.mean(trans_error),
        "median": np.median(trans_error),
        "std": np.std(trans_error),
        "min": np.min(trans_error),
        "max": np.max(trans_error)
    }


class Evaluator(object):
    def __init__(self, checkpoint_path, config_path) -> None:
        self.config = load_config(config_path)
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.device = "cuda"
        self.dataset = get_dataset(self.config["dataset_name"])(
            {**self.config["data"], **self.config["cam"]})
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = torch.tensor(np.array(self.dataset.poses)).to(
            torch.float32).to(self.device)

        self.estimated_c2ws = torch.load(
            self.checkpoint_path / "estimated_c2w.ckpt", weights_only=True).to(self.device)

        sample_rate = self.config["sample_rate"]
        n_views = self.config["n_views"]
        n_frames = len(self.dataset)
        frame_ids = np.arange(n_frames)
        test_frame_ids = frame_ids[int(sample_rate/2)::sample_rate]
        remain_frame_ids = np.array(
            [i for i in frame_ids if i not in test_frame_ids])
        train_frame_ids = remain_frame_ids[np.linspace(
            0, remain_frame_ids.shape[0] - 1, n_views).astype(int)]

        self.test_frame_ids = test_frame_ids
        self.train_frame_ids = train_frame_ids

        ply_path = Path(self.checkpoint_path) / \
            "gaussians" / "global_refined_gs.ply"
        refined_gaussians = GaussianModel(3)
        refined_gaussians.load_ply(ply_path)
        self.refined_gaussians = refined_gaussians

        print(f"Training Frames: {train_frame_ids}")
        print(f"Eval Frames: {test_frame_ids}")

    def eval_train_render(self):
        print("Evaluating train render...")
        train_frame_ids = self.train_frame_ids
        width, height = self.dataset.width, self.dataset.height
        intrinsics = self.dataset.intrinsics
        gaussian_model = self.refined_gaussians
        transform = torchvision.transforms.ToTensor()

        metrics_list = []

        for idx, frame_id in enumerate(train_frame_ids):
            estimated_c2w = self.estimated_c2ws[idx]
            w2c = torch.inverse(estimated_c2w)
            render_settings = get_render_settings(
                width, height, intrinsics, w2c.cpu().numpy())
            render_dict = render_gaussian_model(
                gaussian_model, render_settings)

            gt_color = transform(self.dataset[frame_id][1]).to(self.device)

            rendered_color = render_dict["color"]

            rendered_color = rendered_color.clamp(0, 1)

            psnr = compute_psnr(gt_color, rendered_color)
            ssim = compute_ssim(gt_color, rendered_color)
            lpips = compute_lpips(gt_color, rendered_color)

            metrics_list.append({
                "frame_id": f"{frame_id:04d}",
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
            })
            print(
                f"Frame {frame_id:04d} PSNR: {psnr.item():.2f}, SSIM: {ssim.item():.2f}, LPIPS: {lpips.item():.2f}")

            image_path = self.checkpoint_path / \
                "train" / f"color_{frame_id:04d}.png"
            image_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(rendered_color,
                       image_path)

            image_path = self.checkpoint_path / \
                "gt" / f"gt_{frame_id:04d}.png"
            image_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(gt_color, image_path)

        metrics_path = self.checkpoint_path / "eval_train.json"
        metrics_path.parent.mkdir(exist_ok=True, parents=True)
        avg_psnr = np.array([x["psnr"] for x in metrics_list]).mean()
        avg_ssim = np.array([x["ssim"] for x in metrics_list]).mean()
        avg_lpips = np.array([x["lpips"] for x in metrics_list]).mean()

        metrics = {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_lpips": avg_lpips,
            "metrics": metrics_list,
        }
        print(
            f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Average LPIPS: {avg_lpips}")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

    def find_nearest_train_frame(self, sorted_train_ids, test_ids):
        idx = np.searchsorted(sorted_train_ids, test_ids)
        idx = np.clip(idx, 0, len(sorted_train_ids) - 1)
        candidate_idx = np.stack([
            np.clip(idx - 1, 0, len(sorted_train_ids) - 1),
            np.clip(idx,     0, len(sorted_train_ids) - 1),
            np.clip(idx + 1, 0, len(sorted_train_ids) - 1)], axis=1)

        dist = np.abs(sorted_train_ids[candidate_idx] - test_ids[:, None])
        order = np.argsort(dist, axis=1)
        row = np.arange(test_ids.shape[0])[:, None]
        nearest_idx_pair = candidate_idx[row, order[:, :2]]

        dup_mask = nearest_idx_pair[:, 0] == nearest_idx_pair[:, 1]
        if dup_mask.any():
            nearest_idx_pair[dup_mask,
                             1] = candidate_idx[dup_mask, order[dup_mask, 2]]

        nearest_idx_pair = np.sort(nearest_idx_pair, axis=1)
        nearest_frame_pair = sorted_train_ids[nearest_idx_pair]

        return nearest_idx_pair, nearest_frame_pair

    @torch.no_grad()
    def compute_estimated_test_c2ws(self, gt_c2ws, estimated_train_c2ws, train_frame_ids, test_frame_ids):

        train_id_pairs, train_frame_pairs = self.find_nearest_train_frame(
            train_frame_ids, test_frame_ids)
        # print(f"Train ID Pairs: {train_id_pairs}")
        # print(f"Train Frame Pairs: {train_frame_pairs}")
        ref_gt_train_c2ws = gt_c2ws[train_frame_pairs]
        ref_est_train_c2ws = estimated_train_c2ws[train_id_pairs]

        gt_test_c2ws = gt_c2ws[test_frame_ids]

        rel_gt_train_c2ws = ref_gt_train_c2ws[:,
                                              1] @ torch.inverse(ref_gt_train_c2ws[:, 0])
        rel_est_train_c2ws = ref_est_train_c2ws[:,
                                                1] @ torch.inverse(ref_est_train_c2ws[:, 0])

        rel_gt_test_c2ws = gt_test_c2ws @ torch.inverse(
            ref_gt_train_c2ws[:, 0])

        rel_gt_train_trans = rel_gt_train_c2ws[..., :3, 3]
        rel_est_train_trans = rel_est_train_c2ws[..., :3, 3]

        rel_gt_train_dist = torch.norm(rel_gt_train_trans, dim=-1)
        rel_est_train_dist = torch.norm(rel_est_train_trans, dim=-1)

        scale = rel_est_train_dist / rel_gt_train_dist

        rel_est_test_c2ws = rel_gt_test_c2ws.clone()
        rel_est_test_c2ws[..., :3,
                          3] = rel_est_test_c2ws[..., :3, 3] * scale[:, None]

        est_test_c2ws = rel_est_test_c2ws @ ref_est_train_c2ws[:, 0]

        return est_test_c2ws

    @torch.set_grad_enabled(True)
    def optimize_estimated_c2w(self, model: GaussianModel, est_c2w: torch.Tensor, frame_id: int):
        print(f"Optimizing pose for frame {frame_id:04d}...")
        dataset = self.dataset
        width = dataset.width
        height = dataset.height
        intrinsics = dataset.intrinsics
        device = self.device

        w2c = torch.inverse(est_c2w)

        render_settings = get_render_settings(
            width, height, intrinsics, w2c.cpu().numpy())

        transform = torchvision.transforms.ToTensor()

        gt_color = transform(self.dataset[frame_id][1]).to(device)

        opt_rot = torch.zeros(4)
        opt_rot[-1] = 1.0
        opt_rot = torch.nn.Parameter(
            opt_rot.detach().clone().to(device), requires_grad=True)
        opt_trans = torch.nn.Parameter(
            torch.zeros(3).to(device), requires_grad=True)
        model.training_setup_camera(
            opt_rot, opt_trans, self.config["aligner"])

        min_loss = float("inf")
        n_iter = 400
        pbar = tqdm(range(n_iter))
        for _ in pbar:
            norm_quat = F.normalize(opt_rot, dim=0)
            rel_R = roma.unitquat_to_rotmat(norm_quat)
            rel_T = opt_trans

            xyz = model.get_xyz()
            xyz = (xyz @ rel_R.transpose(1, 0)) + rel_T
            rotations = model.get_rotation()
            rotations = multiply_quaternions(
                rotations, roma.quat_xyzw_to_wxyz(norm_quat))

            render_dict = render_gaussian_model(
                model, render_settings, override_means_3d=xyz, override_rotations=rotations)
            rendered_color = render_dict["color"]
            loss = l1_loss(rendered_color, gt_color, agg="sum")

            with torch.no_grad():
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_w2c = torch.eye(4, dtype=torch.float32, device=device)
                    best_w2c[:3, :3] = roma.unitquat_to_rotmat(
                        norm_quat.clone().detach())
                    best_w2c[:3, 3] = opt_trans.clone().detach()
                pbar.set_postfix(loss=loss.item())
                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

        final_w2c = w2c @ best_w2c
        final_c2w = torch.inverse(final_w2c)
        final_c2w[3, :].fill_(0.0)
        final_c2w[3, 3] = 1.0
        return final_c2w

    def eval_test_render(self):
        print("Evaluating test render...")
        test_frame_ids = self.test_frame_ids
        train_frame_ids = self.train_frame_ids
        width, height = self.dataset.width, self.dataset.height
        intrinsics = self.dataset.intrinsics
        model = self.refined_gaussians
        transform = torchvision.transforms.ToTensor()

        gt_c2ws = torch.tensor(self.dataset.poses).to(self.device)
        estimated_c2ws = self.estimated_c2ws

        est_test_c2ws = self.compute_estimated_test_c2ws(
            gt_c2ws, estimated_c2ws, train_frame_ids, test_frame_ids)

        metrics_list = []
        for idx, frame_id in enumerate(test_frame_ids):
            est_c2w = est_test_c2ws[idx]
            est_c2w = self.optimize_estimated_c2w(model, est_c2w, frame_id)
            w2c = torch.inverse(est_c2w)
            render_settings = get_render_settings(
                width, height, intrinsics, w2c.cpu().numpy())
            render_dict = render_gaussian_model(
                model, render_settings)

            gt_color = transform(self.dataset[frame_id][1]).to(self.device)

            rendered_color = render_dict["color"]

            rendered_color = rendered_color.clamp(0, 1)

            psnr = compute_psnr(gt_color, rendered_color)
            ssim = compute_ssim(gt_color, rendered_color)
            lpips = compute_lpips(gt_color, rendered_color)

            metrics_list.append({
                "frame_id": f"{frame_id:04d}",
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
            })
            print(
                f"Frame {frame_id:04d} PSNR: {psnr.item():.2f}, SSIM: {ssim.item():.2f}, LPIPS: {lpips.item():.2f}")

            image_path = self.checkpoint_path / \
                "test" / f"color_{frame_id:04d}.png"
            image_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(rendered_color,
                       image_path)

            image_path = self.checkpoint_path / \
                "gt" / f"gt_{frame_id:04d}.png"
            image_path.parent.mkdir(exist_ok=True, parents=True)
            save_image(gt_color, image_path)

        metrics_path = self.checkpoint_path / "eval_test.json"
        metrics_path.parent.mkdir(exist_ok=True, parents=True)
        avg_psnr = np.array([x["psnr"] for x in metrics_list]).mean()
        avg_ssim = np.array([x["ssim"] for x in metrics_list]).mean()
        avg_lpips = np.array([x["lpips"] for x in metrics_list]).mean()

        metrics = {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_lpips": avg_lpips,
            "metrics": metrics_list,
        }
        print(
            f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Average LPIPS: {avg_lpips}")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

    @torch.no_grad()
    def eval_trajectory(self):
        print("Evaluating trajectory...")
        est_c2ws = self.estimated_c2ws.cpu().numpy()
        gt_c2ws = self.dataset.poses
        train_frame_ids = self.train_frame_ids
        gt_c2ws = gt_c2ws[train_frame_ids]

        # Truncate the ground truth trajectory if needed
        if gt_c2ws.shape[0] > est_c2ws.shape[0]:
            gt_c2ws = gt_c2ws[:est_c2ws.shape[0]]

        gt_t = gt_c2ws[:, :3, 3]
        est_t = est_c2ws[:, :3, 3]
        est_t_aligned = align_trajectories(est_t, gt_t)
        ate = pose_error(est_t_aligned, gt_t)

        with open(str(self.checkpoint_path / "ate_aligned.json"), "w") as f:
            f.write(json.dumps(ate))

        ate_rmse = ate["rmse"]
        print(f"ATE-RMSE (aligned): {ate_rmse * 100:.2f} cm")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.plot(gt_t[:, 0], gt_t[:, 2], 'k--', label='gt', linewidth=4)
        norm = Normalize(vmin=0, vmax=0.1)
        cmap = plt.get_cmap("jet")
        for i in range(est_t_aligned.shape[0] - 1):
            local_rmse = np.sqrt(np.mean((gt_t[i] - est_t_aligned[i]) ** 2))
            ax.plot(
                [est_t_aligned[i, 0], est_t_aligned[i + 1, 0]],
                [est_t_aligned[i, 2], est_t_aligned[i + 1, 2]],
                color=cmap(norm(local_rmse)),
                linewidth=4
            )
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.tick_params(labelsize=0)  # 颜色条数字字体增大
        # 设置颜色条不显示数字
        cbar.ax.set_yticks([])  # 移除颜色条的数字

        # 设置标题和标签
        ax.set_title(f"ATE RMSE: {ate_rmse:.3f}", fontsize=64)  # 标题字体增大
        # ax.set_xlabel("X", fontsize=24)  # X轴标签字体增大
        # ax.set_ylabel("Y", fontsize=24)  # Y轴标签字体增大
        ax.legend(fontsize=64, loc='upper left',
                  bbox_to_anchor=(0, 1))  # 图例放在左上角

        # 设置统一的 x 和 y 坐标轴范围
        min_x, max_x = np.min(est_t_aligned[:, 0]), np.max(est_t_aligned[:, 0])
        min_y, max_y = np.min(est_t_aligned[:, 2]), np.max(est_t_aligned[:, 2])
        min_x -= np.abs(max_x - min_x) * 0.3
        max_x += np.abs(max_x - min_x) * 0.3
        min_y -= np.abs(max_y - min_y) * 0.3
        max_y += np.abs(max_y - min_y) * 0.3

        ax.set_xlim((min_x, max_x))
        ax.set_ylim((min_y, max_y))

        # 不显示坐标轴的刻度和数字，但保留网格背景
        ax.tick_params(
            axis='both',  # 应用于 X 和 Y 轴
            which='both',  # 同时隐藏主刻度和次刻度
            bottom=False, top=False, left=False, right=False,  # 隐藏刻度线
            labelbottom=False, labelleft=False  # 隐藏刻度数字
        )

        # 调整坐标轴刻度的字体大小
        ax.tick_params(axis='both', which='major', labelsize=24)

        # 移除颜色条的标签
        cbar.set_label("")  # 移除"Error"标签

        # 调整布局以减少右边的空白
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.tight_layout()

        save_path = self.checkpoint_path / Path(f"eval_traj.png")
        plt.savefig(save_path)
        plt.close(fig)
