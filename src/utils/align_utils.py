from typing import Union

import numpy as np
import torch
from einops import rearrange, repeat
from gaussian_rasterizer import GaussianRasterizer
from scipy.spatial.transform import Rotation

from src.entities.gaussian_model import GaussianModel, MiniGaussianModel
from src.utils.nopo_utils import render_gaussians as _nopo_render_gaussians
from src.utils.utils import np2torch


def multiply_quaternions(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    y = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    z = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    return torch.stack((w, x, y, z), dim=-1)


def transformation_to_quaternion(RT: Union[torch.Tensor, np.ndarray]):
    gpu_id = -1
    if isinstance(RT, torch.Tensor):
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    rot = Rotation.from_matrix(R)
    quad = rot.as_quat(canonical=True)
    quad = np.roll(quad, 1)
    tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def extrapolate_poses(poses: np.ndarray) -> np.ndarray:
    return poses[1, :] @ np.linalg.inv(poses[0, :]) @ poses[1, :]


def compute_camera_opt_params(estimate_rel_w2c: np.ndarray) -> tuple:
    quaternion = Rotation.from_matrix(
        estimate_rel_w2c[:3, :3]).as_quat(canonical=True)
    quaternion = quaternion[[3, 0, 1, 2]]
    opt_cam_rot = torch.nn.Parameter(np2torch(quaternion, "cuda"))
    opt_cam_trans = torch.nn.Parameter(
        np2torch(estimate_rel_w2c[:3, 3], "cuda"))
    return opt_cam_rot, opt_cam_trans


def join_render_mini_gaussian(model, mini, render_settings):
    renderer = GaussianRasterizer(raster_settings=render_settings)
    
    mini_means3D = mini.xyz
    mini_means2D = torch.zeros_like(
        mini_means3D, dtype=mini_means3D.dtype, requires_grad=True, device=mini_means3D.device)
    mini_opacities = torch.sigmoid(mini.opacities)
    mini_shs = mini.harmonics
    mini_scales = torch.exp(mini.scales)
    mini_rotations = torch.nn.functional.normalize(mini.rotations)

    main_means3D = model.xyz
    main_means2D = torch.zeros_like(
        main_means3D, dtype=main_means3D.dtype, requires_grad=True, device=main_means3D.device)
    main_opacities = torch.sigmoid(model.opacities)
    main_shs = model.harmonics
    main_scales = torch.exp(model.scales)
    main_rotations = torch.nn.functional.normalize(model.rotations)

    means3D = torch.cat([mini_means3D, main_means3D])
    means2D = torch.cat([mini_means2D, main_means2D])
    opacities = torch.cat([mini_opacities, main_opacities])
    shs = torch.cat([mini_shs, main_shs])
    scales = torch.cat([mini_scales, main_scales])
    rotations = torch.cat([mini_rotations, main_rotations])

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": None,
        "shs": shs,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None
    }

    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def render_mini_gaussian(mini_gaussian, render_settings):
    renderer = GaussianRasterizer(raster_settings=render_settings)

    means3D = mini_gaussian.xyz
    means2D = torch.zeros_like(
        means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device)
    opacities = torch.sigmoid(mini_gaussian.opacities)
    shs = mini_gaussian.harmonics
    scales = torch.exp(mini_gaussian.scales)
    rotations = torch.nn.functional.normalize(mini_gaussian.rotations)

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": None,
        "shs": shs,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None
    }
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def nopo_render_mini_gaussian(device,
                              mini: MiniGaussianModel,
                              nopo_decoder,
                              intrinsics,
                              c2w,
                              width, height):
    nopo_gaussians = mini.to_nopo_gaussians()
    nopo_intrinsics = intrinsics.clone()
    nopo_intrinsics[0, 0] = intrinsics[0, 0] / width
    nopo_intrinsics[1, 1] = intrinsics[1, 1] / height
    nopo_intrinsics[0, 2] = intrinsics[0, 2] / width
    nopo_intrinsics[1, 2] = intrinsics[1, 2] / height
    output = _nopo_render_gaussians(device,
                                    nopo_decoder,
                                    nopo_gaussians,
                                    repeat(nopo_intrinsics, "r c -> 1 1 r c"),
                                    repeat(c2w, "r c -> 1 1 r c"),
                                    width,
                                    height)
    color = rearrange(output[0], "1 1 c h w -> c h w")
    depth = rearrange(output[1], "1 1 h w -> h w")
    return {
        "color": color,
        "depth": depth,
    }


def nopo_render_gaussian(device,
                         model: GaussianModel,
                         nopo_decoder,
                         intrinsics,
                         c2w,
                         width, height):
    nopo_gaussians = model.to_nopo_gaussians()
    nopo_intrinsics = intrinsics.clone()
    nopo_intrinsics[0, 0] = intrinsics[0, 0] / width
    nopo_intrinsics[1, 1] = intrinsics[1, 1] / height
    nopo_intrinsics[0, 2] = intrinsics[0, 2] / width
    nopo_intrinsics[1, 2] = intrinsics[1, 2] / height
    output = _nopo_render_gaussians(device,
                                    nopo_decoder,
                                    nopo_gaussians,
                                    repeat(nopo_intrinsics, "r c -> 1 1 r c"),
                                    repeat(c2w, "r c -> 1 1 r c"),
                                    width,
                                    height)

    color = rearrange(output[0], "1 1 c h w -> c h w")
    depth = rearrange(output[1], "1 1 h w -> h w")
    return {
        "color": color,
        "depth": depth,
    }
