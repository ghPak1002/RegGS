import os
import random

import numpy as np
import open3d as o3d
import torch
from gaussian_rasterizer import (GaussianRasterizationSettings,
                                 GaussianRasterizer)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            dict[k] = v.to(device)
    return dict


def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
                                                  1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    std_fx = 2 * fx / w
    std_fy = 2 * fy / h
    std_cx = -(w - 2 * cx) / w
    std_cy = -(h - 2 * cy) / h
    std_d_scale = far / (far - near)
    std_d_offset = -(far * near) / (far - near)
    opengl_proj = torch.tensor([[std_fx, 0.0, std_cx, 0.0],
                                [0.0, std_fy, std_cy, 0.0],
                                [0.0, 0.0, std_d_scale, std_d_offset],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None
    }
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def batch_search_faiss(indexer, query_points, k):
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float().cpu(), k)
        distances_list.append(torch.tensor(distance))
        ids_list.append(torch.tensor(id))
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids
