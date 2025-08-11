#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from src.utils.gaussian_model_utils import (RGB2SH, build_scaling_rotation,
                                            get_expon_lr_func, inverse_sigmoid,
                                            strip_symmetric, BasicPointCloud)
from src.noposplat.model.types import Gaussians as NoPoGaussians
from einops import rearrange, repeat
from roma import roma
from src.utils.sh_utils import rotate_sh


def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] +
                      r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


class MiniGaussianModel:
    xyz: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    sh_degree: int
    normalize_scale: float

    def __init__(self, xyz: torch.Tensor,
                 harmonics: torch.Tensor,
                 opacities: torch.Tensor,
                 scales: torch.Tensor,
                 rotations: torch.Tensor,
                 sh_degree: int) -> None:
        """
        xyz [N, 3]
        harmonics [N, sh, 3]
        opacities [N, 1]
        scales [N, 3]
        rotations [N, 4]
        sh_degree int
        """
        self.xyz = xyz
        self.harmonics = harmonics
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.sh_degree = sh_degree

    @property
    def number_of_gaussian(self):
        return self.xyz.shape[0]

    def rigid_transform_gaussians(self, rt_mat: torch.Tensor):
        R = rt_mat[0:3, 0:3]
        T = rt_mat[0:3, 3]

        xyz = self.xyz
        rotations = self.rotations
        harmonics = self.harmonics
        sh_degree = self.sh_degree

        new_xyz = (R @ xyz.transpose(1, 0)).transpose(1, 0) + T

        quat = roma.rotmat_to_unitquat(R)
        gs_quat = roma.quat_wxyz_to_xyzw(rotations)
        gs_quat = roma.quat_normalize(gs_quat)
        gs_quat = roma.quat_product(
            repeat(quat, "xyzw -> n xyzw", n=gs_quat.shape[0]), gs_quat)
        new_rotations = roma.quat_xyzw_to_wxyz(gs_quat)

        _, shl, _ = harmonics.shape
        assert shl == (sh_degree + 1) ** 2

        harmonics = rearrange(harmonics, "n sh c -> n c sh")
        new_harmonics = rotate_sh(harmonics, R)
        new_harmonics = rearrange(new_harmonics, "n c sh -> n sh c")

        return MiniGaussianModel(xyz=new_xyz,
                                 harmonics=new_harmonics,
                                 opacities=self.opacities,
                                 scales=self.scales,
                                 rotations=new_rotations,
                                 sh_degree=sh_degree)

    def store_normalize_scale(self, scale):
        self.normalize_scale = scale

    def scale_gaussians(self, factor: torch.Tensor):
        xyz = self.xyz
        scales = self.scales

        new_xyz = xyz * factor
        new_scales = scales + torch.log(factor)

        return MiniGaussianModel(xyz=new_xyz,
                                 harmonics=self.harmonics,
                                 opacities=self.opacities,
                                 scales=new_scales,
                                 rotations=self.rotations,
                                 sh_degree=self.sh_degree)

    def to_nopo_gaussians(self):
        xyz = self.xyz
        harmonics = self.harmonics
        opacities = self.opacities
        scales = self.scales
        rotations = self.rotations

        scales = torch.exp(scales)
        rotations = roma.quat_wxyz_to_xyzw(rotations)
        R = roma.unitquat_to_rotmat(rotations)
        S = scales.diag_embed()
        L = R @ S
        covariances = L @ L.transpose(1, 2)
        opacities = torch.sigmoid(opacities)

        xyz = rearrange(
            xyz, "n xyz -> 1 n xyz")
        scales = rearrange(
            scales, "n s -> 1 n s")
        rotations = rearrange(
            rotations, "n r -> 1 n r")
        opacities = rearrange(
            opacities, "n 1 -> 1 n")
        harmonics = rearrange(
            harmonics, "n sh c -> 1 n c sh")
        covariances = rearrange(
            covariances, "n r c -> 1 n r c")

        return NoPoGaussians(xyz, covariances, harmonics, opacities, scales, rotations)

    @classmethod
    def from_nopo_gaussians(cls, gaussians: NoPoGaussians):
        xyz = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        opacities = gaussians.opacities
        harmonics = gaussians.harmonics

        xyz = rearrange(
            xyz, "b n xyz -> (b n) xyz")
        scales = rearrange(
            scales, "b n s -> (b n) s")
        rotations = rearrange(
            rotations, "b n r -> (b n) r")
        opacities = rearrange(
            opacities, "b n -> (b n) 1")
        harmonics = rearrange(
            harmonics, "b n c sh -> (b n) sh c")
        sh_degree = 3

        scales = torch.log(scales)
        rotations = roma.quat_normalize(rotations)
        rotations = roma.quat_xyzw_to_wxyz(rotations)
        opacities = torch.clip(opacities, 1e-8, 1-1e-8)
        opacities = torch.log((opacities)/((1-opacities)))
        harmonics_size = (sh_degree + 1) ** 2
        harmonics = harmonics[..., 0:harmonics_size, :]
        return cls(xyz, harmonics, opacities, scales, rotations, sh_degree)


class GaussianModel:
    def __init__(self, sh_degree: int = 3, isotropic=False):
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "scaling",
            "rotation",
            "opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # temp
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0, 4).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()
        self.isotropic = isotropic

    def restore_from_params(self, params_dict, training_args):
        self.training_setup(training_args)
        self.densification_postfix(
            params_dict["xyz"],
            params_dict["features_dc"],
            params_dict["features_rest"],
            params_dict["opacity"],
            params_dict["scaling"],
            params_dict["rotation"])

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        actual_covariance = self.build_full_covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation)
        symm = strip_symmetric(actual_covariance)
        return symm

    def build_full_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return actual_covariance

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture_dict(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.clone().detach().cpu(),
            "features_dc": self._features_dc.clone().detach().cpu(),
            "features_rest": self._features_rest.clone().detach().cpu(),
            "scaling": self._scaling.clone().detach().cpu(),
            "rotation": self._rotation.clone().detach().cpu(),
            "opacity": self._opacity.clone().detach().cpu(),
            "max_radii2D": self.max_radii2D.clone().detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().cpu(),
            "denom": self.denom.clone().detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict(),
        }

    def get_size(self):
        return self._xyz.shape[0]

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[
                :, 0:1]  # Extract the first column
            scales = scale.repeat(1, 3)  # Replicate this column three times
            return scales
        return self.scaling_activation(self._scaling)

    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_active_sh_degree(self):
        return self.active_sh_degree

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(self.get_scaling(), scaling_modifier, self._rotation)

    def get_full_covariance(self, scaling_modifier=1):
        return self.build_full_covariance_from_scaling_rotation(self.get_scaling(), scaling_modifier, self._rotation)

    def add_points(self, pcd: o3d.geometry.PointCloud, global_scale_init=True):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(
            np.asarray(pcd.colors)).float().cuda())
        features = (torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda())
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of added points: ", fused_point_cloud.shape[0])

        if global_scale_init:
            global_points = torch.cat(
                (self.get_xyz(), torch.from_numpy(np.asarray(pcd.points)).float().cuda()))
            dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
            dist2 = dist2[self.get_size():]
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(
                np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 3)
        # scales = torch.log(0.001 * torch.ones_like(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(
            1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def training_setup(self, training_args, exposure_ab=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz],
                "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc],
                "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity],
                "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling],
                "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation],
                "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def training_setup_camera(self, cam_rot, cam_trans, cfg, exposure_ab=None):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [self._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [self._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [self._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0.0, "name": "rotation"},
            {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
                "name": "cam_unnorm_rot"},
            {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
                "name": "cam_trans"},
        ]
        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )
        self.optimizer = torch.optim.Adam(params, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.98, patience=10, verbose=False)

    def training_setup_scale_rot_trans(self, gs_scale, cam_rot, cam_trans, cfg):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [self._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [self._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [self._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0.0, "name": "rotation"},
            {"params": [gs_scale], "lr": 1e-3, "name": "gs_scale"},
            {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
                "name": "cam_unnorm_rot"},
            {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
                "name": "cam_trans"},
        ]
        self.optimizer = torch.optim.Adam(params, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.98, patience=10, verbose=False)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(self._scaling.detach().cpu().numpy()[
                            :, 0].reshape(-1, 1), (1, 3))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4")
                      for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def add_gaussians(self, gaussians):
        xyz = gaussians["xyz"]
        features_dc = gaussians["features_dc"]
        features_rest = gaussians["features_rest"]
        opacities = gaussians["opacities"]
        scales = gaussians["scales"]
        rotations = gaussians["rotations"]
        sh_degree = gaussians["sh_degree"]

        new_xyz = nn.Parameter(xyz.clone()
                               .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_features_dc = nn.Parameter(features_dc.clone()
                                       .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_features_rest = nn.Parameter(features_rest.clone()
                                         .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_scaling = nn.Parameter(scales.clone()
                                   .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_rotation = nn.Parameter(rotations.clone()
                                    .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_opacities = nn.Parameter(opacities.clone()
                                     .to(dtype=torch.float32).cuda().requires_grad_(True))
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree

    def add_mini_gaussians(self, mini_gaussians: MiniGaussianModel):
        xyz = mini_gaussians.xyz
        harmonics = mini_gaussians.harmonics
        opacities = mini_gaussians.opacities
        scales = mini_gaussians.scales
        rotations = mini_gaussians.rotations
        sh_degree = mini_gaussians.sh_degree

        features_dc = harmonics[:, 0:1, :].clone()
        features_rest = harmonics[:, 1:, :].clone()

        new_xyz = nn.Parameter(xyz.clone()
                               .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_features_dc = nn.Parameter(features_dc.clone()
                                       .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_features_rest = nn.Parameter(features_rest.clone()
                                         .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_scaling = nn.Parameter(scales.clone()
                                   .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_rotation = nn.Parameter(rotations.clone()
                                    .to(dtype=torch.float32).cuda().requires_grad_(True))
        new_opacities = nn.Parameter(opacities.clone()
                                     .to(dtype=torch.float32).cuda().requires_grad_(True))

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])),
            axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(
            xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(
            opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        if self.optimizer is None:
            return None
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "exposure" not in group["name"]:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        if optimizable_tensors is not None:
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
        else:
            self._xyz = nn.Parameter(
                self._xyz[valid_points_mask].requires_grad_(True))
            self._features_dc = nn.Parameter(
                self._features_dc[valid_points_mask].requires_grad_(True))
            self._features_rest = nn.Parameter(
                self._features_rest[valid_points_mask].requires_grad_(True))
            self._opacity = nn.Parameter(
                self._opacity[valid_points_mask].requires_grad_(True))
            self._scaling = nn.Parameter(
                self._scaling[valid_points_mask].requires_grad_(True))
            self._rotation = nn.Parameter(
                self._rotation[valid_points_mask].requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # 优化器梯度参数更新
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(
                    group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.optimizer is not None:
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = torch.zeros(
                (self.get_xyz().shape[0], 1), device="cuda")
            self.denom = torch.zeros(
                (self.get_xyz().shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros(
                (self.get_xyz().shape[0]), device="cuda")
        else:
            self._xyz = nn.Parameter(
                torch.cat([self._xyz, new_xyz], dim=0).requires_grad_(True))
            self._features_dc = nn.Parameter(
                torch.cat([self._features_dc, new_features_dc], dim=0))
            self._features_rest = nn.Parameter(
                torch.cat([self._features_rest, new_features_rest], dim=0))
            self._opacity = nn.Parameter(
                torch.cat([self._opacity, new_opacities], dim=0))
            self._scaling = nn.Parameter(
                torch.cat([self._scaling, new_scaling], dim=0))
            self._rotation = nn.Parameter(
                torch.cat([self._rotation, new_rotation], dim=0))

            self.xyz_gradient_accum = torch.zeros(
                (self.get_xyz().shape[0], 1), device="cuda")
            self.denom = torch.zeros(
                (self.get_xyz().shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros(
                (self.get_xyz().shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd) is BasicPointCloud:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd.colors)).float().cuda())
        else:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd._xyz)).float().cuda()
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd._rgb)).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ",
              fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(
            (self.get_xyz().shape[0]), device="cuda")

    def to_nopo_gaussians(self) -> NoPoGaussians:
        xyz = self.get_xyz()
        harmonics = self.get_features()
        opacities = self.get_opacity()
        scales = self.get_scaling()
        rotations = self.get_rotation()
        covariances = self.get_full_covariance()

        rotations = roma.quat_wxyz_to_xyzw(rotations)

        xyz = rearrange(
            xyz, "n xyz -> 1 n xyz")
        scales = rearrange(
            scales, "n s -> 1 n s")
        rotations = rearrange(
            rotations, "n r -> 1 n r")
        opacities = rearrange(
            opacities, "n 1 -> 1 n")
        harmonics = rearrange(
            harmonics, "n c sh -> 1 n sh c")
        covariances = rearrange(
            covariances, "n r c -> 1 n r c")

        return NoPoGaussians(xyz, covariances, harmonics, opacities, scales, rotations)

    def to_mini_gaussians(self) -> MiniGaussianModel:
        xyz = self._xyz
        harmonics = self.get_features()
        opacities = self.get_opacity()
        scales = self._scaling
        rotations = self.get_rotation()

        return MiniGaussianModel(
            xyz=xyz,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            sh_degree=self.max_sh_degree
        )

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity(), torch.ones_like(self.get_opacity())*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz().shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling(), dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling()[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_xyz()[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling()[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling(), dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling().max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
