import copy
import os

import numpy as np
import roma
import torch
import torch.nn as nn
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm

from src.noposplat.dataset.data_module import get_data_shim
from src.noposplat.loss import get_losses
from src.noposplat.loss.loss_lpips import LossLpipsCfg, LossLpipsCfgWrapper
from src.noposplat.loss.loss_mse import LossMseCfg, LossMseCfgWrapper
from src.noposplat.loss.loss_ssim import ssim
from src.noposplat.misc.cam_utils import get_pnp_pose, update_pose
from src.noposplat.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA, DecoderSplattingCUDACfg)
from src.noposplat.model.encoder.backbone.backbone_croco import \
    BackboneCrocoCfg
from src.noposplat.model.encoder.common.gaussian_adapter import \
    GaussianAdapterCfg
from src.noposplat.model.encoder.encoder_noposplat import (EncoderNoPoSplat,
                                                           EncoderNoPoSplatCfg,
                                                           OpacityMappingCfg)
from src.noposplat.model.types import Gaussians


def save_render_color(image_path, color) -> None:
    dirname = os.path.dirname(image_path)
    if dirname is not None and dirname != "":
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    color = (rearrange(color.detach().cpu().numpy(), "c h w -> h w c")
             * 255).astype(np.uint8)
    Image.fromarray(color).save(image_path)


def get_nopo_encoder(device: torch.device, checkpoint) -> EncoderNoPoSplat:
    if checkpoint == "acid":
        print("Using acid.ckpt")
        pretrained_weights = "./pretrained_weights/acid.ckpt"
    elif checkpoint == "dl3dv":
        print("Using mixRe10kDl3dv.ckpt")
        pretrained_weights = "./pretrained_weights/mixRe10kDl3dv.ckpt"
    else:
        print("Using re10k.ckpt")
        pretrained_weights = "./pretrained_weights/re10k.ckpt"
    encoder_cfg = EncoderNoPoSplatCfg(
        name="noposplat",
        d_feature=128,
        num_monocular_samples=32,
        backbone=BackboneCrocoCfg(
            name="croco",
            model="ViTLarge_BaseDecoder",
            patch_embed_cls="PatchEmbedDust3R",
            asymmetry_decoder=True,
            intrinsics_embed_loc="encoder",
            intrinsics_embed_degree=4,
            intrinsics_embed_type="token",
        ),
        visualizer=None,
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0,
            sh_degree=4,
        ),
        apply_bounds_shim=True,
        opacity_mapping=OpacityMappingCfg(
            initial=0.0,
            final=0.0,
            warm_up=1,
        ),
        gaussians_per_pixel=1,
        num_surfaces=1,
        gs_params_head_type="dpt_gs",
        pretrained_weights="./pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        pose_free=True,
    )
    encoder = EncoderNoPoSplat(encoder_cfg)
    ckpt_weights = torch.load(pretrained_weights,
                              map_location='cpu')['state_dict']
    # remove the prefix "encoder.", need to judge if is at start of key
    ckpt_weights = {
        k[8:] if k.startswith("encoder.")
        else k: v
        for k, v in ckpt_weights.items()
    }
    missing_keys, unexpected_keys = encoder.load_state_dict(
        ckpt_weights, strict=True)
    return encoder.to(device)


def get_nopo_decoder(device: torch.device) -> DecoderSplattingCUDA:
    decoder_cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda",
        background_color=[0.0, 0.0, 0.0],
        make_scale_invariant=False,
    )
    decoder = DecoderSplattingCUDA(decoder_cfg)
    return decoder.to(device)


def nopo_normalize_intrinsics(intrinsics: torch.Tensor, h_old, w_old, h_new, w_new) -> torch.Tensor:
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    scale_factor = max(h_new / h_old, w_new / w_old)
    h_scaled = round(h_old * scale_factor)
    w_scaled = round(w_old * scale_factor)

    fx = fx * (scale_factor / w_new)
    fy = fy * (scale_factor / h_new)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    # FIX THIS assume always center
    cx = cx / w_old
    cy = cy / h_old
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    return intrinsics


def nopo_normalize_image(image: torch.Tensor, h_new, w_new) -> torch.Tensor:
    device = image.device
    h_old, w_old = image.shape[-2:]
    scale_factor = max(h_new / h_old, w_new / w_old)
    h_scaled = round(h_old * scale_factor)
    w_scaled = round(w_old * scale_factor)
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = Image.fromarray(image_new.numpy())
    image_new = image_new.resize((w_scaled, h_scaled), Image.LANCZOS)
    row = (h_scaled - h_new) // 2
    col = (w_scaled - w_new) // 2
    image_new = image_new.crop((col, row, col + w_new, row + h_new))
    to_tensor = tf.ToTensor()
    return to_tensor(image_new).to(device)


@torch.no_grad()
def compute_gaussians(encoder, images, intrinsics):
    """
    images [batch 2 channel height width]
    intrinsics [batch 2 3 3] (pixel unit, center is half)
    """
    b, v, c, h, w = images.shape

    assert v == 2
    assert h % 16 == 0 and w % 16 == 0

    device = images.device
    data = {
        "image": images,
        "intrinsics": intrinsics,
        "overlap": torch.tensor([[0.5]] * b, dtype=torch.float32, device=device)
    }

    data_shim = get_data_shim(encoder)
    data = data_shim({"context": data})["context"]
    output = encoder(data,
                     0,
                     visualization_dump=None)
    return output


def get_decoder_loss(device):
    loss_cfg = [
        LossMseCfgWrapper(LossMseCfg(weight=1.0)),
        LossLpipsCfgWrapper(LossLpipsCfg(
            weight=0.05,
            apply_after_step=0,
        ))
    ]
    return nn.ModuleList(get_losses(loss_cfg)).to(device)


def compute_pose(decoder, losses, nopo_gaussians, image, intrinsics) -> torch.Tensor:
    """
    image [b c h w]
    intrinsics [b 3 3]
    return
    extrinsics [b 4 4]
    """
    # export_ply(means=nopo_gaussians.means[0],
    #            scales=nopo_gaussians.scales[0],
    #            rotations=nopo_gaussians.rotations[0],
    #            harmonics=nopo_gaussians.harmonics[0],
    #            opacities=nopo_gaussians.opacities[0],
    #            path=Path("./test.ply"))
    device = image.device
    means = nopo_gaussians.means
    b, n, _ = means.shape
    b, c, h, w = image.shape
    assert h * w * 2 == n
    pcd = means[..., n // 2:, :]
    opacities = nopo_gaussians.opacities[..., n // 2:]
    pcd = rearrange(pcd, "b (h w) xyz -> b h w xyz", h=h, w=w)
    opacities = rearrange(opacities, "b (h w) -> b h w", h=h, w=w)
    pose_opts = []
    for i in range(b):
        pose_opt = get_pnp_pose(
            pcd[i],
            opacities[i],
            intrinsics[i], h, w
        )
        pose_opts.append(pose_opt)

    pose_opts = torch.stack(pose_opts, dim=0)
    with torch.set_grad_enabled(True):
        cam_rot_delta = nn.Parameter(torch.zeros(
            [b, 1, 3], requires_grad=True, device=device))
        cam_trans_delta = nn.Parameter(torch.zeros(
            [b, 1, 3], requires_grad=True, device=device))

        opt_params = []
        opt_params.append(
            {
                "params": [cam_rot_delta],
                "lr": 0.005,
            }
        )
        opt_params.append(
            {
                "params": [cam_trans_delta],
                "lr": 0.005,
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        number_steps = 200
        extrinsics = pose_opts.unsqueeze(1).to(device)
        intrinsics = intrinsics.unsqueeze(1)
        # initial pose use pose_opt
        # 用pnp计算出来的位姿，然后用高斯渲染与gt对比，迭代优化位姿
        for i in range(number_steps):
            pose_optimizer.zero_grad()

            output = decoder(
                nopo_gaussians,
                extrinsics,
                intrinsics,
                torch.tensor(
                    [[0.1]] * b, dtype=torch.float32).to(device=device),
                torch.tensor(
                    [[100.0]] * b, dtype=torch.float32).to(device=device),
                (h, w),
                cam_rot_delta=cam_rot_delta,
                cam_trans_delta=cam_trans_delta,
            )

            # Compute and log loss.
            batch = {}
            batch["target"] = {"image": image.unsqueeze(1)}
            total_loss = 0
            for loss_fn in losses:
                loss = loss_fn.forward(
                    output, batch, nopo_gaussians, 0)
                total_loss = total_loss + loss

            # add ssim structure loss
            ssim_, _, _, structure = ssim(
                rearrange(batch["target"]["image"],
                          "b v c h w -> (b v) c h w"),
                rearrange(output.color,
                          "b v c h w -> (b v) c h w"),
                size_average=True, data_range=1.0, retrun_seprate=True, win_size=11
            )
            gt_color = batch["target"]["image"]
            gt_color = rearrange(gt_color, "b v c h w -> (b v) c h w")
            render_color = output.color
            render_color = rearrange(render_color, "b v c h w -> (b v) c h w")

            ssim_loss = (1 - structure) * 1.0
            total_loss = total_loss + ssim_loss

            # backpropagate
            total_loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                new_extrinsic = update_pose(
                    cam_rot_delta=rearrange(
                        cam_rot_delta, "b v i -> (b v) i"),
                    cam_trans_delta=rearrange(
                        cam_trans_delta, "b v i -> (b v) i"),
                    extrinsics=rearrange(
                        extrinsics, "b v i j -> (b v) i j")
                )
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)

                extrinsics = rearrange(
                    new_extrinsic, "(b v) i j -> b v i j", b=b, v=1)
    return rearrange(extrinsics, "b v i j -> (b v) i j")


def compare_scales_rotations_covariances(scales, rotations, covariances):
    b = scales.shape[0]
    device = scales.device
    S = scales.diag_embed()
    # rotations = roma.quat_wxyz_to_xyzw(rotations)
    R = roma.unitquat_to_rotmat(rotations)
    C = R @ S @ S @ rearrange(R, "... r c -> ... c r")
    error = torch.sum(torch.abs(C - covariances))
    pass


@torch.no_grad()
def scale_gaussians_by_render_depth_max(decoder, nopo_gaussians, intrinsics, h, w, depth_max=5.0):
    b = nopo_gaussians.means.shape[0]
    device = nopo_gaussians.means.device
    extrinsics = repeat(torch.eye(4, dtype=torch.float32).to(
        device), "... -> b 1 ...", b=b)
    intrinsics = intrinsics.to(device).unsqueeze(1)
    output = decoder(
        nopo_gaussians,
        extrinsics,
        intrinsics,
        torch.tensor(
            [[0.1]] * b, dtype=torch.float32).to(device=device),
        torch.tensor(
            [[100.0]] * b, dtype=torch.float32).to(device=device),
        (h, w),
    )

    depth = output.depth[:, 0, ...]

    depth_mean = torch.mean(depth, dim=(1, 2))

    scales = depth_max / (depth_mean + 1e-8)

    gaussians_means = nopo_gaussians.means
    gaussians_scales = nopo_gaussians.scales
    gaussians_covariances = nopo_gaussians.covariances

    # compare_scales_rotations_covariances(
    #     gaussians_scales, nopo_gaussians.rotations, gaussians_covariances)

    gaussians_means = gaussians_means * scales
    gaussians_scales = gaussians_scales * scales
    gaussians_covariances = gaussians_covariances * (scales ** 2)

    nopo_gaussians.means = gaussians_means
    nopo_gaussians.scales = gaussians_scales
    nopo_gaussians.covariances = gaussians_covariances
    return scales


def scale_gaussians(nopo_gaussians, scales):
    gaussians_means = nopo_gaussians.means
    gaussians_scales = nopo_gaussians.scales
    gaussians_covariances = nopo_gaussians.covariances
    gaussians_means = gaussians_means * rearrange(scales, "b -> b 1 1")
    gaussians_scales = gaussians_scales * rearrange(scales, "b -> b 1 1")
    gaussians_covariances = gaussians_covariances * rearrange(
        (scales ** 2), "b -> b 1 1 1")
    nopo_gaussians.means = gaussians_means
    nopo_gaussians.scales = gaussians_scales
    nopo_gaussians.covariances = gaussians_covariances
    return nopo_gaussians


def compute_init_scales(pcd1, pcd2):
    mul_sum_1 = torch.sum(pcd1 * pcd2, dim=(1, 2))
    mul_sum_2 = torch.sum(pcd1 * pcd1, dim=(1, 2))
    init_scales = mul_sum_1 / mul_sum_2
    return 1 / init_scales


def compute_depth_loss(depth1, depth2):
    # return torch.mean(torch.abs(depth1 - depth2))
    weight = 1.0
    # weight = 1 / (torch.sqrt(depth1 * depth2) + 1e-8)
    # assert torch.isnan(weight).any() == False
    # assert torch.isinf(weight).any() == False
    delta = torch.abs(depth1 - depth2)
    return torch.mean(delta * weight)


def compute_pcd_loss(pcd1, pcd2):
    weight = 1 / (pcd1[..., 2] * pcd2[..., 2] + 1e-8)
    assert torch.isnan(weight).any() == False
    assert torch.isinf(weight).any() == False
    dist = torch.norm(pcd1 - pcd2, dim=-1)
    # return torch.mean(dist * weight)
    return torch.mean(dist)


def pack_nopo_gaussians(nopo_gaussians) -> Gaussians:
    means_array = []
    covariances_array = []
    harmonics_array = []
    opacities_array = []
    scales_array = []
    rotations_array = []

    for g in nopo_gaussians:
        means = g.means
        covariances = g.covariances
        harmonics = g.harmonics
        opacities = g.opacities
        scales = g.scales
        rotations = g.rotations

        means_array.append(means)
        covariances_array.append(covariances)
        harmonics_array.append(harmonics)
        opacities_array.append(opacities)
        scales_array.append(scales)
        rotations_array.append(rotations)

    means = torch.cat(means_array, dim=0)
    covariances = torch.cat(covariances_array, dim=0)
    harmonics = torch.cat(harmonics_array, dim=0)
    opacities = torch.cat(opacities_array, dim=0)
    scales = torch.cat(scales_array, dim=0)
    rotations = torch.cat(rotations_array, dim=0)

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations
    )


def render_gaussians(device, decoder, nopo_gaussians, intrinsics, extrinsics, h, w):
    """
    nopo_gaussians [b]
    intrinsics [b v 3 3]
    extrinsics [b v 4 4]
    """
    b, v = intrinsics.shape[0:2]
    output = decoder(
        nopo_gaussians,
        extrinsics,
        intrinsics,
        torch.tensor(
            [[0.1] * v] * b, dtype=torch.float32).to(device=device),
        torch.tensor(
            [[100.0] * v] * b, dtype=torch.float32).to(device=device),
        (h, w),
    )
    return output.color, output.depth


def new_align_nopo_gaussians(device, decoder, main_gaussians, nopo_gaussians, intrinsics, estimate_c2w, h, w) -> torch.Tensor:
    """
    intrinsics [3 3]
    estimate_poses [4 4]
    """
    with torch.no_grad():
        main_color, main_depth = \
            render_gaussians(device,
                             decoder,
                             main_gaussians,
                             rearrange(intrinsics, "r c -> 1 1 r c"),
                             rearrange(estimate_c2w, "r c -> 1 1 r c"),
                             h, w)
        nopo_color, nopo_depth = \
            render_gaussians(device,
                             decoder,
                             nopo_gaussians,
                             rearrange(intrinsics, "r c -> 1 1 r c"),
                             rearrange(
                                 torch.eye(4, dtype=torch.float32, device=device), "r c -> 1 1 r c"),
                             h, w)

        main_color = rearrange(main_color, "b v ... -> (b v) ...")
        nopo_color = rearrange(nopo_color, "b v ... -> (b v) ...")

        for i in range(main_color.shape[0]):
            save_render_color(f"main_color_{i}.png", main_color[i])
        for i in range(nopo_color.shape[0]):
            save_render_color(f"nopo_color_{i}.png", nopo_color[i])
        init_scale = torch.median(main_depth) / torch.median(nopo_depth)

    with torch.set_grad_enabled(True):
        scales = nn.Parameter(torch.tensor([init_scale]).to(device))
        scale_optimizer = torch.optim.Adam(
            params=[scales],
            lr=0.01,
        )
        number_steps = 100
        with tqdm(total=number_steps, desc="Training", unit="epoch") as pbar:
            for i in range(number_steps):
                scale_optimizer.zero_grad()
                training_gaussians = copy.deepcopy(nopo_gaussians)
                scale_gaussians(training_gaussians, scales)

                nopo_color, nopo_depth = \
                    render_gaussians(device,
                                     decoder,
                                     training_gaussians,
                                     rearrange(intrinsics, "r c -> 1 1 r c"),
                                     rearrange(
                                         torch.eye(4, dtype=torch.float32, device=device), "r c -> 1 1 r c"),
                                     h, w)

                depth_loss = compute_depth_loss(main_depth, nopo_depth)
                total_loss = depth_loss
                total_loss.backward()
                assert torch.isnan(scales.grad).any() == False
                assert torch.isinf(scales.grad).any() == False
                scale_optimizer.step()
                pbar.set_postfix_str(
                    f"Scale:{scales[0].item():.2f},Total:{total_loss.item():.2f},Depth:{depth_loss.item():.2f}")
                pbar.update(1)
    with torch.no_grad():
        return scales.detach()


def align_nopo_gaussians(device, decoder, nopo_gaussians, intrinsics, estimate_poses, h, w):
    """
    nopo_gaussians array [b]
    intrinsics [b 3 3]
    estimate_poses [b 4 4]
    """
    b = 2
    v = 2

    packed_gaussians = pack_nopo_gaussians(nopo_gaussians)

    extrinsics = repeat(torch.eye(4, dtype=torch.float32),
                        "... -> b v ...", b=b, v=v).to(device)
    extrinsics[:, 1, ...] = estimate_poses

    intrinsics = repeat(intrinsics, "b r c -> b v r c", v=2).to(device)

    num_gaussians = packed_gaussians.means.shape[1] // 2
    pcd1 = packed_gaussians.means[:-1, num_gaussians:, :]
    pcd2 = packed_gaussians.means[1:, :num_gaussians, :]
    extrinsics1 = extrinsics[:-1, 1, :, :]
    viewmaticies1 = torch.inverse(extrinsics1)

    pcd1 = viewmaticies1[:, :3, :3] \
        @ rearrange(pcd1, "b n xyz -> b xyz n")
    pcd1 = pcd1 + viewmaticies1[:, :3, 3].unsqueeze(2)
    pcd1 = rearrange(pcd1, "b xyz n -> b n xyz")

    init_scales = compute_init_scales(pcd1, pcd2)
    init_scales = torch.log(init_scales)

    with torch.set_grad_enabled(True):
        scales = nn.Parameter(init_scales)
        scale_optimizer = torch.optim.Adam(
            params=[scales],
            lr=0.01,
        )

        number_steps = 20
        with tqdm(total=number_steps, desc="Training", unit="epoch") as pbar:
            training_gaussians = Gaussians(
                means=torch.zeros_like(packed_gaussians.means),
                covariances=torch.zeros_like(packed_gaussians.covariances),
                harmonics=packed_gaussians.harmonics,
                opacities=packed_gaussians.opacities,
                scales=torch.zeros_like(packed_gaussians.scales),
                rotations=packed_gaussians.rotations,
            )
            for s in range(number_steps):
                scale_optimizer.zero_grad()
                exp_scales = torch.ones(b, dtype=torch.float32, device=device)
                exp_scales[1:] = torch.exp(scales)
                training_gaussians.means = packed_gaussians.means
                training_gaussians.scales = packed_gaussians.scales
                training_gaussians.covariances = packed_gaussians.covariances
                scale_gaussians(training_gaussians, exp_scales)

                training_extrinsics = extrinsics.clone()
                training_extrinsics[..., 1, 0:3, 3] = \
                    extrinsics[..., 1, 0:3, 3] * \
                    rearrange(exp_scales, "b -> b 1")

                output = decoder(
                    training_gaussians,
                    training_extrinsics,
                    intrinsics,
                    torch.tensor(
                        [[0.1] * 2] * b, dtype=torch.float32).to(device=device),
                    torch.tensor(
                        [[100.0] * 2] * b, dtype=torch.float32).to(device=device),
                    (h, w),
                )
                render_color = output.color
                render_color = rearrange(
                    render_color, "b v c h w -> (b v) c h w")
                for i in range(render_color.shape[0]):
                    save_render_color(f"joint_color_{i}.png", render_color[i])

                render_depth = output.depth
                render_depth1 = render_depth[:-1, 1, :, :]
                render_depth2 = render_depth[1:, 0, :, :]
                num_gaussians = training_gaussians.means.shape[1] // 2
                pcd1 = training_gaussians.means[:-1, num_gaussians:, :]
                pcd2 = training_gaussians.means[1:, :num_gaussians, :]
                extrinsics1 = training_extrinsics[:-1, 1, :, :]
                viewmaticies1 = torch.inverse(extrinsics1)

                pcd1 = viewmaticies1[:, :3, :3] \
                    @ rearrange(pcd1, "b n xyz -> b xyz n")
                pcd1 = pcd1 + viewmaticies1[:, :3, 3].unsqueeze(2)
                pcd1 = rearrange(pcd1, "b xyz n -> b n xyz")

                pcd_loss = compute_pcd_loss(pcd1, pcd2)
                depth_loss = compute_depth_loss(render_depth1, render_depth2)
                total_loss = 0.1 * pcd_loss + 0.1 * depth_loss
                total_loss.backward()
                scale_optimizer.step()
                pbar.set_postfix_str(
                    f"Scale:{exp_scales[b-1].item():.2f},Total:{total_loss.item():.2f},Depth:{depth_loss.item():.2f},PCD Loss: {pcd_loss.item():.2f}")
                pbar.update(1)

    exp_scales = torch.ones(b, dtype=torch.float32, device=device)
    exp_scales[1:] = torch.exp(scales)
    return exp_scales.detach()
