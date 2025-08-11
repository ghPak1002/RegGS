from math import isqrt

import torch
from e3nn.o3 import wigner_D
from einops import einsum
from jaxtyping import Float
from roma import rotmat_to_euler
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    rotations = rotations / (torch.det(rotations) ** (1 / 3))
    try:
        gamma, beta, alpha = rotmat_to_euler(
            "zyz", rotations, as_tuple=True)
    except Exception as e:
        print(
            f"rotations {rotations.detach().cpu().numpy()} {torch.det(rotations).detach().cpu().numpy()}")
        raise e

    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, -beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2: (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


def compute_spherical_harmonics(v: Float[Tensor, "b n"], sh: Float[Tensor, "b n"]):
    device = v.device
    SH_C0 = torch.tensor([
        0.28209479177387814
    ], dtype=torch.float32, device=device)
    SH_C1 = torch.tensor([
        -0.4886025119029199,
        0.4886025119029199,
        -0.4886025119029199
    ], dtype=torch.float32, device=device)
    SH_C2 = torch.tensor([
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ], dtype=torch.float32, device=device)
    SH_C3 = torch.tensor([
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ], dtype=torch.float32, device=device)
    norms = torch.norm(v, p=2, dim=-1, keepdim=True)
    v = v / norms
    x, y, z = (v[:, 0],  v[:, 1],  v[:, 2])
    xx, yy, zz = (x * x, y * y, z * z)
    xy, yz, xz = (x * y, y * z, x * z)
    c0 = SH_C0[0] * sh[0]
    c1 = SH_C1[0] * y * sh[1] + SH_C1[1] * z * sh[2] + SH_C1[2] * x * sh[3]
    c2 = SH_C2[0] * xy * sh[4] + \
        SH_C2[1] * yz * sh[5] + \
        SH_C2[2] * (2.0 * zz - xx - yy) * sh[6] + \
        SH_C2[3] * xz * sh[7] + \
        SH_C2[4] * (xx - yy) * sh[8]
    c3 = SH_C3[0] * y * (3.0 * xx - yy) * sh[9] + \
        SH_C3[1] * xy * z * sh[10] + \
        SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11] + \
        SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12] + \
        SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] + \
        SH_C3[5] * z * (xx - yy) * sh[14] + \
        SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
    return c0 + c1 + c2 + c3


if __name__ == "__main__":
    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 3
    sh = torch.rand(
        (degree + 1) ** 2, dtype=torch.float32, device=device)

    # generate test sample point
    batch = 100
    v = torch.rand((batch, 3), dtype=torch.float32, device=device)
    norms = torch.norm(v, p=2, dim=-1, keepdim=True)
    v = v / norms

    gt_colors = compute_spherical_harmonics(v, sh)
    angles = torch.randn(
        3, dtype=torch.float32, device=device) * torch.pi / 2
    rot_mat = torch.tensor(
        R.from_euler(
            "zyz",
            angles.cpu().detach().numpy(),
            degrees=False).as_matrix(),
        dtype=torch.float32,
        device=device,
    )
    roted_sh = rotate_sh(sh, rot_mat)

    # compute error
    roted_v = torch.matmul(rot_mat, v.transpose(0, 1)).transpose(0, 1)
    colors = compute_spherical_harmonics(roted_v, roted_sh)
    error = torch.max(torch.abs(colors - gt_colors))
    print(f"Error: {error.cpu().detach().numpy()}")
