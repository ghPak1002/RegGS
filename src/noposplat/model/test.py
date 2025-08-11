import torch
from scipy.spatial.transform import Rotation as R
from einops import rearrange

qvec = torch.tensor([[0.1826, 0.3651, 0.5477, 0.7303]], dtype=torch.float32)
svec = torch.tensor([[1.0, 1.0 / 2, 1.0 / 3]], dtype=torch.float32)


def to_covariance(qvec, svec):
    scale = svec.diag_embed()
    rotation = torch.tensor(R.from_quat(
        qvec.detach().cpu().numpy(), scalar_first=True).as_matrix(), dtype=torch.float32
    )
    return rotation @ scale @ rearrange(scale, "... i j -> ... j i") @ rearrange(rotation, "... i j -> ... j i")

def to_qvec_and_svec(covariance):
    u, s, v = torch.svd(covariance)
    rotation = u / torch.det(u)
    svec = torch.sqrt(s)
    qvec = torch.tensor(
        R.from_matrix(rotation.detach().cpu().numpy()).as_quat(scalar_first=True), dtype=torch.float32
    )
    return qvec, svec

covariance = to_covariance(qvec, svec)
print(covariance)
qvec, svec = to_qvec_and_svec(covariance)
print(f"{qvec} {svec}")
covariance = to_covariance(qvec, svec)
print(covariance)
