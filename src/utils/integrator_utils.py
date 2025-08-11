import cv2
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()
