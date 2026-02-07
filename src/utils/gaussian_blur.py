import torch
import torch.nn.functional as F
import math

def gaussian_kernel(kernel_size: int, sigma: float, device):
    """
    Tạo Gaussian kernel 2D
    """
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur(mask, kernel_size=7, sigma=2.0):
    """
    mask: (B, 1, H, W) hoặc (1, H, W)
    return: mask blur cùng shape
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # (1, 1, H, W)

    B, C, H, W = mask.shape
    device = mask.device

    kernel = gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(C, 1, 1, 1)

    padding = kernel_size // 2
    blurred = F.conv2d(mask, kernel, padding=padding, groups=C)
    return blurred
