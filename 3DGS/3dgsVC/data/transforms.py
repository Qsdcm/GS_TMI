"""
MRI Transforms and FFT Operations
用于k-space和图像域之间的转换
"""
 
import torch
import torch.fft as fft
from typing import Tuple


def fft3c(image: torch.Tensor) -> torch.Tensor:
    """
    3D centered FFT (Image → k-space)
    
    对应论文中的 MRI Forward Model: Image → k-space (FFT)
    
    Args:
        image: 图像域数据 (..., x, y, z), complex
        
    Returns:
        k-space数据 (..., kx, ky, kz), complex
    """
    # fftshift → fft3d → ifftshift
    # 使用orthonorm保证能量守恒
    return fft.fftshift(
        fft.fftn(
            fft.ifftshift(image, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
            norm="ortho"
        ),
        dim=(-3, -2, -1)
    )


def ifft3c(kspace: torch.Tensor) -> torch.Tensor:
    """
    3D centered inverse FFT (k-space → Image)
    
    对应论文中的初始化: 使用iFFT从欠采样k-space获取初始图像
    
    Args:
        kspace: k-space数据 (..., kx, ky, kz), complex
        
    Returns:
        图像域数据 (..., x, y, z), complex
    """
    # fftshift → ifft3d → ifftshift
    return fft.fftshift(
        fft.ifftn(
            fft.ifftshift(kspace, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
            norm="ortho"
        ),
        dim=(-3, -2, -1)
    )


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """2D centered FFT"""
    return fft.fftshift(
        fft.fft2(
            fft.ifftshift(image, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho"
        ),
        dim=(-2, -1)
    )


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """2D centered inverse FFT"""
    return fft.fftshift(
        fft.ifft2(
            fft.ifftshift(kspace, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho"
        ),
        dim=(-2, -1)
    )


def normalize_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    归一化k-space数据
    
    Args:
        kspace: k-space数据
        
    Returns:
        归一化后的k-space和归一化因子
    """
    # 使用最大模值归一化
    max_val = torch.abs(kspace).max()
    normalized = kspace / (max_val + 1e-8)
    return normalized, max_val.item()


def denormalize_kspace(kspace: torch.Tensor, norm_factor: float) -> torch.Tensor:
    """反归一化k-space"""
    return kspace * norm_factor


def apply_mask(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    应用欠采样mask
    
    对应论文公式(1): b = A(X) + n
    其中A包含mask操作
    
    Args:
        kspace: 全采样k-space
        mask: 二值mask
        
    Returns:
        欠采样k-space
    """
    return kspace * mask


def rss_combine(images: torch.Tensor) -> torch.Tensor:
    """
    Root Sum of Squares合并多线圈图像
    
    Args:
        images: 多线圈图像 (num_coils, ...)
        
    Returns:
        合并后的图像
    """
    return torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=0))


def complex_to_channels(data: torch.Tensor) -> torch.Tensor:
    """
    将复数tensor转换为双通道实数tensor
    
    Args:
        data: 复数tensor (..., x, y, z)
        
    Returns:
        实数tensor (2, ..., x, y, z) 或 (..., x, y, z, 2)
    """
    return torch.stack([data.real, data.imag], dim=0)


def channels_to_complex(data: torch.Tensor) -> torch.Tensor:
    """
    将双通道实数tensor转换为复数tensor
    
    Args:
        data: 实数tensor (2, ..., x, y, z)
        
    Returns:
        复数tensor (..., x, y, z)
    """
    return data[0] + 1j * data[1]
