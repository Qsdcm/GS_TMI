"""
Evaluation Metrics for MRI Reconstruction

评估指标:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index) 
- NMSE (Normalized Mean Squared Error)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim_sklearn


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None
) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    论文中使用PSNR作为主要评估指标之一
    
    Args:
        pred: 预测图像
        target: 目标图像
        data_range: 数据范围，如果为None则自动计算
        
    Returns:
        PSNR值 (dB)
    """
    # 转换为numpy并取幅度
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.is_complex():
            pred = torch.abs(pred)
        # Handle [2, D, H, W] case (Real/Imag stacked)
        elif pred.ndim == 4 and pred.shape[0] == 2:
            pred = torch.sqrt(pred[0]**2 + pred[1]**2)
        pred = pred.numpy()
    
    if torch.is_tensor(target):
        target = target.detach().cpu()
        if target.is_complex():
            target = torch.abs(target)
        # Handle [2, D, H, W] case (Real/Imag stacked)
        elif target.ndim == 4 and target.shape[0] == 2:
            target = torch.sqrt(target[0]**2 + target[1]**2)
        target = target.numpy()
    
    # 计算MSE
    mse = np.mean((pred - target) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # 计算数据范围
    if data_range is None:
        data_range = target.max() - target.min()
    
    # 计算PSNR
    psnr = 10 * np.log10((data_range ** 2) / mse)
    
    return float(psnr)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    win_size: int = 7
) -> float:
    """
    计算SSIM (Structural Similarity Index)
    
    论文中使用SSIM作为主要评估指标
    
    Args:
        pred: 预测图像
        target: 目标图像
        data_range: 数据范围
        win_size: SSIM窗口大小
        
    Returns:
        SSIM值 [0, 1]
    """
    # 转换为numpy并取幅度
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.is_complex():
            pred = torch.abs(pred)
        # Handle [2, D, H, W] case (Real/Imag stacked)
        elif pred.ndim == 4 and pred.shape[0] == 2:
            pred = torch.sqrt(pred[0]**2 + pred[1]**2)
        pred = pred.numpy()
    
    if torch.is_tensor(target):
        target = target.detach().cpu()
        if target.is_complex():
            target = torch.abs(target)
        # Handle [2, D, H, W] case (Real/Imag stacked)
        elif target.ndim == 4 and target.shape[0] == 2:
            target = torch.sqrt(target[0]**2 + target[1]**2)
        target = target.numpy()
    
    # 计算数据范围
    if data_range is None:
        data_range = target.max() - target.min()
    
    # 确保窗口大小不超过图像维度
    min_dim = min(pred.shape)
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        win_size = max(3, win_size)
    
    # 使用skimage计算SSIM
    ssim_value = ssim_sklearn(
        pred, target,
        data_range=data_range,
        win_size=win_size,
        channel_axis=None  # 3D数据
    )
    
    return float(ssim_value)


def compute_nmse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    计算NMSE (Normalized Mean Squared Error)
    
    NMSE = ||pred - target||^2 / ||target||^2
    
    Args:
        pred: 预测图像
        target: 目标图像
        
    Returns:
        NMSE值
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.is_complex():
            pred = torch.abs(pred)
        pred = pred.numpy()
    
    if torch.is_tensor(target):
        target = target.detach().cpu()
        if target.is_complex():
            target = torch.abs(target)
        target = target.numpy()
    
    # 计算NMSE
    diff_norm = np.sum((pred - target) ** 2)
    target_norm = np.sum(target ** 2)
    
    if target_norm == 0:
        return float('inf')
    
    nmse = diff_norm / target_norm
    
    return float(nmse)


def compute_ssim_3d_slicewise(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis: int = 0,
    data_range: Optional[float] = None
) -> Tuple[float, float]:
    """
    按切片计算3D SSIM的平均值和标准差
    
    Args:
        pred: 预测图像 (D, H, W)
        target: 目标图像 (D, H, W)
        axis: 切片轴
        data_range: 数据范围
        
    Returns:
        (mean_ssim, std_ssim)
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.is_complex():
            pred = torch.abs(pred)
        pred = pred.numpy()
    
    if torch.is_tensor(target):
        target = target.detach().cpu()
        if target.is_complex():
            target = torch.abs(target)
        target = target.numpy()
    
    if data_range is None:
        data_range = target.max() - target.min()
    
    ssim_values = []
    num_slices = pred.shape[axis]
    
    for i in range(num_slices):
        if axis == 0:
            pred_slice = pred[i, :, :]
            target_slice = target[i, :, :]
        elif axis == 1:
            pred_slice = pred[:, i, :]
            target_slice = target[:, i, :]
        else:
            pred_slice = pred[:, :, i]
            target_slice = target[:, :, i]
        
        # 跳过空切片
        if target_slice.max() - target_slice.min() < 1e-8:
            continue
            
        ssim_val = ssim_sklearn(
            pred_slice, target_slice,
            data_range=data_range,
            win_size=7
        )
        ssim_values.append(ssim_val)
    
    if len(ssim_values) == 0:
        return 0.0, 0.0
    
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    
    return float(mean_ssim), float(std_ssim)


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_3d_ssim: bool = True
) -> Dict[str, float]:
    """
    综合评估重建结果
    
    Args:
        pred: 预测图像
        target: 目标图像
        compute_3d_ssim: 是否计算3D SSIM
        
    Returns:
        包含各项指标的字典
    """
    metrics = {}
    
    # PSNR
    metrics['psnr'] = compute_psnr(pred, target)
    
    # SSIM
    if compute_3d_ssim:
        try:
            metrics['ssim'] = compute_ssim(pred, target)
        except Exception as e:
            # 如果3D SSIM失败，使用切片平均
            mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
            metrics['ssim'] = mean_ssim
            metrics['ssim_std'] = std_ssim
    else:
        mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
        metrics['ssim'] = mean_ssim
        metrics['ssim_std'] = std_ssim
    
    # NMSE
    metrics['nmse'] = compute_nmse(pred, target)
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """打印评估指标"""
    print(f"{prefix}PSNR: {metrics['psnr']:.2f} dB")
    print(f"{prefix}SSIM: {metrics['ssim']:.4f}")
    print(f"{prefix}NMSE: {metrics['nmse']:.6f}")
