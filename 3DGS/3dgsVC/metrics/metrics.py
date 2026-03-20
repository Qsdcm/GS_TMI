"""Evaluation metrics and plateau monitoring for 3DGSMR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_sklearn


def _to_magnitude_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu()
        if data.is_complex():
            data = torch.abs(data)
        elif data.ndim >= 1 and data.shape[0] == 2:
            data = torch.sqrt(data[0] ** 2 + data[1] ** 2 + 1e-12)
        return data.numpy()

    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        return np.abs(arr)
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2 + 1e-12)
    return arr


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: Optional[float] = None) -> float:
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(target_np.max() - target_np.min())
    if data_range <= 0:
        data_range = 1.0
    return float(10 * np.log10((data_range ** 2) / mse))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    win_size: int = 7,
) -> float:
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    if data_range is None:
        data_range = float(target_np.max() - target_np.min())
    if data_range <= 0:
        data_range = 1.0
    min_dim = min(pred_np.shape)
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        win_size = max(3, win_size)
    return float(
        ssim_sklearn(
            pred_np,
            target_np,
            data_range=data_range,
            win_size=win_size,
            channel_axis=None,
        )
    )


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    numerator = np.sum((pred_np - target_np) ** 2)
    denominator = np.sum(target_np ** 2)
    if denominator == 0:
        return float("inf")
    return float(numerator / denominator)


def compute_ssim_3d_slicewise(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis: int = 0,
    data_range: Optional[float] = None,
) -> tuple[float, float]:
    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    if data_range is None:
        data_range = float(target_np.max() - target_np.min())
    if data_range <= 0:
        data_range = 1.0

    values = []
    for idx in range(pred_np.shape[axis]):
        pred_slice = np.take(pred_np, idx, axis=axis)
        target_slice = np.take(target_np, idx, axis=axis)
        if target_slice.max() - target_slice.min() < 1e-8:
            continue
        values.append(ssim_sklearn(pred_slice, target_slice, data_range=data_range, win_size=7))
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


_LPIPS_MODELS: Dict[str, torch.nn.Module] = {}
_LPIPS_WARNING_EMITTED = False


def _get_lpips_model(device: torch.device, net: str = "alex"):
    key = f"{net}:{device}"
    if key in _LPIPS_MODELS:
        return _LPIPS_MODELS[key]
    try:
        import lpips  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "LPIPS evaluation requires the `lpips` package. Install it in the active environment to run paper-faithful reproduction metrics."
        ) from exc
    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    _LPIPS_MODELS[key] = model
    return model


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: Optional[torch.device] = None,
    axis: int = 0,
    require: bool = False,
) -> Optional[float]:
    try:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _get_lpips_model(dev)
    except ModuleNotFoundError:
        global _LPIPS_WARNING_EMITTED
        if require:
            raise
        if not _LPIPS_WARNING_EMITTED:
            print('[Metrics] LPIPS package not available; skipping LPIPS computation in this environment.')
            _LPIPS_WARNING_EMITTED = True
        return None

    pred_np = _to_magnitude_numpy(pred)
    target_np = _to_magnitude_numpy(target)
    t_min = float(target_np.min())
    t_max = float(target_np.max())
    scale = max(t_max - t_min, 1e-8)
    pred_norm = ((pred_np - t_min) / scale) * 2.0 - 1.0
    target_norm = ((target_np - t_min) / scale) * 2.0 - 1.0

    values = []
    with torch.no_grad():
        for idx in range(pred_norm.shape[axis]):
            pred_slice = np.take(pred_norm, idx, axis=axis)
            target_slice = np.take(target_norm, idx, axis=axis)
            pred_tensor = torch.from_numpy(pred_slice).float().to(dev).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            target_tensor = torch.from_numpy(target_slice).float().to(dev).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            values.append(float(model(pred_tensor, target_tensor).item()))
    if not values:
        return None
    return float(np.mean(values))


@dataclass
class MetricPlateauStopper:
    patience_evals: int
    min_iterations_before_stop: int
    metric_min_delta: Dict[str, float]
    best_metric: str = "psnr"
    best_values: Dict[str, float] = field(default_factory=dict)
    bad_eval_count: int = 0

    def _improved(self, name: str, value: float) -> bool:
        delta = self.metric_min_delta.get(name, 0.0)
        if name not in self.best_values:
            return True
        best = self.best_values[name]
        if name == "lpips":
            return value < best - delta
        return value > best + delta

    def update(self, iteration: int, metrics: Dict[str, float]) -> bool:
        improved = False
        for name, value in metrics.items():
            if value is None:
                continue
            if self._improved(name, value):
                self.best_values[name] = value
                improved = True
        if improved:
            self.bad_eval_count = 0
        else:
            self.bad_eval_count += 1
        return iteration >= self.min_iterations_before_stop and self.bad_eval_count >= self.patience_evals


def evaluate_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_3d_ssim: bool = True,
    compute_lpips_metric: bool = False,
    lpips_device: Optional[torch.device] = None,
    require_lpips: bool = False,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["psnr"] = compute_psnr(pred, target)
    if compute_3d_ssim:
        try:
            metrics["ssim"] = compute_ssim(pred, target)
        except Exception:
            mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
            metrics["ssim"] = mean_ssim
            metrics["ssim_std"] = std_ssim
    else:
        mean_ssim, std_ssim = compute_ssim_3d_slicewise(pred, target)
        metrics["ssim"] = mean_ssim
        metrics["ssim_std"] = std_ssim
    metrics["nmse"] = compute_nmse(pred, target)
    if compute_lpips_metric:
        lpips_value = compute_lpips(pred, target, device=lpips_device, require=require_lpips)
        if lpips_value is not None:
            metrics["lpips"] = lpips_value
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(f"{prefix}PSNR: {metrics['psnr']:.2f} dB")
    print(f"{prefix}SSIM: {metrics['ssim']:.4f}")
    print(f"{prefix}NMSE: {metrics['nmse']:.6f}")
    if "lpips" in metrics:
        print(f"{prefix}LPIPS: {metrics['lpips']:.4f}")
