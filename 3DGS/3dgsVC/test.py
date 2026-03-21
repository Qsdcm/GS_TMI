"""Testing and inference entry point for 3DGSMR."""

import argparse
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:  # pragma: no cover
    HAS_NIBABEL = False

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, TileVoxelizer, Voxelizer
from metrics import evaluate_reconstruction


def parse_args():
    parser = argparse.ArgumentParser(description="Test 3DGSMR for MRI reconstruction")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--acceleration", type=float, default=1.0, help="Acceleration factor override")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_volume", action="store_true", default=True)
    parser.add_argument("--save_slices", action="store_true", default=True)
    parser.add_argument("--slices_axial", nargs="+", type=int)
    parser.add_argument("--slices_coronal", nargs="+", type=int)
    parser.add_argument("--slices_sagittal", nargs="+", type=int)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GaussianTester:
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any] = None,
        device: torch.device = None,
        data_path: str = None,
        acceleration_override: float = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acceleration_override = acceleration_override
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loading weights from: {checkpoint_path}")
        if config is not None:
            self.config = config
        elif "config" in checkpoint:
            self.config = checkpoint["config"]
        else:
            raise ValueError("Config not found in checkpoint and not provided externally.")
        if data_path is not None:
            self.config["data"]["data_path"] = data_path
        self._setup_data()
        self._setup_model(checkpoint)

    def _setup_data(self):
        data_config = self.config["data"]
        acc_factor = self.acceleration_override if self.acceleration_override is not None else data_config["acceleration_factor"]
        self.dataset = MRIDataset(
            data_path=data_config["data_path"],
            acceleration_factor=acc_factor,
            mask_type=data_config.get("mask_type", "stacked_2d_gaussian"),
            use_acs=data_config.get("use_acs", True),
            acs_lines=data_config.get("acs_lines", 24),
            csm_path=data_config.get("csm_path"),
            mask_path=data_config.get("mask_path"),
            readout_axis=data_config.get("readout_axis"),
            phase_axes=tuple(data_config["phase_axes"]) if data_config.get("phase_axes") is not None else None,
            normalize_kspace=data_config.get("normalize", data_config.get("normalize_kspace", True)),
            normalization_percentile=data_config.get("normalization_percentile", 99.9),
            device=str(self.device),
        )
        data = self.dataset.get_data()
        self.volume_shape = tuple(data["volume_shape"])
        self.target_image = data["ground_truth"].to(self.device)
        self.zero_filled = data.get("zero_filled_complex", data["zero_filled"]).to(self.device)
        self.kspace_undersampled_complex = data.get("kspace_undersampled_complex")
        if self.kspace_undersampled_complex is None:
            kspace_ri = data["kspace_undersampled"]
            num_coils = kspace_ri.shape[0] // 2
            self.kspace_undersampled_complex = torch.complex(kspace_ri[:num_coils], kspace_ri[num_coils:])
        self.kspace_undersampled_complex = self.kspace_undersampled_complex.to(self.device)
        self.mask = data["mask"].to(self.device)
        self.mask_coils = self.mask.unsqueeze(0)
        self.sensitivity_maps = data["sensitivity_maps"].to(self.device)
        self.normalization_scale = float(data.get("normalization_scale", 1.0))

    def _setup_model(self, checkpoint: Dict[str, Any]):
        gaussian_state = checkpoint["gaussian_state"]
        num_points = gaussian_state["positions"].shape[0]
        self.gaussian_model = GaussianModel3D(num_points=num_points, volume_shape=tuple(self.volume_shape), device=str(self.device))
        self.gaussian_model.load_state_dict(gaussian_state)
        vox_config = self.config.get("voxelizer", {})
        vox_type = vox_config.get("type", "chunk")
        if vox_type == "tile_cuda":
            self.voxelizer = TileVoxelizer(
                tuple(self.volume_shape),
                tile_size=vox_config.get("tile_size", 8),
                max_radius=vox_config.get("max_radius", 20),
                use_cuda=True,
                strict_cuda=self.config.get("mode", {}).get("strict_cuda", False),
                device=str(self.device),
            )
        elif vox_type == "tile":
            self.voxelizer = TileVoxelizer(
                tuple(self.volume_shape),
                tile_size=vox_config.get("tile_size", 8),
                max_radius=vox_config.get("max_radius", 20),
                use_cuda=False,
                strict_cuda=False,
                device=str(self.device),
            )
        else:
            self.voxelizer = Voxelizer(tuple(self.volume_shape), device=str(self.device))

    def _apply_hard_data_consistency(self, volume: torch.Tensor) -> torch.Tensor:
        if not self.config.get("mode", {}).get("apply_hard_data_consistency", True):
            return volume
        pred_kspace = fft3c(volume.unsqueeze(0) * self.sensitivity_maps)
        kspace_dc = pred_kspace * (1.0 - self.mask_coils) + self.kspace_undersampled_complex
        coil_images_dc = ifft3c(kspace_dc)
        return torch.sum(torch.conj(self.sensitivity_maps) * coil_images_dc, dim=0)

    def reconstruct(self) -> torch.Tensor:
        self.gaussian_model.eval()
        with torch.no_grad():
            start = time.time()
            volume = self.voxelizer(
                positions=self.gaussian_model.positions,
                scales=self.gaussian_model.get_scales(),
                rotations=self.gaussian_model.rotations,
                density=self.gaussian_model.get_densities(),
            )
            if isinstance(volume, tuple):
                volume = torch.complex(volume[0], volume[1])
            volume = self._apply_hard_data_consistency(volume)
            print(f"Reconstruction took {time.time() - start:.2f}s")
        return volume

    def save_results(
        self,
        output_dir: str,
        save_volume: bool = True,
        save_slices: bool = True,
        custom_slices: Optional[Dict[str, List[int]]] = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        recon_volume = self.reconstruct()
        mode = self.config.get("mode", {})
        metrics_payload: Dict[str, Any] = {}
        if not mode.get("self_supervised_deploy", False):
            metrics = evaluate_reconstruction(
                recon_volume,
                self.target_image,
                compute_3d_ssim=True,
                compute_lpips_metric=self.config.get("metrics", {}).get("compute_lpips_during_test", False),
                lpips_device=self.device,
                require_lpips=mode.get("require_lpips", False),
            )
            zf_metrics = evaluate_reconstruction(
                self.zero_filled,
                self.target_image,
                compute_3d_ssim=True,
                compute_lpips_metric=self.config.get("metrics", {}).get("compute_lpips_during_test", False),
                lpips_device=self.device,
                require_lpips=mode.get("require_lpips", False),
            )
            metrics_payload = {"3dgsmr": metrics, "zero_filled": zf_metrics}
            print(f"Metrics: {metrics_payload}")
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as handle:
            yaml.dump(metrics_payload, handle)

        recon_np = recon_volume.detach().cpu().numpy()
        target_np = self.target_image.detach().cpu().numpy()
        zf_np = self.zero_filled.detach().cpu().numpy()
        mask_np = self.mask.detach().cpu().numpy()

        if save_volume:
            if savemat is None:
                raise ImportError("Saving .mat requires scipy")
            savemat(os.path.join(output_dir, "reconstruction.mat"), {"reconstruction": recon_np}, do_compression=True)
            savemat(os.path.join(output_dir, "target.mat"), {"target": target_np}, do_compression=True)
            savemat(os.path.join(output_dir, "zero_filled.mat"), {"zero_filled": zf_np}, do_compression=True)
            savemat(os.path.join(output_dir, "mask.mat"), {"mask": mask_np}, do_compression=True)
            if HAS_NIBABEL:
                self._save_nifti(np.abs(recon_np), os.path.join(output_dir, "reconstruction.nii.gz"))
                self._save_nifti(np.abs(target_np), os.path.join(output_dir, "target.nii.gz"))
                self._save_nifti(np.abs(zf_np), os.path.join(output_dir, "zero_filled.nii.gz"))
                self._save_nifti(np.abs(mask_np), os.path.join(output_dir, "mask.nii.gz"))

        if save_slices:
            self._save_comparison_slices(target_np, zf_np, recon_np, mask_np, output_dir, custom_slices)

    @staticmethod
    def _save_nifti(volume_abs, path):
        img = nib.Nifti1Image(volume_abs, np.eye(4))
        nib.save(img, path)

    def _save_comparison_slices(self, target, zf, recon, mask, output_dir, custom_slices=None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        slice_dir = os.path.join(output_dir, "slices")
        os.makedirs(slice_dir, exist_ok=True)

        def get_mag(arr):
            if arr.ndim == 4 and arr.shape[0] == 2:
                return np.sqrt(arr[0] ** 2 + arr[1] ** 2 + 1e-12)
            return np.abs(arr)

        t_mag = get_mag(target)
        z_mag = get_mag(zf)
        r_mag = get_mag(recon)
        m_mag = get_mag(mask)
        vmax = np.percentile(t_mag, 99.9)
        d_size, h_size, w_size = t_mag.shape
        slices_cfg = self.config.get("test", {}).get("slices", {})

        def get_indices(dim_name, max_dim, arg_val):
            if arg_val is not None:
                return arg_val
            if dim_name in slices_cfg:
                return slices_cfg[dim_name]
            return [max_dim // 2]

        slices_map = {
            "Axial": (0, get_indices("axial", d_size, custom_slices.get("axial") if custom_slices else None)),
            "Coronal": (1, get_indices("coronal", h_size, custom_slices.get("coronal") if custom_slices else None)),
            "Sagittal": (2, get_indices("sagittal", w_size, custom_slices.get("sagittal") if custom_slices else None)),
        }

        for view_name, (dim, indices) in slices_map.items():
            for idx in indices:
                if idx >= t_mag.shape[dim]:
                    continue
                if dim == 0:
                    images = [t_mag[idx, :, :], z_mag[idx, :, :], r_mag[idx, :, :], m_mag[idx, :, :]]
                elif dim == 1:
                    images = [t_mag[:, idx, :], z_mag[:, idx, :], r_mag[:, idx, :], m_mag[:, idx, :]]
                else:
                    images = [t_mag[:, :, idx], z_mag[:, :, idx], r_mag[:, :, idx], m_mag[:, :, idx]]
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                titles = ["Original (GT)", "Undersampled (ZF)", "Reconstruction", "K-Space Mask"]
                for ax, img, title in zip(axes, images, titles):
                    value_limit = 1.0 if title == "K-Space Mask" else vmax
                    ax.imshow(img, cmap="gray", vmin=0, vmax=value_limit)
                    ax.set_title(title)
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(slice_dir, f"{view_name.lower()}_{idx:03d}.png"), dpi=200)
                plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    config = None
    if args.config is not None:
        with open(args.config, "r") as handle:
            config = yaml.safe_load(handle)
    tester = GaussianTester(
        checkpoint_path=args.weights,
        config=config,
        device=device,
        data_path=args.dataset,
        acceleration_override=args.acceleration,
    )
    custom_slices = {
        "axial": args.slices_axial,
        "coronal": args.slices_coronal,
        "sagittal": args.slices_sagittal,
    }
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.weights), "test_results")
    tester.save_results(output_dir, save_volume=args.save_volume, save_slices=args.save_slices, custom_slices=custom_slices)


if __name__ == "__main__":
    main()
