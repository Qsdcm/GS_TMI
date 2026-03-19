"""
MRI Dataset for 3DGSMR (Fixed Mask Generation)
修正了Mask生成逻辑：从3D随机点采样改为符合MRI物理的2D相位编码方向欠采样。
这会产生正确的混叠伪影(Aliasing artifacts)而非仅仅是模糊。
"""
 
import os
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
from .transforms import ifft3c, fft3c, normalize_kspace

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


class MRIDataset(Dataset):
    """
    MRI数据集类
    
    从h5/.mat文件加载k-space数据，支持：
    - 多线圈数据合并
    - k-space归一化
    - 欠采样mask生成 (Fixed: 2D Phase Encoding Mask)
    """
    
    def __init__(
        self,
        data_path: str,
        acceleration_factor: int = 4,
        mask_type: str = "gaussian",
        use_acs: bool = True,
        acs_lines: int = 24,
        csm_path: Optional[str] = None,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.data_path = data_path
        self.acceleration_factor = acceleration_factor
        self.mask_type = mask_type
        self.use_acs = use_acs
        self.acs_lines = acs_lines
        self.csm_path = csm_path
        self.device = device
        
        self._load_data()
        
    def _load_data(self):
        """加载h5/.mat文件中的k-space数据"""
        print(f"Loading data from {self.data_path}")

        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == '.mat':
            kspace_data = self._load_mat_kspace()
        else:
            kspace_data = self._load_h5_kspace()

        kspace_data = self._normalize_kspace_layout(kspace_data)

        # kspace_data shape: (num_coils, kx, ky, kz)
        self.kspace_multicoil = torch.from_numpy(kspace_data).to(torch.complex64)
        self.num_coils = self.kspace_multicoil.shape[0]
        self.volume_shape = self.kspace_multicoil.shape[1:]  # (kx, ky, kz)
        
        print(f"Loaded k-space data with shape: {self.kspace_multicoil.shape}")
        print(f"Number of coils: {self.num_coils}")
        print(f"Volume shape: {self.volume_shape}")
        
        self._process_data()

    def _load_h5_kspace(self) -> np.ndarray:
        if h5py is None:
            raise ImportError("Reading .h5 k-space files requires h5py to be installed.")
        with h5py.File(self.data_path, 'r') as f:
            keys = list(f.keys())
            preferred_keys = ['kspace', 'ksp']
            key = next((name for name in preferred_keys if name in f), None)
            if key is None:
                key = keys[0]
            print(f"Using H5 key: {key}")
            return f[key][:]

    def _load_mat_kspace(self) -> np.ndarray:
        variables = scipy.io.whosmat(self.data_path)
        if len(variables) == 0:
            raise ValueError(f"No variables found in MATLAB file: {self.data_path}")

        preferred_keys = ['kspace', 'ksp', 'raw_kspace', 'rawdata', 'data']
        available_names = [name for name, _, _ in variables]

        key = None
        for preferred in preferred_keys:
            match = next((name for name in available_names if name.lower() == preferred), None)
            if match is not None:
                key = match
                break

        if key is None:
            key = max(variables, key=lambda item: np.prod(item[1]))[0]

        print(f"Using MATLAB variable: {key}")
        mat = scipy.io.loadmat(self.data_path, variable_names=[key])
        return mat[key]

    def _normalize_kspace_layout(self, kspace_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(kspace_data)
        arr = np.squeeze(arr)

        if not np.iscomplexobj(arr):
            arr = self._maybe_convert_real_imag_to_complex(arr)

        if arr.ndim != 4:
            raise ValueError(
                f"Expected a 4D k-space array after normalization, got shape {arr.shape}"
            )

        coil_axis = self._infer_coil_axis(arr.shape)
        if coil_axis != 0:
            old_shape = arr.shape
            arr = np.moveaxis(arr, coil_axis, 0)
            print(f"Moved coil axis from dim {coil_axis} to dim 0: {old_shape} -> {arr.shape}")
        else:
            print(f"Detected coil axis at dim 0: {arr.shape}")

        if not np.iscomplexobj(arr):
            raise ValueError(
                f"K-space data must be complex-valued, but got dtype {arr.dtype}"
            )

        return np.ascontiguousarray(arr)

    def _maybe_convert_real_imag_to_complex(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 5:
            complex_axis = None
            if arr.shape[-1] == 2:
                complex_axis = arr.ndim - 1
            elif arr.shape[0] == 2:
                complex_axis = 0
            else:
                candidates = [axis for axis, size in enumerate(arr.shape) if size == 2]
                if len(candidates) == 1:
                    complex_axis = candidates[0]

            if complex_axis is not None:
                arr = np.moveaxis(arr, complex_axis, -1)
                print(f"Merged real/imag channels from dim {complex_axis}: {arr.shape[:-1]}")
                return arr[..., 0] + 1j * arr[..., 1]

        return arr

    def _infer_coil_axis(self, shape: Tuple[int, ...]) -> int:
        candidates = [axis for axis, size in enumerate(shape) if 1 <= size <= 64]
        if len(candidates) == 0:
            return 0
        if len(candidates) == 1:
            return candidates[0]

        if shape[-1] <= 64 and shape[0] > 64:
            return len(shape) - 1
        if shape[0] <= 64 and shape[-1] > 64:
            return 0

        return min(candidates, key=lambda axis: shape[axis])
        
    def _process_data(self):
        """Standardized Data Processing Pipeline"""
        
        # 1. Acquire HIGH-RES CSM for Ground Truth generation
        #    Priority: External Path > Estimated from Full k-space
        csm_gt = self._get_best_available_csm()
        
        # 2. Generate Ground Truth Image (SENSE-1 combination)
        #    Use Full K-space + High-Res CSM
        images_multicoil = ifft3c(self.kspace_multicoil)
        self.ground_truth_image = torch.sum(
            torch.conj(csm_gt) * images_multicoil, 
            dim=0
        )
        self.kspace_full = fft3c(self.ground_truth_image)
        print(f"Ground truth generated using High-Res CSM. Shape: {self.ground_truth_image.shape}")
        
        # 3. Generate Sampling Mask
        self._generate_mask()
        
        # 4. Determine Input CSM for Reconstruction (Avoid Data Leakage)
        #    If external CSM provided: Use it (Assume it's intended for recon)
        #    If NO external CSM: Use ACS-estimated CSM (Self-Calibration)
        if self.csm_path and self.csm_path != "None":
             self.sensitivity_maps = csm_gt # External CSM is used for both
             print("Using External CSM for input.")
        else:
             self.sensitivity_maps = self._estimate_csm_from_acs()
             print("Using ACS-estimated CSM for input (No Data Leakage).")

        # 5. Apply Undersampling & Generate Zero-Filled
        self._apply_undersampling()

    def _get_best_available_csm(self):
        """Get best possible CSM for GT generation"""
        # Try loading external
        if self.csm_path and self.csm_path != "None":
            try:
                csm = self._load_external_csm(self.csm_path)
                return csm
            except Exception as e:
                print(f"Failed to load external CSM for GT: {e}")
        
        # Fallback: Estimate from FULL k-space
        print("Estimating High-Res CSM from Full k-space (For GT generation ONLY)...")
        images_multicoil = ifft3c(self.kspace_multicoil)
        rss = torch.sqrt(torch.sum(torch.abs(images_multicoil) ** 2, dim=0))
        rss = rss + 1e-12
        return images_multicoil / rss.unsqueeze(0)

    def _estimate_csm_from_acs(self):
        """Estimate CSM from ACS region (Low-Res)"""
        print(f"Estimating Low-Res CSM from ACS region (center {self.acs_lines} lines)...")
        
        # Create ACS mask
        acs_mask = torch.zeros_like(self.kspace_multicoil, dtype=torch.float32)
        _, kx, ky, kz = self.kspace_multicoil.shape
        cy, cz = ky // 2, kz // 2
        half_acs = self.acs_lines // 2
        
        # Keep full Readout kx, crop Phase Encoding ky, kz
        y_start = max(0, cy - half_acs)
        y_end = min(ky, cy + half_acs)
        z_start = max(0, cz - half_acs)
        z_end = min(kz, cz + half_acs)
        
        acs_mask[:, :, y_start:y_end, z_start:z_end] = 1.0
        
        # Get ACS data
        kspace_acs = self.kspace_multicoil * acs_mask
        
        # Estiamte
        images_acs = ifft3c(kspace_acs)
        rss = torch.sqrt(torch.sum(torch.abs(images_acs) ** 2, dim=0))
        rss = rss + 1e-12
        return images_acs / rss.unsqueeze(0)

    def _load_external_csm(self, path):
        print(f"Loading external CSM from {path}")
        mat = scipy.io.loadmat(path)
        possible_keys = ['csm', 'sensitivity_maps', 'maps', 'S']
        csm_key = next((k for k in possible_keys if k in mat), None)
        
        if csm_key is None:
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(keys) > 0:
                csm_key = max(keys, key=lambda k: mat[k].size)
        
        if csm_key:
            csm_data = mat[csm_key]
        else:
            raise ValueError("Could not find CSM variable in .mat file")

        csm = torch.from_numpy(csm_data)
        if not torch.is_complex(csm):
                csm = csm.to(torch.complex64)

        if csm.shape != self.kspace_multicoil.shape:
            # Simple Permute Check
            if csm.shape[0] != self.num_coils and csm.shape[-1] == self.num_coils:
                 csm = csm.permute(3, 0, 1, 2)
        
        return csm

    def _combine_coils(self):
        # Deprecated by _process_data, keeping for compatibility if called externally
        pass 
        
    def _estimate_sensitivity_maps(self):
         # Deprecated by _process_data, keeping for compatibility if called externally
        self._process_data()
        
    def _generate_mask(self):
        """生成欠采样mask"""
        shape = self.volume_shape
        
        if self.mask_type == "gaussian":
            mask = self._generate_gaussian_mask(shape)
        elif self.mask_type == "poisson":
            mask = self._generate_poisson_mask(shape)
        else:  # random
            mask = self._generate_random_mask(shape)
            
        if self.use_acs:
            mask = self._add_acs_region(mask)
            
        self.mask = mask
        actual_acc = mask.numel() / mask.sum().item()
        print(f"Generated {self.mask_type} mask (2D Phase Encoding) with actual acceleration: {actual_acc:.2f}x")
        
    def _generate_gaussian_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        生成Gaussian采样mask (2D Phase Encoding)
        保留kx(Readout)全采样，只在ky-kz平面欠采样
        """
        kx, ky, kz = shape
        
        # 只在 ky-kz 平面计算
        total_lines = ky * kz
        target_lines = int(total_lines / self.acceleration_factor)
        
        cy, cz = ky // 2, kz // 2
        
        y = torch.arange(ky) - cy
        z = torch.arange(kz) - cz
        
        yy, zz = torch.meshgrid(y, z, indexing='ij')
        
        # 2D Gaussian PDF
        sigma = min(ky, kz) / 4
        prob = torch.exp(-(yy**2 + zz**2) / (2 * sigma**2))
        
        # Normalize to match target acceleration
        prob = prob / prob.sum() * target_lines
        prob = torch.clamp(prob, 0, 1)
        
        # 2D Sampling
        mask_2d = (torch.rand((ky, kz)) < prob).float()
        
        # Expand to 3D (kx is fully sampled)
        mask_3d = mask_2d.unsqueeze(0).expand(kx, ky, kz)
        
        return mask_3d
    
    def _generate_poisson_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        生成Poisson-Disc mask (Variable Density) - Imitating vdPoisMex
        使用 Variable Density Poisson Disc Sampling 算法
        """
        from .mask_generator import gen_poisson_mask
        kx, ky, kz = shape
        
        # Generate 2D mask using the Poisson Disc algorithm
        # Returns shape (ky, kz)
        mask_2d_np = gen_poisson_mask((ky, kz), self.acceleration_factor, self.acs_lines)
        
        mask_2d = torch.from_numpy(mask_2d_np).float()
        
        # Expand to 3D (kx is fully sampled)
        mask_3d = mask_2d.unsqueeze(0).expand(kx, ky, kz)
        
        return mask_3d
    
    def _generate_random_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机欠采样mask (2D Phase Encoding)"""
        kx, ky, kz = shape
        prob = 1.0 / self.acceleration_factor
        
        # 2D Sampling
        mask_2d = (torch.rand((ky, kz)) < prob).float()
        
        # Expand to 3D
        mask_3d = mask_2d.unsqueeze(0).expand(kx, ky, kz)
        
        return mask_3d
    
    def _add_acs_region(self, mask: torch.Tensor) -> torch.Tensor:
        """添加ACS（全采样中心）区域"""
        kx, ky, kz = mask.shape
        cy, cz = ky // 2, kz // 2
        half_acs = self.acs_lines // 2
        
        # 修复：mask可能是expand生成的视图，必须clone后才能进行in-place修改
        # 如果不clone，会报 "more than one element... refers to a single memory location"
        mask = mask.clone()
        
        # 只在 ky-kz 平面中心添加，贯穿所有 kx
        mask[:, cy-half_acs:cy+half_acs, cz-half_acs:cz+half_acs] = 1.0
        
        return mask
        
    def _apply_undersampling(self):
        """应用欠采样并生成Zero-filled图像"""
        # mask shape: (kx, ky, kz)
        # kspace_multicoil shape: (coils, kx, ky, kz)
        
        # Expand mask for coils
        mask_expanded = self.mask.unsqueeze(0)
        
        # Undersampled multi-coil k-space
        kspace_undersampled_complex = self.kspace_multicoil * mask_expanded
        
        # Zero-filled reconstruction (SENSE-1 like)
        # sum(conj(csm) * ifft3d(x), chan)
        image_multicoil = ifft3c(kspace_undersampled_complex)
        zero_filled_complex = torch.sum(
            torch.conj(self.sensitivity_maps) * image_multicoil,
            dim=0
        ) # (kx, ky, kz)
        
        # Prepare outputs in user specified format
        # Zero-filled: [2, kx, ky, kz]
        self.zero_filled_image = torch.stack([
            zero_filled_complex.real,
            zero_filled_complex.imag
        ], dim=0)
        
        # K-space undersampled: [2*coils, kx, ky, kz]
        # User said [24, ...], assuming 12 coils.
        # kspace_undersampled_complex is (coils, kx, ky, kz)
        self.kspace_undersampled = torch.cat([
            kspace_undersampled_complex.real,
            kspace_undersampled_complex.imag
        ], dim=0)
        
        print(f"Applied undersampling. Non-zero ratio: {self.mask.sum()/self.mask.numel():.4f}")
        print(f"Zero-filled shape: {self.zero_filled_image.shape}")
        print(f"Undersampled k-space shape: {self.kspace_undersampled.shape}")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        return {
            'kspace_full': self.kspace_full,
            'kspace_undersampled': self.kspace_undersampled,
            'mask': self.mask,
            'ground_truth': self.ground_truth_image,
            'zero_filled': self.zero_filled_image,
            'sensitivity_maps': self.sensitivity_maps,
            'volume_shape': self.volume_shape
        }
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.get_data()
