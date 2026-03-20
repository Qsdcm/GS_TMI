"""
3DGSMR Testing/Inference Entry Point (Enhanced)

功能:
1. 加载训练好的模型
2. 执行重建
3. 自动保存: 原图(GT), 欠采样(ZF), 重建(Recon), Mask
4. 生成对比切片图像 (支持自定义切片位置)
"""

import os
import argparse
import yaml
import torch
import numpy as np
import time
import random
from typing import Dict, Any, List

try:
    from scipy.io import savemat
except Exception:  # pragma: no cover
    savemat = None

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from data import MRIDataset
from gaussian import GaussianModel3D, Voxelizer, TileVoxelizer
from metrics import evaluate_reconstruction

def parse_args():
    parser = argparse.ArgumentParser(description='Test 3DGSMR for MRI Reconstruction')
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--acceleration', type=float, default=1.0, help='Acceleration factor')
    
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for mask generation')
    
    parser.add_argument('--save_volume', action='store_true', default=True, help='Save volumes as .mat/.nii')
    parser.add_argument('--save_slices', action='store_true', default=True, help='Save slice comparison images')
    
    # 新增: 自定义切片索引参数
    parser.add_argument('--slices_axial', nargs='+', type=int, help='Slice indices for Axial view (e.g. 50 100 150)')
    parser.add_argument('--slices_coronal', nargs='+', type=int, help='Slice indices for Coronal view')
    parser.add_argument('--slices_sagittal', nargs='+', type=int, help='Slice indices for Sagittal view')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class GaussianTester:
    def __init__(self, checkpoint_path: str, config: Dict[str, Any] = None, device: torch.device = None, 
                 data_path: str = None, acceleration_override: float = None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acceleration_override = acceleration_override
        
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if config is not None:
            self.config = config
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found.")
        
        if data_path is not None:
            self.config['data']['data_path'] = data_path
        
        self._setup_data()
        self._setup_model(checkpoint)
        
    def _setup_data(self):
        data_config = self.config['data']
        acc_factor = self.acceleration_override if self.acceleration_override is not None else data_config['acceleration_factor']
        
        print(f"Loading data from: {data_config['data_path']}")
        print(f"Using Acceleration Factor: {acc_factor}x")
        
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=acc_factor,
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=int(data_config.get('acs_lines', int(data_config.get('center_fraction', 0.08) * 256)))
        )
        
        data = self.dataset.get_data()
        self.volume_shape = data['volume_shape']
        self.target_image = data['ground_truth'].to(self.device)
        self.zero_filled = data['zero_filled'].to(self.device)
        # 保存 Mask 用于可视化
        self.mask = data['mask'].to(self.device)
        
    def _setup_model(self, checkpoint: Dict):
        gaussian_state = checkpoint['gaussian_state']
        num_points = gaussian_state['positions'].shape[0]
        
        self.gaussian_model = GaussianModel3D(
            num_points=num_points,
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        
        self.voxelizer = self._create_voxelizer(tuple(self.volume_shape))

    def _create_voxelizer(self, volume_shape):
        """Create voxelizer from config, matching trainer logic."""
        vox_config = self.config.get('voxelizer', {})
        vox_type = vox_config.get('type', 'chunk')
        tile_size = vox_config.get('tile_size', 8)
        max_radius = vox_config.get('max_radius', 20)
        if vox_type == 'tile_cuda':
            return TileVoxelizer(volume_shape, tile_size, max_radius, use_cuda=True, device=str(self.device))
        elif vox_type == 'tile':
            return TileVoxelizer(volume_shape, tile_size, max_radius, use_cuda=False, device=str(self.device))
        else:
            return Voxelizer(volume_shape, device=str(self.device))
        
    def reconstruct(self) -> torch.Tensor:
        self.gaussian_model.eval()
        with torch.no_grad():
            t0 = time.time()
            volume = self.voxelizer(
                positions=self.gaussian_model.positions,
                scales=self.gaussian_model.get_scales(),
                rotations=self.gaussian_model.rotations,
                density=self.gaussian_model.get_densities()
            )
            print(f"Reconstruction took {time.time()-t0:.2f}s")
        return volume
    
    def save_results(self, output_dir: str, save_volume: bool = True, save_slices: bool = True, 
                     custom_slices: Dict[str, List[int]] = None):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running reconstruction...")
        recon_volume = self.reconstruct()
        
        metrics = evaluate_reconstruction(recon_volume, self.target_image, compute_3d_ssim=True)
        zf_metrics = evaluate_reconstruction(self.zero_filled, self.target_image, compute_3d_ssim=True)
        
        print("\n" + "="*50)
        print(f"PSNR: {zf_metrics['psnr']:.2f} -> {metrics['psnr']:.2f} dB")
        print(f"SSIM: {zf_metrics['ssim']:.4f} -> {metrics['ssim']:.4f}")
        print("="*50)
        
        with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
            yaml.dump({'3dgsmr': metrics, 'zero_filled': zf_metrics}, f)

        recon_np = recon_volume.detach().cpu().numpy()
        target_np = self.target_image.detach().cpu().numpy()
        zf_np = self.zero_filled.detach().cpu().numpy()
        mask_np = self.mask.detach().cpu().numpy() # Mask 也转为 numpy
        
        if save_volume:
            print(f"\nSaving volumes to {output_dir} ...")
            if savemat is None:
                raise ImportError(
                    "Saving .mat requires scipy. Please install it via: pip install scipy"
                )

            savemat(os.path.join(output_dir, 'reconstruction.mat'), {'reconstruction': recon_np}, do_compression=True)
            savemat(os.path.join(output_dir, 'target.mat'), {'target': target_np}, do_compression=True)
            savemat(os.path.join(output_dir, 'zero_filled.mat'), {'zero_filled': zf_np}, do_compression=True)
            savemat(os.path.join(output_dir, 'mask.mat'), {'mask': mask_np}, do_compression=True)
            
            if HAS_NIBABEL:
                self._save_nifti(np.abs(recon_np), os.path.join(output_dir, 'reconstruction.nii.gz'))
                self._save_nifti(np.abs(target_np), os.path.join(output_dir, 'target.nii.gz'))
                self._save_nifti(np.abs(zf_np), os.path.join(output_dir, 'zero_filled.nii.gz'))
                # 保存 mask 的 nifti，通常看幅度即可 (0或1)
                self._save_nifti(np.abs(mask_np), os.path.join(output_dir, 'mask.nii.gz'))
        
        if save_slices:
            print("Generating comparison slices...")
            self._save_comparison_slices(target_np, zf_np, recon_np, mask_np, output_dir, custom_slices)
            
    def _save_nifti(self, volume_abs, path):
        img = nib.Nifti1Image(volume_abs, np.eye(4))
        nib.save(img, path)

    def _save_comparison_slices(self, target, zf, recon, mask, output_dir, custom_slices=None):
        """生成四图对比切片: GT | Zero-Filled | Recon | Mask"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        slice_dir = os.path.join(output_dir, 'slices')
        os.makedirs(slice_dir, exist_ok=True)
        
        # Helper to get magnitude
        def get_mag(arr):
            if arr.ndim == 4 and arr.shape[0] == 2:
                return np.sqrt(arr[0]**2 + arr[1]**2)
            return np.abs(arr)

        t_mag = get_mag(target)
        z_mag = get_mag(zf)
        r_mag = get_mag(recon)
        m_mag = get_mag(mask)
        
        vmax = np.percentile(t_mag, 99.9)
        D, H, W = t_mag.shape
        
        # 确定切片索引：Args > Config > Default(Center)
        slices_cfg = self.config.get('test', {}).get('slices', {})
        
        def get_indices(dim_name, max_dim, arg_val):
            if arg_val is not None: return arg_val
            if dim_name in slices_cfg: return slices_cfg[dim_name]
            return [max_dim // 2] # Fallback to center
            
        slices_map = {
            'Axial': (0, get_indices('axial', D, custom_slices.get('axial') if custom_slices else None)),
            'Coronal': (1, get_indices('coronal', H, custom_slices.get('coronal') if custom_slices else None)),
            'Sagittal': (2, get_indices('sagittal', W, custom_slices.get('sagittal') if custom_slices else None))
        }
        
        for view_name, (dim, idxs) in slices_map.items():
            for idx in idxs:
                if idx >= target.shape[dim]: continue
                
                if dim == 0:
                    imgs = [t_mag[idx,:,:], z_mag[idx,:,:], r_mag[idx,:,:], m_mag[idx,:,:]]
                elif dim == 1:
                    imgs = [t_mag[:,idx,:], z_mag[:,idx,:], r_mag[:,idx,:], m_mag[:,idx,:]]
                else:
                    imgs = [t_mag[:,:,idx], z_mag[:,:,idx], r_mag[:,:,idx], m_mag[:,:,idx]]
                
                # 修改为 1行4列
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                titles = ['Original (GT)', 'Undersampled (ZF)', 'Reconstruction', 'K-Space Mask']
                
                for ax, img, title in zip(axes, imgs, titles):
                    # Mask 显示 0-1
                    v_lim = 1.0 if title == 'K-Space Mask' else vmax
                    ax.imshow(img, cmap='gray', vmin=0, vmax=v_lim)
                    ax.set_title(title)
                    ax.axis('off')
                
                plt.suptitle(f"{view_name} Slice {idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(slice_dir, f'{view_name}_slice_{idx:03d}.png'), dpi=150)
                plt.close()
        print(f"Saved slice comparisons to {slice_dir}")

def main():
    args = parse_args()
    
    # 强制同步种子
    set_seed(args.seed)
    
    # 路径逻辑 (Task 2: Testing output)
    if args.output_dir:
        save_dir = args.output_dir
    else:
        # 默认保存到 weights 所在目录的上级目录下的 test_results
        # e.g. .../exp_name/checkpoints/best.pth -> .../exp_name/test_results_8x
        weights_dir = os.path.dirname(args.weights)
        exp_dir = os.path.dirname(weights_dir)
        acc_tag = int(args.acceleration) if args.acceleration.is_integer() else args.acceleration
        save_dir = os.path.join(exp_dir, f"test_results_{acc_tag}x")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
    tester = GaussianTester(
        checkpoint_path=args.weights,
        config=config, 
        device=device, 
        data_path=args.dataset,
        acceleration_override=args.acceleration
    )
    
    # 收集自定义切片参数
    custom_slices = {
        'axial': args.slices_axial,
        'coronal': args.slices_coronal,
        'sagittal': args.slices_sagittal
    }
    
    tester.save_results(save_dir, args.save_volume, args.save_slices, custom_slices)

if __name__ == '__main__':
    main()