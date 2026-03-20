import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '3dgsVC'))

from data.dataset import MRIDataset
from data.transforms import fft3c
from gaussian.gaussian_model import GaussianModel3D
from gaussian.tile_voxelizer import TileVoxelizer
from gaussian.voxelizer import Voxelizer
from trainers.trainer import GaussianTrainer


class PaperFaithfulTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = ROOT / 'tests' / '_tmp'
        cls.tmp_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        real = rng.standard_normal((2, 8, 8, 8)).astype(np.float32)
        imag = rng.standard_normal((2, 8, 8, 8)).astype(np.float32)
        cls.kspace = real + 1j * imag
        cls.data_path = cls.tmp_dir / 'synthetic_kspace.mat'
        savemat(cls.data_path, {'kspace': cls.kspace})

    def test_complex_gaussian_parameterization(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        positions = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        scales = torch.tensor([[0.20, 0.20, 0.20]], device=device)
        density = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device=device)
        model = GaussianModel3D(
            num_points=1,
            volume_shape=(8, 8, 8),
            initial_positions=positions,
            initial_densities=density,
            initial_scales=scales,
            device=device,
        )
        voxelizer = Voxelizer((8, 8, 8), device=device)
        volume = voxelizer(model.positions, model.get_scale_values(), model.rotations, model.density)
        peak = volume.flatten()[torch.abs(volume).argmax()]
        self.assertGreater(abs(float(peak.real)), 0.0)
        self.assertGreater(abs(float(peak.imag)), 0.0)
        self.assertAlmostEqual(float(peak.imag / peak.real), 2.0, places=3)

    def test_initialization_from_ifft_random_grid_and_exact_scale(self):
        torch.manual_seed(0)
        image = torch.complex(
            torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6),
            torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6) * 0.5,
        )
        model = GaussianModel3D.from_image(image=image, num_points=10, density_scale_k=0.2, init_mode='random', device='cpu')
        rotations = model.rotations.detach()
        self.assertTrue(torch.allclose(rotations[:, 0], torch.ones_like(rotations[:, 0])))
        self.assertTrue(torch.allclose(rotations[:, 1:], torch.zeros_like(rotations[:, 1:])))

        shape = torch.tensor(image.shape, dtype=torch.float32)
        voxel_coords = torch.round((model.positions.detach() + 1.0) * 0.5 * shape - 0.5).long()
        expected_scales = GaussianModel3D._exact_grid_scale_init(voxel_coords, image.shape, torch.device('cpu'))
        self.assertTrue(torch.allclose(model.get_scale_values().detach(), expected_scales, atol=1e-6, rtol=1e-6))

        z, y, x = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
        expected_density = image[z, y, x] * 0.2
        self.assertTrue(torch.allclose(model.density.detach(), expected_density, atol=1e-6, rtol=1e-6))

    def test_mask_geometry(self):
        torch.manual_seed(0)
        dataset = MRIDataset(
            data_path=str(self.data_path),
            acceleration_factor=4,
            mask_type='stacked_2d_gaussian',
            use_acs=True,
            acs_lines=2,
            readout_axis=0,
            phase_axes=(1, 2),
            device='cpu',
        )
        mask = dataset.mask
        readout_size = mask.shape[0]
        collapsed = mask.sum(dim=0)
        self.assertTrue(torch.all((collapsed == 0) | (collapsed == readout_size)))
        cy = mask.shape[1] // 2
        cz = mask.shape[2] // 2
        self.assertTrue(torch.all(mask[:, cy - 1:cy + 1, cz - 1:cz + 1] == 1.0))
        self.assertLess(mask.sum().item(), mask.numel())

    def test_forward_model_consistency(self):
        volume = torch.complex(torch.randn(8, 8, 8), torch.randn(8, 8, 8))
        csm = torch.complex(torch.ones(2, 8, 8, 8), torch.zeros(2, 8, 8, 8))
        mask = torch.zeros(8, 8, 8)
        mask[:, 2:6, 2:6] = 1.0
        coil_kspaces = []
        for coil_idx in range(csm.shape[0]):
            coil_kspace = fft3c((volume * csm[coil_idx]).unsqueeze(0)).squeeze(0)
            coil_kspaces.append(coil_kspace * mask)
        stacked = torch.stack(coil_kspaces, dim=0)
        self.assertEqual(stacked.shape, (2, 8, 8, 8))
        self.assertTrue(torch.is_complex(stacked))
        self.assertEqual(stacked.dtype, torch.complex64)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for tile_cuda parity test')
    def test_voxelizer_parity_tile_cuda_vs_reference(self):
        torch.manual_seed(0)
        device = 'cuda:0'
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.25, -0.25, 0.25]], device=device)
        scales = torch.tensor([[0.20, 0.15, 0.10], [0.18, 0.18, 0.12]], device=device)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
        density = torch.tensor([1.0 + 0.2j, 0.5 - 0.3j], dtype=torch.complex64, device=device)
        reference = Voxelizer((8, 8, 8), device=device)(positions, scales, rotations, density)
        tile_cuda = TileVoxelizer((8, 8, 8), tile_size=4, max_radius=8, use_cuda=True, strict_cuda=True, device=device)
        cuda_out = tile_cuda(positions, scales, rotations, density)
        cuda_out = torch.complex(cuda_out[0], cuda_out[1])
        max_err = torch.max(torch.abs(reference - cuda_out)).item()
        self.assertLess(max_err, 5e-3)

    def test_original_clone_split_and_prune(self):
        model = GaussianModel3D(
            num_points=1,
            volume_shape=(8, 8, 8),
            initial_positions=torch.zeros(1, 3),
            initial_densities=torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            initial_scales=torch.tensor([[0.20, 0.20, 0.20]]),
            device='cpu',
        )
        split_count = model.densify_and_split(torch.tensor([1.0]), grad_threshold=0.1, scale_threshold=0.05, use_long_axis_splitting=False)
        self.assertEqual(split_count, 1)
        self.assertEqual(model.num_points, 2)
        self.assertTrue(torch.allclose(model.get_scale_values(), torch.full((2, 3), 0.20 / 1.6), atol=1e-6))
        self.assertTrue(torch.allclose(model.density, torch.full((2,), 0.5 + 0.0j, dtype=torch.complex64), atol=1e-6))

        clone_model = GaussianModel3D(
            num_points=1,
            volume_shape=(8, 8, 8),
            initial_positions=torch.zeros(1, 3),
            initial_densities=torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            initial_scales=torch.tensor([[0.01, 0.01, 0.01]]),
            device='cpu',
        )
        clone_count = clone_model.densify_and_clone(torch.tensor([1.0]), grad_threshold=0.1, scale_threshold=0.05)
        self.assertEqual(clone_count, 1)
        self.assertEqual(clone_model.num_points, 2)
        self.assertTrue(torch.allclose(clone_model.density, torch.full((2,), 0.5 + 0.0j, dtype=torch.complex64), atol=1e-6))

        prune_model = GaussianModel3D(
            num_points=2,
            volume_shape=(8, 8, 8),
            initial_positions=torch.zeros(2, 3),
            initial_densities=torch.tensor([0.001 + 0.0j, 0.1 + 0.0j], dtype=torch.complex64),
            initial_scales=torch.full((2, 3), 0.05),
            device='cpu',
        )
        pruned = prune_model.prune(0.01)
        self.assertEqual(pruned, 1)
        self.assertEqual(prune_model.num_points, 1)

    def test_long_axis_split_behavior(self):
        model = GaussianModel3D(
            num_points=1,
            volume_shape=(8, 8, 8),
            initial_positions=torch.zeros(1, 3),
            initial_densities=torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            initial_scales=torch.tensor([[0.40, 0.20, 0.10]]),
            device='cpu',
        )
        split_count = model.densify_and_split(torch.tensor([1.0]), grad_threshold=0.1, scale_threshold=0.05, use_long_axis_splitting=True, long_axis_offset_factor=1.0)
        self.assertEqual(split_count, 1)
        self.assertEqual(model.num_points, 2)
        child_scales = model.get_scale_values()
        self.assertTrue(torch.allclose(child_scales[:, 0], torch.full((2,), 0.20), atol=1e-6))
        self.assertTrue(torch.allclose(child_scales[:, 1], torch.full((2,), 0.17), atol=1e-6))
        self.assertTrue(torch.allclose(child_scales[:, 2], torch.full((2,), 0.085), atol=1e-6))
        self.assertTrue(torch.allclose(model.density, torch.full((2,), 0.6 + 0.0j, dtype=torch.complex64), atol=1e-6))

    def test_densify_every_100_and_cap(self):
        cfg = yaml.safe_load((ROOT / '3dgsVC' / 'configs' / 'legacy_nonpaper.yaml').read_text())
        cfg['data']['data_path'] = str(self.data_path)
        cfg['output']['output_dir'] = str(self.tmp_dir / 'trainer_out')
        cfg['output']['use_tensorboard'] = False
        cfg['data']['acceleration_factor'] = 8
        cfg['gaussian']['initial_num_points'] = 1
        cfg['gaussian']['max_num_points'] = 4
        cfg['gaussian']['init_mode'] = 'random'
        cfg['data']['mask_type'] = 'stacked_2d_gaussian'
        cfg['data']['use_acs'] = False
        cfg['voxelizer']['type'] = 'chunk'
        cfg['adaptive_control']['strategy'] = 'original'
        trainer = GaussianTrainer(cfg, device=torch.device('cpu'))
        trainer.gaussian_model.scales = torch.nn.Parameter(torch.log(torch.tensor([[0.20, 0.20, 0.20]])))
        trainer._position_grads_for_densify = torch.tensor([[1.0, 1.0, 1.0]])
        stats_50 = trainer.adaptive_density_control(50)
        self.assertEqual(stats_50, {'split': 0, 'clone': 0, 'prune': 0})
        stats_100 = trainer.adaptive_density_control(100)
        self.assertGreaterEqual(stats_100['split'] + stats_100['clone'], 1)
        trainer.config['gaussian']['max_num_points'] = trainer.gaussian_model.num_points
        trainer._position_grads_for_densify = torch.ones_like(trainer.gaussian_model.positions)
        stats_cap = trainer.adaptive_density_control(200)
        self.assertEqual(stats_cap, {'split': 0, 'clone': 0, 'prune': 0})

    def test_config_branches(self):
        lowmid = yaml.safe_load((ROOT / '3dgsVC' / 'configs' / 'paper_lowmid.yaml').read_text())
        highacc = yaml.safe_load((ROOT / '3dgsVC' / 'configs' / 'paper_highacc.yaml').read_text())
        legacy = yaml.safe_load((ROOT / '3dgsVC' / 'configs' / 'legacy_nonpaper.yaml').read_text())
        self.assertEqual(lowmid['adaptive_control']['strategy'], 'original')
        self.assertEqual(lowmid['gaussian']['initial_num_points'], 200000)
        self.assertEqual(highacc['adaptive_control']['strategy'], 'long_axis')
        self.assertEqual(highacc['gaussian']['initial_num_points'], 500)
        self.assertEqual(legacy['data']['mask_type'], 'poisson')
        self.assertEqual(legacy['voxelizer']['type'], 'chunk')


if __name__ == '__main__':
    unittest.main()
