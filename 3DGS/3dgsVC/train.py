"""
3DGSMR Training Entry Point

用法:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/latest.pth

论文: Three-Dimensional MRI Reconstruction with Gaussian Representations:
      Tackling the Undersampling Problem
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random

from trainers import GaussianTrainer
# 引入 Tester 以实现自动测试
from test import GaussianTester


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3DGSMR for MRI Reconstruction')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data_path', type=str, default=None, help='Override data path in config')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory base')
    parser.add_argument('--acceleration', type=int, default=None, help='Override acceleration factor')
    parser.add_argument('--max_iterations', type=int, default=None, help='Override max iterations')
    parser.add_argument('--initial_points', type=int, default=None, help='Override initial number of Gaussian points')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # [新增] 训练时的自动测试切片参数
    parser.add_argument('--slices_axial', nargs='+', type=int, help='Test slices for Axial view')
    parser.add_argument('--slices_coronal', nargs='+', type=int, help='Test slices for Coronal view')
    parser.add_argument('--slices_sagittal', nargs='+', type=int, help='Test slices for Sagittal view')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args) -> dict:
    """使用命令行参数覆盖配置"""
    if args.data_path is not None:
        config['data']['data_path'] = args.data_path
    
    # 注意：这里只覆盖 base output dir，具体的子文件夹在 main 中生成
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    
    if args.acceleration is not None:
        config['data']['acceleration_factor'] = args.acceleration
    
    if args.max_iterations is not None:
        config['training']['max_iterations'] = args.max_iterations
    
    if args.initial_points is not None:
        config['gaussian']['initial_num_points'] = args.initial_points
    
    if args.seed is not None:
        config['training']['seed'] = args.seed
        
    return config


def main():
    args = parse_args()
    
    # 1. 设置随机种子
    set_seed(args.seed)
    
    # 2. 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 3. 加载并覆盖配置
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    config = override_config(config, args)
    
    # 4. 自动生成详细的子文件夹名称 (Task 2)
    # 格式: outputs/acc{X}_pts{Y}_grad{Z}_seed{W}
    base_output_dir = config['output']['output_dir']
    acc = config['data']['acceleration_factor']
    pts = config['gaussian']['initial_num_points']
    grad_thresh = config['adaptive_control']['grad_threshold']
    seed = config['training']['seed']
    
    # 处理 grad_thresh 显示，避免过长的小数
    exp_subfolder = f"acc{acc}_pts{pts}_grad{grad_thresh}_seed{seed}"
    full_output_dir = os.path.join(base_output_dir, exp_subfolder)
    
    # 更新 config 中的 output_dir，这样 Trainer 内部会直接使用这个完整路径
    config['output']['output_dir'] = full_output_dir
    
    # 打印关键配置
    print("\n" + "=" * 50)
    print("3DGSMR Training Configuration")
    print("=" * 50)
    print(f"Data path:       {config['data']['data_path']}")
    print(f"Acceleration:    {acc}x")
    print(f"Initial Points:  {pts}")
    print(f"Grad Threshold:  {grad_thresh}")
    print(f"Output Directory:{full_output_dir}")
    print("=" * 50 + "\n")
    
    # 5. 创建并运行 Trainer
    trainer = GaussianTrainer(config=config, device=device)
    best_psnr, best_ssim = trainer.train(resume_from=args.resume)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Results saved to: {full_output_dir}")
    print("=" * 50)

    # 6. 自动测试 (Task 3 & Slice Update)
    print("\n" + "=" * 50)
    print("Starting Automatic Testing (Best Checkpoint)")
    print("=" * 50)
    
    # 构建测试路径和参数
    best_checkpoint_path = os.path.join(full_output_dir, 'checkpoints', 'best.pth')
    
    # 测试结果放在训练目录下的 test_results_best 文件夹
    test_output_dir = os.path.join(full_output_dir, 'test_results_best')
    
    if os.path.exists(best_checkpoint_path):
        # 实例化 Tester
        tester = GaussianTester(
            checkpoint_path=best_checkpoint_path,
            config=config,  # 传递当前包含完整配置的 config
            device=device,  # 使用同一个 GPU
            data_path=config['data']['data_path'],
            acceleration_override=acc # 保持相同的加速倍数
        )
        
        # [新增] 收集命令行传入的切片参数
        custom_slices = {
            'axial': args.slices_axial,
            'coronal': args.slices_coronal,
            'sagittal': args.slices_sagittal
        }
        
        # 运行测试并保存结果
        tester.save_results(
            output_dir=test_output_dir, 
            save_volume=True, 
            save_slices=True,
            custom_slices=custom_slices
        )
        print(f"Automatic testing completed. Results saved to: {test_output_dir}")
    else:
        print(f"Error: Best checkpoint not found at {best_checkpoint_path}")

    # 7. 对 custom_checkpoints 逐个自动测试
    custom_checkpoints = config.get('training', {}).get(
        'custom_checkpoints',
        [10, 50, 100, 300, 600, 800, 1000, 1200]
    )
    # 只测试训练范围内的步数
    max_iters = config.get('training', {}).get('max_iterations', None)
    if max_iters is not None:
        custom_checkpoints = [ckpt for ckpt in custom_checkpoints if ckpt < max_iters]

    if len(custom_checkpoints) > 0:
        print("\n" + "=" * 50)
        print("Starting Automatic Testing (Custom Checkpoints)")
        print("=" * 50)

    for step in custom_checkpoints:
        ckpt_path = os.path.join(full_output_dir, 'checkpoints', f'checkpoint_{step:06d}.pth')
        step_output_dir = os.path.join(full_output_dir, f'test_results_{step}')

        if not os.path.exists(ckpt_path):
            print(f"[Skip] Checkpoint not found: {ckpt_path}")
            continue

        tester = GaussianTester(
            checkpoint_path=ckpt_path,
            config=config,
            device=device,
            data_path=config['data']['data_path'],
            acceleration_override=acc
        )
        tester.save_results(
            output_dir=step_output_dir,
            save_volume=True,
            save_slices=True,
            custom_slices=custom_slices
        )
        print(f"Custom checkpoint testing completed: step={step} -> {step_output_dir}")


if __name__ == '__main__':
    main()