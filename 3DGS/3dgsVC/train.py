"""Training entry point for 3DGSMR."""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from test import GaussianTester
from trainers import GaussianTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3DGSMR for MRI reconstruction")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path in config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory base")
    parser.add_argument("--acceleration", type=int, default=None, help="Override acceleration factor")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--initial_points", type=int, default=None, help="Override initial Gaussian count")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--voxelizer", type=str, default=None, choices=["chunk", "tile", "tile_cuda"])
    parser.add_argument("--strategy", type=str, default=None, choices=["auto", "original", "long_axis"])
    parser.add_argument("--skip_auto_test", action="store_true", help="Disable post-training automatic testing")
    parser.add_argument("--slices_axial", nargs="+", type=int, help="Test slices for axial view")
    parser.add_argument("--slices_coronal", nargs="+", type=int, help="Test slices for coronal view")
    parser.add_argument("--slices_sagittal", nargs="+", type=int, help="Test slices for sagittal view")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def override_config(config: dict, args) -> dict:
    if args.data_path is not None:
        config["data"]["data_path"] = args.data_path
    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir
    if args.acceleration is not None:
        config["data"]["acceleration_factor"] = args.acceleration
    if args.max_iterations is not None:
        config["training"]["max_iterations"] = args.max_iterations
    if args.initial_points is not None:
        config["gaussian"]["initial_num_points"] = args.initial_points
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.voxelizer is not None:
        config.setdefault("voxelizer", {})["type"] = args.voxelizer
    if args.strategy is not None:
        config.setdefault("adaptive_control", {})["strategy"] = args.strategy
    if args.skip_auto_test:
        config.setdefault("mode", {})["auto_test_after_train"] = False
    return config


def make_output_dir(config: dict) -> str:
    base_output_dir = config["output"]["output_dir"]
    acc = config["data"]["acceleration_factor"]
    pts = config["gaussian"]["initial_num_points"]
    strategy = config["adaptive_control"].get("strategy", "auto")
    voxelizer = config.get("voxelizer", {}).get("type", "chunk")
    seed = config["training"]["seed"]
    config_name = os.path.splitext(os.path.basename(config.get("_config_path", "default.yaml")))[0]
    run_name = f"{config_name}_acc{acc}_pts{pts}_{strategy}_{voxelizer}_seed{seed}"
    return os.path.join(base_output_dir, run_name)


def maybe_run_post_train_test(config: dict, device: torch.device, args, output_dir: str):
    mode = config.get("mode", {})
    if not mode.get("auto_test_after_train", True):
        print("Post-train automatic testing disabled by mode/config.")
        return
    if mode.get("self_supervised_deploy", False):
        print("Skipping GT-based post-train testing in self-supervised deploy mode.")
        return

    best_checkpoint_path = os.path.join(output_dir, "checkpoints", "best.pth")
    if not os.path.exists(best_checkpoint_path):
        print(f"Best checkpoint not found at {best_checkpoint_path}; skipping automatic testing.")
        return

    tester = GaussianTester(
        checkpoint_path=best_checkpoint_path,
        config=config,
        device=device,
        data_path=config["data"]["data_path"],
        acceleration_override=config["data"]["acceleration_factor"],
    )
    custom_slices = {
        "axial": args.slices_axial,
        "coronal": args.slices_coronal,
        "sagittal": args.slices_sagittal,
    }
    test_output_dir = os.path.join(output_dir, "test_results_best")
    tester.save_results(output_dir=test_output_dir, save_volume=True, save_slices=True, custom_slices=custom_slices)
    print(f"Automatic testing completed. Results saved to: {test_output_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    config["_config_path"] = args.config
    config = override_config(config, args)
    config["output"]["output_dir"] = make_output_dir(config)

    print("\n" + "=" * 60)
    print("3DGSMR Training Configuration")
    print("=" * 60)
    print(f"Data path:        {config['data']['data_path']}")
    print(f"Acceleration:     {config['data']['acceleration_factor']}x")
    print(f"Initial points:   {config['gaussian']['initial_num_points']}")
    print(f"Strategy:         {config['adaptive_control'].get('strategy')}")
    print(f"Voxelizer:        {config.get('voxelizer', {}).get('type')}")
    print(f"Output directory: {config['output']['output_dir']}")
    print("=" * 60 + "\n")

    trainer = GaussianTrainer(config=config, device=device)
    best_metrics = trainer.train(resume_from=args.resume)

    print("\n" + "=" * 60)
    print("Training complete")
    print(f"Best metrics: {best_metrics}")
    print(f"Results saved to: {config['output']['output_dir']}")
    print("=" * 60)

    maybe_run_post_train_test(config, device, args, config["output"]["output_dir"])


if __name__ == "__main__":
    main()
