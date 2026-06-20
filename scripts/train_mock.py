#!/usr/bin/env python3
"""
模拟训练脚本
============

模拟 3D Gaussian Splatting 训练流程，用于演示流水线。
不执行真实 CUDA 训练，但输出与真实训练一致的中间产物。

用法:
    python scripts/train_mock.py --config configs/default.yaml
    python scripts/train_mock.py --config configs/default.yaml --frames outputs/frames --depth outputs/depth
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import yaml

from gaussian_scan.gaussian_model import GaussianCloud
from gaussian_scan.utils import ensure_dir
from gaussian_scan.visualization import plot_point_cloud_2d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def simulate_training(
    config: dict, output_dir: str
) -> dict:
    """模拟 3DGS 训练过程。

    生成：
    - 随机初始化的 GaussianCloud（模拟训练后状态）
    - 模拟 loss 曲线
    - checkpoint 文件 (mock_model.npz, mock_model.ply)

    Returns
    -------
    dict
        训练统计信息。
    """
    gs_cfg = config.get("gaussian", {})
    train_cfg = config.get("training", {})
    num_gaussians = gs_cfg.get("num_init", 50000)
    iterations = train_cfg.get("iterations", 10000)
    scene_bounds = gs_cfg.get("scene_bounds", [2.0, 2.0, 2.0])

    logger.info(f"Gaussian 点数: {num_gaussians}")
    logger.info(f"模拟迭代数: {iterations}")

    # --- 生成随机点云作为"训练后"结果 ---
    logger.info("初始化 Gaussian 点云...")
    cloud = GaussianCloud.random_cloud(
        num_points=num_gaussians,
        bounds=tuple(scene_bounds),
        seed=42,
    )
    logger.info(f"已生成 {len(cloud)} 个 Gaussian 点")

    # --- 模拟训练循环 ---
    logger.info("模拟训练中...")
    loss_history = []
    start_time = time.time()

    for step in range(1, iterations + 1):
        # 模拟 loss 下降：指数衰减 + 噪声
        progress = step / iterations
        main_loss = 0.5 * np.exp(-4 * progress) + 0.02 * np.exp(-2 * progress)
        noise = np.random.randn() * 0.003 * (1 - progress)
        loss = main_loss + noise
        loss_history.append(float(loss))

        if step % (iterations // 10) == 0 or step == iterations:
            elapsed = time.time() - start_time
            eta = (elapsed / step) * (iterations - step)
            logger.info(
                f"  [Step {step:6d}/{iterations}] "
                f"loss={loss:.6f}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s"
            )

    total_time = time.time() - start_time
    logger.info(f"训练完成，耗时 {total_time:.1f}s (模拟)")

    # --- 保存 checkpoint ---
    ensure_dir(output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    # NPZ
    npz_path = os.path.join(ckpt_dir, "mock_model.npz")
    cloud.export_npz(npz_path)
    logger.info(f"Checkpoint (NPZ): {npz_path}")

    # PLY
    ply_path = os.path.join(ckpt_dir, "mock_model.ply")
    cloud.export_ply(ply_path)
    logger.info(f"Checkpoint (PLY): {ply_path}")

    # Loss 曲线数据
    loss_path = os.path.join(output_dir, "loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f)
    logger.info(f"Loss 曲线: {loss_path}")

    # 3D 散点图
    plot_path = os.path.join(output_dir, "gaussian_points_3d.png")
    plot_point_cloud_2d(cloud, save_path=plot_path, title="GaussianScan-Recon: Simulated Gaussian Cloud")
    logger.info(f"可视化: {plot_path}")

    stats = {
        "num_gaussians": len(cloud),
        "iterations": iterations,
        "total_time_seconds": total_time,
        "final_loss": float(loss_history[-1]),
        "best_loss": float(min(loss_history)),
        "checkpoint_npz": npz_path,
        "checkpoint_ply": ply_path,
        "loss_json": loss_path,
    }

    # 保存统计信息
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"统计信息: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="GaussianScan-Recon 模拟训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="outputs/frames",
        help="帧目录 (用于流水线一致性，mock 模式下可忽略)",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="outputs/depth",
        help="深度图目录 (用于流水线一致性，mock 模式下可忽略)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="输出根目录 (默认: outputs)",
    )

    args = parser.parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    logger.info(f"配置: {args.config}")
    logger.info(f"场景: {config.get('scene', {}).get('name', 'default')}")

    # 显示警告
    logger.warning("=" * 60)
    logger.warning("⚠️  MOCK TRAINING MODE")
    logger.warning("这是模拟训练，不执行真实 3DGS 训练。")
    logger.warning("生成的 Gaussian 点云为随机数据，仅供流水线演示。")
    logger.warning("真实训练请集成 CUDA 可微光栅化器。")
    logger.warning("=" * 60)

    # 执行模拟训练
    stats = simulate_training(config, args.output)

    # 输出摘要
    logger.info("=" * 60)
    logger.info("训练摘要:")
    logger.info(f"  Gaussian 点数: {stats['num_gaussians']:,}")
    logger.info(f"  迭代数:       {stats['iterations']:,}")
    logger.info(f"  最终 Loss:    {stats['final_loss']:.6f}")
    logger.info(f"  最优 Loss:    {stats['best_loss']:.6f}")
    logger.info(f"  耗时:         {stats['total_time_seconds']:.1f}s (模拟)")
    logger.info(f"  Checkpoint:   {stats['checkpoint_npz']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
