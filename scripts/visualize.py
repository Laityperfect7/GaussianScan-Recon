#!/usr/bin/env python3
"""
可视化脚本
==========

从 checkpoint 加载 Gaussian 点云，生成：
- 3D 点云散点图
- Orbit 环绕视角渲染
- 深度图可视化
- HTML 交互预览
- 结果网格图

用法:
    python scripts/visualize.py --checkpoint outputs/checkpoints/mock_model.npz
    python scripts/visualize.py --checkpoint outputs/checkpoints/mock_model.npz --mode all
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from gaussian_scan.gaussian_model import GaussianCloud
from gaussian_scan.camera import generate_orbit_cameras, generate_linear_path
from gaussian_scan.renderer import SimpleRenderer, RenderConfig
from gaussian_scan.visualization import (
    plot_point_cloud_2d,
    plot_orbit_grid,
    depth_map_visualization,
    generate_html_preview,
)
from gaussian_scan.utils import ensure_dir, write_image, create_grid_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="GaussianScan-Recon 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="outputs/checkpoints/mock_model.npz",
        help="Checkpoint 路径 (.npz) (默认: outputs/checkpoints/mock_model.npz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/renders",
        help="输出目录 (默认: outputs/renders)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["scatter", "orbit", "depth", "html", "all"],
        help="可视化模式 (默认: all)",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=24,
        help="Orbit 视图数量 (默认: 24)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Orbit 半径 (默认: 3.0)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("W", "H"),
        help="渲染分辨率 (默认: 640 480)",
    )

    args = parser.parse_args()
    ensure_dir(args.output)

    # --- 加载 checkpoint ---
    if not os.path.exists(args.checkpoint):
        logger.warning(f"Checkpoint 不存在: {args.checkpoint}")
        logger.info("使用随机生成的 GaussianCloud 替代...")
        cloud = GaussianCloud.random_cloud(
            num_points=5000,
            bounds=(2.0, 2.0, 2.0),
            seed=42,
        )
    else:
        logger.info(f"加载 checkpoint: {args.checkpoint}")
        cloud = GaussianCloud.from_npz(args.checkpoint)
    logger.info(f"Gaussian 点数: {len(cloud):,}")

    W, H = args.resolution

    # --- 模式：3D 散点图 ---
    if args.mode in ("scatter", "all"):
        logger.info("生成 3D 散点图...")
        path = os.path.join(args.output, "gaussian_scatter.png")
        plot_point_cloud_2d(cloud, save_path=path, title="GaussianScan-Recon: 3D Gaussian Cloud")
        logger.info(f"  保存: {path}")

    # --- 模式：Orbit 环绕视图 ---
    if args.mode in ("orbit", "all"):
        logger.info("生成环绕视角...")
        cameras = generate_orbit_cameras(
            center=np.array([0.0, 0.0, 0.0]),
            radius=args.radius,
            num_views=args.num_views,
            width=W,
            height=H,
            fx=W * 1.2,
            fy=W * 1.2,
        )

        # 渲染网格图
        grid_path = os.path.join(args.output, "orbit_grid.png")
        plot_orbit_grid(cloud, cameras, save_path=grid_path, cols=6)
        logger.info(f"  网格图: {grid_path}")

        # 渲染几张单独的视角
        renderer = SimpleRenderer(cloud)
        key_views = [0, args.num_views // 4, args.num_views // 2, 3 * args.num_views // 4]
        for idx in key_views:
            if idx < len(cameras):
                img = renderer.render(cameras[idx])
                img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                view_path = os.path.join(args.output, f"orbit_view_{idx:03d}.png")
                write_image(view_path, img_uint8)
        logger.info(f"  单独视角: {len(key_views)} 张")

    # --- 模式：深度图可视化 ---
    if args.mode in ("depth", "all"):
        logger.info("生成深度图可视化演示...")
        H_d, W_d = 240, 320
        yy, xx = np.mgrid[0:H_d, 0:W_d].astype(np.float32)
        cx, cy = W_d / 2, H_d / 2
        dist = np.sqrt(((xx - cx) / W_d) ** 2 + ((yy - cy) / H_d) ** 2)
        mock_depth = 1.0 - dist * 0.7 + np.random.randn(H_d, W_d) * 0.02
        mock_depth = np.clip(mock_depth, 0, 1)

        depth_path = os.path.join(args.output, "depth_visualization.png")
        depth_map_visualization(mock_depth, save_path=depth_path)
        logger.info(f"  保存: {depth_path}")

    # --- 模式：HTML 预览 ---
    if args.mode in ("html", "all"):
        logger.info("生成 HTML 交互预览...")
        html_path = os.path.join(args.output, "preview.html")
        generate_html_preview(cloud, html_path)
        logger.info(f"  保存: {html_path}")
        logger.info(f"  用浏览器打开: file://{os.path.abspath(html_path)}")

    logger.info(f"可视化完成 → {args.output}")


if __name__ == "__main__":
    main()
