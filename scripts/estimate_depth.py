#!/usr/bin/env python3
"""
深度估计接口脚本
=================

提供模拟深度估计功能，用于流水线演示。
真实使用时替换为 Depth Anything V2 / ZoeDepth / MiDaS。

用法:
    # Mock 深度估计（默认）
    python scripts/estimate_depth.py --input outputs/frames --output outputs/depth

    # 指定模拟模式
    python scripts/estimate_depth.py --input outputs/frames --output outputs/depth --mode gradient

    # 可选：指定真实模型接口（预留，不执行）
    # python scripts/estimate_depth.py --input ... --model depth-anything-v2
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from gaussian_scan.utils import ensure_dir, write_image, normalize_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock 深度估计策略
# ---------------------------------------------------------------------------


def mock_depth_radial(
    image: np.ndarray, noise_level: float = 0.02
) -> np.ndarray:
    """径向渐近深度：中心近，边缘远。

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
    noise_level : float

    Returns
    -------
    np.ndarray, shape (H, W)
    """
    H, W = image.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    # 到中心的归一化距离
    dist = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
    depth = 1.0 - dist * 0.7  # 中心 ~1.0，边缘 ~0.3
    depth += np.random.randn(H, W) * noise_level
    return np.clip(depth, 0, 1).astype(np.float32)


def mock_depth_gradient(
    image: np.ndarray, direction: str = "horizontal"
) -> np.ndarray:
    """渐变深度。

    Parameters
    ----------
    image : np.ndarray
    direction : str
        'horizontal' 或 'vertical'

    Returns
    -------
    np.ndarray
    """
    H, W = image.shape[:2]
    if direction == "horizontal":
        depth = np.tile(np.linspace(0.2, 1.0, W), (H, 1))
    else:
        depth = np.tile(np.linspace(0.2, 1.0, H)[:, None], (1, W))
    return depth.astype(np.float32)


def mock_depth_structured(
    image: np.ndarray, num_objects: int = 3, base_depth: float = 0.5
) -> np.ndarray:
    """结构化场景模拟深度：随机放置物体。

    Parameters
    ----------
    image : np.ndarray
    num_objects : int
        模拟物体数量。
    base_depth : float
        背景深度。

    Returns
    -------
    np.ndarray
    """
    H, W = image.shape[:2]
    depth = np.full((H, W), base_depth, dtype=np.float32)

    rng = np.random.RandomState(42)
    for _ in range(num_objects):
        cx = rng.randint(W // 4, 3 * W // 4)
        cy = rng.randint(H // 4, 3 * H // 4)
        rx = rng.randint(40, 120)
        ry = rng.randint(40, 100)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        obj_depth = rng.uniform(0.1, 0.35)
        depth[mask] = obj_depth

    # 添加噪声
    depth += np.random.randn(H, W) * 0.005
    return np.clip(depth, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 深度估计入口
# ---------------------------------------------------------------------------

MODES = {
    "radial": mock_depth_radial,
    "gradient": mock_depth_gradient,
    "structured": mock_depth_structured,
}


def main():
    parser = argparse.ArgumentParser(
        description="深度估计（Mock / 接口预留）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入帧目录或单张图片路径",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/depth",
        help="输出深度图目录 (默认: outputs/depth)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="radial",
        choices=list(MODES.keys()),
        help=f"模拟模式: {list(MODES.keys())} (默认: radial)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help=(
            "深度模型名称 (默认: mock)。"
            "真实使用时可替换为: depth-anything-v2, zoedepth, midas"
        ),
    )

    args = parser.parse_args()

    # --- 只支持 mock 模式 ---
    if args.model != "mock":
        logger.warning(
            f"模型 '{args.model}' 尚未集成。"
            f"当前仅支持 mock 模式用于流水线演示。"
        )
        logger.info("切换到 mock 模式...")

    depth_fn = MODES[args.mode]
    logger.info(f"深度估计模式: mock/{args.mode}")
    logger.info(f"输入: {args.input}")

    input_path = Path(args.input)
    ensure_dir(args.output)

    if input_path.is_dir():
        image_files = sorted(
            [f for f in input_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not image_files:
            logger.error(f"目录 {args.input} 中没有图像文件")
            sys.exit(1)

        logger.info(f"处理 {len(image_files)} 张图片...")
        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"跳过无法读取: {img_path.name}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth = depth_fn(img_rgb)

            # 保存：深度图可视化（伪彩色）+ 原始深度值
            depth_vis = (depth * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

            out_name = img_path.stem
            write_image(os.path.join(args.output, f"{out_name}_depth_vis.png"), depth_color, rgb=False)
            np.save(os.path.join(args.output, f"{out_name}_depth.npy"), depth)

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(image_files)}")

    elif input_path.is_file():
        img = cv2.imread(str(input_path))
        if img is None:
            logger.error(f"无法读取图像: {args.input}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = depth_fn(img_rgb)

        depth_vis = (depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        write_image(os.path.join(args.output, "depth_vis.png"), depth_color, rgb=False)
        np.save(os.path.join(args.output, "depth.npy"), depth)
    else:
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)

    logger.info(f"深度估计完成 → {args.output}")
    logger.info("💡 提示：这是模拟深度估计。真实使用请替换为 Depth Anything V2 等模型。")


if __name__ == "__main__":
    main()
