#!/usr/bin/env python3
"""
预处理脚本
==========

从视频/图像中提取关键帧，执行缩放、去模糊、去重等操作。

用法:
    # 从视频提取帧
    python scripts/preprocess.py --input demo.mp4 --output outputs/frames --max-frames 200

    # 从图像目录加载
    python scripts/preprocess.py --input data/samples/object_scan --output outputs/frames

    # 生成示例图像（用于测试）
    python scripts/preprocess.py --input DEMO --output data/samples/object_scan --num-demo 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 将项目根目录添加到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaussian_scan.dataset import (
    ImageDataset,
    VideoFrameExtractor,
    generate_sample_images,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
)
from gaussian_scan.utils import ensure_dir, write_image, resize_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="GaussianScan-Recon 预处理：提取关键帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入路径：视频文件、图像目录，或 'DEMO' 生成示例图像",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/frames",
        help="输出目录 (默认: outputs/frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="最大提取帧数 (默认: 200)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="目标尺寸：短边像素数 (默认: 不缩放)",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=50.0,
        help="模糊过滤阈值，低于此分数的帧被丢弃 (默认: 50.0)",
    )
    parser.add_argument(
        "--num-demo",
        type=int,
        default=10,
        help="生成示例图像的数量 (仅 --input DEMO 时生效)",
    )

    args = parser.parse_args()

    # --- 输入为 DEMO：生成示例图像 ---
    if args.input.upper() == "DEMO":
        logger.info("生成示例图像...")
        paths = generate_sample_images(
            output_dir=args.output,
            num_images=args.num_demo,
            size=(640, 480),
        )
        logger.info(f"已生成 {len(paths)} 张示例图像 → {args.output}")
        sys.exit(0)

    input_path = Path(args.input)

    # --- 输入为视频 ---
    if input_path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
        logger.info(f"输入类型: 视频 ({input_path.name})")

        extractor = VideoFrameExtractor(str(input_path))
        logger.info(
            f"  分辨率: {extractor.original_size[0]}x{extractor.original_size[1]}, "
            f"FPS: {extractor.fps:.1f}, 总帧数: {extractor.total_frames}"
        )

        frames = extractor.extract_keyframes(
            max_frames=args.max_frames,
            blur_threshold=args.blur_threshold,
            output_dir=args.output,
        )

        logger.info(f"提取完成: {len(frames)} 帧 → {args.output}")

        # 可选缩放
        if args.target_size:
            logger.info(f"缩放到短边 {args.target_size}px...")
            for f in os.listdir(args.output):
                fpath = os.path.join(args.output, f)
                if os.path.isfile(fpath):
                    img = resize_image(
                        fpath, target_size=args.target_size
                    )
                    write_image(fpath, img, rgb=False)
            logger.info("缩放完成")

        sys.exit(0)

    # --- 输入为图片目录 ---
    if input_path.is_dir():
        logger.info(f"输入类型: 图像目录 ({input_path})")

        dataset = ImageDataset(str(input_path))
        logger.info(f"找到 {dataset.num_images} 张图片")

        if dataset.num_images == 0:
            logger.error("目录中没有支持的图像文件")
            logger.info(f"支持的格式: {SUPPORTED_IMAGE_EXTS}")
            sys.exit(1)

        ensure_dir(args.output)
        saved_count = 0
        for i, frame in enumerate(dataset):
            if i >= args.max_frames:
                break
            img = frame.image
            if args.target_size:
                img = resize_image(img, target_size=args.target_size)
            out_path = os.path.join(args.output, f"frame_{i:06d}.jpg")
            write_image(out_path, img)
            saved_count += 1

        logger.info(f"已复制/处理 {saved_count} 张图片 → {args.output}")
        sys.exit(0)

    # --- 未识别的输入 ---
    logger.error(f"无法识别的输入: {args.input}")
    logger.error("请提供：视频文件、图像目录，或 'DEMO'")
    sys.exit(1)


if __name__ == "__main__":
    main()
