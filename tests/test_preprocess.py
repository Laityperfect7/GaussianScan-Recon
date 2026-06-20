"""
测试: 预处理与数据集模块
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaussian_scan.dataset import (
    ImageDataset,
    VideoFrameExtractor,
    generate_sample_images,
    FrameInfo,
    SUPPORTED_IMAGE_EXTS,
)
from gaussian_scan.utils import ensure_dir


class TestImageDataset:
    """ImageDataset 单元测试。"""

    def test_empty_directory(self):
        """测试空目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ImageDataset(tmpdir)
            assert dataset.num_images == 0
            assert len(dataset) == 0

    def test_load_images(self):
        """测试加载图片。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试图片
            for i in range(3):
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(tmpdir, f"img_{i}.jpg"), img)

            dataset = ImageDataset(tmpdir)
            assert dataset.num_images == 3

            frame = dataset[0]
            assert isinstance(frame, FrameInfo)
            assert frame.image.shape == (64, 64, 3)

    def test_iteration(self):
        """测试迭代。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(tmpdir, f"img_{i:02d}.png"), img)

            dataset = ImageDataset(tmpdir)
            frames = list(dataset)
            assert len(frames) == 5

    def test_nonexistent_directory(self):
        """测试不存在的目录。"""
        with pytest.raises(FileNotFoundError):
            ImageDataset("/nonexistent/path/12345")


class TestGenerateSampleImages:
    """示例图像生成测试。"""

    def test_generate(self):
        """测试生成示例图像。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_sample_images(tmpdir, num_images=5)
            assert len(paths) == 5
            for p in paths:
                assert os.path.exists(p)
                img = cv2.imread(p)
                assert img is not None
                assert img.shape == (480, 640, 3)


class TestFrameInfo:
    """FrameInfo 数据类测试。"""

    def test_creation(self):
        """测试创建。"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        frame = FrameInfo(image=img, index=0, timestamp=1.5, source="test.mp4")
        assert frame.index == 0
        assert frame.timestamp == 1.5
        assert frame.source == "test.mp4"


class TestVideoFrameExtractor:
    """VideoFrameExtractor 单元测试。"""

    def test_nonexistent_file(self):
        """测试不存在的文件。"""
        with pytest.raises(FileNotFoundError):
            VideoFrameExtractor("/nonexistent/video.mp4")

    def test_create_sample_video_and_extract(self):
        """测试从生成的视频中提取帧。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建简短的测试视频
            video_path = os.path.join(tmpdir, "test.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))

            for i in range(30):
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                writer.write(img)
            writer.release()

            assert os.path.exists(video_path)

            # 提取帧
            extractor = VideoFrameExtractor(video_path)
            assert extractor.total_frames >= 29
            assert extractor.fps > 0

            frames = extractor.extract_keyframes(
                max_frames=10,
                blur_threshold=None,
            )
            assert 0 < len(frames) <= 10
            assert all(isinstance(f, FrameInfo) for f in frames)

    def test_sharpness_score(self):
        """测试清晰度评分。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 5.0, (160, 120))

            # 清晰帧
            sharp = np.ones((120, 160, 3), dtype=np.uint8) * 128
            cv2.putText(sharp, "SHARP", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(sharp)
            writer.release()

            extractor = VideoFrameExtractor(video_path)
            score = extractor._sharpness_score(sharp)
            # 清晰图像应有正分数
            assert isinstance(score, float)
            assert score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
