"""
数据集模块
==========

提供图像数据集加载和视频帧提取功能。
支持多种输入格式，用于预处理和后续流程。
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass

from gaussian_scan.utils import ensure_dir


# 支持的图像扩展名
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# 支持的视频扩展名
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


@dataclass
class FrameInfo:
    """单帧信息。

    Attributes
    ----------
    image : np.ndarray
        图像数组 (H, W, 3)。
    index : int
        帧索引。
    timestamp : float
        帧时间戳（秒），图片为 0。
    source : str
        来源文件路径。
    """

    image: np.ndarray
    index: int
    timestamp: float = 0.0
    source: str = ""


class ImageDataset:
    """图像数据集加载器。

    从目录中加载多张图片，支持排序和过滤。

    Parameters
    ----------
    image_dir : str or Path
        图像目录路径。

    Examples
    --------
    >>> dataset = ImageDataset("data/samples/object_scan")
    >>> for frame in dataset:
    ...     print(frame.image.shape)
    """

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"目录不存在: {image_dir}")

        self._image_paths: List[Path] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """扫描目录中的所有图像文件并排序。"""
        self._image_paths = []
        for ext in SUPPORTED_IMAGE_EXTS:
            self._image_paths.extend(self.image_dir.glob(f"*{ext}"))
            self._image_paths.extend(self.image_dir.glob(f"*{ext.upper()}"))
        self._image_paths = sorted(set(self._image_paths), key=lambda p: p.name)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> FrameInfo:
        """读取单张图像。

        Parameters
        ----------
        idx : int
            图像索引。

        Returns
        -------
        FrameInfo
        """
        path = self._image_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"无法读取图像: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return FrameInfo(
            image=image, index=idx, timestamp=0.0, source=str(path)
        )

    def __iter__(self) -> Generator[FrameInfo, None, None]:
        for i in range(len(self)):
            yield self[i]

    @property
    def image_paths(self) -> List[Path]:
        """返回所有图像路径。"""
        return self._image_paths

    @property
    def num_images(self) -> int:
        """返回图像数量。"""
        return len(self._image_paths)

    def __repr__(self):
        return f"ImageDataset(n={len(self)}, dir={self.image_dir.name})"


class VideoFrameExtractor:
    """视频关键帧提取器。

    从视频文件中按时间间隔或清晰度评分提取关键帧。

    Parameters
    ----------
    video_path : str
        视频文件路径。
    target_size : tuple (width, height), optional
        统一缩放尺寸，保持比例（短边对齐）。

    Examples
    --------
    >>> extractor = VideoFrameExtractor("demo.mp4", target_size=(1280, 720))
    >>> frames = extractor.extract_keyframes(max_frames=200)
    """

    def __init__(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        ext = self.video_path.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            raise ValueError(f"不支持的视频格式: {ext}。支持: {SUPPORTED_VIDEO_EXTS}")

        self.target_size = target_size

        # 读取视频属性
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = (
            self.total_frames / self.fps if self.fps > 0 else 0
        )
        self.original_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.cap.release()

    def __repr__(self):
        return (
            f"VideoFrameExtractor({self.video_path.name}, "
            f"{self.original_size[0]}x{self.original_size[1]}, "
            f"{self.fps:.1f}fps, {self.total_frames} frames)"
        )

    def _resize_image(
        self, image: np.ndarray
    ) -> np.ndarray:
        """按目标尺寸缩放图像（保持比例，短边对齐）。"""
        if self.target_size is None:
            return image
        h, w = image.shape[:2]
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))

    def _sharpness_score(self, image: np.ndarray) -> float:
        """计算图像清晰度（拉普拉斯方差）。

        Parameters
        ----------
        image : np.ndarray
            灰度图像或 BGR 图像。

        Returns
        -------
        float
            清晰度分数。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def extract_keyframes(
        self,
        max_frames: int = 200,
        min_interval: int = 1,
        blur_threshold: Optional[float] = None,
        output_dir: Optional[str] = None,
        progress_callback=None,
    ) -> List[FrameInfo]:
        """提取关键帧。

        使用均匀采样 + 清晰度筛选策略：
        1. 按时间间隔均匀采样；
        2. 计算清晰度分数；
        3. 过滤模糊帧；
        4. 去重相近帧。

        Parameters
        ----------
        max_frames : int
            最大输出帧数。
        min_interval : int
            最小帧间隔。
        blur_threshold : float, optional
            模糊阈值，低于此值的帧被丢弃。None 则不过滤。
        output_dir : str, optional
            若提供，将关键帧保存到该目录。
        progress_callback : callable, optional
            进度回调函数，接收 (current, total)。

        Returns
        -------
        list of FrameInfo
        """
        cap = cv2.VideoCapture(str(self.video_path))
        total = min(self.total_frames, max_frames * min_interval * 2)
        step = max(min_interval, self.total_frames // max_frames)

        frames: List[FrameInfo] = []
        frame_idx = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                score = self._sharpness_score(frame)

                if blur_threshold is not None and score < blur_threshold:
                    frame_idx += 1
                    continue

                # 缩放
                frame = self._resize_image(frame)

                timestamp = frame_idx / self.fps if self.fps > 0 else 0.0
                info = FrameInfo(
                    image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    index=frame_idx,
                    timestamp=timestamp,
                    source=str(self.video_path),
                )
                frames.append(info)
                extracted += 1

                if output_dir:
                    self._save_frame(frame, frame_idx, output_dir)

                if progress_callback:
                    progress_callback(extracted, max_frames)

                if extracted >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return frames

    def _save_frame(
        self, image: np.ndarray, idx: int, output_dir: str
    ) -> None:
        """保存单帧图像。"""
        ensure_dir(output_dir)
        out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, image)

    def read_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """读取指定索引的帧。

        Parameters
        ----------
        frame_idx : int
            帧索引。

        Returns
        -------
        np.ndarray or None
        """
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = self._resize_image(frame)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None


def generate_sample_images(
    output_dir: str, num_images: int = 5, size: Tuple[int, int] = (640, 480)
) -> List[str]:
    """生成示例图像（用于演示和测试）。

    生成带有彩色渐变和圆形的合成图像，
    模拟多视角物体扫描输入。

    Parameters
    ----------
    output_dir : str
        输出目录。
    num_images : int
        生成图像数量。
    size : tuple
        图像尺寸 (width, height)。

    Returns
    -------
    list of str
        生成的文件路径列表。
    """
    ensure_dir(output_dir)
    paths = []
    for i in range(num_images):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # 渐变背景
        for y in range(size[1]):
            color = np.array(
                [
                    int(30 + 30 * np.sin(i * 0.8 + y * 0.02)),
                    int(30 + 30 * np.sin(i * 0.5 + y * 0.03)),
                    int(60 + 40 * np.sin(i * 0.6 + y * 0.01)),
                ]
            )
            img[y, :] = np.clip(color, 0, 255)

        # 中心圆形"物体"
        cx, cy = size[0] // 2, size[1] // 2
        # 物体位置随视角变化
        offset_x = int(30 * np.sin(i * 2 * np.pi / num_images))
        offset_y = int(20 * np.cos(i * 2 * np.pi / num_images))
        cx += offset_x
        cy += offset_y

        cv2.circle(img, (cx, cy), 80, (200, 150, 50), -1)
        cv2.circle(img, (cx - 30, cy - 30), 30, (220, 200, 100), -1)
        cv2.circle(img, (cx + 20, cy + 20), 20, (180, 130, 40), -1)

        # 标签
        cv2.putText(
            img,
            f"View {i+1}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        path = os.path.join(output_dir, f"sample_{i:03d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(path)

    return paths
