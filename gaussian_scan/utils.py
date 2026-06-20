"""
通用工具模块
============

提供文件 I/O、目录管理、图像处理等通用功能。
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建。

    Parameters
    ----------
    path : str
        目录路径。

    Returns
    -------
    str
        规范化后的路径。
    """
    os.makedirs(path, exist_ok=True)
    return path


def read_image(
    path: str, rgb: bool = True
) -> Optional[np.ndarray]:
    """读取图像文件。

    Parameters
    ----------
    path : str
        图像路径。
    rgb : bool
        True 返回 RGB，False 返回 BGR。

    Returns
    -------
    np.ndarray or None
    """
    image = cv2.imread(path)
    if image is None:
        return None
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_image(
    path: str, image: np.ndarray, rgb: bool = True, quality: int = 95
) -> None:
    """写入图像文件。

    Parameters
    ----------
    path : str
        输出路径。
    image : np.ndarray
        图像数组。
    rgb : bool
        True 表示输入为 RGB。
    quality : int
        JPEG 质量 1-100。
    """
    ensure_dir(os.path.dirname(path))
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])


def resize_image(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]],
    keep_aspect: bool = True,
) -> np.ndarray:
    """缩放图像。

    Parameters
    ----------
    image : np.ndarray
        输入图像。
    target_size : int or tuple
        目标尺寸。int 表示短边长度；tuple 表示 (w, h)。
    keep_aspect : bool
        是否保持比例。

    Returns
    -------
    np.ndarray
    """
    h, w = image.shape[:2]
    if isinstance(target_size, int):
        scale = target_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = target_size
        if keep_aspect:
            scale = min(new_w / w, new_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """将单通道灰度图转换为三通道 RGB 图。

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, 1)

    Returns
    -------
    np.ndarray, shape (H, W, 3)
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return np.stack([image.squeeze()] * 3, axis=-1)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """将图像归一化到 [0, 1]。

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    np.ndarray
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)


def create_grid_image(
    images: list,
    cols: int = 4,
    pad: int = 4,
    bg_color: Tuple[int, int, int] = (20, 20, 30),
) -> np.ndarray:
    """将多张图像拼接为网格图。

    Parameters
    ----------
    images : list of np.ndarray
        图像列表。
    cols : int
        每行列数。
    pad : int
        图像间距（像素）。
    bg_color : tuple
        背景色 (R, G, B)，范围 [0, 255]。

    Returns
    -------
    np.ndarray
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n = len(images)
    rows = int(np.ceil(n / cols))

    # 统一缩放到第一张图的大小
    h0, w0 = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != (h0, w0):
            img = cv2.resize(img, (w0, h0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        resized.append(img)

    grid_h = rows * h0 + (rows + 1) * pad
    grid_w = cols * w0 + (cols + 1) * pad
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for i, img in enumerate(resized):
        r, c = i // cols, i % cols
        y = pad + r * (h0 + pad)
        x = pad + c * (w0 + pad)
        grid[y : y + h0, x : x + w0] = img

    return grid


# ---------------------------------------------------------------------------
# 向后兼容：将 save_ply / save_npz / load_npz 引到这里
# ---------------------------------------------------------------------------

from gaussian_scan.gaussian_model import save_ply, save_npz, load_npz  # noqa: E402, F401
