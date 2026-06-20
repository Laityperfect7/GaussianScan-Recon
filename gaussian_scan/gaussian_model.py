"""
Gaussian 模型模块
================

定义 3D Gaussian 数据结构，支持从图像 + 深度图生成简化 Gaussian 点云，
以及导入/导出 PLY、NPZ 格式。

数据结构参考：
  Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
  ACM Trans. Graph., 2023
"""

import os
import struct
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------


@dataclass
class GaussianPoint:
    """单个 3D Gaussian 点。

    每个点至少包含位置、颜色、尺度、透明度，
    可选包含旋转（四元数）和球谐系数。

    Attributes
    ----------
    position : np.ndarray, shape (3,)
        3D 世界坐标 (x, y, z)。
    color : np.ndarray, shape (3,)
        RGB 颜色，范围 [0, 1]。
    scale : np.ndarray, shape (3,)
        沿三个轴的缩放因子 (sx, sy, sz)。
    opacity : float
        不透明度 α，范围 [0, 1]。
    rotation : np.ndarray, shape (4,) or None
        旋转四元数 (qw, qx, qy, qz)，默认为单位四元数。
    sh_coeffs : np.ndarray or None
        球谐系数，当前版本保留占位。
    """

    position: np.ndarray  # (3,)
    color: np.ndarray  # (3,)
    scale: np.ndarray  # (3,)
    opacity: float
    rotation: Optional[np.ndarray] = None  # (4,) quaternion
    sh_coeffs: Optional[np.ndarray] = None  # spherical harmonics

    def __post_init__(self):
        """确保数组类型正确。"""
        self.position = np.asarray(self.position, dtype=np.float32).flatten()
        self.color = np.asarray(self.color, dtype=np.float32).flatten()
        self.scale = np.asarray(self.scale, dtype=np.float32).flatten()
        self.opacity = float(np.clip(self.opacity, 0.0, 1.0))
        if self.rotation is None:
            self.rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32).flatten()

    @property
    def xyz(self) -> np.ndarray:
        """返回位置坐标。"""
        return self.position

    @property
    def rgb(self) -> np.ndarray:
        """返回 RGB 颜色。"""
        return self.color

    def __repr__(self):
        return (
            f"GaussianPoint(pos={self.position.round(3)}, "
            f"color={self.color.round(3)}, "
            f"scale={self.scale.round(4)}, "
            f"α={self.opacity:.3f})"
        )


# ---------------------------------------------------------------------------
# Gaussian 点云
# ---------------------------------------------------------------------------


class GaussianCloud:
    """3D Gaussian 点云容器。

    管理一组 GaussianPoint，支持批量操作、从深度图生成、
    PLY/NPZ 序列化等。

    Parameters
    ----------
    points : list of GaussianPoint, optional
        初始化点集。

    Examples
    --------
    >>> cloud = GaussianCloud()
    >>> cloud.add_point(position=[0,0,0], color=[1,0,0], scale=[0.1,0.1,0.1], opacity=0.9)
    >>> cloud.export_ply("output.ply")
    """

    def __init__(self, points: Optional[List[GaussianPoint]] = None):
        self._points: List[GaussianPoint] = points if points is not None else []

    # -- 基本操作 ----------------------------------------------------------

    def add_point(
        self,
        position: np.ndarray,
        color: np.ndarray,
        scale: np.ndarray = None,
        opacity: float = 0.9,
        rotation: Optional[np.ndarray] = None,
    ) -> None:
        """添加一个 Gaussian 点。

        Parameters
        ----------
        position : array-like, shape (3,)
            3D 位置。
        color : array-like, shape (3,)
            RGB 颜色 [0, 1]。
        scale : array-like, shape (3,), optional
            缩放因子，默认 [0.02, 0.02, 0.02]。
        opacity : float
            不透明度，默认 0.9。
        rotation : array-like, shape (4,), optional
            旋转四元数。
        """
        if scale is None:
            scale = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self._points.append(
            GaussianPoint(
                position=np.asarray(position, dtype=np.float32),
                color=np.asarray(color, dtype=np.float32),
                scale=np.asarray(scale, dtype=np.float32),
                opacity=float(opacity),
                rotation=(
                    np.asarray(rotation, dtype=np.float32)
                    if rotation is not None
                    else None
                ),
            )
        )

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, idx: int) -> GaussianPoint:
        return self._points[idx]

    def __iter__(self):
        return iter(self._points)

    @property
    def points(self) -> List[GaussianPoint]:
        """返回点列表。"""
        return self._points

    # -- 批量属性 ----------------------------------------------------------

    def get_positions(self) -> np.ndarray:
        """返回所有点的位置数组 (N, 3)。"""
        return np.stack([p.position for p in self._points], axis=0)

    def get_colors(self) -> np.ndarray:
        """返回所有点的颜色数组 (N, 3)。"""
        return np.stack([p.color for p in self._points], axis=0)

    def get_scales(self) -> np.ndarray:
        """返回所有点的尺度数组 (N, 3)。"""
        return np.stack([p.scale for p in self._points], axis=0)

    def get_opacities(self) -> np.ndarray:
        """返回所有点的不透明度数组 (N,)。"""
        return np.array([p.opacity for p in self._points], dtype=np.float32)

    def get_rotations(self) -> np.ndarray:
        """返回所有点的旋转数组 (N, 4)。"""
        return np.stack([p.rotation for p in self._points], axis=0)

    # -- I/O ---------------------------------------------------------------

    def to_dict(self) -> dict:
        """序列化为字典（用于 NPZ 保存）。"""
        return {
            "positions": self.get_positions(),
            "colors": self.get_colors(),
            "scales": self.get_scales(),
            "opacities": self.get_opacities(),
            "rotations": self.get_rotations(),
            "num_points": len(self._points),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GaussianCloud":
        """从字典恢复。"""
        cloud = cls()
        positions = data["positions"]
        colors = data["colors"]
        scales = data["scales"]
        opacities = data["opacities"]
        rotations = data.get("rotations", None)
        for i in range(data["num_points"]):
            rot = rotations[i] if rotations is not None else None
            cloud.add_point(
                position=positions[i],
                color=colors[i],
                scale=scales[i],
                opacity=float(opacities[i]),
                rotation=rot,
            )
        return cloud

    def export_ply(self, filepath: str) -> None:
        """导出为 PLY 格式（带 Gaussian 自定义属性）。

        Parameters
        ----------
        filepath : str
            输出 .ply 文件路径。
        """
        save_ply(
            filepath=filepath,
            positions=self.get_positions(),
            colors=self.get_colors(),
            scales=self.get_scales(),
            opacities=self.get_opacities(),
            rotations=self.get_rotations(),
        )

    def export_npz(self, filepath: str) -> None:
        """导出为 NPZ 压缩格式。

        Parameters
        ----------
        filepath : str
            输出 .npz 文件路径。
        """
        save_npz(filepath, self.to_dict())

    @classmethod
    def from_npz(cls, filepath: str) -> "GaussianCloud":
        """从 NPZ 文件加载。

        Parameters
        ----------
        filepath : str
            .npz 文件路径。

        Returns
        -------
        GaussianCloud
        """
        data = load_npz(filepath)
        return cls.from_dict(data)

    # -- 生成方法 -----------------------------------------------------------

    @classmethod
    def from_depth_and_image(
        cls,
        image: np.ndarray,
        depth_map: np.ndarray,
        camera_intrinsics: np.ndarray,
        scale_base: float = 0.02,
        opacity: float = 0.9,
        sample_step: int = 4,
    ) -> "GaussianCloud":
        """从 RGB 图像和深度图生成 Gaussian 点云。

        对图像进行均匀采样，每个采样像素对应一个 Gaussian 点，
        位置由相机模型反投影得到。

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3)
            RGB 图像。
        depth_map : np.ndarray, shape (H, W)
            深度图（0-1 归一化或实际深度值）。
        camera_intrinsics : np.ndarray, shape (3, 3)
            相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]。
        scale_base : float
            基础高斯尺度。
        sample_step : int
            采样步长（每隔 N 像素采样一次）。

        Returns
        -------
        GaussianCloud
        """
        cloud = cls()
        H, W = depth_map.shape[:2]
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        # 归一化深度到 0.5~3.0 米模拟范围
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_normalized = depth_map
        depth_scaled = 0.5 + depth_normalized * 2.5  # 0.5 ~ 3.0 米

        for y in range(0, H, sample_step):
            for x in range(0, W, sample_step):
                d = depth_scaled[y, x]
                # 反投影：pixel -> camera -> world (此处简化为 camera = world)
                X = (x - cx) * d / fx
                Y = (y - cy) * d / fy
                Z = d

                if image.ndim == 3 and image.shape[2] >= 3:
                    color = image[y, x, :3].astype(np.float32)
                    if color.max() > 1.0:
                        color = color / 255.0
                else:
                    color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

                # 尺度与深度相关：远处更大
                s = scale_base * (1.0 + d * 0.5)
                scale_vec = np.array([s, s, s], dtype=np.float32)

                cloud.add_point(
                    position=np.array([X, Y, Z], dtype=np.float32),
                    color=color,
                    scale=scale_vec,
                    opacity=opacity,
                )
        return cloud

    @classmethod
    def random_cloud(
        cls,
        num_points: int = 5000,
        bounds: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        seed: int = 42,
    ) -> "GaussianCloud":
        """生成随机 Gaussian 点云用于测试和演示。

        Parameters
        ----------
        num_points : int
            点数量。
        bounds : tuple
            (x_range, y_range, z_range)，各维度范围为 [-b, b]。
        seed : int
            随机种子。

        Returns
        -------
        GaussianCloud
        """
        rng = np.random.RandomState(seed)
        cloud = cls()
        for _ in range(num_points):
            pos = (rng.rand(3) - 0.5) * 2.0 * np.array(bounds)
            color = np.clip(rng.rand(3) * 0.5 + 0.3, 0, 1)
            scale = np.abs(rng.randn(3) * 0.03 + 0.04)
            cloud.add_point(
                position=pos,
                color=color,
                scale=scale,
                opacity=float(np.clip(rng.rand() * 0.5 + 0.5, 0, 1)),
            )
        return cloud

    def __repr__(self):
        return f"GaussianCloud(n_points={len(self._points)})"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def save_npz(filepath: str, data: dict) -> None:
    """保存字典数据为 NPZ 文件。

    Parameters
    ----------
    filepath : str
        输出路径。
    data : dict
        键值对，值需为 numpy 数组。
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savez_compressed(filepath, **data)


def load_npz(filepath: str) -> dict:
    """从 NPZ 文件加载数据。

    Parameters
    ----------
    filepath : str
        .npz 文件路径。

    Returns
    -------
    dict
    """
    return dict(np.load(filepath))


def save_ply(
    filepath: str,
    positions: np.ndarray,
    colors: np.ndarray,
    scales: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
) -> None:
    """将 Gaussian 点云保存为 PLY 文件。

    使用自定义属性字段兼容 3D Gaussian Splatting 标准格式。

    Parameters
    ----------
    filepath : str
        .ply 文件路径。
    positions : np.ndarray, shape (N, 3)
    colors : np.ndarray, shape (N, 3)
    scales : np.ndarray, shape (N, 3) or None
    opacities : np.ndarray, shape (N,) or None
    rotations : np.ndarray, shape (N, 4) or None
    """
    N = positions.shape[0]
    has_scales = scales is not None
    has_opacities = opacities is not None
    has_rotations = rotations is not None

    # 构建 PLY 头
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if has_opacities:
        header_lines.append("property float opacity")
    if has_scales:
        header_lines += [
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
        ]
    if has_rotations:
        header_lines += [
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            # xyz
            f.write(struct.pack("<fff", *positions[i]))
            # rgb
            rgb = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
            f.write(struct.pack("<BBB", *rgb))
            # opacity
            if has_opacities:
                f.write(struct.pack("<f", float(opacities[i])))
            # scale
            if has_scales:
                f.write(struct.pack("<fff", *scales[i]))
            # rotation
            if has_rotations:
                f.write(struct.pack("<ffff", *rotations[i]))
