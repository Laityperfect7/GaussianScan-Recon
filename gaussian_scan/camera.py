"""
相机模型与轨迹生成
==================

提供针孔相机模型和环绕轨迹生成功能，
用于新视角渲染和可视化。
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PinholeCamera:
    """针孔相机模型。

    定义相机内参和外参，支持世界到像素/像素到世界的投影。

    Parameters
    ----------
    width : int
        图像宽度（像素）。
    height : int
        图像高度（像素）。
    fx : float
        X 方向焦距（像素）。
    fy : float
        Y 方向焦距（像素）。
    cx : float
        主点 X 坐标。
    cy : float
        主点 Y 坐标。
    position : np.ndarray
        相机在世界坐标系中的位置 (3,)。
    look_at : np.ndarray
        相机注视目标点 (3,)。
    up : np.ndarray
        上方向向量 (3,)。
    """

    width: int = 1280
    height: int = 720
    fx: float = 960.0
    fy: float = 960.0
    cx: float = 640.0
    cy: float = 360.0

    # 外参
    position: Optional[np.ndarray] = None  # (3,)
    look_at: Optional[np.ndarray] = None  # (3,)
    up: Optional[np.ndarray] = None  # (3,)

    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        if self.look_at is None:
            self.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.position = np.asarray(self.position, dtype=np.float32)
        self.look_at = np.asarray(self.look_at, dtype=np.float32)
        self.up = np.asarray(self.up, dtype=np.float32)

    @property
    def K(self) -> np.ndarray:
        """内参矩阵 (3, 3)。"""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )

    @property
    def R(self) -> np.ndarray:
        """旋转矩阵 (3, 3)：从世界坐标系到相机坐标系。"""
        z_axis = self.position - self.look_at
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        x_axis = np.cross(self.up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=0).T

    @property
    def t(self) -> np.ndarray:
        """平移向量 t = -R @ position。"""
        return -self.R @ self.position

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """将世界坐标点转为相机坐标。

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            世界坐标系中的点。

        Returns
        -------
        np.ndarray, shape (N, 3)
            相机坐标系中的点。
        """
        return (self.R @ points.T).T + self.t

    def camera_to_pixel(self, points_camera: np.ndarray) -> np.ndarray:
        """将相机坐标点投影到像素坐标。

        Parameters
        ----------
        points_camera : np.ndarray, shape (N, 3)
            相机坐标系中的点。

        Returns
        -------
        np.ndarray, shape (N, 2)
            像素坐标 (u, v)。
        """
        pts = points_camera.copy()
        z = pts[:, 2:3]
        z = np.where(np.abs(z) < 1e-8, 1e-8, z)
        uv = pts[:, :2] / z
        uv = uv * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        return uv

    def project(self, world_points: np.ndarray) -> np.ndarray:
        """将世界 3D 点投影到像素坐标（完整流水线）。

        Parameters
        ----------
        world_points : np.ndarray, shape (N, 3)

        Returns
        -------
        np.ndarray, shape (N, 2)
        """
        cam = self.world_to_camera(world_points)
        return self.camera_to_pixel(cam)

    def get_extrinsic_matrix(self) -> np.ndarray:
        """返回 4x4 外参矩阵 [R|t; 0|1]。"""
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "position": self.position.tolist(),
            "look_at": self.look_at.tolist(),
            "up": self.up.tolist(),
        }

    def __repr__(self):
        return (
            f"PinholeCamera({self.width}x{self.height}, "
            f"f=({self.fx:.1f},{self.fy:.1f}), "
            f"pos={self.position.round(2)})"
        )


def generate_orbit_cameras(
    center: np.ndarray = None,
    radius: float = 3.0,
    num_views: int = 36,
    height_range: Tuple[float, float] = (-1.5, 1.5),
    width: int = 1280,
    height: int = 720,
    fx: float = 960.0,
    fy: float = 960.0,
) -> List[PinholeCamera]:
    """生成环绕物体的相机轨迹。

    相机以 center 为中心，在 sphere 表面采样位置，
    look_at 始终指向 center。

    Parameters
    ----------
    center : np.ndarray, shape (3,), optional
        环绕中心点，默认为原点。
    radius : float
        轨道半径（米）。
    num_views : int
        相机数量。
    height_range : tuple
        Y 轴高度范围 (min, max)。
    width, height : int
        输出图像分辨率。
    fx, fy : float
        焦距。

    Returns
    -------
    list of PinholeCamera
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    cameras = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        # 高度在范围内正弦变化
        phi = height_range[0] + (height_range[1] - height_range[0]) * (
            0.5 + 0.5 * np.sin(i / num_views * np.pi * 2)
        )
        x = center[0] + radius * np.cos(theta)
        y = center[1] + phi
        z = center[2] + radius * np.sin(theta)

        cam = PinholeCamera(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            position=np.array([x, y, z], dtype=np.float32),
            look_at=center.copy(),
        )
        cameras.append(cam)
    return cameras


def generate_linear_path(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    look_at: np.ndarray,
    num_views: int = 30,
    width: int = 1280,
    height: int = 720,
    fx: float = 960.0,
    fy: float = 960.0,
) -> List[PinholeCamera]:
    """生成直线运动相机轨迹（推轨镜头效果）。

    Parameters
    ----------
    start_pos : np.ndarray, shape (3,)
        起始位置。
    end_pos : np.ndarray, shape (3,)
        结束位置。
    look_at : np.ndarray, shape (3,)
        注视点。
    num_views : int
        采样数量。
    width, height : int
        图像分辨率。
    fx, fy : float
        焦距。

    Returns
    -------
    list of PinholeCamera
    """
    cameras = []
    for i in range(num_views):
        t = i / (num_views - 1) if num_views > 1 else 0
        pos = start_pos + (end_pos - start_pos) * t
        cam = PinholeCamera(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            position=pos.astype(np.float32).copy(),
            look_at=look_at.copy(),
        )
        cameras.append(cam)
    return cameras
