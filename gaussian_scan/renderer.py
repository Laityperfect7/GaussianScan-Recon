"""
简化渲染器模块
===============

提供一个简化的 Gaussian 点云光栅化渲染器（非 CUDA），
用于演示和可视化。真实 3DGS 训练需使用 CUDA 可微光栅化器。

渲染策略：
  - 使用 painter's algorithm 简化近似
  - 将 Gaussian 椭球投影为屏幕椭圆
  - 适用于小规模点云的实时预览
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from gaussian_scan.gaussian_model import GaussianCloud, GaussianPoint
from gaussian_scan.camera import PinholeCamera


@dataclass
class RenderConfig:
    """渲染配置。

    Attributes
    ----------
    background_color : tuple
        背景颜色 (R, G, B)，范围 [0, 1]。
    alpha_threshold : float
        alpha 低于此值的 Gaussian 不渲染。
    near_plane : float
        近裁剪面距离。
    far_plane : float
        远裁剪面距离。
    """

    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.08)
    alpha_threshold: float = 0.01
    near_plane: float = 0.1
    far_plane: float = 50.0


class SimpleRenderer:
    """简化 Gaussian 渲染器。

    通过将 3D Gaussian 投影到 2D 屏幕并用椭圆近似 splatting，
    实现无需 CUDA 的可视化渲染。

    Parameters
    ----------
    cloud : GaussianCloud
        待渲染的高斯点云。
    config : RenderConfig
        渲染配置。

    Examples
    --------
    >>> cloud = GaussianCloud.random_cloud(5000)
    >>> renderer = SimpleRenderer(cloud)
    >>> image = renderer.render(camera)
    """

    def __init__(
        self,
        cloud: GaussianCloud,
        config: Optional[RenderConfig] = None,
    ):
        self.cloud = cloud
        self.config = config or RenderConfig()

    def render(
        self,
        camera: PinholeCamera,
    ) -> np.ndarray:
        """从指定相机渲染一幅 RGB 图像。

        Parameters
        ----------
        camera : PinholeCamera
            渲染相机。

        Returns
        -------
        np.ndarray, shape (H, W, 3)
            RGB 图像，值范围 [0, 1]。
        """
        W, H = camera.width, camera.height

        # 初始化画布
        image = np.ones((H, W, 3), dtype=np.float32)
        for c in range(3):
            image[:, :, c] = self.config.background_color[c]

        if len(self.cloud) == 0:
            return image

        # 获取所有点属性
        positions = self.cloud.get_positions()  # (N, 3)
        colors = self.cloud.get_colors()  # (N, 3)
        scales = self.cloud.get_scales()  # (N, 3)
        opacities = self.cloud.get_opacities()  # (N,)

        # 转换到相机坐标系
        cam_points = camera.world_to_camera(positions)  # (N, 3)
        z_vals = cam_points[:, 2]

        # 裁剪
        valid = (z_vals > self.config.near_plane) & (
            z_vals < self.config.far_plane
        )
        valid &= opacities > self.config.alpha_threshold
        if not valid.any():
            return image

        cam_points = cam_points[valid]
        colors = colors[valid]
        scales = scales[valid]
        opacities = opacities[valid]
        z_vals = z_vals[valid]

        # 投影到像素坐标
        uv = camera.camera_to_pixel(cam_points)  # (N, 2)

        # 按深度从远到近排序（painter's algorithm）
        sort_idx = np.argsort(z_vals)[::-1]  # 远的先画
        uv = uv[sort_idx]
        colors = colors[sort_idx]
        scales = scales[sort_idx]
        opacities = opacities[sort_idx]
        cam_points = cam_points[sort_idx]
        z_vals = z_vals[sort_idx]

        # 简化 splatting：对每个点画一个椭圆
        for i in range(len(uv)):
            u, v = uv[i]
            if u < 0 or u >= W or v < 0 or v >= H:
                continue

            # 屏幕空间椭圆半径（与深度和尺度相关）
            s = scales[i].mean()
            radius = max(1, int(s * camera.fx / max(z_vals[i], 0.1)))

            # 限制半径避免过大开销
            radius = min(radius, 15)

            color = colors[i]
            alpha = opacities[i]

            # 在椭圆区域内混合
            y_min = max(0, int(v) - radius)
            y_max = min(H, int(v) + radius + 1)
            x_min = max(0, int(u) - radius)
            x_max = min(W, int(u) + radius + 1)

            if y_min >= y_max or x_min >= x_max:
                continue

            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            dist_sq = ((xx - u) ** 2 + (yy - v) ** 2) / (radius**2 + 1e-8)
            gauss_weight = np.exp(-dist_sq) * alpha

            for c in range(3):
                patch = image[y_min:y_max, x_min:x_max, c]
                image[y_min:y_max, x_min:x_max, c] = (
                    patch * (1 - gauss_weight) + color[c] * gauss_weight
                )

        return np.clip(image, 0, 1)

    def render_multiview(
        self,
        cameras: List[PinholeCamera],
        progress: bool = True,
    ) -> List[np.ndarray]:
        """从多个相机渲染图像序列。

        Parameters
        ----------
        cameras : list of PinholeCamera
            相机列表。
        progress : bool
            是否显示进度条。

        Returns
        -------
        list of np.ndarray
            渲染图像列表，每幅 (H, W, 3)，范围 [0, 1]。
        """
        images = []
        iterator = cameras
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(cameras, desc="Rendering")
            except ImportError:
                pass

        for cam in iterator:
            img = self.render(cam)
            images.append(img)
        return images


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def render_demo_image(cloud: GaussianCloud, camera: PinholeCamera) -> np.ndarray:
    """使用默认配置渲染一张演示图像。

    Parameters
    ----------
    cloud : GaussianCloud
        点云。
    camera : PinholeCamera
        相机。

    Returns
    -------
    np.ndarray, shape (H, W, 3)
    """
    renderer = SimpleRenderer(cloud)
    return renderer.render(camera)
