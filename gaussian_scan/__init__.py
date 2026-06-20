"""
GaussianScan-Recon: 高斯泼溅物体扫描与场景复现系统
====================================================

A practical 3D Gaussian Splatting pipeline for object scanning
and scene reconstruction from images or monocular video.

核心模块：
- gaussian_model : 3D Gaussian 数据结构与点云生成
- camera        : 相机模型与轨迹生成
- dataset       : 数据集加载与预处理
- renderer      : 简化渲染器（splatting 近似）
- visualization : 可视化工具
- utils         : 通用工具函数
"""

__version__ = "0.1.0"
__author__ = "Laityperfect7"

from gaussian_scan.gaussian_model import GaussianPoint, GaussianCloud
from gaussian_scan.camera import PinholeCamera, generate_orbit_cameras
from gaussian_scan.dataset import ImageDataset, VideoFrameExtractor
from gaussian_scan.utils import ensure_dir, save_ply, save_npz, load_npz

__all__ = [
    "GaussianPoint",
    "GaussianCloud",
    "PinholeCamera",
    "generate_orbit_cameras",
    "ImageDataset",
    "VideoFrameExtractor",
    "ensure_dir",
    "save_ply",
    "save_npz",
    "load_npz",
]
