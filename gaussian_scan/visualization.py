"""
可视化模块
==========

提供 3D Gaussian 点云的可视化功能，包括：
- 静态图像（matplotlib）
- 多视角对比
- 深度图可视化
- HTML 交互式预览（可选）

依赖：matplotlib, open3d（可选用于 3D 交互）
"""

import os
import math
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

from gaussian_scan.gaussian_model import GaussianCloud
from gaussian_scan.camera import PinholeCamera, generate_orbit_cameras
from gaussian_scan.renderer import SimpleRenderer, RenderConfig
from gaussian_scan.utils import ensure_dir


def plot_point_cloud_2d(
    cloud: GaussianCloud,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Gaussian Point Cloud",
    dpi: int = 120,
) -> None:
    """绘制 Gaussian 点云的 2D 散点图（XY 投影 + 颜色）。

    Parameters
    ----------
    cloud : GaussianCloud
        待可视化的高斯点云。
    save_path : str, optional
        保存路径，None 则显示。
    figsize : tuple
        图像尺寸。
    title : str
        图表标题。
    dpi : int
        分辨率。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    positions = cloud.get_positions()
    colors = cloud.get_colors()
    scales = cloud.get_scales()
    opacities = cloud.get_opacities()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # 按透明度排序，透明点后画
    sort_idx = np.argsort(opacities)
    positions = positions[sort_idx]
    colors = colors[sort_idx]
    sizes = scales[sort_idx].mean(axis=1) * 200  # 点大小与尺度相关

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=np.clip(colors, 0, 1),
        s=np.clip(sizes, 1, 100),
        alpha=0.6,
        edgecolors="none",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 等比例缩放
    max_range = np.max([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min(),
    ]) / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
    if max_range > 0:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_orbit_grid(
    cloud: GaussianCloud,
    cameras: List[PinholeCamera],
    save_path: Optional[str] = None,
    cols: int = 6,
    figsize_per_image: Tuple[int, int] = (3, 2),
    title: str = "Orbit View Renderings",
) -> np.ndarray:
    """生成环绕视角渲染网格图。

    Parameters
    ----------
    cloud : GaussianCloud
        高斯点云。
    cameras : list of PinholeCamera
        轨道相机列表。
    save_path : str, optional
        保存路径。
    cols : int
        每行图像数量。
    figsize_per_image : tuple
        每子图尺寸 (w, h) 英寸。
    title : str
        标题。

    Returns
    -------
    np.ndarray
        网格图像 (H, W, 3)，范围 [0, 1]。
    """
    import matplotlib.pyplot as plt

    renderer = SimpleRenderer(cloud)
    images = renderer.render_multiview(cameras, progress=False)

    n = len(images)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1]),
        dpi=100,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1))
            ax.set_title(f"View {i+1}", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    # 先渲染画布
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    grid_img = buf[:, :, :3].astype(np.float32) / 255.0

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()
    return grid_img


def depth_map_visualization(
    depth_map: np.ndarray,
    save_path: Optional[str] = None,
    colormap: str = "inferno",
    title: str = "Depth Map",
) -> None:
    """可视化深度图，叠加色谱。

    Parameters
    ----------
    depth_map : np.ndarray, shape (H, W)
        深度图。
    save_path : str, optional
        保存路径。
    colormap : str
        matplotlib 色谱名称。
    title : str
        标题。
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 灰度图
    axes[0].imshow(depth_map, cmap="gray")
    axes[0].set_title("Grayscale", fontsize=12)
    axes[0].axis("off")

    # 伪彩色图
    im = axes[1].imshow(depth_map, cmap=colormap)
    axes[1].set_title(f"{colormap.capitalize()} Colormap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
    else:
        plt.show()


def generate_html_preview(
    cloud: GaussianCloud,
    output_path: str,
    title: str = "GaussianScan-Recon Preview",
) -> str:
    """生成交互式 HTML 预览页（使用 Three.js CDN 的点云可视化）。

    Parameters
    ----------
    cloud : GaussianCloud
        高斯点云。
    output_path : str
        输出 HTML 文件路径。
    title : str
        页面标题。

    Returns
    -------
    str
        输出文件路径。
    """
    positions = cloud.get_positions()
    colors = cloud.get_colors()

    # 转换为 JavaScript 数组
    points_json = []
    for i in range(len(positions)):
        x, y, z = positions[i]
        r, g, b = np.clip(colors[i], 0, 1)
        points_json.append(
            f"{{'x':{x:.4f},'y':{y:.4f},'z':{z:.4f},'r':{r:.3f},'g':{g:.3f},'b':{b:.3f}}}"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #0a0a14; font-family: 'Segoe UI', sans-serif; }}
  #info {{ position: absolute; top: 20px; left: 20px; color: white; z-index: 100; pointer-events: none; }}
  #info h1 {{ font-size: 1.5em; margin: 0; font-weight: 300; letter-spacing: 2px; }}
  #info p {{ font-size: 0.8em; opacity: 0.6; margin: 5px 0 0 0; }}
  #controls {{ position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); color: rgba(255,255,255,0.5); font-size: 0.75em; z-index: 100; }}
</style>
</head>
<body>
<div id="info">
  <h1>🔮 {title}</h1>
  <p>{len(cloud):,} Gaussian points | Drag to rotate | Scroll to zoom</p>
</div>
<div id="controls">🖱️ Left-drag: Rotate &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Right-drag: Pan</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(3, 2, 4);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 0);

// Grid helper
const grid = new THREE.GridHelper(5, 20, 0x333344, 0x1a1a2e);
scene.add(grid);

const geometry = new THREE.BufferGeometry();
const pointData = [{','.join(points_json[:500])}];
const positions = new Float32Array(pointData.length * 3);
const colors = new Float32Array(pointData.length * 3);
pointData.forEach((p, i) => {{
  positions[i*3] = p.x; positions[i*3+1] = p.y; positions[i*3+2] = p.z;
  colors[i*3] = p.r; colors[i*3+1] = p.g; colors[i*3+2] = p.b;
}});
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const material = new THREE.PointsMaterial({{ size: 0.03, vertexColors: true, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.8 }});
const points = new THREE.Points(geometry, material);
scene.add(points);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""

    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
