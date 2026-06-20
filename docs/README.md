# GaussianScan-Recon Documentation

## 目录

- [项目简介](#项目简介)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [配置文件](#配置文件)
- [真实 3DGS 训练指南](#真实-3dgs-训练指南)
- [常见问题](#常见问题)

---

## 项目简介

GaussianScan-Recon 是一个面向 3D Gaussian Splatting 的模块化管线项目，支持从图像或单目视频进行物体扫描与场景复现。

当前版本提供完整的 **流水线框架** 和 **模拟训练流程**，方便学习、课程设计和二次开发。

---

## 安装指南

```bash
# 克隆仓库
git clone https://github.com/Laityperfect7/GaussianScan-Recon.git
cd GaussianScan-Recon

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

---

## 快速开始

一键运行完整流水线（模拟模式）：

```bash
# Step 1: 生成示例数据
python scripts/preprocess.py --input DEMO --output outputs/frames --num-demo 10

# Step 2: 模拟深度估计
python scripts/estimate_depth.py --input outputs/frames --output outputs/depth --mode radial

# Step 3: 模拟训练
python scripts/train_mock.py --config configs/default.yaml

# Step 4: 可视化
python scripts/visualize.py --checkpoint outputs/checkpoints/mock_model.npz --mode all
```

---

## 模块说明

### gaussian_scan/ — 核心库

| 模块 | 说明 |
|------|------|
| `gaussian_model.py` | GaussianPoint / GaussianCloud 数据结构、PLY/NPZ I/O |
| `camera.py` | 针孔相机模型、轨道/直线轨迹生成 |
| `dataset.py` | 图像数据集加载、视频帧提取 |
| `renderer.py` | 简化光栅化渲染器（非 CUDA） |
| `visualization.py` | matplotlib 2D/3D 可视化、HTML 交互预览 |
| `utils.py` | 图像读写、缩放、网格拼接 |

### scripts/ — 流水线脚本

| 脚本 | 功能 |
|------|------|
| `preprocess.py` | 视频抽帧 / 图片加载 / 示例生成 |
| `estimate_depth.py` | Mock 深度估计接口 |
| `train_mock.py` | 模拟 3DGS 训练 |
| `visualize.py` | 结果可视化 |

---

## 配置文件

配置文件位于 `configs/default.yaml`，包含完整的参数说明。

所有参数均可通过命令行 `--config` 指定自定义配置文件。

---

## 真实 3DGS 训练指南

当前项目为 **模拟训练模式**。要执行真实 3D Gaussian Splatting 训练，需要：

### 1. 安装 CUDA 依赖

```bash
# 安装 PyTorch (CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 克隆 3DGS 官方仓库
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git submodules/gaussian-splatting

# 安装 CUDA 光栅化器
pip install submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install submodules/gaussian-splatting/submodules/simple-knn
```

### 2. 替换深度估计

在 `scripts/estimate_depth.py` 中集成 Depth Anything V2：

```python
# 安装
pip install torch hubconf

# 加载模型
import torch
model = torch.hub.load('depth-anything/Depth-Anything-V2', 'Depth-Anything-V2-Large')
```

### 3. 替换训练逻辑

将 `scripts/train_mock.py` 替换为基于 `diff-gaussian-rasterization` 的真实训练脚本。

### 4. 可选：集成 COLMAP

对于多视角静态场景，可使用 COLMAP 获取更精确的相机位姿和稀疏点云。

---

## 常见问题

### Q: 为什么使用模拟训练而不是真实训练？

A: 真实 3DGS 训练需要 CUDA 环境和较大的 GPU 显存（≥8GB），本项目提供模拟训练用于学习和代码架构展示。真实训练的接口已预留。

### Q: 模拟训练产生的点云有意义吗？

A: 模拟训练产生的 Gaussian 点云为随机数据，用于验证 I/O、可视化、导出等流程是否正常工作。要获得有意义的重建结果，需执行真实训练。

### Q: 项目支持哪些输入格式？

A: 图片：JPG、PNG、BMP、TIFF、WebP；视频：MP4、AVI、MOV、MKV、WebM、FLV。

### Q: 如何导出结果？

A: 支持 PLY（二进制）和 NPZ（压缩数组）两种格式。PLY 兼容 CloudCompare、MeshLab、Blender 等工具。
