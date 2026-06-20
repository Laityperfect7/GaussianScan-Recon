"""
测试: Gaussian 模型模块
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaussian_scan.gaussian_model import (
    GaussianPoint,
    GaussianCloud,
    save_ply,
    save_npz,
    load_npz,
)


class TestGaussianPoint:
    """GaussianPoint 单元测试。"""

    def test_creation_defaults(self):
        """测试默认创建。"""
        p = GaussianPoint(
            position=[1.0, 2.0, 3.0],
            color=[0.5, 0.5, 0.5],
            scale=[0.1, 0.1, 0.1],
            opacity=0.8,
        )
        assert p.position.shape == (3,)
        assert p.color.shape == (3,)
        assert p.scale.shape == (3,)
        assert 0.0 <= p.opacity <= 1.0
        assert p.rotation.shape == (4,)  # 默认单位四元数

    def test_creation_with_rotation(self):
        """测试带旋转的创建。"""
        rot = np.array([0.707, 0.707, 0.0, 0.0])
        p = GaussianPoint(
            position=[0, 0, 0],
            color=[1, 0, 0],
            scale=[0.05, 0.05, 0.05],
            opacity=0.5,
            rotation=rot,
        )
        np.testing.assert_array_almost_equal(p.rotation, rot)

    def test_xyz_property(self):
        """测试 xyz 属性。"""
        p = GaussianPoint(
            position=[1.0, 2.0, 3.0],
            color=[0, 0, 0],
            scale=[0.1, 0.1, 0.1],
            opacity=1.0,
        )
        np.testing.assert_array_equal(p.xyz, [1.0, 2.0, 3.0])

    def test_opacity_clip(self):
        """测试不透明度裁剪。"""
        p = GaussianPoint(
            position=[0, 0, 0],
            color=[0, 0, 0],
            scale=[0.1, 0.1, 0.1],
            opacity=2.5,
        )
        assert p.opacity == 1.0

        p2 = GaussianPoint(
            position=[0, 0, 0],
            color=[0, 0, 0],
            scale=[0.1, 0.1, 0.1],
            opacity=-0.5,
        )
        assert p2.opacity == 0.0

    def test_repr(self):
        """测试字符串表示。"""
        p = GaussianPoint(
            position=[1.0, 2.0, 3.0],
            color=[0.5, 0.5, 0.5],
            scale=[0.1, 0.1, 0.1],
            opacity=0.9,
        )
        s = repr(p)
        assert "GaussianPoint" in s


class TestGaussianCloud:
    """GaussianCloud 单元测试。"""

    def test_empty_cloud(self):
        """测试空点云。"""
        cloud = GaussianCloud()
        assert len(cloud) == 0

    def test_add_point(self):
        """测试添加点。"""
        cloud = GaussianCloud()
        cloud.add_point(
            position=[0, 0, 0],
            color=[1, 0, 0],
            scale=[0.1, 0.1, 0.1],
            opacity=0.9,
        )
        assert len(cloud) == 1

    def test_get_positions(self):
        """测试批量获取位置。"""
        cloud = GaussianCloud()
        for i in range(10):
            cloud.add_point(
                position=[i, i, i],
                color=[1, 1, 1],
                scale=[0.1, 0.1, 0.1],
                opacity=1.0,
            )
        positions = cloud.get_positions()
        assert positions.shape == (10, 3)

    def test_get_colors(self):
        """测试批量获取颜色。"""
        cloud = GaussianCloud()
        cloud.add_point(position=[0, 0, 0], color=[0.2, 0.4, 0.6], scale=[0.1, 0.1, 0.1], opacity=1.0)
        cloud.add_point(position=[1, 1, 1], color=[0.8, 0.6, 0.4], scale=[0.1, 0.1, 0.1], opacity=1.0)
        colors = cloud.get_colors()
        assert colors.shape == (2, 3)

    def test_random_cloud(self):
        """测试随机点云生成。"""
        cloud = GaussianCloud.random_cloud(num_points=1000, seed=123)
        assert len(cloud) == 1000
        positions = cloud.get_positions()
        # 位置应该在 bounds 范围内
        assert np.all(np.abs(positions) <= 3.0)  # 默认 bounds 导致

    def test_random_cloud_reproducibility(self):
        """测试随机点云可复现性。"""
        cloud1 = GaussianCloud.random_cloud(num_points=100, seed=42)
        cloud2 = GaussianCloud.random_cloud(num_points=100, seed=42)
        np.testing.assert_array_almost_equal(
            cloud1.get_positions(), cloud2.get_positions()
        )

    def test_from_depth_and_image(self):
        """测试从深度图和图像生成点云。"""
        H, W = 60, 80
        image = np.random.rand(H, W, 3).astype(np.float32)
        depth_map = np.random.rand(H, W).astype(np.float32)
        K = np.array([[100, 0, 40], [0, 100, 30], [0, 0, 1]], dtype=np.float32)

        cloud = GaussianCloud.from_depth_and_image(
            image, depth_map, K, sample_step=10
        )
        # 60/10 * 80/10 = 6*8 = 48 个点
        assert len(cloud) == 48

    def test_to_dict_and_from_dict(self):
        """测试序列化往返。"""
        cloud = GaussianCloud.random_cloud(num_points=50, seed=1)
        data = cloud.to_dict()
        restored = GaussianCloud.from_dict(data)
        assert len(restored) == len(cloud)
        np.testing.assert_array_almost_equal(
            cloud.get_positions(), restored.get_positions()
        )

    def test_export_import_npz(self):
        """测试 NPZ 导出/导入。"""
        cloud = GaussianCloud.random_cloud(num_points=50, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            cloud.export_npz(path)
            assert os.path.exists(path)

            loaded = GaussianCloud.from_npz(path)
            assert len(loaded) == len(cloud)
            np.testing.assert_array_almost_equal(
                cloud.get_positions(), loaded.get_positions()
            )

    def test_export_ply(self):
        """测试 PLY 导出。"""
        cloud = GaussianCloud.random_cloud(num_points=50, seed=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.ply")
            cloud.export_ply(path)
            assert os.path.exists(path)
            # 验证文件非空
            assert os.path.getsize(path) > 0

    def test_iter(self):
        """测试迭代。"""
        cloud = GaussianCloud.random_cloud(num_points=10)
        points = list(cloud)
        assert len(points) == 10
        assert all(isinstance(p, GaussianPoint) for p in points)

    def test_scale_default(self):
        """测试添加点时默认尺度。"""
        cloud = GaussianCloud()
        cloud.add_point(position=[0, 0, 0], color=[1, 1, 1])
        assert len(cloud) == 1
        assert cloud[0].scale.shape == (3,)


class TestPLYFormat:
    """PLY 格式相关测试。"""

    def test_binary_ply_header(self):
        """测试 PLY 头写入。"""
        cloud = GaussianCloud.random_cloud(num_points=5, seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.ply")
            cloud.export_ply(path)
            with open(path, "r", encoding="ascii", errors="ignore") as f:
                header = f.read(500)
            assert "ply" in header.lower()
            assert "element vertex 5" in header
            assert "end_header" in header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
