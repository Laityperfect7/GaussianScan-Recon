"""
测试: 配置模块
"""

import os
import sys
import tempfile
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 模拟 train_mock 中的 load_config
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDefaultConfig:
    """默认配置测试。"""

    def test_config_exists(self):
        """测试配置文件存在。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"

    def test_config_is_valid_yaml(self):
        """测试配置文件是有效的 YAML。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)
        assert isinstance(config, dict)

    def test_required_sections(self):
        """测试必需的配置段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)

        required_sections = [
            "scene",
            "input",
            "preprocessing",
            "depth",
            "gaussian",
            "training",
            "visualization",
        ]
        for section in required_sections:
            assert section in config, f"配置缺少段: {section}"

    def test_scene_section(self):
        """测试 scene 配置段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)
        scene = config["scene"]
        assert "name" in scene
        assert "type" in scene

    def test_gaussian_section(self):
        """测试 gaussian 配置段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)
        gs = config["gaussian"]
        assert gs["num_init"] > 0
        assert 0 <= gs["sh_degree"] <= 4
        assert len(gs["scene_bounds"]) == 3

    def test_training_section(self):
        """测试 training 配置段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)
        train = config["training"]
        assert train["iterations"] > 0
        assert "learning_rate" in train
        assert "loss" in train

    def test_depth_section(self):
        """测试 depth 配置段。"""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        config = load_config(config_path)
        depth = config["depth"]
        assert depth["model"] == "mock"
        assert "mock" in depth


class TestConfigRoundTrip:
    """配置读写测试。"""

    def test_write_and_read_config(self):
        """测试 YAML 配置写入和读取。"""
        config = {
            "test_key": "test_value",
            "nested": {"a": 1, "b": 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.yaml")
            with open(path, "w") as f:
                yaml.dump(config, f)
            loaded = load_config(path)
            assert loaded == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
