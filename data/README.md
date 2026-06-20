# 数据目录

## 目录结构

```
data/
├── README.md                     # 本文件
└── samples/
    ├── object_scan/              # 物体扫描示例
    │   └── (放置多视角物体图片)
    └── room_scene/               # 室内场景示例
        └── (放置室内场景图片或视频)
```

## 使用说明

### 物体扫描

将多张不同角度的物体照片放入 `data/samples/object_scan/`，运行：

```bash
python scripts/preprocess.py --input data/samples/object_scan --output outputs/frames
```

### 室内场景

将室内场景照片或视频放入 `data/samples/room_scene/`，运行：

```bash
python scripts/preprocess.py --input data/samples/room_scene --output outputs/frames
```

### 快速演示（无需真实数据）

```bash
# 生成 10 张示例图像
python scripts/preprocess.py --input DEMO --output data/samples/object_scan --num-demo 10
```

## 支持的格式

- 图片：JPG (.jpg, .jpeg), PNG (.png), BMP (.bmp), TIFF (.tif, .tiff), WebP (.webp)
- 视频：MP4 (.mp4), AVI (.avi), MOV (.mov), MKV (.mkv), WebM (.webm), FLV (.flv)

## 注意事项

- 视频文件默认被 .gitignore 忽略，不会提交到仓库
- 大文件建议放在 `outputs/` 目录下（也在 .gitignore 中）
- 如需提交小尺寸示例数据，请手动 `git add -f`
