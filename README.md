# monocular-3d-detection-v1

这是一个离线 monocular / surround-view 3D box 重建实验工程。当前主线流程是：

1. 已有 2D track / mask / depth 初始化 / 标定输入；
2. 分别在 `rear`、`left_rear`、`right_rear` 三个视角内做单视角 3D box 轨迹优化；
3. 使用 LoMa/外观/极线几何把不同视角的同一辆车合成 `global_track_id`；
4. 基于单视角优化结果，对同一个 `global_track_id` 做跨视角联合后优化；
5. 输出 CSV 和视频可视化。

更详细的中文流程说明见：

```text
documentation/current_multiview_3d_pipeline_cn.md
```

## 1. 这个 GitHub 包里有什么

包含：

```text
code/                         核心 Python 代码
configs/                      当前主线配置和通用配置
documentation/                中文流程文档
scripts/                      Ubuntu/WSL 启动脚本和可视化脚本
environment-ubuntu-gpu.yml    Ubuntu GPU 环境参考
environment-original-2dbox-full-cpu.yml  CPU 环境参考
```

不包含：

```text
data/             原始图像、标定、场景数据
preprocessed/     2D track、mask、depth 初始化等中间结果
outputs/          实验输出、视频、日志
checkpoints/      SAM / Depth Anything / LoMa / YOLO 等模型权重
external/LoMa     LoMa 外部仓库代码
third_party/      SAM、Depth Anything 等外部仓库
```

这些目录在仓库中只保留空目录占位，需要你下载或手动拷贝。

## 2. Ubuntu 环境配置

我们在 WSL Ubuntu 上测试过当前流程。推荐新机器使用 Miniforge/Conda。

### 2.1 安装 Miniforge

如果机器上没有 conda：

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/miniforge3/etc/profile.d/conda.sh
```

如果你安装到了别的位置，后面运行脚本时可以设置：

```bash
export CONDA_SH=/your/miniforge3/etc/profile.d/conda.sh
```

项目自带脚本默认先找：

```text
/opt/miniforge3/etc/profile.d/conda.sh
```

### 2.2 创建 GPU 环境

```bash
cd /path/to/monocular-3d-detection-v1
conda env create -f environment-ubuntu-gpu.yml
conda activate mono-detect-original-2dbox-full-gpu
```

安装 CUDA 版 PyTorch。当前 WSL 测试用的是 CUDA 12.8 对应的 PyTorch wheel；Ubuntu 新机器建议根据实际 NVIDIA driver 选择版本。若使用 CUDA 12.8：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps numpy==1.26.4
```

第二行是必要的：部分 PyTorch/torchvision wheel 会把 numpy 升到 2.x；当前代码和 OpenCV 组合按 `numpy==1.26.4` 测试。

验证：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

如果只想先 CPU 检查代码：

```bash
conda env create -f environment-original-2dbox-full-cpu.yml
conda activate mono-detect-original-2dbox-full-cpu
```

CPU 可以跑小样本和可视化，但全量 3D 优化会很慢。

## 3. 外部代码和模型权重放置

### 3.1 LoMa

跨视角 2D track 匹配需要 LoMa。放到：

```text
external/LoMa
```

示例：

```bash
cd /path/to/monocular-3d-detection-v1
mkdir -p external
git clone https://github.com/davnords/LoMa.git external/LoMa
```

如果 LoMa 需要额外模型权重，请按 LoMa 官方 README 下载，并放在 LoMa 默认查找的位置，或在 `configs/loma_global_2d_repair_params_tracks_v1.yaml` 中修改。

### 3.2 SAM / Depth Anything / 其他权重

本项目支持两种运行方式：

1. **已有预处理结果模式**：已经有 2D track、mask、depth 初始化 CSV，直接进入 3D box 优化和跨视角联合优化。
2. **原始 2D box 端到端模式**：原始数据里只有 annotation JSON 中的 2D box，需要先跑 SAM 生成 mask、跑 DA3 生成 depth，再做 2D tracking 和 3D box 优化。

如果使用第 2 种模式，需要额外放置这些外部仓库和权重：

```text
third_party/segment_anything/
third_party/Depth-Anything-3/
checkpoints/
```

当前代码默认读取：

```text
checkpoints/
  sam_vit_h_4b8939.pth

third_party/Depth-Anything-3/checkpoints/
  da3metric-large/
```

下载示例：

```bash
cd /path/to/monocular-3d-detection-v1

mkdir -p third_party checkpoints
git clone https://github.com/facebookresearch/segment-anything.git third_party/segment_anything
git clone https://github.com/DepthAnything/Depth-Anything-3.git third_party/Depth-Anything-3

wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

pip install -U huggingface_hub
mkdir -p third_party/Depth-Anything-3/checkpoints/da3metric-large
huggingface-cli download depth-anything/DA3METRIC-LARGE \
  --local-dir third_party/Depth-Anything-3/checkpoints/da3metric-large
```

如果 Hugging Face 下载慢，可以临时使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download depth-anything/DA3METRIC-LARGE \
  --local-dir third_party/Depth-Anything-3/checkpoints/da3metric-large
```

由于权重体积大且授权各不相同，本仓库不提交权重。建议优先从官方地址下载；如果网络不稳定，也可以手动拷贝 `checkpoints/` 和 `third_party/Depth-Anything-3/checkpoints/`。

## 4. 数据和预处理结果放置

当前配置默认读取以下路径。新机器上需要保持相同相对目录，或修改配置文件。

### 4.1 图像

```text
data/camera/rear_camera/*.jpg
data/camera/left_rear_camera/*.jpg
data/camera/right_rear_camera/*.jpg
```

### 4.2 标定

```text
data/calib/rear_camera/rear_camera-intrinsic.json
data/calib/rear_camera/rear_camera-to-car_center-extrinsic.json

data/calib/left_rear_camera/left_rear_camera-intrinsic.json
data/calib/left_rear_camera/left_rear_camera-to-car_center-extrinsic.json

data/calib/right_rear_camera/right_rear_camera-intrinsic.json
data/calib/right_rear_camera/right_rear_camera-to-car_center-extrinsic.json
```

外参文件名里的 `camera-to-car_center` 表示：

```text
P_car = T_camera_to_car @ P_camera
```

渲染/投影到相机时会使用反方向变换。

### 4.3 2D track + depth 初始化

```text
preprocessed/tracks/robust_botsort_hybrid_depth_v1/rear_camera/tracks_hybrid_depth.csv
preprocessed/tracks/robust_botsort_hybrid_depth_v1/left_rear_camera/tracks_hybrid_depth.csv
preprocessed/tracks/robust_botsort_hybrid_depth_v1/right_rear_camera/tracks_hybrid_depth.csv
```

这些 CSV 应至少包含：

- `frame`
- `timestamp`
- `track_id`
- `x1, y1, x2, y2`
- 类别字段，例如 `class_name`
- 图像路径字段
- depth / 3D 初始化相关字段，具体以当前代码读取为准

### 4.4 mask / detection CSV

单视角 3D 优化当前默认使用 ensured/cropped mask：

```text
preprocessed/masks_robust_botsort_ensured_v1/rear_camera/gt2d_sam_masks_ensured_cropped.csv
preprocessed/masks_robust_botsort_ensured_v1/left_rear_camera/gt2d_sam_masks_ensured_cropped.csv
preprocessed/masks_robust_botsort_ensured_v1/right_rear_camera/gt2d_sam_masks_ensured_cropped.csv
```

LoMa/外观匹配配置中还会读取 detection/mask CSV：

```text
preprocessed/masks/rear_camera/gt2d_sam_masks.csv
preprocessed/masks/left_rear_camera/gt2d_sam_masks.csv
preprocessed/masks/right_rear_camera/gt2d_sam_masks.csv
```

mask 图像建议放在：

```text
preprocessed/masks_robust_botsort_ensured_v1/rear_camera/masks/
preprocessed/masks_robust_botsort_ensured_v1/left_rear_camera/masks/
preprocessed/masks_robust_botsort_ensured_v1/right_rear_camera/masks/

preprocessed/masks/rear_camera/masks/
preprocessed/masks/left_rear_camera/masks/
preprocessed/masks/right_rear_camera/masks/
```

CSV 中的 `mask_path` 可以是相对仓库根目录的路径。

### 4.5 LoMa/外观匹配需要的 detection embedding

当前 LoMa 配置默认读取：

```text
outputs/robust_botsort_2d_rear_views/rear_camera/detection_embeddings.npz
outputs/robust_botsort_2d_rear_views/left_rear_camera/detection_embeddings.npz
outputs/robust_botsort_2d_rear_views/right_rear_camera/detection_embeddings.npz
```

虽然路径在 `outputs/` 下，但它是 LoMa 匹配阶段的输入缓存。你可以：

1. 手动把已有缓存拷贝到上述路径；
2. 或者修改 `configs/loma_global_2d_repair_params_tracks_v1.yaml` 中的 `embedding_cache` 路径。

## 5. 运行主线流程

先进入仓库并激活环境：

```bash
cd /path/to/monocular-3d-detection-v1
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate mono-detect-original-2dbox-full-gpu
```

如果 conda 安装在别处：

```bash
export CONDA_SH=~/miniforge3/etc/profile.d/conda.sh
```

### 5.1 可选前处理：从原始 2D box 生成 mask / depth / track

如果你的输入是原始场景数据，且里面已经有 2D box annotation，但还没有 mask、DA3 depth、2D track，则先跑这一段。

原始数据目录需要包含：

```text
data/
  camera/
    rear_camera/
    left_rear_camera/
    right_rear_camera/
  calib/
    rear_camera/
    left_rear_camera/
    right_rear_camera/
  format_output/annotations/NV/
```

其中 `format_output/annotations/NV/*.json` 是原始 2D box 来源；本流程不会跑目标检测。

一键运行：

```bash
cd /path/to/monocular-3d-detection-v1
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate mono-detect-original-2dbox-full-gpu

bash scripts/run_original_2dbox_full_pipeline.sh \
  /path/to/scene_data \
  configs/original_2dbox_full_pipeline_gpu.yaml
```

后台运行：

```bash
mkdir -p outputs/original_2dbox_full_gpu_v1/logs
nohup bash scripts/run_original_2dbox_full_pipeline.sh \
  /path/to/scene_data \
  configs/original_2dbox_full_pipeline_gpu.yaml \
  > outputs/original_2dbox_full_gpu_v1/logs/e2e_run.log 2>&1 &
```

看日志：

```bash
tail -f outputs/original_2dbox_full_gpu_v1/logs/e2e_run.log
```

这一步会依次执行：

```text
1. export_original_2dbox_csv.py
   从 annotation JSON 导出每个相机的原始 2D box CSV。

2. sam_segment_gt2d_boxes.py
   用 SAM 对每个 2D box 生成前景 mask。

3. da3_metric_rear_depth_export.py
   用 DA3 Metric 为每个相机导出 metric depth。

4. retrack_sort_2d
   基于原始 2D box 重新做 2D SORT tracking。

5. attach_da3_depth_to_tracks.py
   把 DA3 depth 绑定到 track CSV，作为 3D 初始化深度来源。

6. ensure_masks_for_all_tracks
   确保每个 tracked 2D box 都能找到一个对应 mask；必要时会补齐 cropped mask。

7. rebuild_3d_box_optimizer.run
   对每个 track 做单视角 3D box 优化，并输出展示版视频。
```

主要输出目录：

```text
outputs/original_2dbox_full_gpu_v1/
  tracking_input/          原始 2D box 导出的 CSV
  masks/raw_sam/           SAM 原始 mask 结果
  depth/                   DA3 metric depth
  tracking/sort2d_tracks/  2D tracking 结果
  tracks_with_depth/       绑定 depth 后的 track CSV
  masks/ensured/           与 track 对齐后的 mask
  optimized_3d/            单视角 3D box 优化结果和展示版视频
```

如果你只想先小样本验证，可以复制 `configs/original_2dbox_full_pipeline_gpu.yaml`，把：

```yaml
runtime:
  max_frames: 50
```

设成较小帧数。

### 5.2 第一步：三视角单视角 3D 优化

```bash
bash scripts/start_rebuild_repair_params_full_wsl.sh
```

看日志：

```bash
tail -f outputs/rebuild_three_views_repair_params_full_wsl_v1/logs/run.log
```

主要输出：

```text
outputs/rebuild_three_views_repair_params_full_wsl_v1/frame_3d_boxes_world_track_joint.csv
outputs/rebuild_three_views_repair_params_full_wsl_v1/frame_loss_diagnostics.csv
outputs/rebuild_three_views_repair_params_full_wsl_v1/*_overlay.mp4
```

### 5.3 第二步：LoMa 跨视角 2D track 关联

```bash
bash scripts/start_loma_repair_params_tracks_wsl.sh
```

看日志：

```bash
tail -f outputs/loma_global_2d_repair_params_tracks_v1/logs/run.log
```

主要输出：

```text
outputs/loma_global_2d_repair_params_tracks_v1/global_track_assignments.csv
outputs/loma_global_2d_repair_params_tracks_v1/accepted_edges.csv
outputs/loma_global_2d_repair_params_tracks_v1/candidate_diagnostics.csv
```

如果你已经手动准备好了 `global_track_assignments.csv`，可以跳过这一步，但需要把路径写进：

```text
configs/multiview_joint_loma_repair_from_singleview_v1.yaml
```

### 5.4 第三步：跨视角联合后优化

```bash
bash scripts/start_multiview_joint_loma_repair_wsl.sh
```

看日志：

```bash
tail -f outputs/multiview_joint_loma_repair_from_singleview_v1/logs/run.log
```

主要输出：

```text
outputs/multiview_joint_loma_repair_from_singleview_v1/frame_3d_boxes_multiview_joint_from_singleview.csv
outputs/multiview_joint_loma_repair_from_singleview_v1/frame_loss_diagnostics.csv
outputs/multiview_joint_loma_repair_from_singleview_v1/global_track_optimization_summary.csv
outputs/multiview_joint_loma_repair_from_singleview_v1/*_overlay.mp4
```

## 6. 单独渲染某几个 global track

用于检查跨视角同 ID 是否合理：

```bash
export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
python scripts/render_multiview_joint_track_panels.py \
  --config outputs/multiview_joint_loma_repair_from_singleview_v1/resolved_config.yaml \
  --diagnostics outputs/multiview_joint_loma_repair_from_singleview_v1/frame_loss_diagnostics.csv \
  --output-dir outputs/multiview_joint_loma_repair_from_singleview_v1/joint_track_videos \
  --global-track-ids 126,135,145,165,170 \
  --panel-width 640 \
  --panel-height 360 \
  --bev-width 520 \
  --fps 10
```

输出示例：

```text
outputs/multiview_joint_loma_repair_from_singleview_v1/joint_track_videos/global_126_rear_right_rear_bev.mp4
```

## 7. 当前重要假设

- 3D box size 顺序固定为 `[length, width, height]`。
- yaw 当前不优化，默认与 rear camera / 自车运动方向平行。
- box upright，只允许 yaw，不允许 pitch/roll。
- 当前截断判断包含 track-level 启发式规则，这部分很重要但仍有待商榷。
- 当前主线关闭 `depth_safety` / `center_depth_safety`，使用 `ego_box_safety` 处理近距离互穿。
- `bbox_fit` 和 `top_bottom_edges` 主要用于远距离；近距离强贴 2D box 容易造成 box 变小并拉近。

## 8. 上传 GitHub 前检查

确认不要提交：

```text
data/
preprocessed/
outputs/
checkpoints/
models/
external/
third_party/
*.pt / *.pth / *.ckpt / *.safetensors
```

本包已经提供 `.gitignore`，但上传前建议再检查：

```bash
git status --short
```

如果需要创建新仓库：

```bash
git init
git add README.md code configs documentation scripts environment-*.yml requirements-*.txt .gitignore
git commit -m "initial monocular 3d pipeline"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```
