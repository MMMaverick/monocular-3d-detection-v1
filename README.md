# monocular-3d-detection-v1

Monocular / surround-view 3D box reconstruction experiments for rear, left-rear, and right-rear cameras.

> 当前推荐入口：请先看
> [`documentation/current_multiview_3d_pipeline_cn.md`](documentation/current_multiview_3d_pipeline_cn.md)。
> 该文档记录目前正在使用的“单视角优化 → LoMa 跨视角匹配 → 跨视角联合后优化”主线流程。
>
> 最新本地成功复现版请看
> [`documentation/reproduce_latest_success_botsort_fixed_depth_10m_cn.md`](documentation/reproduce_latest_success_botsort_fixed_depth_10m_cn.md)。
> 该文档冻结了 BoT-SORT + 修正 DA3 depth + label fix + 10m 上下贴边规则这版实验的配置和输入 CSV。

The current main pipeline is a two-stage offline baseline:

1. associate single-view tracks across `rear`, `left_rear`, and `right_rear`;
2. jointly optimize each global track in world coordinates, sharing one 3D size and one world trajectory across all matched camera observations.

Yaw is currently fixed to the rear-camera / ego-motion reference direction. The box is upright; pitch and roll are not optimized.

## 新机器优先复现这版：track 也重新生成

如果目标是确认“本地最新成功复现版”能不能在新机器恢复，优先跑完整流程，不要直接拿仓库里提交的 track 当最终输入。

也就是说，新机器上默认应该重新生成：

```text
annotation JSON
  ↓
SAM mask
  ↓
DA3 metric depth
  ↓
DINOv2 + BoT-SORT 重新追踪
  ↓
把 DA3 depth 绑定到新生成的 tracks.csv
  ↓
确保每个 tracked 2D box 都有 mask
  ↓
3D box 优化
```

仓库里提交的两套 track CSV 只用于对照和排查：

```text
preprocessed/tracks/robust_botsort_hybrid_depth_v1/*/tracks.csv
```

这是纯 BoT-SORT 2D track 对照文件。

```text
preprocessed/tracks/robust_botsort_fixed_da3_depth_v1/*/tracks.csv
```

这是本地成功实验曾经使用过的 fixed DA3 depth track 对照文件，包含：

```text
depth_median / x_cam / y_cam / z_cam / depth_source_image_width / depth_source_image_height
```

但如果你要验证“新机器完整流程是否能恢复”，应该使用下面 A 路径，让 track 在新机器重新生成。

### A. 从原始数据完整重跑，也重新生成 track

这条路径会重新生成 SAM mask、DA3 depth、BoT-SORT track、depth-track、mask 对齐和 3D 优化。适合检查整套流程是否能在新机器跑通：

```bash
cd /path/to/monocular-3d-detection-v1
export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
conda activate mono-detect-original-2dbox-full-cpu

bash scripts/run_original_2dbox_full_pipeline.sh \
  /path/to/scene_data \
  configs/original_2dbox_full_pipeline_gpu.yaml
```

这里的 `/path/to/scene_data` 需要包含：

```text
camera/
calib/
format_output/annotations/NV/
```

这条是当前最重要的“完整流程复现”。它会在新机器输出新的 track：

```text
outputs/original_2dbox_full_gpu_v1/tracking/robust_botsort_tracks/
  rear_camera/tracks.csv
  left_rear_camera/tracks.csv
  right_rear_camera/tracks.csv
```

然后继续生成带 depth 的 track：

```text
outputs/original_2dbox_full_gpu_v1/tracks_with_depth/
  rear_camera/tracks.csv
  left_rear_camera/tracks.csv
  right_rear_camera/tracks.csv
```

3D 优化默认使用 `tracks_with_depth` 里的这三份新生成 CSV。

由于 SAM / DA3 / DINOv2 / BoT-SORT 都会重新推理，结果应接近本地版本，但不承诺逐字节一致。如果效果明显不对，优先检查新生成的 track 数量、depth 分布和 mask 对齐行数。

### B. 已提交 fixed-depth tracks 只作为 debug 对照

如果怀疑新机器重新 tracking 结果不稳定，可以临时使用仓库里提交的 fixed-depth tracks 做 debug。这个路径不是默认完整流程，只用于隔离问题：判断问题来自 tracking/depth 生成，还是来自 3D 优化代码。

先把原始场景数据放到仓库的 `data/` 下：

```text
monocular-3d-detection-v1/
  data/
    camera/
    calib/
    format_output/annotations/NV/
```

然后先生成 raw SAM mask：

```bash
python code/examples/sam_segment_gt2d_boxes.py \
  --annotations data/format_output/annotations/NV \
  --camera-root data/camera \
  --cameras rear_camera,left_rear_camera,right_rear_camera \
  --output outputs/robust_botsort_2d_rear_views_repro_local_v1/masks/raw_sam \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --box-scale 1.5 \
  --positive-points 5 \
  --max-frames -1 \
  --device cuda \
  --fps 10 \
  --mask-format png \
  --min-gt2d-score 0.0
```

再把 mask 对齐到已提交的 fixed-depth tracks：

```bash
python -m rebuild_3d_box_optimizer.ensure_masks_for_all_tracks \
  --config configs/ensure_masks_for_reproduce_latest_success_v1.yaml \
  --output-dir outputs/robust_botsort_2d_rear_views_repro_local_v1/masks_ensured_for_3d \
  --min-iou 0.30
```

最后跑冻结 3D 配置：

```bash
python -m rebuild_3d_box_optimizer.run \
  --config configs/reproduce_latest_success_botsort_fixed_depth_10m_v1.yaml
```

输出在：

```text
outputs/reproduce_latest_success_botsort_fixed_depth_10m_v1
```

这条路径固定了本地成功实验的 depth-track CSV 和 3D 参数；重新生成 mask。如果 A 路径不行但 B 路径正常，说明问题主要在新机器重新生成的 tracking/depth；如果 B 路径也不正常，说明问题更可能在 mask 生成/对齐、环境版本或 3D 优化代码。

如果这条仍明显不对，优先对比：

```text
outputs/robust_botsort_2d_rear_views_repro_local_v1/masks_ensured_for_3d/*/gt2d_sam_masks_ensured_cropped.csv
```

本地成功版 mask 对齐行数应为：

```text
rear_camera       3246
left_rear_camera  1801
right_rear_camera 1190
```

### C. 最新成功版关键参数

冻结配置：

```text
configs/reproduce_latest_success_botsort_fixed_depth_10m_v1.yaml
```

关键参数：

```yaml
solver:
  device: cuda
  learning_rate: 0.1
  max_iterations: 3000
  track_parallel:
    workers: 4

observations:
  top_bottom_edges:
    activate_untruncated: false   # 保持 10m 规则
  ground_plane:
    camera_height_m: 0.8
    camera_height_m_by_view:
      rear: 0.6
```

本地成功版规模：

```text
有效优化轨迹：145
rear        66
left_rear   26
right_rear  53
```

完整重跑时，新机器会重新生成 track，数量可能和本地略有差异。已提交的 fixed-depth tracks 里包含短轨迹，完整数量更多；3D 优化时通过 `min_track_frames: 3` 筛成上面的 145 条。

## 原始三路后视数据：新机器部署入口

本节针对项目最初的三路后视数据，不针对 Waymo。这里的三路相机是：

```text
rear_camera
left_rear_camera
right_rear_camera
```

当前原始数据端到端流程默认不运行 YOLO 或其他 detector。输入假设已经有逐帧 2D box annotation：

```text
<scene_data>/format_output/annotations/NV/*.json
```

如果新数据只有图片、没有 2D box，需要先在项目外部跑检测器，并导出等价的逐帧 2D box CSV。项目内部负责从已有 2D box 生成 mask、depth、稳定 2D track，以及后续 3D box 优化。

### 1. 目录结构

新机器上建议把仓库和数据保持成下面的结构：

```text
monocular-3d-detection-v1/
  code/
  configs/
  scripts/
  checkpoints/
    sam_vit_h_4b8939.pth
  third_party/
    segment_anything/
    Depth-Anything-3/
      checkpoints/
        da3metric-large/
  external/
    LoMa/

<scene_data>/
  camera/
    rear_camera/*.jpg
    left_rear_camera/*.jpg
    right_rear_camera/*.jpg
  calib/
    rear_camera/
      rear_camera-intrinsic.json
      rear_camera-to-car_center-extrinsic.json
    left_rear_camera/
      left_rear_camera-intrinsic.json
      left_rear_camera-to-car_center-extrinsic.json
    right_rear_camera/
      right_rear_camera-intrinsic.json
      right_rear_camera-to-car_center-extrinsic.json
  format_output/
    annotations/
      NV/*.json
```

外参文件名里的 `camera-to-car_center` 表示：

```text
P_car = T_camera_to_car @ P_camera
```

### 2. 环境安装

推荐使用 WSL Ubuntu + Miniforge/Conda。CPU 环境可直接用仓库里的文件创建：

```bash
cd /path/to/monocular-3d-detection-v1
conda env create -f environment-original-2dbox-full-cpu.yml
conda activate mono-detect-original-2dbox-full-cpu
```

如果要跑全量 SAM / DA3 / 3D 优化，建议使用 GPU。当前根目录没有单独维护 GPU yml，可以先创建 CPU 环境，再替换安装 CUDA 版 PyTorch。例如 CUDA 12.8：

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall --no-deps numpy==1.26.4
```

验证 CUDA：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

项目运行时通常需要：

```bash
export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
```

### 3. 外部代码和权重

#### 3.1 SAM

用于从 2D box 生成前景 mask。

```bash
cd /path/to/monocular-3d-detection-v1
mkdir -p third_party checkpoints
git clone https://github.com/facebookresearch/segment-anything.git third_party/segment_anything
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

默认配置读取：

```text
checkpoints/sam_vit_h_4b8939.pth
```

#### 3.2 Depth Anything 3 Metric

用于为每个相机生成 metric depth，再绑定到 2D track 作为 3D 初始化深度来源。

```bash
cd /path/to/monocular-3d-detection-v1
git clone https://github.com/DepthAnything/Depth-Anything-3.git third_party/Depth-Anything-3
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

默认配置读取：

```text
third_party/Depth-Anything-3/checkpoints/da3metric-large
```

#### 3.3 LoMa

用于后续跨视角 2D track 匹配，把不同相机里的同一目标合成 global track。

```bash
cd /path/to/monocular-3d-detection-v1
mkdir -p external
git clone https://github.com/davnords/LoMa.git external/LoMa
```

如果 LoMa 需要额外权重，请按 LoMa 官方 README 放到其默认路径，或修改：

```text
configs/loma_global_2d_repair_params_tracks_v1.yaml
```

### 4. 2D box 到 2D track 的实际流程

原始数据流程的“检测到追踪”更准确地说是“已有 2D annotation 到稳定 2D track”：

```text
annotation JSON
  ↓
scripts/export_original_2dbox_csv.py
  ↓
outputs/original_2dbox_full_*/tracking_input/<camera>/tracks.csv
  # 注意：这里虽然叫 tracks.csv，但此时仍是逐帧 2D box，不是稳定 tracking
  ↓
SAM mask CSV / cropped boxes
  ↓
python -m rebuild_3d_box_optimizer.robust_track_2d
  ↓
outputs/original_2dbox_full_*/tracking/robust_botsort_tracks/<camera>/tracks.csv
  # 这里才是重新追踪后的 2D track，包含稳定 track_id
```

当前本地复现实验使用的不是 SORT-lite，而是 appearance-assisted BoT-SORT：

```text
2D box / SAM CSV
  ↓
DINOv2 crop embedding
  ↓
BoxMOT BoT-SORT + CMC(ECC)
  ↓
稳定单视角 tracks.csv
```

这对应本地复现配置：

```text
configs/robust_botsort_2d_rear_views_repro_local_v1.yaml
```

以及历史复现输出：

```text
outputs/robust_botsort_2d_rear_views_repro_local_v1/
  rear_camera/tracks.csv
  left_rear_camera/tracks.csv
  right_rear_camera/tracks.csv
```

注意：SORT-lite (`retrack_sort_2d`) 仍然保留在仓库中，但它只是轻量 fallback / debug 方案，不是当前推荐复现主线。之前复现效果更接近历史实验的是 BoT-SORT 这一版。

如果手动执行导出：

```bash
python scripts/export_original_2dbox_csv.py \
  --annotations /path/to/scene_data/format_output/annotations/NV \
  --camera-root /path/to/scene_data/camera \
  --cameras rear_camera,left_rear_camera,right_rear_camera \
  --output-root outputs/original_2dbox_full_gpu_v1/tracking_input \
  --max-frames -1
```

推荐 BoT-SORT 追踪入口：

```bash
python -m rebuild_3d_box_optimizer.robust_track_2d \
  --config outputs/original_2dbox_full_gpu_v1/configs/robust_botsort_2d.yaml
```

推荐 BoT-SORT 参数：

```yaml
tracking:
  method: robust_botsort
  appearance:
    type: dinov2
    model: dinov2_vitl14
    device: cuda
    batch_size: 24
    crop_size: 224
    box_padding: 0.08
  botsort:
    frame_rate: 10
    track_high_thresh: 0.45
    track_low_thresh: 0.08
    new_track_thresh: 0.50
    track_buffer: 60
    match_thresh: 0.80
    proximity_thresh: 0.70
    appearance_thresh: 0.35
    second_match_thresh: 0.50
    unconfirmed_match_thresh: 0.70
    min_hits: 1
    max_obs: 100
    fuse_first_associate: true
    use_cmc: true
    cmc_method: ecc
```

说明：

- `appearance.type: dinov2`：用 DINOv2 对每个 2D box crop 提外观特征；
- `track_buffer: 60`：目标短暂遮挡或漏检时可保活的帧数；
- `proximity_thresh` / `appearance_thresh`：BoT-SORT 第一阶段匹配的空间和外观门限；
- `use_cmc: true`、`cmc_method: ecc`：启用相机运动补偿；
- 当前 `robust_track_2d` 是按 canonical class 分组跑 BoT-SORT，然后给 `track_id` 加类别段偏移，例如 `class_id * 1_000_000 + local_track_id`；
- 追踪后仍会对每条 track 做多数帧类别投票，写入 `track_majority_label` / `track_majority_raw_label`，用于后续 3D size prior。

追踪输出会保留原始 detection 行，并新增/覆盖：

```text
track_id                  新的稳定 2D track id
source_track_id           原始 annotation/detection 中的 id，如果有
raw_label                 当前帧原始类别
raw_canonical_label       当前帧映射后的类别
track_majority_label      当前 track 多数帧类别
track_majority_raw_label  当前 track 多数帧原始类别
label / gt_label / prompt 统一设置为 track_majority_label
```

单独渲染 2D tracking 视频：

```bash
python -m rebuild_3d_box_optimizer.render_sort2d_tracks \
  --config outputs/original_2dbox_full_gpu_v1/configs/render_robust_botsort_tracks.yaml \
  --output-dir outputs/original_2dbox_full_gpu_v1/tracking/robust_botsort_track_videos \
  --fps 10 \
  --thickness 1
```

输出：

```text
outputs/original_2dbox_full_gpu_v1/tracking/robust_botsort_track_videos/
  rear_camera_sort2d_tracks.mp4
  left_rear_camera_sort2d_tracks.mp4
  right_rear_camera_sort2d_tracks.mp4
```

### 5. 一键跑原始 2D box 端到端流程

全量 GPU 入口：

```bash
cd /path/to/monocular-3d-detection-v1
export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
conda activate mono-detect-original-2dbox-full-cpu

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

这一步依次执行：

```text
1. export_original_2dbox_csv.py
   从 annotation JSON 导出每个相机的逐帧 2D box。

2. sam_segment_gt2d_boxes.py
   用 SAM 对每个 2D box 生成前景 mask。

3. da3_metric_rear_depth_export.py
   用 DA3 Metric 为每个相机导出 metric depth。

4. rebuild_3d_box_optimizer.robust_track_2d
   基于 SAM/2D box CSV、DINOv2 外观特征和 BoT-SORT 重新做稳定 2D tracking。

5. attach_da3_depth_to_tracks.py
   把 DA3 depth 绑定到 track CSV，作为 3D 初始化深度来源。

6. ensure_masks_for_all_tracks
   确保每个 tracked 2D box 都能找到对应 mask；必要时补齐 cropped mask。

7. rebuild_3d_box_optimizer.run
   对每个 track 做单视角 3D box 优化，并输出 CSV / 视频。
```

主要输出：

```text
outputs/original_2dbox_full_gpu_v1/
  tracking_input/          原始 2D box 导出的逐帧 CSV
  masks/raw_sam/           SAM 原始 mask
  depth/                   DA3 metric depth
  tracking/robust_botsort_tracks/  BoT-SORT 2D tracking 结果
  tracking/sort2d_tracks/          fallback SORT-lite 结果，仅在 tracking.method != robust_botsort 时使用
  tracks_with_depth/       绑定 depth 后的 track CSV
  masks/ensured/           与 track 对齐后的 mask
  optimized_3d/            单视角 3D box 优化结果和视频
```

如果只想在新机器上先快速冒烟测试，可以复制配置并设置：

```yaml
runtime:
  max_frames: 50
```

或者直接使用 smoke 配置：

```text
configs/original_2dbox_full_pipeline_smoke.yaml
configs/original_2dbox_full_pipeline_smoke_gpu.yaml
```

### 6. 端到端之后的跨视角主线

原始 2D box 端到端流程先得到每个视角内的单视角 3D 优化结果。之后可继续跑当前三视角主线：

```bash
bash scripts/start_loma_repair_params_tracks_wsl.sh
bash scripts/start_multiview_joint_loma_repair_wsl.sh
```

LoMa 阶段输出：

```text
outputs/loma_global_2d_repair_params_tracks_v1/
  global_track_assignments.csv
  accepted_edges.csv
  candidate_diagnostics.csv
```

跨视角联合优化输出：

```text
outputs/multiview_joint_loma_repair_from_singleview_v1/
  frame_3d_boxes_multiview_joint_from_singleview.csv
  frame_loss_diagnostics.csv
  global_track_optimization_summary.csv
  *_overlay.mp4
```

### 7. 依赖关系小结

推荐 2D tracking 主线依赖：

```text
torch / torchvision
boxmot
opencv-python
numpy
PyYAML
DINOv2 本地 torch hub cache
```

BoT-SORT 主线需要提前缓存 DINOv2。推荐在有网络的机器上执行一次：

```bash
python - <<'PY'
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
print(torch.hub.get_dir())
PY
```

之后 `robust_track_2d.py` 会从下面的本地缓存读取：

```text
~/.cache/torch/hub/facebookresearch_dinov2_main
```

如果新机器不能访问网络，可以把这整个 cache 文件夹手动拷过去。快速 smoke 或无 DINO 环境可以在配置中临时使用：

```yaml
tracking:
  appearance:
    type: hsv_histogram
```

fallback SORT-lite 只依赖：

```text
numpy
scipy        # Hungarian assignment
PyYAML
opencv-python # 仅渲染 tracking 视频需要
```

完整原始三路后视端到端流程还需要：

```text
torch / torchvision
segment-anything + sam_vit_h_4b8939.pth
Depth-Anything-3 + da3metric-large
LoMa
opencv-python
Pillow
pandas
moviepy
pycolmap / evo / e3nn 等 environment-original-2dbox-full-cpu.yml 中列出的包
```

注意：YOLO / ultralytics 是 Waymo 检测实验使用的依赖；原始三路后视数据流程默认不使用 YOLO。

## Run the full multiview joint optimization

PowerShell:

```powershell
Set-Location D:\mono-detect
$env:PYTHONPATH="D:\mono-detect\code"
C:\ProgramData\miniforge3\envs\dvgt\python.exe -m rebuild_3d_box_optimizer.run_multiview_joint --config configs\multiview_joint_track_optimization_v1.yaml
```

Watch progress in another PowerShell window:

```powershell
Set-Location D:\mono-detect
Get-Content outputs\multiview_joint_track_optimization_v1\progress.log -Wait -Tail 40
```

The full config uses convergence mode:

- Adam on CUDA when available
- float32
- max safety iterations: 3000
- minimum iterations: 500
- patience: 250
- mask foreground point containment enabled
- support-edge, bbox-fit, depth-safety, mask-containment, mask-oversize, and temporal smoothness losses
- videos enabled with 3D box, 2D box, support edges, mask pixels, and per-frame loss panel

## Main outputs

`outputs/multiview_joint_track_optimization_v1/`

- `global_track_assignments.csv` — source view/track to global track id
- `global_track_components.csv` — tracks inside each global component, with conflict flags
- `match_edges.csv` — accepted cross-camera matching edges
- `candidate_match_diagnostics.csv` — top candidate matches and rejection reasons
- `frame_3d_boxes_multiview_joint.csv` — optimized per-frame 3D boxes
- `frame_loss_diagnostics.csv` — per-frame loss breakdown used by videos
- `global_track_optimization_summary.csv` — per-global-track optimization summary
- `*_overlay.mp4` — visualization videos
- `progress.log` — live progress log

## Useful configs

- `configs/multiview_joint_track_optimization_v1.yaml` — full three-view association + joint optimization
- `configs/multiview_track_association_v1.yaml` — association-only baseline
- `configs/multiview_joint_track_optimization_smoke.yaml` — quick smoke test

## Notes

Large local data, preprocessing products, and generated outputs are intentionally ignored by git. Keep them in the workspace when running locally.

## 配置工作流

通用优化参数放在 `configs/common/`。实验配置文件应尽量保持很小，通常只负责：

- 选择本次要跑的 view / track；
- 覆盖本次实验专用输入；
- 指定输出目录。

每次实验运行时，输出目录都会保存一份完整展开后的：

```text
experiment_config_snapshot.yaml
```

可以把它看作“本次实验使用的通用配置副本”。之后即使继续修改 `configs/common/`，已经完成实验的快照也不会变化，方便复现和对比。

新的调参优先修改通用配置文件，然后启动一个新实验，让新实验目录自动保存新的配置快照。
