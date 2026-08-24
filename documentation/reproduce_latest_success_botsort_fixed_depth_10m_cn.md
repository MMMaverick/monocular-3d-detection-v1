# 最新成功复现版：BoT-SORT + 修正 DA3 depth + label fix + 10m 上下贴边规则

本说明对应本地已经跑通的实验：

```text
D:\mono-detect\outputs\rebuild_three_views_botsort_repro_local_3d_fixed_depth_label_fix_10m_v1
```

本地结果摘要：

```text
frames=5963
tracks=145
diagnostic_rows=5963
rear / left_rear / right_rear 三路视频均已输出
```

## 1. 这版成功复现的关键设置

- 2D tracking：`robust_track_2d.py`
  - DINOv2 crop embedding
  - BoxMOT BoT-SORT
  - CMC/ECC
- depth 初始化：DA3 metric bbox median
  - 已修正 bbox 到 depth map 的图像尺寸映射 bug
  - depth-track CSV 中包含：
    - `depth_source_image_width`
    - `depth_source_image_height`
    - `depth_source_image_size_source`
- label 读取：已修复 `raw_label / class_name` 没被 3D 优化读取的问题
- 3D 优化：
  - yaw 固定，不优化
  - size 每个 track 共享
  - center 每帧优化，轨迹联合优化
  - `top_bottom_edges.activate_untruncated: false`
  - 即上下贴边保持 10m 规则：只在 `initial_distance > 10m` 时触发
- 可视化：展示版
  - mask
  - 2D box
  - 3D box
  - BEV
  - 不显示 loss 面板和贴边辅助线

## 2. 已随仓库固化的输入

为了避免新机器重新跑 tracking/depth 后 id 或 depth 绑定不一致，已经提交本次实验使用的两类 track CSV：

```text
preprocessed/tracks/robust_botsort_hybrid_depth_v1/
  rear_camera/tracks.csv
  left_rear_camera/tracks.csv
  right_rear_camera/tracks.csv
```

这是纯 BoT-SORT track 输出。

```text
preprocessed/tracks/robust_botsort_fixed_da3_depth_v1/
  rear_camera/tracks.csv
  left_rear_camera/tracks.csv
  right_rear_camera/tracks.csv
```

这是本次 3D 成功复现实验实际使用的、已经绑定修正 DA3 depth 的 track CSV。

三路 depth-track 行数：

```text
rear_camera       3882
left_rear_camera  2335
right_rear_camera 1906
```

## 3. 新机器目录要求

为了让已提交的 CSV 中的相对路径能直接解析，推荐把原始场景数据放到仓库的 `data/` 下：

```text
monocular-3d-detection-v1/
  data/
    camera/
      rear_camera/*.jpg
      left_rear_camera/*.jpg
      right_rear_camera/*.jpg
    calib/
      rear_camera/rear_camera-intrinsic.json
      rear_camera/rear_camera-to-car_center-extrinsic.json
      left_rear_camera/...
      right_rear_camera/...
    format_output/
      annotations/
        NV/*.json
```

权重仍然不放进 GitHub，需要按 README 下载：

```text
checkpoints/sam_vit_h_4b8939.pth
third_party/Depth-Anything-3/checkpoints/da3metric-large
~/.cache/torch/hub/facebookresearch_dinov2_main
```

## 4. 复现路径 A：从原始 2D box 重新跑完整流程

这是最完整路径，会重新生成 SAM mask、DA3 depth、BoT-SORT track、depth-track、mask 对齐和 3D 优化：

```bash
cd /path/to/monocular-3d-detection-v1
export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
conda activate mono-detect-original-2dbox-full-cpu

bash scripts/run_original_2dbox_full_pipeline.sh \
  "$(pwd)/data" \
  configs/original_2dbox_full_pipeline_gpu.yaml
```

注意：这条路径会重新跑模型，结果应该尽量接近，但不保证和本地冻结实验逐字节一致。

## 5. 复现路径 B：使用已提交 depth-track CSV 复现 3D 阶段

这条路径固定使用本次成功实验的 depth-track CSV。先生成 raw SAM masks：

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

然后把 raw SAM masks 对齐到已提交的 depth-track CSV：

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

输出目录：

```text
outputs/reproduce_latest_success_botsort_fixed_depth_10m_v1
```

## 6. 本次冻结配置文件

```text
configs/reproduce_latest_success_botsort_fixed_depth_10m_v1.yaml
configs/ensure_masks_for_reproduce_latest_success_v1.yaml
configs/robust_botsort_2d_rear_views_repro_local_v1.yaml
configs/rebuild_three_views_botsort_repro_local_3d_fixed_depth_v1.yaml
configs/common/rebuild_single_view_ensured_masks_common.yaml
```

其中最重要的是：

```yaml
observations:
  top_bottom_edges:
    activate_untruncated: false
```

这保证当前成功复现版仍然使用 10m 上下贴边触发规则。
