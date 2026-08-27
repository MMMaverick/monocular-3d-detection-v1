# 多视角联合优化与 3D 动静状态交付文档

本文档描述当前可交付流程：

```text
单视角 3D 优化结果
  + LoMa 跨视角同车匹配结果
  ↓
多视角联合后优化
  ↓
3D 轨迹鲁棒平滑
  ↓
每个 track 输出 moving/static/uncertain 状态
  ↓
输出 CSV 与视频可视化
```

这套流程不依赖光流。光流实验已经验证对当前结果没有明显收益，且会引入额外误差源。

## 1. 目录与输入

默认以后视三路为例：

```text
rear
left_rear
right_rear
```

需要准备以下结果。

### 1.1 单视角 3D 优化结果

默认读取：

```text
outputs/rebuild_three_views_repair_params_full_wsl_v1/
  frame_3d_boxes_world_track_joint.csv
  frame_loss_diagnostics.csv
```

其中：

- `frame_3d_boxes_world_track_joint.csv`：每帧 3D box 优化结果；
- `frame_loss_diagnostics.csv`：每帧可视化/诊断结果，包含 2D box、mask 路径、3D 投影角点等。

### 1.2 跨视角同车匹配结果

默认读取：

```text
outputs/loma_global_2d_repair_params_tracks_v1/
  global_track_assignments.csv
  accepted_edges.csv
```

这些文件由 LoMa 跨视角匹配脚本生成：

```text
code/rebuild_3d_box_optimizer/associate_2d_tracks_loma.py
```

### 1.3 annotation / 标定

动静状态如果需要使用 world 坐标，建议提供 annotation JSON：

```text
data/format_output/annotations/NV/*.json
```

其中需要包含：

- ego pose：`ego2global_rotation`、`ego2global_translation`
- camera extrinsic

如果 3D 优化结果 CSV 已经包含 `world_x/world_y/world_z`，后处理可以直接使用；如果没有，则会从 `cx/cy/cz` 加 annotation 外参转到 world 坐标。

## 2. 第一步：生成跨视角同车匹配

配置文件：

```text
configs/loma_global_2d_repair_params_tracks_v1.yaml
```

运行：

```powershell
cd D:\mono-detect
$env:PYTHONPATH="D:\mono-detect\code"
C:\ProgramData\miniforge3\envs\dvgt\python.exe -m rebuild_3d_box_optimizer.associate_2d_tracks_loma `
  --config D:\mono-detect\configs\loma_global_2d_repair_params_tracks_v1.yaml
```

输出：

```text
outputs/loma_global_2d_repair_params_tracks_v1/
  global_track_assignments.csv
  accepted_edges.csv
  candidate_diagnostics.csv
  match_visualizations/
```

最重要的是：

- `global_track_assignments.csv`
  - 每行表示一个 source track 属于哪个 `global_track_id`；
  - 后续联合优化严格使用这个文件，不再根据单视角 3D 距离重新猜匹配关系。

## 3. 第二步：多视角联合后优化

配置文件：

```text
configs/multiview_joint_loma_repair_from_singleview_v1.yaml
```

运行：

```powershell
cd D:\mono-detect
$env:PYTHONPATH="D:\mono-detect\code"
C:\ProgramData\miniforge3\envs\dvgt\python.exe -m rebuild_3d_box_optimizer.run_multiview_joint_from_singleview `
  --config D:\mono-detect\configs\multiview_joint_loma_repair_from_singleview_v1.yaml
```

输出：

```text
outputs/multiview_joint_loma_repair_from_singleview_v1/
  frame_3d_boxes_multiview_joint_from_singleview.csv
  frame_loss_diagnostics.csv
  global_track_optimization_summary.csv
  global_track_assignments.csv
  global_track_components.csv
  match_edges.csv
  candidate_match_diagnostics.csv
  progress.log
  *_overlay.mp4
```

### 3.1 只跑某些 global track

修改配置：

```yaml
multiview_joint:
  only_global_track_ids: [116]
```

或者按 source track 筛选：

```yaml
multiview_joint:
  only_source_tracks: ["rear:5000022", "right_rear:4000000"]
```

然后重新运行同一条命令。

### 3.2 重要参数

```yaml
multiview_joint:
  enabled: true
  multi_view_only: true
```

- `enabled: true`
  - 使用 LoMa/fixed assignment 模式；
  - 不再根据 3D 距离重新做跨视角关联。
- `multi_view_only: true`
  - 只优化至少被两个视角匹配到的 global track。

初始化融合：

```yaml
initialization:
  timestamp_merge_tolerance_ms: 20.0
  view_weights:
    rear: 1.0
    left_rear: 0.8
    right_rear: 0.8
  truncation_weight:
    none: 1.0
    single_side: 0.5
    multi_side: 0.25
  soft_outlier_m: 3.0
  hard_outlier_m: 8.0
```

含义：

- 同一时刻不同相机的观测会合成一个 world center；
- rear 视角默认权重略高；
- 截断 bbox 的初始化权重降低；
- 多视角初始 3D 中心差异太大时会鲁棒降权，避免一个坏视角把整条 track 拉飞。

## 4. 第三步：3D 轨迹动静状态后处理

入口脚本：

```text
scripts/robust_world_motion_visualize.py
```

以后视三路单视角优化结果为例：

```powershell
cd D:\mono-detect
$env:PYTHONPATH="D:\mono-detect\code"
C:\ProgramData\miniforge3\envs\dvgt\python.exe D:\mono-detect\scripts\robust_world_motion_visualize.py `
  --world-csv D:\mono-detect\outputs\final_three_views_fixed_v1\frame_3d_boxes_world_track_joint.csv `
  --diagnostics-csv D:\mono-detect\outputs\final_three_views_fixed_v1\frame_loss_diagnostics.csv `
  --annotations D:\vggt-omega\data\format_output\annotations\NV `
  --views rear left_rear right_rear `
  --output-dir D:\mono-detect\outputs\rear_3d_motion_state_delivery_v1 `
  --fps 10 `
  --speed-window-s 1.0 `
  --moving-threshold-mps 2.0 `
  --motion-policy ratio `
  --moving-ratio-threshold 0.30 `
  --smooth-window 7 `
  --hampel-window 5 `
  --hampel-sigma 3.0 `
  --max-residual-step-m 3.0 `
  --min-track-frames 10 `
  --min-track-duration-s 1.0 `
  --repo-root D:\mono-detect
```

如果要对多视角联合优化结果做动静状态判断，替换输入：

```powershell
--world-csv D:\mono-detect\outputs\multiview_joint_loma_repair_from_singleview_v1\frame_3d_boxes_multiview_joint_from_singleview.csv `
--diagnostics-csv D:\mono-detect\outputs\multiview_joint_loma_repair_from_singleview_v1\frame_loss_diagnostics.csv `
--output-dir D:\mono-detect\outputs\multiview_joint_motion_state_v1
```

## 5. 动静状态判定

默认推荐：

```text
1 秒窗口计算 world 平面速度。
speed > 2m/s 的时间片记为 moving。
moving 时间片比例 >= 0.30 的 track 判为 moving。
否则判为 static。
如果轨迹太短或残余跳变过大，则判为 uncertain。
```

关键参数：

- `speed-window-s`
  - 速度计算窗口，默认 1 秒；
- `moving-threshold-mps`
  - 单个速度片段超过该阈值算“这一段在动”；
- `motion-policy`
  - `ratio`：按动态片段比例判断，当前推荐；
  - `median`：按中位速度判断；
  - `any`：只要动过就算动态，不推荐直接用于 noisy monocular 3D；
  - `majority`：帧级 moving 数量不少于 static 数量则判动态；
- `moving-ratio-threshold`
  - `ratio` 策略下的动态片段比例阈值；
- `smooth-window`
  - world 坐标平滑窗口；
- `hampel-window` / `hampel-sigma`
  - 局部异常点替换；
- `max-residual-step-m`
  - 平滑后相邻帧仍然跳太大时，track 会被认为不可靠。

## 6. 输出字段

`track_world_motion_summary_smoothed.csv` 每个 track 一行，核心字段：

```text
view
track_id
class
num_frames
first_frame
last_frame
duration_s
motion_state
motion_state_reason
moving_ratio
median_speed_1s_mps
p90_speed_1s_mps
max_speed_1s_mps
distance_to_ego_min_m
distance_to_ego_median_m
distance_to_ego_max_m
distance_to_ego_first_m
distance_to_ego_last_m
depth_min_m
depth_median_m
depth_max_m
world_displacement_xy_m
world_path_xy_m
max_smoothed_step_m
jitter_ratio
outlier_frames
usable_for_motion
```

字段解释：

- `motion_state`
  - 最终每个 track 的状态；
- `moving_ratio`
  - 有多少比例的有效速度片段超过动态阈值；
- `distance_to_ego_*`
  - 该目标相对自车/当前相机的距离统计，辅助判断远距离目标是否值得关注；
- `depth_*`
  - camera z 深度统计；
- `world_displacement_xy_m`
  - 平滑后起点到终点的 world 平面位移；
- `world_path_xy_m`
  - 平滑后轨迹路径长度；
- `jitter_ratio`
  - 路径长度 / 起终点位移，越大说明轨迹越抖；
- `outlier_frames`
  - Hampel filter 替换过的异常帧数量；
- `usable_for_motion`
  - 该轨迹是否满足最小帧数、最小时长、最大残余跳变约束。

## 7. 可视化

脚本会输出：

```text
videos/rear_world_motion.mp4
videos/left_rear_world_motion.mp4
videos/right_rear_world_motion.mp4
```

视频显示：

- 2D box；
- 3D box；
- 当前 1 秒速度；
- 帧级状态；
- track 级状态。

颜色：

- 绿色：moving；
- 蓝橙色：static；
- 黄色：uncertain。

## 8. 当前本地验证结果

以后视三路 `final_three_views_fixed_v1` 为输入，本地已跑通：

```text
output: D:\mono-detect\outputs\rear_3d_motion_state_delivery_v1
states:
  moving: 80
  static: 10
  uncertain: 50
```

前视三路也跑通过：

```text
output: D:\mono-detect\outputs\front_3d_motion_state_delivery_v1
states:
  moving: 41
  static: 22
  uncertain: 70
```

## 9. 调参建议

如果“静态车误判为动态车”的代价更大：

```text
提高 moving_ratio_threshold，例如 0.30 → 0.40 / 0.50
或提高 moving_threshold_mps，例如 2.0 → 2.5
```

如果“动态车误判为静态车”的代价更大：

```text
降低 moving_ratio_threshold，例如 0.30 → 0.20
```

目前观察：很多误判来自远距离目标，而这些目标对当前需求不太重要。因此可以在下游使用：

```text
distance_to_ego_median_m
distance_to_ego_min_m
jitter_ratio
usable_for_motion
```

对远距离、强抖动或不可靠 track 降权。
