# 3D 轨迹动静状态后处理交付说明

本文档说明如何在已经完成 3D box 优化之后，基于优化后的 3D 轨迹判断每个 tracking object 是动态车还是静态车，并输出可检查的 CSV 与视频。

## 1. 输入

该流程不重新做检测、分割、深度估计或 3D 优化，只使用已经算好的优化结果：

- `frame_3d_boxes_world_track_joint.csv`
  - 每帧每个 track 的 3D box 结果。
  - 如果里面已有 `world_x/world_y/world_z`，会直接用世界坐标。
  - 如果没有世界坐标，会读取 annotation 里的相机外参和 ego pose，将 `cx/cy/cz` 从 camera frame 转到 global/world frame。
- `frame_loss_diagnostics.csv`
  - 用于视频可视化，包含图像路径、2D box、3D box 投影角点、mask 路径等。
- annotation JSON 目录，可选但强烈建议提供
  - 用于把 camera 坐标转换到 world 坐标。
  - 如果没有 annotation，脚本只能退化到 camera-local 坐标，这种结果不适合判断自车运动场景里的绝对动静状态。

## 2. 输出

输出目录下会生成：

```text
frame_world_motion_smoothed.csv
track_world_motion_summary_smoothed.csv
videos/
```

其中最重要的是：

### `track_world_motion_summary_smoothed.csv`

每个 track 一行，包含最终状态和辅助判断字段：

- `view`
- `track_id`
- `class`
- `num_frames`
- `first_frame`
- `last_frame`
- `duration_s`
- `motion_state`
  - `moving`
  - `static`
  - `uncertain`
- `motion_preference`
  - 给下游使用的软倾向：`moving` / `static` / `unknown`。
- `motion_confidence`
  - 0~1 的可信度分数，用于排序/筛选，不是严格标定概率。
- `moving_probability` / `static_probability`
  - 归一化动静证据分数，两者相加等于 1。
- `motion_state_reason`
  - 状态由哪条规则得到。
- `moving_ratio`
  - 有多少比例的 1 秒速度片段超过动态阈值。
- `median_speed_1s_mps`
- `p90_speed_1s_mps`
- `max_speed_1s_mps`
- `distance_to_ego_min_m`
- `distance_to_ego_median_m`
- `distance_to_ego_max_m`
- `distance_to_ego_first_m`
- `distance_to_ego_last_m`
- `depth_min_m`
- `depth_median_m`
- `depth_max_m`
- `world_displacement_xy_m`
- `world_path_xy_m`
- `max_smoothed_step_m`
- `jitter_ratio`
- `outlier_frames`
- `usable_for_motion`

### `frame_world_motion_smoothed.csv`

每帧每个 track 一行，包含平滑前后的坐标、每帧速度和帧级状态，适合 debug 某一帧为什么被判为 moving/static。

### `videos/*_world_motion.mp4`

每个视角一个视频，显示：

- 2D box
- 3D box
- 每帧速度
- 帧级动静状态
- track 级动静状态

## 3. 动静状态判定逻辑

默认使用比较稳的 ratio 策略：

```text
每隔约 1 秒计算一次 world 平面位移速度。
如果 speed > moving_threshold_mps，则这个时间片认为在动。
如果一条 track 中 moving 时间片比例 >= moving_ratio_threshold，则整条 track 判为 moving。
否则判为 static。
```

当前建议参数：

```text
speed_window_s = 1.0
moving_threshold_mps = 2.0
motion_policy = ratio
moving_ratio_threshold = 0.30
static_preference_ratio_threshold = 0.10
near_distance_m = 20.0
far_distance_m = 50.0
smooth_window = 7
hampel_window = 5
hampel_sigma = 3.0
max_residual_step_m = 3.0
min_track_frames = 10
min_track_duration_s = 1.0
```

解释：

- `moving_threshold_mps=2.0`
  - 1 秒速度超过 2m/s 才认为这一段在动。
- `moving_ratio_threshold=0.30`
  - 至少 30% 的有效速度片段在动，整条 track 才判为动态。
  - 这样可以避免 3D box 偶发抖动把静态车误判成动态。
- `static_preference_ratio_threshold=0.10`
  - 对 `uncertain` track，如果 moving 片段比例低于 10%，倾向输出 `motion_preference=static`。
- `near_distance_m=20.0` / `far_distance_m=50.0`
  - 近距离轨迹可信度更高，远距离轨迹可信度降低。
- `smooth_window=7`
  - 对 world 坐标做轻量平滑，降低单帧跳变。
- `hampel_window=5` / `hampel_sigma=3.0`
  - 用 Hampel filter 替换局部异常点。
- `max_residual_step_m=3.0`
  - 平滑后如果相邻帧仍有过大的跳变，则该 track 更容易被标记为 `uncertain`。

## 4. 推荐运行命令

以后视三路为例：

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
  --static-preference-ratio-threshold 0.10 `
  --near-distance-m 20.0 `
  --far-distance-m 50.0 `
  --smooth-window 7 `
  --hampel-window 5 `
  --hampel-sigma 3.0 `
  --max-residual-step-m 3.0 `
  --min-track-frames 10 `
  --min-track-duration-s 1.0 `
  --repo-root D:\mono-detect
```

前视三路只需要替换输入输出和 views：

```powershell
--views center_front left_front right_front
```

## 5. 当前本地 smoke 结果

以后视三路 `final_three_views_fixed_v1` 为输入，当前输出：

```text
D:\mono-detect\outputs\rear_3d_motion_state_delivery_v1
```

状态统计：

```text
moving: 80
static: 10
uncertain: 50
```

前视三路 `rebuild_front_three_views_gt_eval_full_novideo_v1` 也跑通过，状态统计：

```text
moving: 41
static: 22
uncertain: 70
```

## 6. 注意事项

- 该流程依赖 3D 优化结果质量。如果远距离车的 3D box 抖动明显，可能会造成静态误判动态。
- 对下游任务来说，如果“静态误判动态”的代价更大，可以提高 `moving_ratio_threshold`，例如从 `0.30` 提到 `0.40` 或 `0.50`。
- 如果“动态误判静态”的代价更大，可以降低 `moving_ratio_threshold`，例如 `0.20`。
- 不建议使用“只要一帧速度超过阈值就动态”的规则，因为当前 monocular 3D 结果会有偶发深度跳变。

## 7. `uncertain` 的可用倾向与归一化概率

`motion_state=uncertain` 表示这条轨迹不适合给硬判定，常见原因是帧数太少、持续时间太短、远距离抖动明显，或平滑后仍有较大残余跳变。

但下游通常仍然需要一个可用倾向，因此额外输出：

```text
motion_preference: moving / static / unknown
motion_confidence: 0~1
moving_probability + static_probability = 1
```

当前规则：

```text
如果 motion_state 是 moving/static：
    motion_preference = motion_state

如果 motion_state 是 uncertain：
    moving_ratio >= 0.30:
        motion_preference = moving
    moving_ratio <= 0.10:
        motion_preference = static
    其他：
        motion_preference = unknown
```

归一化概率：

```text
moving_probability = clamp(moving_ratio, 0, 1)
static_probability = 1 - moving_probability
```

注意：这里的 probability 是速度片段比例形成的“证据分数”，不是经过校准的真实概率。

`motion_confidence` 的计算：

```text
motion_confidence = 规则可靠性 × 比例确定性 × 距离可靠性
```

其中：

```text
规则可靠性:
  usable_for_motion=True   -> 1.0
  usable_for_motion=False  -> 0.5

比例确定性:
  abs(moving_ratio - 0.30) / 0.30
  然后 clamp 到 0~1

距离可靠性:
  distance < 20m   -> 1.0
  20m~50m          -> 0.7
  >50m             -> 0.4
```

如果 `motion_preference=unknown`，最终 confidence 再乘 0.5。
