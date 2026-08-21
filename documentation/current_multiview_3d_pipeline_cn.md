# 当前 2D Box → 单视角 3D 优化 → 跨视角联合后优化流程说明

本文档描述的是当前项目里正在使用的主线流程。历史实验、Waymo 检测探索、早期几何 debug 脚本仍保留在仓库中，但不属于这条主线，除非后续明确重新启用。

## 1. 当前目标

输入已有的多相机 2D box / track / mask / 标定 / depth 初始化结果，先分别在每个视角内优化 3D box 轨迹，再用 LoMa/特征匹配得到的跨视角同 ID 关系，把同一辆车在不同相机里的轨迹做二次联合优化。

当前主线分三段：

1. 单视角 3D box 优化：每个 camera view 内，每个 2D track 独立优化一段 3D 轨迹。
2. 跨视角 2D track 关联：用 LoMa 几何匹配 + DINO/外观特征，把不同视角下同一辆车的 track 合成 global id。
3. 跨视角联合后优化：以上一步 global id 为固定关联，把多个视角对同一辆车的观测放到同一个优化问题里。

## 2. 目录与主入口

### 主配置

- `configs/rebuild_three_views_repair_params_full_wsl_v1.yaml`
  - 当前三视角单视角优化主配置。
  - 输出到 `outputs/rebuild_three_views_repair_params_full_wsl_v1`。

- `configs/loma_global_2d_repair_params_tracks_v1.yaml`
  - 当前 LoMa 跨视角 2D track 匹配配置。
  - 输出到 `outputs/loma_global_2d_repair_params_tracks_v1`。

- `configs/multiview_joint_loma_repair_from_singleview_v1.yaml`
  - 当前跨视角联合后优化配置。
  - 从单视角优化 CSV 和 LoMa global id 文件读取输入。
  - 输出到 `outputs/multiview_joint_loma_repair_from_singleview_v1`。

### 主脚本

- `code/rebuild_3d_box_optimizer/run.py`
  - 单视角 3D box 优化入口。

- `code/rebuild_3d_box_optimizer/associate_2d_tracks_loma.py`
  - LoMa 跨视角 track 匹配入口。

- `code/rebuild_3d_box_optimizer/run_multiview_joint_from_singleview.py`
  - 跨视角联合后优化入口。

- `scripts/render_multiview_joint_track_panels.py`
  - 单独查看某些 global track 的联合视图视频。
  - 一个视频内左侧放 2/3 个相机视角，右侧放 BEV。

### WSL 后台启动脚本

- `scripts/start_rebuild_repair_params_full_wsl.sh`
- `scripts/start_loma_repair_params_tracks_wsl.sh`
- `scripts/start_multiview_joint_loma_repair_wsl.sh`

这些脚本会把日志写到对应输出目录下的 `logs/` 里，并记录 pid，避免重复启动。

## 3. 运行顺序

### 3.1 单视角三路优化

在 WSL 中运行：

```bash
cd /mnt/d/mono-detect
bash scripts/start_rebuild_repair_params_full_wsl.sh
```

主要输出：

- `outputs/rebuild_three_views_repair_params_full_wsl_v1/frame_3d_boxes_world_track_joint.csv`
- `outputs/rebuild_three_views_repair_params_full_wsl_v1/frame_loss_diagnostics.csv`
- `outputs/rebuild_three_views_repair_params_full_wsl_v1/track_summary.csv`
- `outputs/rebuild_three_views_repair_params_full_wsl_v1/rear_overlay.mp4`
- `outputs/rebuild_three_views_repair_params_full_wsl_v1/left_rear_overlay.mp4`
- `outputs/rebuild_three_views_repair_params_full_wsl_v1/right_rear_overlay.mp4`

这一阶段仍然是“单视角内优化”，不同视角之间还没有共同约束。

### 3.2 跨视角 2D track 关联

```bash
cd /mnt/d/mono-detect
bash scripts/start_loma_repair_params_tracks_wsl.sh
```

主要输出：

- `outputs/loma_global_2d_repair_params_tracks_v1/global_track_assignments.csv`
- `outputs/loma_global_2d_repair_params_tracks_v1/accepted_edges.csv`
- `outputs/loma_global_2d_repair_params_tracks_v1/global_2d_observations.csv`
- `outputs/loma_global_2d_repair_params_tracks_v1/candidate_diagnostics.csv`

其中 `global_track_assignments.csv` 是跨视角联合优化的核心输入。当前联合优化默认固定使用这个文件，不再临时按 3D 距离重做匹配。

### 3.3 跨视角联合后优化

```bash
cd /mnt/d/mono-detect
bash scripts/start_multiview_joint_loma_repair_wsl.sh
```

主要输出：

- `outputs/multiview_joint_loma_repair_from_singleview_v1/frame_3d_boxes_multiview_joint_from_singleview.csv`
- `outputs/multiview_joint_loma_repair_from_singleview_v1/frame_loss_diagnostics.csv`
- `outputs/multiview_joint_loma_repair_from_singleview_v1/global_track_optimization_summary.csv`
- `outputs/multiview_joint_loma_repair_from_singleview_v1/global_track_assignments.csv`
- `outputs/multiview_joint_loma_repair_from_singleview_v1/global_track_components.csv`
- `outputs/multiview_joint_loma_repair_from_singleview_v1/match_edges.csv`

这一阶段做的是“二次优化”：先读取单视角优化好的 3D box 作为初值，再把同一个 global id 的多视角观测合在一起优化。

## 4. 当前优化变量

单视角和联合后优化都以轨迹为单位处理：

- 每一帧一个 3D center。
- 每条 track 共享一个 size。
- yaw 当前默认不优化，使用以后视相机/自车运动方向为参考的 yaw-only upright box。
- 坐标系以 world/ego 相关坐标作为优化主坐标，渲染时投回各相机。

size 顺序统一为：

```text
[length, width, height]
```

不要再用“投影后哪条边最长”判断车长。长边语义固定就是车辆朝向方向的 length。

## 5. 当前主要 loss

### 单视角优化阶段

当前主配置 `rebuild_three_views_repair_params_full_wsl_v1.yaml` 中主要启用：

- `supporting_edges`
  - 左右贴边约束。
  - 不截断时，要求 3D box 投影后的左右支撑边贴近 2D box 左右边。
  - 截断时使用单边约束。

- `mask.contain`
  - mask 前景像素应被 3D box 投影区域包含。
  - 当前为了速度仍然使用采样前景点，主配置里 `max_foreground_points: 512`。

- `mask.oversize`
  - 在保证包含 mask 的前提下，限制 3D box 投影区域不要过大。
  - 当前权重较小，用来辅助紧致，不应该主导优化。

- `bbox_fit`
  - 远距离启用，辅助 3D box 投影和 2D box 更贴。
  - 当前使用 `far_start_m: 35.0`。

- `top_bottom_edges`
  - 距离超过阈值时加入上下贴边。
  - 当前使用 `far_start_m: 10.0`。

- `ego_box_safety`
  - 自车 box 和目标 box 不应互相穿插。
  - 用来代替早期的 `depth_safety` / `center_depth_safety` 近距离安全项。

- `ground_plane`
  - 目标 box 底部应接近地面。
  - 当前相机高度主配置为 `0.8m`。

当前关闭：

- `depth_safety`
- `center_depth_safety`

原因：目前倾向用 ego box 近距离几何约束替代这两个深度安全项。

### 跨视角联合后优化阶段

跨视角阶段仍复用单视角观测项，同时额外关注：

- 同一个 global id 下，不同视角观测共享同一段 3D 轨迹。
- 不同视角 timestamp 在一定阈值内会合并成同一个时间节点。
- `joint_from_singleview.temporal_smoothness`
  - 在联合后优化中加入轨迹平滑。
  - 当前默认权重较轻，用于抑制跨视角融合后的跳变。

## 6. 截断判断逻辑：重要且有待商榷

当前截断判断不只看 box 是否直接贴到图像边界，还加入了 track 级别的补判：

- 如果同一条 track 大部分时间 box 很大；
- 某些帧突然在图像边缘附近变小；
- 且 width/height/area 明显低于该 track 的参考尺寸；
- 则这些边缘附近帧会被认为可能是截断。

这个逻辑是为了解决“2D label 没贴到边界但实际已经截断”的情况。但它非常关键，也有风险：如果误判，会影响贴边、height prior、oversize 等项，可能把近距离大车拉坏。

因此这部分目前应视为“待进一步验证的重要启发式规则”，不要把它当成完全可靠的真值。

相关配置位置：

```yaml
observations:
  supporting_edges:
    track_level_truncation:
      enabled: true
      ...
      size_anomaly_near_border:
        enabled: true
```

相关代码位置：

- `code/rebuild_3d_box_optimizer/data.py`

## 7. 跨视角匹配依据

当前跨视角匹配不是直接用 3D box 距离完成，而是使用 LoMa 和外观/几何筛选：

1. 找相近 timestamp 下的候选 2D track 帧。
2. 用极线几何先过滤不合理候选。
3. 用 LoMa 做图像特征匹配。
4. 可选 triangulation/reprojection 检查。
5. 用 track 级别外观特征辅助评分。
6. 每个 view pair 里用二分图选择较可信的边。
7. 根据 accepted edges 合成 global id。

最终给联合优化使用的是固定 global id 文件：

```text
outputs/loma_global_2d_repair_params_tracks_v1/global_track_assignments.csv
```

## 8. 单独查看某些 global track

用于检查跨视角同 ID 是否合理、联合优化后 3D box 是否一致：

```powershell
cd D:\mono-detect
$env:PYTHONPATH='D:\mono-detect\code'
C:\ProgramData\miniforge3\envs\dvgt\python.exe scripts\render_multiview_joint_track_panels.py --config outputs\multiview_joint_loma_repair_from_singleview_v1\resolved_config.yaml --diagnostics outputs\multiview_joint_loma_repair_from_singleview_v1\frame_loss_diagnostics.csv --output-dir outputs\multiview_joint_loma_repair_from_singleview_v1\joint_track_videos --global-track-ids 126,135,145,165,170 --panel-width 640 --panel-height 360 --bev-width 520 --fps 10
```

输出示例：

```text
outputs/multiview_joint_loma_repair_from_singleview_v1/joint_track_videos/global_126_rear_right_rear_bev.mp4
```

画面布局：

- 左侧：同一 global id 在 2 个或 3 个相机视角下的 mask、2D box、3D box。
- 右侧：BEV，坐标轴采用 rear-reference。

## 9. 当前暂不属于主线的内容

以下内容仍保留，但当前主线不依赖：

- `waymo/` 以及 `waymo_*` 脚本/配置
  - Waymo 场景检测、连通域、YOLO/SORT 探索。
  - 这是新数据源探索，不应混入当前原数据主线。

- 大量 `rebuild_*_v1.yaml`
  - 多数是历史消融或单个 track debug 配置。
  - 需要复现实验时可以参考，但新实验优先从当前主配置复制。

- `depth_safety` / `center_depth_safety`
  - 早期近距离安全约束。
  - 当前主线关闭，倾向使用 `ego_box_safety`。

- `bbox_fit` 的近距离版本
  - 当前只用于远距离辅助。
  - 近距离强行 bbox fit 容易把 box 拉近/缩小。

- yaw 优化
  - 当前未启用。
  - 暂时默认车辆 yaw 与后视相机/自车运动方向一致。

## 10. 接手时建议先看哪些文件

如果只想跑主线：

1. `configs/rebuild_three_views_repair_params_full_wsl_v1.yaml`
2. `configs/loma_global_2d_repair_params_tracks_v1.yaml`
3. `configs/multiview_joint_loma_repair_from_singleview_v1.yaml`
4. `scripts/start_rebuild_repair_params_full_wsl.sh`
5. `scripts/start_loma_repair_params_tracks_wsl.sh`
6. `scripts/start_multiview_joint_loma_repair_wsl.sh`

如果想改优化逻辑：

1. `code/rebuild_3d_box_optimizer/optimizer.py`
2. `code/rebuild_3d_box_optimizer/torch_geometry.py`
3. `code/rebuild_3d_box_optimizer/data.py`

如果想改跨视角关联：

1. `code/rebuild_3d_box_optimizer/associate_2d_tracks_loma.py`
2. `configs/loma_global_2d_repair_params_tracks_v1.yaml`

如果想改跨视角联合后优化：

1. `code/rebuild_3d_box_optimizer/run_multiview_joint_from_singleview.py`
2. `configs/multiview_joint_loma_repair_from_singleview_v1.yaml`

## 11. 目前最需要小心的问题

- 2D label 本身可能错，类别先验不能过强。
- 截断判断很重要，但现在仍是启发式。
- 大车、小车在投影上可能相似，单靠贴边/mask 容易产生“变小 + 拉近”的退化。
- mask 如果不准，会直接影响 contain / oversize。
- 跨视角 global id 如果匹配错，联合优化会把两辆车硬拉到一起。
- 当前 yaw 不优化，因此侧后视下朝向是否合理依赖统一 rear-reference 约定。
