from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach DA3 depth statistics to per-camera 2D track CSVs.")
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--calib-root", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--image-root",
        default="",
        help=(
            "Root directory used to resolve relative row['image'] paths. "
            "For this dataset this should normally be D:/vggt-omega, because "
            "track CSV rows contain paths like data/camera/rear_camera/*.jpg."
        ),
    )
    parser.add_argument("--image-width", type=float, default=0.0, help="Fallback source image width in pixels, e.g. 1920.")
    parser.add_argument("--image-height", type=float, default=0.0, help="Fallback source image height in pixels, e.g. 1080.")
    parser.add_argument(
        "--allow-bbox-size-fallback",
        action="store_true",
        help="Dangerous legacy behavior: if image size is unknown, use x2/y2 as image size. Disabled by default.",
    )
    parser.add_argument("--bbox-scale", type=float, default=1.0)
    parser.add_argument("--conf-percentile", type=float, default=0.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=250.0)
    args = parser.parse_args()

    track_root = Path(args.track_root)
    depth_root = Path(args.depth_root)
    calib_root = Path(args.calib_root)
    out_root = Path(args.output_root)
    cameras = [x.strip() for x in str(args.cameras).split(",") if x.strip()]

    summary = []
    for camera in cameras:
        src = track_root / camera / "tracks.csv"
        if not src.exists():
            # SORT writes by compact view name; keep this fallback for generated configs.
            compact = camera.replace("_camera", "")
            src = track_root / compact / "tracks.csv"
        depth_dir = depth_root / camera
        out_csv = out_root / camera / "tracks.csv"
        rows, stats = attach_camera(src, depth_dir, calib_root / camera / f"{camera}-intrinsic.json", args)
        write_csv(out_csv, rows)
        stats.update({"camera": camera, "rows": len(rows), "csv": str(out_csv)})
        summary.append(stats)
        print(
            "ATTACH_DEPTH "
            f"camera={camera} rows={len(rows)} depth_rows={stats['depth_attached']} "
            f"missing_depth={stats['missing_depth']} "
            f"image_size_csv={stats['image_size_from_csv']} "
            f"image_size_root={stats['image_size_from_image_root']} "
            f"image_size_fixed={stats['image_size_from_fixed_arg']} "
            f"image_size_bbox_fallback={stats['image_size_from_bbox_fallback']} "
            f"csv={out_csv}",
            flush=True,
        )
    write_csv(out_root / "summary.csv", summary)
    return 0


def attach_camera(src: Path, depth_dir: Path, intrinsic_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_csv(src)
    if not rows:
        return [], {
            "depth_attached": 0,
            "missing_depth": 0,
            "image_size_from_csv": 0,
            "image_size_from_image_root": 0,
            "image_size_from_fixed_arg": 0,
            "image_size_from_bbox_fallback": 0,
        }
    depth_pack = np.load(depth_dir / "depth.npz")
    depth = np.asarray(depth_pack["depth"], dtype=np.float32)
    conf = np.asarray(depth_pack["depth_conf"], dtype=np.float32) if "depth_conf" in depth_pack else np.ones_like(depth, dtype=np.float32)
    image_list = read_image_list(depth_dir / "image_list.txt")
    depth_by_image = {norm_path(path): idx for idx, path in enumerate(image_list)}
    intrinsic = load_intrinsic(intrinsic_path)

    attached = 0
    missing = 0
    size_sources = {"csv": 0, "image_root": 0, "fixed_arg": 0, "bbox_fallback": 0}
    for row in rows:
        image_key = norm_path(Path(str(row.get("image", ""))))
        frame = int_float(row.get("frame", -1))
        depth_idx = depth_by_image.get(image_key, frame if 0 <= frame < depth.shape[0] else -1)
        if depth_idx < 0 or depth_idx >= depth.shape[0]:
            missing += 1
            continue
        ok, size_source = attach_row_depth(row, depth[depth_idx], conf[depth_idx], intrinsic, args)
        if size_source:
            size_sources[size_source] = size_sources.get(size_source, 0) + 1
        attached += int(ok)
        missing += int(not ok)
    return rows, {
        "depth_attached": attached,
        "missing_depth": missing,
        "image_size_from_csv": size_sources.get("csv", 0),
        "image_size_from_image_root": size_sources.get("image_root", 0),
        "image_size_from_fixed_arg": size_sources.get("fixed_arg", 0),
        "image_size_from_bbox_fallback": size_sources.get("bbox_fallback", 0),
    }


def attach_row_depth(row: dict[str, object], depth: np.ndarray, conf: np.ndarray, intrinsic: np.ndarray, args: argparse.Namespace) -> tuple[bool, str]:
    try:
        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
    except Exception:
        return False, ""
    if x2 <= x1 or y2 <= y1:
        return False, ""
    h, w = depth.shape[:2]
    src_w, src_h, size_source = source_image_size(row, args)
    if src_w <= 1 or src_h <= 1:
        if not bool(args.allow_bbox_size_fallback):
            raise ValueError(
                "Could not resolve source image size for row "
                f"image={row.get('image')!r}. Pass --image-root pointing at the dataset root "
                "or pass --image-width/--image-height. Refusing legacy bbox-size fallback."
            )
        src_w = max(src_w, x2 + 1.0)
        src_h = max(src_h, y2 + 1.0)
        size_source = "bbox_fallback"
    sx = w / src_w
    sy = h / src_h
    bx1, by1, bx2, by2 = expand_box(x1, y1, x2, y2, float(args.bbox_scale))
    ix1 = int(np.floor(np.clip(bx1 * sx, 0, w - 1)))
    iy1 = int(np.floor(np.clip(by1 * sy, 0, h - 1)))
    ix2 = int(np.ceil(np.clip(bx2 * sx, 0, w)))
    iy2 = int(np.ceil(np.clip(by2 * sy, 0, h)))
    if ix2 <= ix1 or iy2 <= iy1:
        return False, size_source
    z_patch = depth[iy1:iy2, ix1:ix2].reshape(-1)
    c_patch = conf[iy1:iy2, ix1:ix2].reshape(-1)
    valid = np.isfinite(z_patch) & np.isfinite(c_patch) & (z_patch > float(args.min_depth)) & (z_patch < float(args.max_depth))
    if np.any(valid) and float(args.conf_percentile) > 0:
        valid &= c_patch >= np.percentile(c_patch[valid], float(args.conf_percentile))
    if not np.any(valid):
        return False, size_source
    vals = z_patch[valid]
    z_med = float(np.median(vals))
    z_mean = float(np.mean(vals))
    z_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    px, py = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    row["cx"] = cx
    row["cy"] = cy
    row["depth_median"] = z_med
    row["depth_weighted_mean"] = z_mean
    row["depth_iqr"] = z_iqr
    row["conf_mean"] = float(np.mean(c_patch[valid]))
    row["x_cam"] = (cx - px) / fx * z_med
    row["y_cam"] = (cy - py) / fy * z_med
    row["z_cam"] = z_med
    row["depth_source"] = "da3_bbox_median"
    row["depth_source_image_width"] = src_w
    row["depth_source_image_height"] = src_h
    row["depth_source_image_size_source"] = size_source
    return True, size_source


def source_image_size(row: dict[str, object], args: argparse.Namespace) -> tuple[float, float, str]:
    for wk, hk in (("img_new_w", "img_new_h"), ("image_width", "image_height"), ("width", "height")):
        try:
            w = float(row.get(wk) or 0)
            h = float(row.get(hk) or 0)
        except Exception:
            w = h = 0.0
        if w > 1 and h > 1:
            return w, h, "csv"
    image = str(row.get("image") or "")
    image_path = resolve_image_path(image, str(getattr(args, "image_root", "") or ""))
    if image_path is not None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            ih, iw = frame.shape[:2]
            return float(iw), float(ih), "image_root"
    fixed_w = float(getattr(args, "image_width", 0.0) or 0.0)
    fixed_h = float(getattr(args, "image_height", 0.0) or 0.0)
    if fixed_w > 1 and fixed_h > 1:
        return fixed_w, fixed_h, "fixed_arg"
    return 0.0, 0.0, ""


def resolve_image_path(image: str, image_root: str) -> Path | None:
    if not image:
        return None
    raw = Path(image)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if image_root:
            root = Path(image_root)
            candidates.append(root / raw)
        candidates.append(raw)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def expand_box(x1: float, y1: float, x2: float, y2: float, scale: float) -> tuple[float, float, float, float]:
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h


def read_image_list(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_path(path: Path) -> str:
    return path.as_posix().lower()


def load_intrinsic(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.values() if isinstance(payload, dict) else []
    for item in values:
        param = item.get("param", {}) if isinstance(item, dict) else {}
        for key in ("cam_K_new", "cam_K"):
            data = param.get(key, {}).get("data")
            if data is not None:
                return np.asarray(data, dtype=np.float32).reshape(3, 3)
    raise ValueError(f"Could not find cam_K_new/cam_K in {path}")


def int_float(value: object) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return -1


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
