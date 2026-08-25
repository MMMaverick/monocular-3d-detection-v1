from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DA3_ROOT = Path(r"C:\Users\Administrator\Documents\3ddt\map-anything\Depth-Anything-3")
DEFAULT_MODEL_DIR = DEFAULT_DA3_ROOT / "checkpoints" / "da3metric-large"
DEFAULT_INTRINSIC = PROJECT_ROOT / "data" / "calib" / "rear_camera" / "rear_camera-intrinsic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rear-camera single-view DA3METRIC-LARGE depth.")
    parser.add_argument("--da3-root", default=str(DEFAULT_DA3_ROOT), help="Depth-Anything-3 source directory.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="DA3METRIC-LARGE model directory.")
    parser.add_argument("--input", default="data/camera/rear_camera", help="Rear camera image directory.")
    parser.add_argument("--output", default="outputs/da3metric_rear_full", help="Output directory.")
    parser.add_argument("--intrinsic", default=str(DEFAULT_INTRINSIC), help="Rear camera intrinsic JSON.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit frames. Use -1 for all.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--chunk-size", type=int, default=80, help="Images per metric inference chunk.")
    parser.add_argument("--process-res", type=int, default=336, help="DA3 processing resolution.")
    parser.add_argument("--process-res-method", default="upper_bound_resize", help="DA3 resize method.")
    parser.add_argument("--device", default="cuda", help="Inference device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_da3_imports(Path(args.da3_root))
    from depth_anything_3.api import DepthAnything3

    image_paths = _collect_image_sequence(
        args.input,
        stride=args.stride,
        max_frames=None if args.max_frames is None or args.max_frames < 0 else args.max_frames,
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_size = Image.open(image_paths[0]).size
    original_k = _load_intrinsic(Path(args.intrinsic))

    print(f"Loading DA3 metric model: {args.model_dir}")
    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device).eval()

    depth_chunks: list[np.ndarray] = []
    conf_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    intrinsic_chunks: list[np.ndarray] = []
    extrinsic_chunks: list[np.ndarray] = []

    for start in range(0, len(image_paths), args.chunk_size):
        chunk_paths = image_paths[start : start + args.chunk_size]
        print(f"Running metric chunk {start}-{start + len(chunk_paths) - 1} ({len(chunk_paths)} images)")
        with torch.inference_mode():
            prediction = model.inference(
                [str(path) for path in chunk_paths],
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                export_dir=None,
            )

        raw_depth = np.asarray(prediction.depth, dtype=np.float32)
        proc_images = np.asarray(prediction.processed_images, dtype=np.uint8)
        out_h, out_w = raw_depth.shape[1:]
        scaled_k = _scale_intrinsic(original_k, original_size, (out_w, out_h))
        focal = float((scaled_k[0, 0] + scaled_k[1, 1]) * 0.5)
        metric_depth = raw_depth * (focal / 300.0)

        sky = getattr(prediction, "sky", None)
        if sky is None:
            conf = np.ones_like(metric_depth, dtype=np.float32)
        else:
            conf = (~np.asarray(sky, dtype=bool)).astype(np.float32)

        depth_chunks.append(metric_depth.astype(np.float32))
        conf_chunks.append(conf)
        image_chunks.append(proc_images)
        intrinsic_chunks.append(np.repeat(scaled_k[None, ...], len(chunk_paths), axis=0))
        extrinsic_chunks.append(np.repeat(np.eye(4, dtype=np.float32)[None, :3, :4], len(chunk_paths), axis=0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    depth = np.concatenate(depth_chunks, axis=0)
    conf = np.concatenate(conf_chunks, axis=0)
    images = np.concatenate(image_chunks, axis=0)
    intrinsics = np.concatenate(intrinsic_chunks, axis=0)
    extrinsics = np.concatenate(extrinsic_chunks, axis=0)

    np.savez_compressed(output_dir / "depth.npz", depth=depth, depth_conf=conf)
    np.savez_compressed(output_dir / "images.npz", images=images)
    np.savez_compressed(output_dir / "cameras.npz", intrinsic=intrinsics, extrinsic=extrinsics)
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(path) for path in image_paths) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model": str(args.model_dir),
        "num_images": int(depth.shape[0]),
        "original_size_wh": list(original_size),
        "processed_shape_hwc": list(images.shape[1:]),
        "intrinsic_source": str(args.intrinsic),
        "scaled_intrinsic": intrinsics[0].tolist(),
        "metric_scaling": "depth_m = da3metric_raw_depth * mean(fx, fy) / 300 using scaled processed-image intrinsics",
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _prepare_da3_imports(da3_root: Path) -> None:
    for path in (da3_root / ".local_deps", da3_root / "src"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))


def _collect_image_sequence(input_dir: str | Path, stride: int = 1, max_frames: int | None = None) -> list[Path]:
    """List one camera image sequence without depending on the original vggt-omega repo."""

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    step = max(1, int(stride))
    paths = paths[::step]
    if max_frames is not None and int(max_frames) >= 0:
        paths = paths[: int(max_frames)]
    return paths


def _load_intrinsic(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.values() if isinstance(payload, dict) else []
    for item in values:
        param = item.get("param", {}) if isinstance(item, dict) else {}
        for key in ("cam_K_new", "cam_K"):
            data = param.get(key, {}).get("data")
            if data is not None:
                return np.asarray(data, dtype=np.float32).reshape(3, 3)
    raise ValueError(f"Could not find cam_K_new/cam_K in {path}")


def _scale_intrinsic(k: np.ndarray, original_size_wh: tuple[int, int], output_size_wh: tuple[int, int]) -> np.ndarray:
    original_w, original_h = original_size_wh
    output_w, output_h = output_size_wh
    scaled = k.copy().astype(np.float32)
    scaled[0, :] *= output_w / original_w
    scaled[1, :] *= output_h / original_h
    scaled[2, :] = [0.0, 0.0, 1.0]
    return scaled


if __name__ == "__main__":
    main()
