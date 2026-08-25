from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "boxmot": "22.0.0",
    "ultralytics": "8.4.115",
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "lap": "0.5.13",
    "lapx": "0.9.4",
    "filterpy": "1.4.5",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check key environment versions for reproducing the BoxMOT tracking experiment.")
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if a key version/path mismatches.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {},
        "paths": {},
        "ok": True,
    }

    ok = True
    packages: dict[str, dict[str, object]] = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        matched = actual is not None and (actual == expected or actual.startswith(expected + "+"))
        ok = ok and matched
        packages[name] = {"expected": expected, "actual": actual, "matched": matched}

    paths = {
        "sam_checkpoint": repo / "checkpoints" / "sam_vit_h_4b8939.pth",
        "da3_metric_model_dir": repo / "third_party" / "Depth-Anything-3" / "checkpoints" / "da3metric-large",
    }
    try:
        import torch

        paths["dinov2_cache"] = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    except Exception:
        paths["dinov2_cache"] = Path("~/.cache/torch/hub/facebookresearch_dinov2_main").expanduser()

    path_report: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        exists = path.exists()
        ok = ok and exists
        path_report[name] = {"path": str(path), "exists": exists}

    report["packages"] = packages
    report["paths"] = path_report
    report["ok"] = ok
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
