#!/usr/bin/env python3
"""Verify the offline original-2D-box full pipeline runtime.

This checks the heavy pieces that usually break on a fresh Ubuntu box:
NumPy/OpenCV/Torch, classic SAM import, and local DA3 metric checkpoint load.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    da3_ckpt = repo / "third_party" / "Depth-Anything-3" / "checkpoints" / "da3metric-large"
    sam_ckpt = repo / "checkpoints" / "sam_vit_h_4b8939.pth"

    import numpy
    import cv2
    import torch
    import pandas
    import scipy

    print(f"repo={repo}")
    print(f"numpy={numpy.__version__}")
    print(f"cv2={cv2.__version__}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"pandas={pandas.__version__}")
    print(f"scipy={scipy.__version__}")

    import segment_anything  # noqa: F401

    print(f"sam_import=ok checkpoint_exists={sam_ckpt.exists()} path={sam_ckpt}")

    from depth_anything_3.api import DepthAnything3

    print(f"da3_import=ok checkpoint_exists={da3_ckpt.exists()} path={da3_ckpt}")
    model = DepthAnything3.from_pretrained(str(da3_ckpt))
    print(f"da3_load=ok model={type(model).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
