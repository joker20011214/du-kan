"""Smoke test: run the Oxford grader on an existing ground-truth label PNG.

Useful to verify the framework end-to-end without loading the
Swin-UMamba model.  It uses ``grader.grade_one`` with ``seg_path=...``,
which bypasses the predictor.

Run with::

    python -m oxford_classification.grader.tests.smoke_test_grade
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    label = REPO / "data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr/KIDNEY_0001.png"
    image = REPO / "data/nnUNet_raw/Dataset800_Kidney_811Split/imagesTr/KIDNEY_0001_0000.png"
    if not label.exists():
        print(f"Label not found: {label}", file=sys.stderr)
        return 1
    if not image.exists():
        print(f"Image not found: {image}", file=sys.stderr)
        return 1

    from oxford_classification.grader.grader import grade_one
    out_dir = REPO / "oxford_classification" / "grader" / "output" / "smoke"
    record = grade_one(
        image_path=str(image),
        seg_path=str(label),
        out_dir=str(out_dir),
        thresholds_yaml=str(REPO / "oxford_classification" / "configs" / "thresholds.yaml"),
    )
    print("=" * 60)
    print(f"Case: {record['case']}")
    print(f"Oxford: {record['oxford']['summary']}")
    print(f"Confidence: {record['oxford']['confidence']}")
    print(f"Detail: {json.dumps(record['oxford']['detail'], indent=2, ensure_ascii=False)}")
    print(f"Glomeruli: {len(record['per_glomerulus'])}")
    print(f"Outputs in: {out_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())