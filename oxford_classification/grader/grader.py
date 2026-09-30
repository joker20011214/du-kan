"""End-to-end single-case grading.

Usage (programmatic)::

    from oxford_grader.grader import grade_one
    result = grade_one(
        image_path=".../KIDNEY_0001_0000.png",
        seg_path=".../KIDNEY_0001.png",          # or pass predictor=...
        out_dir="oxford_grader/output",
        thresholds_yaml="oxford_grader/thresholds.yaml",
    )
    print(result["summary"])   # M0E1S1T0C0
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .instance_utils import CLASS, estimate_cortex_area, per_glomerulus_stats, split_glomeruli
from .oxford_rules import grade_case, load_thresholds


def _load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def _load_or_predict_seg(
    image_path: str,
    seg_path: Optional[str] = None,
    predictor=None,
) -> np.ndarray:
    if seg_path and os.path.exists(seg_path):
        return np.array(Image.open(seg_path)).astype(np.uint8)
    if predictor is None:
        raise ValueError(f"seg_path={seg_path} does not exist and predictor is None; provide one of them.")
    from inference import predict_full_mask
    img = _load_image(image_path)
    return predict_full_mask(predictor, img)


def grade_one(
    image_path: str,
    seg_path: Optional[str] = None,
    out_dir: str = 'oxford_grader/output',
    thresholds_yaml: str = 'oxford_grader/thresholds.yaml',
    save_per_glomerulus_csv: bool = True,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grade a single slide and write JSON (+ optional CSV) to ``out_dir``.

    Parameters
    ----------
    image_path : str
        Path to the RGB slide image (``*_0000.png`` for nnUNet raw data).
    seg_path : str, optional
        Path to a precomputed 8-class segmentation PNG.  If it does not
        exist or is ``None``, ``predictor`` must be supplied.
    predictor : optional
        Callable ``predictor(img) -> seg`` (e.g. a Swin-UMambaD wrapper).
    out_dir : str
        Directory to write ``<case>_oxford.json`` and the per-glomerulus
        CSV into.  Created if missing.
    thresholds_yaml : str
        Path to the Oxford thresholds YAML file.  Ignored if ``thresholds``
        is supplied directly.
    save_per_glomerulus_csv : bool
        Whether to also write ``<case>_per_glomerulus.csv``.
    thresholds : dict, optional
        Pre-loaded Oxford threshold dict (overrides ``thresholds_yaml``).
    """
    if thresholds is None:
        thresholds = load_thresholds(thresholds_yaml)
    os.makedirs(out_dir, exist_ok=True)
    seg = _load_or_predict_seg(image_path, seg_path, predictor=None)
    glom_lbl = split_glomeruli(
        seg,
        min_glomerulus_pixels=thresholds['instance']['min_glomerulus_pixels'],
    )
    per_glom = per_glomerulus_stats(
        seg,
        glom_lbl,
        thresholds['sclerosis']['global_sclerosis_ratio'],
        min_glomerulus_pixels=thresholds['instance']['min_glomerulus_pixels'],
    )
    cortex_px = estimate_cortex_area(
        seg,
        dilation_px=thresholds['tubular_atrophy']['cortex_dilation_px'],
    )
    atrophy_px = int((seg == CLASS['tubular_atrophy']).sum())
    score = grade_case(per_glom, atrophy_px, cortex_px, thresholds=thresholds)
    case_id = Path(image_path).stem
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    record = {
        'case': case_id,
        'image': str(image_path),
        'oxford': score.to_dict(),
        'per_glomerulus': [g.to_dict() for g in per_glom],
    }
    with open(os.path.join(out_dir, f'{case_id}_oxford.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    if save_per_glomerulus_csv:
        import csv
        csv_path = os.path.join(out_dir, f'{case_id}_per_glomerulus.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(('glom_id', 'glom_pixels', 'lesion_pixels', 'cell_pixels',
                        'sclerosis_pixels', 'crescent_pixels', 'endothelial_pixels',
                        'is_global_sclerosis'))
            for g in per_glom:
                w.writerow([
                    g.glom_id, g.glom_pixels, g.lesion_pixels, g.cell_pixels,
                    g.sclerosis_pixels, g.crescent_pixels, g.endothelial_pixels,
                    int(g.is_global_sclerosis),
                ])
    return record