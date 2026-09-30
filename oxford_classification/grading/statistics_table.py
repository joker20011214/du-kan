"""Statistics table generation for Oxford grading."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage
from PIL import Image

from .label_extractor import LabelExtractor, LABEL_NAMES_CN


def count_regions_per_label(label_path: str) -> Dict[int, int]:
    arr = np.array(Image.open(label_path))
    out = {}
    for label_id in LABEL_NAMES_CN.keys():
        mask = (arr == label_id)
        if not mask.any():
            out[label_id] = 0
            continue
        _, n = ndimage.label(mask)
        out[label_id] = int(n)
    return out


def generate_statistics_table(
    labels_dir: str,
    image_names: Optional[List[str]] = None,
    top_n: int = 10,
    output_file: Optional[str] = None,
) -> None:
    p = Path(labels_dir)
    files = sorted(p.glob('*.png'))
    if image_names:
        files = [f for f in files if f.stem in image_names]
    files = files[:top_n]
    headers = ['case'] + [LABEL_NAMES_CN[k] for k in sorted(LABEL_NAMES_CN.keys())]
    rows = [','.join(headers)]
    for fp in files:
        counts = count_regions_per_label(str(fp))
        cells = [fp.stem] + [str(counts.get(k, 0)) for k in sorted(LABEL_NAMES_CN.keys())]
        rows.append(','.join(cells))
    text = '\n'.join(rows)
    if output_file:
        Path(output_file).write_text(text, encoding='utf-8')
        print(f"Wrote {output_file}")
    else:
        print(text)


def generate_statistics_table_with_pixels(
    labels_dir: str,
    image_names: Optional[List[str]] = None,
    top_n: int = 10,
    output_file: Optional[str] = None,
) -> None:
    extractor = LabelExtractor(labels_dir)
    p = Path(labels_dir)
    files = sorted(p.glob('*.png'))
    if image_names:
        files = [f for f in files if f.stem in image_names]
    files = files[:top_n]
    headers = ['case', 'total_px'] + [LABEL_NAMES_CN[k] for k in sorted(LABEL_NAMES_CN.keys())]
    rows = [','.join(headers)]
    for fp in files:
        counts = count_regions_per_label(str(fp))
        ls = extractor.extract_single(str(fp))
        non_background_stats = [
            counts.get(k, 0) for k in sorted(LABEL_NAMES_CN.keys())
        ]
        cells = [fp.stem, str(ls.total_pixels)] + [str(s) for s in non_background_stats]
        rows.append(','.join(cells))
    text = '\n'.join(rows)
    if output_file:
        Path(output_file).write_text(text, encoding='utf-8')
        print(f"Wrote {output_file}")
    else:
        print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oxford grading statistics tables')
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--output', default=None)
    parser.add_argument('--with-pixels', action='store_true')
    args = parser.parse_args()
    if args.with_pixels:
        generate_statistics_table_with_pixels(args.labels_dir, top_n=args.top_n, output_file=args.output)
    else:
        generate_statistics_table(args.labels_dir, top_n=args.top_n, output_file=args.output)