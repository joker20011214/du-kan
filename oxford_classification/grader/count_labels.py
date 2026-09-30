"""Quick per-label pixel counter for an 8-class segmentation PNG.

Usage (CLI)::

    python -m oxford_grader.count_labels         --labels-dir data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr         --limit 10         --out oxford_grader/output/label_counts/label_counts_first10.json

Or programmatically::

    from oxford_grader.count_labels import count_label_pixels, count_dir
    counts = count_label_pixels(".../KIDNEY_0001.png")
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from .instance_utils import CLASS


LABEL_NAMES = {v: k for k, v in CLASS.items()}


def count_label_pixels(seg_path: str) -> Dict[str, int]:
    """Return ``{label_name: pixel_count}`` for an 8-class label PNG.

    The PNG is read as-is (no conversion), so the integer class values
    stored in the file are preserved.
    """
    arr = np.array(Image.open(seg_path))
    out = {}
    total = int(arr.size)
    out['__total_px__'] = total
    for cls, name in LABEL_NAMES.items():
        out[name] = int((arr == cls).sum())
    return out


def count_dir(
    labels_dir: str,
    limit: int | None = None,
    pattern: str = '*.png',
    sort_key=None,
) -> List[Dict[str, object]]:
    """Count labels for every label PNG in ``labels_dir``.

    Returns a list of ``{"case": ..., "counts": {...}}`` records.  The
    ``counts`` dict has the same structure as ``count_label_pixels``.
    """
    p = Path(labels_dir)
    files = sorted(p.glob(pattern), key=sort_key)
    if limit:
        files = files[:limit]
    out = []
    for fp in files:
        c = count_label_pixels(str(fp))
        out.append({'case': fp.stem, 'image': str(fp), 'counts': c})
    return out


def _print_table(records: List[Dict[str, object]]) -> None:
    """Pretty-print the records as a fixed-width table to stdout."""
    label_order = []
    label_order.extend(('background', 'crescent', 'sclerosis', 'mesangial_lesion',
                        'tubular_atrophy', 'mesangial_cell', 'glomerulus',
                        'endothelial_cell'))
    headers = ['case', 'total_px'] + label_order
    col_w = [14, 10] + [12] * len(label_order)

    def _row(cells):
        return '  '.join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print(_row(headers))
    print(_row(['-' * w for w in col_w]))
    for r in records:
        c = r['counts']
        cells = [r['case'], c.get('__total_px__', '')]
        for n in label_order:
            cells.append(c.get(n, 0))
        print(_row(cells))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    ap.add_argument('--labels-dir',
                    default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                    help='Directory containing <case>.png label maps.')
    ap.add_argument('--limit', type=int, default=10,
                    help='Max number of files.')
    ap.add_argument('--out',
                    default='oxford_grader/output/label_counts/label_counts.json',
                    help='Output JSON path.')
    ap.add_argument('--csv',
                    default='oxford_grader/output/label_counts/label_counts.csv',
                    help='Output CSV path.')
    ap.add_argument('--no-csv', action='store_true',
                    help='Skip the CSV output.')
    ap.add_argument('--no-json', action='store_true',
                    help='Skip the JSON output.')
    args = ap.parse_args(argv)
    records = count_dir(args.labels_dir, limit=args.limit)
    _print_table(records)
    if records:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_json:
            with open(out_p, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f'\nWrote {out_p}')
        if not args.no_csv:
            label_order = []
            label_order.extend(('background', 'crescent', 'sclerosis', 'mesangial_lesion',
                                'tubular_atrophy', 'mesangial_cell', 'glomerulus',
                                'endothelial_cell'))
            csv_p = Path(args.csv)
            csv_p.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_p, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['case', 'total_px'] + label_order)
                for r in records:
                    c = r['counts']
                    row = [r['case'], c['__total_px__']] + [c.get(n, 0) for n in label_order]
                    w.writerow(row)
            print(f'Wrote {csv_p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))