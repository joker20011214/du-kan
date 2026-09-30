"""Visualization card for a single Oxford MEST-C grade.

Produces a 2-row x 3-column PNG that contains:

  * Top-left: the original RGB image (downsampled to fit)
  * Top-middle: the segmentation overlay (8 classes colour-coded)
  * Top-right: a per-glomerulus table with the worst offender highlighted
  * Bottom-left..middle..right: the 5 Oxford letters (M / E / S / T / C)
    with Chinese annotations and a "summary" string

The card is informational; it does not affect the score.

This module supports two backends: matplotlib when available, otherwise a
graceful fallback that uses only PIL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib.patches import FancyBboxPatch

    _CANDIDATE_FONTS = (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    )

    for fp in _CANDIDATE_FONTS:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                cjk_name = font_manager.FontProperties(fname=fp).get_name()
                plt.rcParams['font.sans-serif'] = [cjk_name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


COLOR_TABLE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (255, 165, 0),
    3: (255, 255, 0),
    4: (0, 255, 255),
    5: (0, 0, 255),
    6: (0, 255, 0),
    7: (255, 0, 255),
}

LABEL_CN: Dict[int, str] = {
    1: '新月体',
    2: '硬化',
    3: '系膜病变',
    4: '肾小管萎缩',
    5: '系膜细胞',
    6: '肾小球',
    7: '内皮细胞',
}


def _overlay(seg: np.ndarray, alpha: float = 0.55, downsample: int = 4) -> np.ndarray:
    """Render the segmentation as an RGB image with a light gray background.

    For every non-background class we paint its colour on top with weight
    ``alpha``; background is left gray (240).
    """
    h, w = seg.shape
    H, W = h // downsample, w // downsample
    out = np.zeros((H, W, 3), dtype=np.uint8)
    small = seg[::downsample, ::downsample]
    for cls, color in COLOR_TABLE.items():
        if cls == 0:
            continue
        out[small == cls] = color
    bg = np.full_like(out, 240)
    return ((1 - alpha) * bg + alpha * out).astype(np.uint8)


def _downsample_img(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new = (int(w * scale), int(h * scale))
    return np.array(Image.fromarray(img).resize(new, Image.BILINEAR))


def render_card(
    record: Dict[str, Any],
    image_path: Optional[str] = None,
    seg: Optional[np.ndarray] = None,
    out_path: str = 'oxford_card.png',
) -> str:
    """Render the Oxford MEST-C visualization card to ``out_path``.

    Returns the path of the rendered PNG.
    """
    oxford = record['oxford']
    per_glom = record['per_glomerulus']
    base_img = None
    overlay = None
    if image_path and os.path.exists(image_path):
        base_img = _downsample_img(np.array(Image.open(image_path).convert('RGB')))
    if seg is None and image_path is not None:
        sibling = Path(image_path).parent / (Path(image_path).stem.replace('_0000', '') + '.png')
        if sibling.exists():
            seg = np.array(Image.open(sibling)).astype(np.uint8)
    if seg is not None:
        overlay = _overlay(seg, downsample=max(1, seg.shape[0] // 800))
    if _HAS_MPL:
        return _render_with_mpl(record, base_img, overlay, per_glom, out_path)
    return _render_with_pil(record, base_img, overlay, per_glom, out_path)


def _render_with_mpl(
    record: Dict[str, Any],
    base_img: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    per_glom: List[Dict[str, Any]],
    out_path: str,
) -> str:
    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.2, 1.0],
        hspace=0.25,
        wspace=0.2,
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_ovl = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[0, 2])
    ax_letters = fig.add_subplot(gs[1, :])

    for ax in (ax_img, ax_ovl, ax_tbl, ax_letters):
        ax.set_xticks([])
        ax.set_yticks([])

    if base_img is not None:
        ax_img.imshow(base_img)
        ax_img.set_title('原图', fontsize=14)
    else:
        ax_img.text(0.5, 0.5, '无原图', ha='center', va='center', fontsize=18)
    ax_img.set_title('原图' if base_img is not None else '原图 (无)', fontsize=14)
    if overlay is not None:
        ax_ovl.imshow(overlay)
    ax_ovl.set_title('分割叠加', fontsize=14)

    if per_glom:
        worst = max(per_glom, key=lambda g: g.get('sclerosis_pixels', 0) + g.get('lesion_pixels', 0))
        rows = []
        for g in per_glom:
            rows.append([
                str(g.get('glom_id', '')),
                str(g.get('glom_pixels', '')),
                str(g.get('lesion_pixels', '')),
                str(g.get('cell_pixels', '')),
                str(g.get('sclerosis_pixels', '')),
                str(g.get('crescent_pixels', '')),
                str(g.get('endothelial_pixels', '')),
            ])
        ax_tbl.axis('off')
        ax_tbl.table(
            cellText=rows,
            colLabels=['ID', 'px', 'les', 'cell', 'scl', 'crsc', 'endo'],
            loc='center',
        )
        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1)

    ox = record.get('oxford', {})
    letters = ['M', 'E', 'S', 'T', 'C']
    labels = [
        f"M: {ox.get('M', '?')}  (系膜增生)",
        f"E: {ox.get('E', '?')}  (内皮增生)",
        f"S: {ox.get('S', '?')}  (节段硬化)",
        f"T: {ox.get('T', '?')}  (肾小管萎缩)",
        f"C: {ox.get('C', '?')}  (新月体)",
    ]
    for i, (letter, label) in enumerate(zip(letters, labels)):
        ax_letters.add_patch(FancyBboxPatch(
            (0.05 + i * 0.19, 0.4), 0.17, 0.4,
            boxstyle='round,pad=0.02',
            facecolor='#f0f8ff',
            edgecolor='navy',
        ))
        ax_letters.text(0.135 + i * 0.19, 0.7, letter, fontsize=20, ha='center')
        ax_letters.text(0.135 + i * 0.19, 0.5, label, fontsize=11, ha='center')
    ax_letters.text(
        0.5, 0.15,
        f"summary: {ox.get('summary', '')}    confidence: {ox.get('confidence', '')}",
        fontsize=14, ha='center',
    )
    ax_letters.set_xlim(0, 1)
    ax_letters.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _render_with_pil(
    record: Dict[str, Any],
    base_img: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    per_glom: List[Dict[str, Any]],
    out_path: str,
) -> str:
    """Render a lightweight visualization card with PIL only."""
    W, H = 1600, 1000
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    except Exception:
        font = ImageFont.load_default()
        font_big = ImageFont.load_default()

    panels = []
    if base_img is not None:
        h, w = base_img.shape[:2]
        scale = min(500 / w, 400 / h)
        panels.append(Image.fromarray(base_img).resize((int(w * scale), int(h * scale))))
    else:
        panels.append(None)
    if overlay is not None:
        h, w = overlay.shape[:2]
        scale = min(500 / w, 400 / h)
        panels.append(Image.fromarray(overlay).resize((int(w * scale), int(h * scale))))
    else:
        panels.append(None)

    for i, p in enumerate(panels):
        x = 30 + i * 530
        y = 30
        draw.rectangle([x, y, x + 500, y + 400], outline='black')
        if p is not None:
            canvas.paste(p, (x + (500 - p.width) // 2, y + (400 - p.height) // 2))
        title = 'Original' if i == 0 else 'Segmentation Overlay'
        draw.text((x + 10, y - 25), title, fill='black', font=font_big)

    draw.text((30, 460), 'Per-glomerulus stats', fill='black', font=font_big)
    if per_glom:
        headers = ['ID', 'pixels', 'lesion', 'cell', 'scl', 'crsc', 'endo']
        col_w = 60
        x = 30
        y = 490
        for j, h in enumerate(headers):
            draw.text((x + j * col_w, y), h, fill='black', font=font_big)
        for i, g in enumerate(per_glom[:8]):
            yy = y + 30 + i * 25
            cells = [
                str(g.get('glom_id', '')),
                str(g.get('glom_pixels', '')),
                str(g.get('lesion_pixels', '')),
                str(g.get('cell_pixels', '')),
                str(g.get('sclerosis_pixels', '')),
                str(g.get('crescent_pixels', '')),
                str(g.get('endothelial_pixels', '')),
            ]
            for j, c in enumerate(cells):
                draw.text((x + j * col_w, yy), c, fill='black', font=font)

    ox = record.get('oxford', {})
    letters = ['M', 'E', 'S', 'T', 'C']
    cn = {
        'M': '系膜增生', 'E': '内皮增生', 'S': '节段硬化',
        'T': '肾小管萎缩', 'C': '新月体',
    }
    for i, letter in enumerate(letters):
        x = 30 + i * 220
        y = 760
        draw.rectangle([x, y, x + 200, y + 180], outline='navy', fill='#f0f8ff')
        draw.text((x + 90, y + 10), letter, fill='black', font=font_big)
        draw.text((x + 50, y + 50), f"{letter}: {ox.get(letter, '?')}", fill='black', font=font)
        draw.text((x + 50, y + 80), cn[letter], fill='black', font=font)
    draw.text(
        (30, 940),
        f"summary: {ox.get('summary', '')}    confidence: {ox.get('confidence', '')}",
        fill='black', font=font_big,
    )
    canvas.save(out_path)
    return out_path


def render_cards_from_dir(
    out_dir: str,
    image_dir: Optional[str] = None,
) -> List[str]:
    """Render cards for every <case>_oxford.json in ``out_dir``."""
    paths = sorted(Path(out_dir).glob('*_oxford.json'))
    out = []
    for p in paths:
        with open(p) as f:
            record = json.loads(f.read())
        out_path = str(p).replace('_oxford.json', '_oxford_card.png')
        if image_dir and not os.path.exists(out_path):
            case = p.stem.replace('_oxford', '')
            candidate = str(Path(image_dir) / f"{case}_0000.png")
            if not os.path.exists(candidate):
                candidate = str(Path(image_dir) / f"{case}.png")
            render_card(record, image_path=candidate, out_path=out_path)
        out.append(out_path)
    return out


import json  # noqa: E402