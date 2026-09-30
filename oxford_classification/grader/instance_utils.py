"""Glomerulus instance split + multi-class pixel attribution.

Given a single full-resolution 8-class segmentation map (H, W) uint8, this
module:
  1. splits the connected components of class 6 (glomerulus) into
     individual instances and discards tiny noise;
  2. counts, for every instance, the number of pixels belonging to each
     of the other classes that fall inside it;
  3. (NEW) "captures" pixels of crescent / sclerosis / mesangial_lesion
     that lie just outside the glomerulus mask (Dataset800 follows the
     convention that crescents and lesions are drawn adjacent to the
     glomerulus, not on top of it).  We dilate the glomerulus mask by
     a few pixels and re-attribute the captured pixels;
  4. estimates the cortical area (used by the T-score) from the union
     of glomerulus + mesangial + endothelial pixels, dilated by a fixed
     margin.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy import ndimage

from .oxford_rules import PerGlomerulus


CLASS = {
    'background': 0,
    'crescent': 1,
    'sclerosis': 2,
    'mesangial_lesion': 3,
    'tubular_atrophy': 4,
    'mesangial_cell': 5,
    'glomerulus': 6,
    'endothelial_cell': 7,
}

ATTRACTION_DILATION_PX = 8
ATTRACTION_CLASSES = ('crescent', 'sclerosis', 'mesangial_lesion', 'endothelial_cell')


def split_glomeruli(
    seg: np.ndarray,
    min_glomerulus_pixels: int = 2000,
) -> np.ndarray:
    """Return a (H, W) int32 map of glomerulus instance ids; 0 = not-glom.

    Connected components of class 6 smaller than ``min_glomerulus_pixels``
    are removed (treated as noise).
    """
    glom = (seg == CLASS['glomerulus']).astype(np.uint8)
    lbl, n = ndimage.label(glom, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(seg, dtype=np.int32)
    sizes = ndimage.sum(glom, lbl, range(1, n + 1), index=range(1, n + 1))
    for i, sz in enumerate(sizes, start=1):
        if sz < min_glomerulus_pixels:
            lbl[lbl == i] = 0
    return lbl.astype(np.int32)


def per_glomerulus_stats(
    seg: np.ndarray,
    glom_lbl: np.ndarray,
    sclerosis_global_ratio: float = 0.8,
    min_glomerulus_pixels: int = 2000,
    attraction_dilation_px: int = ATTRACTION_DILATION_PX,
) -> List[PerGlomerulus]:
    """For each glomerulus instance, compute per-class pixel counts.

    Two attribution passes are used:
      * "Strict" - only pixels of the class that fall *inside* the glom
        mask are counted.  Used for ``cell`` and ``sclerosis``.
      * "Attraction" - the glomerulus mask is dilated by
        ``attraction_dilation_px`` (default 8) and adjacent
        crescent / sclerosis / mesangial_lesion pixels are captured.
    """
    out = []
    n_inst = int(glom_lbl.max())
    if n_inst == 0:
        return out

    masks = {k: (seg == v) for k, v in CLASS.items() if v != 0}
    glom_masks = {gid: (glom_lbl == gid) for gid in range(1, n_inst + 1)}

    if attraction_dilation_px > 0:
        attraction = np.zeros_like(glom_lbl, dtype=bool)
        for gid, m in glom_masks.items():
            dil = ndimage.binary_dilation(m, iterations=attraction_dilation_px)
            attraction = attraction | dil
        outside = ~(attraction | (glom_lbl > 0))
        dil_labels = ndimage.binary_dilation(
            glom_lbl > 0,
            iterations=attraction_dilation_px,
            mask=None,
            border_value=0,
            origin=0,
            brute_force=False,
        ).astype(np.int32)
        if outside.any():
            filled = glom_lbl.copy()
            for _ in range(attraction_dilation_px):
                dil_filled = ndimage.maximum_filter(filled, size=3)
                filled = np.where(filled > 0, filled, dil_filled)
            glom_lbl_attr = filled
        else:
            glom_lbl_attr = glom_lbl
    else:
        glom_lbl_attr = glom_lbl

    for gid in range(1, n_inst + 1):
        strict = glom_masks[gid]
        attr = glom_lbl_attr == gid
        gpix = int(strict.sum())
        if gpix < min_glomerulus_pixels:
            continue
        cell = int((masks['mesangial_cell'] & strict).sum())
        scl = int((masks['sclerosis'] & strict).sum())
        lesion = int((masks['mesangial_lesion'] & attr).sum())
        crsc = int((masks['crescent'] & attr).sum())
        endo = int((masks['endothelial_cell'] & attr).sum())
        scl_ratio = scl / max(gpix, 1)
        out.append(PerGlomerulus(
            glom_id=gid,
            glom_pixels=gpix,
            lesion_pixels=lesion,
            cell_pixels=cell,
            sclerosis_pixels=scl,
            crescent_pixels=crsc,
            endothelial_pixels=endo,
            is_global_sclerosis=(scl_ratio >= sclerosis_global_ratio),
        ))
    return out


def estimate_cortex_area(
    seg: np.ndarray,
    dilation_px: int = 200,
) -> int:
    """Approximate the cortical area in pixels.

    The heuristic:
      * Build a "kidney tissue" mask as anything not background.
      * Build a "core" mask from glomerulus + mesangial_cell + mesangial_lesion
        + endothelial_cell, take the largest connected component of core
        (the "best" cortex piece), dilate it by ``dilation_px`` and count
        the intersection with kidney tissue.
      * If there is no core at all, fall back to the whole kidney.
    """
    kidney = seg != CLASS['background']
    if not kidney.any():
        return 0
    core = (
        (seg == CLASS['glomerulus'])
        | (seg == CLASS['mesangial_cell'])
        | (seg == CLASS['mesangial_lesion'])
        | (seg == CLASS['endothelial_cell'])
    )
    if not core.any():
        return int(kidney.sum())
    lbl, n = ndimage.label(core, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return int(kidney.sum())
    sizes = ndimage.sum(core, lbl, range(1, n + 1), index=range(1, n + 1))
    big_id = int(np.argmax(sizes)) + 1
    big = lbl == big_id
    big_dil = ndimage.binary_dilation(big, iterations=dilation_px)
    return int((big_dil & kidney).sum())