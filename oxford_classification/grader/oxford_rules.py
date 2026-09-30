"""MEST-C scoring rules (pure functions, no I/O).

Each rule consumes:
  * a list of PerGlomerulus statistics
  * the threshold config dict
  * optionally, the slice-level atrophy / cortex pixel counts

and returns the Oxford letter for that dimension.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PerGlomerulus:
    """Per-glomerulus aggregated statistics from a segmentation map."""
    glom_id: int
    glom_pixels: int
    lesion_pixels: int = 0
    cell_pixels: int = 0
    sclerosis_pixels: int = 0
    crescent_pixels: int = 0
    endothelial_pixels: int = 0
    is_global_sclerosis: bool = False

    @property
    def lesion_ratio(self) -> float:
        return self.lesion_pixels / max(self.glom_pixels, 1)

    @property
    def sclerosis_ratio(self) -> float:
        return self.sclerosis_pixels / max(self.glom_pixels, 1)

    @property
    def cell_density(self) -> float:
        return self.cell_pixels / max(self.glom_pixels, 1)

    @property
    def crescent_ratio(self) -> float:
        return self.crescent_pixels / max(self.glom_pixels, 1)

    @property
    def endothelial_ratio(self) -> float:
        return self.endothelial_pixels / max(self.glom_pixels, 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OxfordScore:
    """Slice-level Oxford MEST-C score."""
    M: str
    E: str
    S: str
    T: str
    C: str
    summary: str
    confidence: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_thresholds(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['oxford']


def grade_M(glomeruli: List[PerGlomerulus], thr: Dict[str, Any]) -> str:
    """Mesangial hypercellularity.

    The slice-level M is taken as the worst among non-globally-sclerosed
    glomeruli.  Any single glomerulus satisfying either the lesion-area
    ratio or the cell-density threshold bumps the slice to M1.
    """
    lesion_thr = thr['mesangial']['lesion_ratio_threshold']
    cell_thr = thr['mesangial']['cell_density_threshold']
    for g in glomeruli:
        if g.is_global_sclerosis:
            continue
        if g.lesion_ratio >= lesion_thr or g.cell_density >= cell_thr:
            return 'M1'
    return 'M0'


def grade_E(glomeruli: List[PerGlomerulus], thr: Dict[str, Any]) -> str:
    """Endocapillary hypercellularity (slice-level).
    
    优化策略：
    1. 降低require_copresent_lesion的强制要求（提高敏感度）
    2. 增加内皮比例阈值判断
    """
    require_copl = thr['endocapillary'].get('require_copresent_lesion', False)
    min_endo_ratio = thr['endocapillary'].get('min_endothelial_ratio', 0.015)
    
    for g in glomeruli:
        if g.endothelial_pixels == 0:
            continue
        
        # 检查内皮比例是否超过阈值
        if g.endothelial_ratio >= min_endo_ratio:
            # 如果不强制要求共存病变，直接判定E1
            if not require_copl:
                return 'E1'
            # 如果强制要求，则需要同时有系膜病变
            if g.lesion_pixels > 0:
                return 'E1'
    
    return 'E0'


def grade_S(glomeruli: List[PerGlomerulus], thr: Dict[str, Any]) -> str:
    """Segmental sclerosis (slice-level)."""
    smin = thr['sclerosis']['segmental_min_ratio']
    gmax = thr['sclerosis']['global_sclerosis_ratio']
    for g in glomeruli:
        if g.is_global_sclerosis:
            continue
        if smin <= g.sclerosis_ratio < gmax:
            return 'S1'
    return 'S0'


def grade_T(atrophy_pixels: int, cortex_pixels: int, thr: Dict[str, Any]) -> str:
    """Tubular atrophy / interstitial fibrosis (slice-level ratio).
    
    优化策略：降低阈值以提高敏感度
    """
    if cortex_pixels <= 0:
        return 'T0'
    r = atrophy_pixels / cortex_pixels
    
    # 使用优化后的阈值
    t0_max = thr['tubular_atrophy'].get('t0_max', 0.15)  # 默认15%
    t1_max = thr['tubular_atrophy'].get('t1_max', 0.40)  # 默认40%
    
    if r < t0_max:
        return 'T0'
    if r <= t1_max:
        return 'T1'
    return 'T2'


def grade_C(glomeruli: List[PerGlomerulus], thr: Dict[str, Any]) -> str:
    """Crescents (slice-level, fraction of all glomeruli)."""
    n_total = sum(1 for g in glomeruli if g.glom_pixels > 0)
    n_cresc = sum(1 for g in glomeruli if g.crescent_pixels > 0)
    if n_cresc == 0:
        return 'C0'
    if n_cresc >= thr['crescent']['c1_min_glomeruli'] and \
            n_cresc / max(n_total, 1) < thr['crescent']['c2_min_ratio']:
        return 'C1'
    return 'C2'


def grade_case(
    glomeruli: List[PerGlomerulus],
    atrophy_pixels: int,
    cortex_pixels: int,
    thresholds: Optional[Dict[str, Any]] = None,
    thresholds_path: Optional[str] = None,
) -> OxfordScore:
    """Compute the full MEST-C score for a single slide."""
    if thresholds is None:
        if thresholds_path is None:
            raise ValueError("Either `thresholds` or `thresholds_path` is required.")
        thresholds = load_thresholds(thresholds_path)
    M = grade_M(glomeruli, thresholds)
    E = grade_E(glomeruli, thresholds)
    S = grade_S(glomeruli, thresholds)
    T = grade_T(atrophy_pixels, cortex_pixels, thresholds)
    C = grade_C(glomeruli, thresholds)
    n_total = sum(1 for g in glomeruli if g.glom_pixels > 0)
    n_global_scl = sum(1 for g in glomeruli if g.is_global_sclerosis)
    n_seg_scl = sum(
        1 for g in glomeruli
        if thresholds['sclerosis']['segmental_min_ratio']
           <= g.sclerosis_ratio
           < thresholds['sclerosis']['global_sclerosis_ratio']
    )
    n_cresc = sum(1 for g in glomeruli if g.crescent_pixels > 0)
    n_endo = sum(
        1 for g in glomeruli
        if g.endothelial_pixels > 0 and g.lesion_pixels > 0
    )
    conf = 'high'
    if n_total < thresholds['confidence']['min_total_glomeruli']:
        conf = 'low'
    if cortex_pixels < thresholds['confidence']['min_cortex_area_px']:
        conf = 'low'
    detail = {
        'n_glomeruli_total': n_total,
        'n_glomeruli_global_sclerosis': n_global_scl,
        'n_glomeruli_segmental_sclerosis': n_seg_scl,
        'n_glomeruli_crescent': n_cresc,
        'n_glomeruli_endocapillary': n_endo,
        'atrophy_pixels': atrophy_pixels,
        'cortex_pixels': cortex_pixels,
        'atrophy_ratio': atrophy_pixels / max(cortex_pixels, 1),
    }
    return OxfordScore(
        M=M, E=E, S=S, T=T, C=C,
        summary=f"M:{M} E:{E} S:{S} T:{T} C:{C}",
        confidence=conf,
        detail=detail,
    )