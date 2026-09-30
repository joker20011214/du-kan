"""Oxford MEST-C grader - dataclasses and OxfordGrader class."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class OxfordScore(str, Enum):
    M0 = 'M0'; M1 = 'M1'
    E0 = 'E0'; E1 = 'E1'
    S0 = 'S0'; S1 = 'S1'
    T0 = 'T0'; T1 = 'T1'; T2 = 'T2'
    C0 = 'C0'; C1 = 'C1'; C2 = 'C2'


@dataclass
class OxfordGrade:
    image_name: str
    M_score: str
    E_score: str
    S_score: str
    T_score: str
    C_score: str
    interpretation: str
    risk_level: float
    total_glomeruli: int = 0
    grade_details: Dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return f"M:{self.M_score} E:{self.E_score} S:{self.S_score} T:{self.T_score} C:{self.C_score}"

    def to_dict(self) -> dict:
        return {
            'image_name': self.image_name,
            'M': self.M_score, 'E': self.E_score, 'S': self.S_score,
            'T': self.T_score, 'C': self.C_score,
            'summary': self.summary,
            'interpretation': self.interpretation,
            'risk_level': self.risk_level,
            'total_glomeruli': self.total_glomeruli,
            'grade_details': self.grade_details,
        }


class OxfordGrader:
    """Compute MEST-C scores from pre-extracted glomerulus statistics."""

    def __init__(
        self,
        m_threshold: float = 0.25,
        e_threshold: float = 0.0125,
        s_threshold: float = 0.10,
        t0_max: float = 0.25,
        t1_max: float = 0.50,
    ) -> None:
        self.m_threshold = m_threshold
        self.e_threshold = e_threshold
        self.s_threshold = s_threshold
        self.t0_max = t0_max
        self.t1_max = t1_max

    def calculate_M_score(self, analysis) -> str:
        """M0 if mesangial_ratio < threshold, else M1."""
        ratio = getattr(analysis, 'mesangial_ratio', 0.0)
        return 'M1' if ratio > self.m_threshold else 'M0'

    def calculate_E_score(self, analysis) -> str:
        """E1 if any glomerulus has endothelial cells + mesangial_lesion."""
        other = getattr(analysis, 'other_components', {}) or {}
        endo = other.get(7, [])
        mes = other.get(3, [])
        return 'E1' if (endo and mes) else 'E0'

    def calculate_S_score(self, analysis) -> str:
        """S1 if any segmental sclerosis detected."""
        return 'S1' if getattr(analysis, 'focal_sclerosis_count', 0) > 0 else 'S0'

    def calculate_T_score(self, atrophy_pixels: int, total_pixels: int) -> str:
        r = atrophy_pixels / total_pixels if total_pixels > 0 else 0
        if r < self.t0_max:
            return 'T0'
        if r <= self.t1_max:
            return 'T1'
        return 'T2'

    def calculate_C_score(self, analysis) -> str:
        return 'C1' if getattr(analysis, 'crescent_glomeruli', 0) > 0 else 'C0'

    def assess_risk(self, grade: OxfordGrade) -> float:
        risk = 0.0
        if grade.M_score == 'M1': risk += 0.25
        if grade.E_score == 'E1': risk += 0.25
        if grade.S_score == 'S1': risk += 0.25
        if grade.T_score == 'T1': risk += 0.15
        if grade.T_score == 'T2': risk += 0.30
        if grade.C_score == 'C1': risk += 0.10
        if grade.C_score == 'C2': risk += 0.30
        return min(risk, 1.0)

    def interpret_grade(self, grade: OxfordGrade) -> str:
        risk = self.assess_risk(grade)
        if risk < 0.2:
            return '低风险 - 病理改变轻微,预后良好'
        if risk < 0.5:
            return '中风险 - 需要定期随访'
        if risk < 0.8:
            return '高风险 - 建议积极治疗'
        return '极高风险 - 需要立即干预'

    def grade_from_analysis(
        self,
        analysis,
        atrophy_pixels: Optional[int] = None,
        total_pixels: Optional[int] = None,
    ) -> OxfordGrade:
        image_name = getattr(analysis, 'image_name', 'unknown')
        total_glomeruli = getattr(analysis, 'total_glomeruli', 0)
        M = self.calculate_M_score(analysis)
        E = self.calculate_E_score(analysis)
        S = self.calculate_S_score(analysis)
        if atrophy_pixels is None or total_pixels is None:
            T = 'T0'
        else:
            T = self.calculate_T_score(atrophy_pixels, total_pixels)
        C = self.calculate_C_score(analysis)
        grade = OxfordGrade(
            image_name=image_name,
            M_score=M, E_score=E, S_score=S, T_score=T, C_score=C,
            interpretation='', risk_level=0.0, total_glomeruli=total_glomeruli,
        )
        grade.risk_level = self.assess_risk(grade)
        grade.interpretation = self.interpret_grade(grade)
        return grade

    def grade_from_dict(self, label_stats: Dict) -> OxfordGrade:
        return self.grade_from_analysis(label_stats)


def quick_grade(label_stats) -> OxfordGrade:
    grader = OxfordGrader()
    return grader.grade_from_dict(label_stats)