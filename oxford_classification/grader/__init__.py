"""Oxford MEST-C grading framework.

This package provides:

  * ``oxford_rules`` - pure scoring functions (grade_M, grade_E, grade_S,
    grade_T, grade_C, grade_case) and data classes.
  * ``instance_utils`` - glomerulus instance segmentation and
    per-instance pixel attribution.
  * ``grader`` - end-to-end single-slide ``grade_one`` helper.
  * ``count_labels`` - quick per-class pixel counter utility.
  * ``report`` - visualization card renderer (matplotlib + PIL fallback).
"""
from .oxford_rules import (
    OxfordScore,
    PerGlomerulus,
    grade_C,
    grade_E,
    grade_M,
    grade_S,
    grade_T,
    grade_case,
    load_thresholds,
)

__all__ = [
    'OxfordScore',
    'PerGlomerulus',
    'grade_M',
    'grade_E',
    'grade_S',
    'grade_T',
    'grade_C',
    'grade_case',
    'load_thresholds',
]