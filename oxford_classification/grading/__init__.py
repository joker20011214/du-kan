"""
Oxford Grading Framework - 牛津分级辅助提取框架

该框架用于从肾脏病理图像分割结果中提取标签统计信息,
辅助进行Oxford肾脏病理分级。

标签编码:
    0: background    - 背景
    1: crescent      - 新月体
    2: sclerosis     - 硬化
    3: mesangial_lesion  - 系膜区病变
    4: tubular_atrophy   - 肾小管萎缩
    5: mesangial_cell    - 系膜细胞
    6: glomerulus       - 肾小球
    7: endothelial_cell  - 内皮细胞
"""

from .label_extractor import LabelExtractor, LabelStatistics
from .glomeruli_analyzer import GlomeruliAnalyzer, GlomerulusAnalysis
from .oxford_grader import OxfordGrader, OxfordGrade
from .visualizer import OxfordVisualizer
from .main import extract_oxford_features
from .statistics_table import (
    generate_statistics_table,
    generate_statistics_table_with_pixels,
)


def quick_extract(path):
    """兼容接口 - 实际调用label_extractor中的函数"""
    from .label_extractor import quick_extract as _quick_extract
    return _quick_extract(path)


__version__ = '1.0.0'