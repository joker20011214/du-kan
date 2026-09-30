"""
Oxford Classification Integration - 牛津分级集成包
=====================================================

整合了所有与 Oxford MEST-C 分级相关的代码,统一对外暴露接口。

目录结构:

    oxford_classification/
    ├── __init__.py                  # 顶层包入口(本文件)
    │
    ├── analyzers/                   # 单一指标独立分析器(4 个)
    │   ├── mesangial_hyperplasia_analyzer.py
    │   ├── endothelial_hyperplasia_analyzer.py
    │   ├── sclerosis_crescent_analyzer.py
    │   └── tubular_atrophy_analyzer.py
    │
    ├── smart_counters/              # 智能统计工具(2 个)
    │   ├── smart_glomeruli_counter.py
    │   └── smart_mesangial_counter.py
    │
    ├── grader/                      # 完整牛津分级核心库(从 .pyc 还原)
    │   ├── oxford_rules.py          # MEST-C 分级规则 + 数据类
    │   ├── grader.py                # 单切片分级主流程
    │   ├── instance_utils.py        # 肾小球实例分析
    │   ├── count_labels.py          # 标签统计(CLI)
    │   ├── report.py                # 视觉卡片生成
    │   └── tests/
    │       └── smoke_test_grade.py
    │
    ├── grading/                     # 高层封装(从 .pyc 还原)
    │   ├── label_extractor.py
    │   ├── glomeruli_analyzer.py
    │   ├── oxford_grader.py
    │   ├── visualizer.py
    │   ├── main.py
    │   └── statistics_table.py
    │
    ├── configs/
    │   └── thresholds.yaml
    │
    ├── run_oxford_classification.py # 综合分级入口脚本
    └── README.md

使用方式:

    # 顶层导入
    import sys; sys.path.insert(0, '/path/to/Swin-UMamba-main')
    import oxford_classification as oc

    # 单切片 MEST-C 评分
    from oxford_classification import PerGlomerulus, grade_case, load_thresholds
    thr = load_thresholds('oxford_classification/configs/thresholds.yaml')
    record = grade_case(per_glomerulus_list, thr)

    # 批量特征提取
    oc.extract_oxford_features(
        labels_dir='...',
        output_dir='./oxford_output',
    )
"""

__version__ = '1.0.0'

try:
    from .grader.oxford_rules import (
        OxfordScore,
        PerGlomerulus,
        grade_M,
        grade_E,
        grade_S,
        grade_T,
        grade_C,
        grade_case,
        load_thresholds,
    )
except Exception:  # pragma: no cover
    pass

try:
    from .grading.oxford_grader import OxfordGrader, OxfordGrade
    from .grading.glomeruli_analyzer import (
        GlomeruliAnalyzer,
        GlomerulusAnalysis,
        ConnectedComponent,
    )
    from .grading.label_extractor import LabelExtractor, LabelStatistics
    from .grading.visualizer import OxfordVisualizer
    from .grading.main import extract_oxford_features, quick_analysis
    from .grading.statistics_table import (
        generate_statistics_table,
        generate_statistics_table_with_pixels,
    )
except Exception:  # pragma: no cover
    pass


__all__ = [
    '__version__',
    'OxfordScore',
    'PerGlomerulus',
    'grade_M', 'grade_E', 'grade_S', 'grade_T', 'grade_C',
    'grade_case', 'load_thresholds',
    'OxfordGrader', 'OxfordGrade',
    'GlomeruliAnalyzer', 'GlomerulusAnalysis', 'ConnectedComponent',
    'LabelExtractor', 'LabelStatistics',
    'OxfordVisualizer',
    'extract_oxford_features', 'quick_analysis',
    'generate_statistics_table', 'generate_statistics_table_with_pixels',
]