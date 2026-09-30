"""Visualization helpers for Oxford grading results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Configure a CJK-capable font if available so Chinese labels render
# without producing UserWarnings. We try a list of common candidates
# and silently fall back to the default font if none are present.
for _cjk_font in (
    'Noto Sans CJK SC',
    'Noto Sans CJK',
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'SimHei',
    'PingFang SC',
    'Microsoft YaHei',
    'DejaVu Sans',
):
    try:
        matplotlib.rcParams['font.sans-serif'] = [_cjk_font]
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

from .label_extractor import LabelExtractor, LABEL_NAMES_CN, LabelStatistics
from .glomeruli_analyzer import GlomeruliAnalyzer, GlomerulusAnalysis
from .oxford_grader import OxfordGrader, OxfordGrade


class OxfordVisualizer:
    """Generate summary plots for the Oxford grading workflow."""

    def __init__(self, output_dir: str = './oxford_output') -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_label_distribution(self, stats: LabelStatistics) -> plt.Figure:
        """Bar chart of label pixel counts."""
        names = [s.label_name_cn for s in stats.label_stats]
        counts = [s.pixel_count for s in stats.label_stats]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, counts)
        ax.set_title(f"标签分布 - {stats.image_name}")
        ax.set_xlabel('标签')
        ax.set_ylabel('像素数')
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        return fig

    def visualize_batch_comparison(
        self,
        results: Dict[str, LabelStatistics],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        labels = list(LABEL_NAMES_CN.values())
        image_names = list(results.keys())
        n_images = len(image_names)
        n_labels = len(labels)
        data = np.zeros((n_labels, n_images))
        for j, name in enumerate(image_names):
            s = results[name]
            for i, label_id in enumerate(sorted(LABEL_NAMES_CN.keys())):
                data[i, j] = s.label_counts.get(label_id, 0) / max(s.total_pixels, 1) * 100
        fig, ax = plt.subplots(figsize=(max(8, n_images * 0.6), 6))
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(n_images))
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, label='占比 (%)')
        ax.set_title('批次标签分布对比')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    def visualize_oxford_grades(
        self,
        grades: Dict[str, OxfordGrade],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        letters = ['M', 'E', 'S', 'T', 'C']
        image_names = list(grades.keys())
        data = []
        for name in image_names:
            g = grades[name]
            row = [int(g.M_score[1]), int(g.E_score[1]),
                   int(g.S_score[1]),
                   int(g.T_score[1]) if g.T_score != 'T0' else 0,
                   int(g.C_score[1]) if g.C_score != 'C0' else 0]
            data.append(row)
        data = np.array(data)
        fig, ax = plt.subplots(figsize=(8, max(3, len(image_names) * 0.5)))
        ax.imshow(data, aspect='auto', cmap='Reds', vmin=0, vmax=2)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters)
        ax.set_yticks(range(len(image_names)))
        ax.set_yticklabels(image_names)
        for i in range(len(image_names)):
            for j in range(len(letters)):
                ax.text(j, i, str(data[i, j]), ha='center', va='center')
        ax.set_title('牛津分级热图')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    def visualize_glomeruli_heatmap(
        self,
        analysis: GlomerulusAnalysis,
        image_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 10))
        if image_path and Path(image_path).exists():
            img = np.array(Image.open(image_path))
            ax.imshow(img)
        for g in analysis.glomeruli_details:
            y0, x0, y1, x1 = g.bbox
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor='red', linewidth=1.5))
            ax.text(x0, y0, str(g.component_id), color='yellow', fontsize=8)
        ax.set_title(f"肾小球分布 - {analysis.image_name} (总数: {analysis.total_glomeruli})")
        ax.axis('off')
        if output_path:
            fig.savefig(output_path)
        return fig

    def create_summary_report(
        self,
        results: Dict[str, OxfordGrade],
        output_path: Optional[str] = None,
    ) -> str:
        lines = [
            "=" * 70,
            "牛津分级汇总报告",
            "=" * 70,
        ]
        for name, g in results.items():
            lines.extend([
                '',
                f"图像: {name}",
                f"  分级: {g.summary}",
                f"  风险等级: {g.risk_level:.2f}",
                f"  解读: {g.interpretation}",
                f"  肾小球总数: {g.total_glomeruli}",
            ])
        text = '\n'.join(lines)
        if output_path:
            Path(output_path).write_text(text, encoding='utf-8')
        return text

    def save_results_to_json(
        self,
        results: Dict[str, OxfordGrade],
        output_path: str,
    ) -> None:
        data = {k: g.to_dict() for k, g in results.items()}
        Path(output_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )


def visualize_label_overlay(
    image_path: str,
    label_path: str,
    save_path: Optional[str] = None,
    alpha: float = 0.5,
) -> plt.Figure:
    img = np.array(Image.open(image_path))
    lbl = np.array(Image.open(label_path))
    overlay = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    palette = {
        0: (0, 0, 0), 1: (255, 0, 0), 2: (255, 165, 0),
        3: (255, 255, 0), 4: (0, 255, 255), 5: (0, 0, 255),
        6: (0, 255, 0), 7: (255, 0, 255),
    }
    for k, color in palette.items():
        overlay[lbl == k] = color
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(overlay, alpha=alpha)
    patches = [mpatches.Patch(color=np.array(c) / 255, label=LABEL_NAMES_CN[k])
               for k, c in palette.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8)
    ax.set_title('Label Overlay')
    ax.axis('off')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig