"""High-level entry point: extract Oxford features for a directory of labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List

from .label_extractor import LabelExtractor, LabelStatistics
from .glomeruli_analyzer import GlomeruliAnalyzer, GlomerulusAnalysis
from .oxford_grader import OxfordGrader, OxfordGrade
from .visualizer import OxfordVisualizer


def extract_oxford_features(
    labels_dir: str,
    image_names: Optional[List[str]] = None,
    output_dir: str = './oxford_output',
    generate_visualizations: bool = True,
    generate_report: bool = True,
) -> dict:
    """Extract all Oxford features for a directory of label PNGs.

    Steps:
      1. LabelExtractor.extract_all()
      2. GlomeruliAnalyzer.analyze_batch()
      3. OxfordGrader.grade_from_analysis()
      4. Save JSON + (optional) visualizations and summary report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    extractor = LabelExtractor(labels_dir)
    analyzer = GlomeruliAnalyzer()
    grader = OxfordGrader()
    visualizer = OxfordVisualizer(output_dir=output_dir) if generate_visualizations else None

    files = sorted(Path(labels_dir).glob('*.png'))
    if image_names:
        files = [Path(labels_dir) / f"{n}.png" for n in image_names if (Path(labels_dir) / f"{n}.png").exists()]

    label_stats_map = {}
    glomerulus_map = {}
    grade_map = {}
    for i, label_path in enumerate(files):
        stem = label_path.stem
        try:
            ls = extractor.extract_single(str(label_path))
            ga = analyzer.analyze_single(str(label_path))
            grade = grader.grade_from_analysis(
                ga,
                atrophy_pixels=ls.label_counts.get(4, 0),
                total_pixels=ls.total_pixels,
            )
            label_stats_map[stem] = ls
            glomerulus_map[stem] = ga
            grade_map[stem] = grade
            print(f"  [{i + 1}/{len(files)}] {stem}: {grade.summary}")
        except Exception as e:
            print(f"  [{i + 1}/{len(files)}] {stem}: ERROR {e}")

    if generate_visualizations and visualizer is not None:
        try:
            visualizer.visualize_batch_comparison(
                label_stats_map,
                str(Path(output_dir) / 'batch_comparison.png'),
            )
        except Exception:
            pass
        try:
            visualizer.visualize_oxford_grades(
                grade_map,
                str(Path(output_dir) / 'oxford_grades_distribution.png'),
            )
        except Exception:
            pass
        for stem, ga in glomerulus_map.items():
            try:
                visualizer.visualize_glomeruli_heatmap(
                    ga,
                    output_path=str(Path(output_dir) / f'{stem}_glomeruli_heatmap.png'),
                )
            except Exception:
                pass
            try:
                visualizer.visualize_label_distribution(
                    label_stats_map[stem],
                )
                plt.savefig(str(Path(output_dir) / f'{stem}_label_distribution.png'))
                plt.close()
            except Exception:
                pass

    if generate_report and visualizer is not None:
        try:
            visualizer.create_summary_report(
                grade_map,
                str(Path(output_dir) / 'oxford_report.txt'),
            )
        except Exception:
            pass

    try:
        visualizer.save_results_to_json(
            grade_map,
            str(Path(output_dir) / 'oxford_results.json'),
        )
    except Exception:
        pass

    return {
        'grades': {k: g.to_dict() for k, g in grade_map.items()},
        'label_stats': {k: s.to_dict() for k, s in label_stats_map.items()},
        'glomeruli': {k: g.to_dict() for k, g in glomerulus_map.items()},
    }


def quick_analysis(label_path: str) -> dict:
    """Quick single-image analysis."""
    p = Path(label_path)
    extractor = LabelExtractor(str(p.parent))
    analyzer = GlomeruliAnalyzer()
    grader = OxfordGrader()
    ls = extractor.extract_single(label_path)
    ga = analyzer.analyze_single(label_path)
    grade = grader.grade_from_analysis(
        ga,
        atrophy_pixels=ls.label_counts.get(4, 0),
        total_pixels=ls.total_pixels,
    )
    return {
        'label_stats': ls.to_dict(),
        'glomeruli': ga.to_dict(),
        'grade': grade.to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--labels-dir', type=str, required=True,
                    help='Directory containing <case>.png label maps.')
    ap.add_argument('--images', nargs='+', default=None,
                    help='Restrict to specific image stems.')
    ap.add_argument('--output-dir', type=str, default='./oxford_output')
    ap.add_argument('--no-visualization', action='store_true')
    ap.add_argument('--no-report', action='store_true')
    args = ap.parse_args()
    extract_oxford_features(
        labels_dir=args.labels_dir,
        image_names=args.images,
        output_dir=args.output_dir,
        generate_visualizations=not args.no_visualization,
        generate_report=not args.no_report,
    )


if __name__ == '__main__':
    main()