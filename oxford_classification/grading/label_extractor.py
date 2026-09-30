"""
标签提取模块 - 基础标签像素级统计
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


LABEL_NAMES = {
    0: 'background',
    1: 'crescent',
    2: 'sclerosis',
    3: 'mesangial_lesion',
    4: 'tubular_atrophy',
    5: 'mesangial_cell',
    6: 'glomerulus',
    7: 'endothelial_cell',
}

LABEL_NAMES_CN = {
    0: '背景',
    1: '新月体',
    2: '硬化',
    3: '系膜区病变',
    4: '肾小管萎缩',
    5: '系膜细胞',
    6: '肾小球',
    7: '内皮细胞',
}


@dataclass
class PixelStats:
    """像素级统计结果"""
    label_id: int
    label_name: str
    label_name_cn: str
    pixel_count: int
    area_percentage: float
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabelStatistics:
    """图像标签统计结果"""
    image_name: str
    image_shape: Tuple[int, int]
    total_pixels: int
    label_stats: List[PixelStats]
    label_counts: Dict[int, int] = None

    def to_dict(self) -> dict:
        result = {
            'image_name': self.image_name,
            'image_shape': self.image_shape,
            'total_pixels': self.total_pixels,
            'label_stats': [s.to_dict() for s in self.label_stats],
        }
        if self.label_counts is not None:
            result['label_counts'] = self.label_counts
        return result

    def summary(self) -> str:
        return '\n'.join([
            f"图像: {self.image_name}",
            f"尺寸: {self.image_shape}",
            f"总像素: {self.total_pixels:,}",
        ] + [
            f"  {s.label_name} ({s.label_name_cn}): {s.pixel_count:,} px ({s.area_percentage:.2f}%)"
            for s in self.label_stats
        ])


class LabelExtractor:
    """标签提取器"""

    def __init__(self, labels_dir: str):
        self.labels_dir = Path(labels_dir)
        self._verify_labels_dir()

    def _verify_labels_dir(self):
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")

    def extract_single(self, label_path: str) -> LabelStatistics:
        """Extract statistics for a single label PNG."""
        path = Path(label_path)
        if not path.exists():
            raise FileNotFoundError(f"标签文件不存在: {label_path}")
        img = np.array(Image.open(path))
        total = img.size
        h, w = img.shape[:2]
        stats = []
        counts = {}
        for label_id, name in LABEL_NAMES.items():
            mask = (img == label_id)
            count = int(mask.sum())
            counts[label_id] = count
            bbox = None
            if count > 0:
                rows, cols = np.where(mask)
                bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
            stats.append(PixelStats(
                label_id=label_id,
                label_name=name,
                label_name_cn=LABEL_NAMES_CN[label_id],
                pixel_count=count,
                area_percentage=count / total * 100,
                bbox=bbox,
            ))
        return LabelStatistics(
            image_name=path.stem,
            image_shape=(h, w),
            total_pixels=total,
            label_stats=stats,
            label_counts=counts,
        )

    def extract_batch(self, label_paths: List[str]) -> Dict[str, LabelStatistics]:
        results = {}
        for p in label_paths:
            try:
                stats = self.extract_single(p)
                results[stats.image_name] = stats
            except Exception as e:
                print(f"处理 {p} 时出错: {e}")
        return results

    def extract_all(self) -> Dict[str, LabelStatistics]:
        files = sorted(self.labels_dir.glob('*.png'))
        return self.extract_batch([str(f) for f in files])

    def get_label_density(self, label_id: int, image_names: Optional[List[str]] = None) -> float:
        if image_names is None:
            stats = self.extract_all()
        else:
            stats = self.extract_batch([self.labels_dir / f"{n}.png" for n in image_names])
        densities = [s.label_counts.get(label_id, 0) / s.total_pixels for s in stats.values()]
        return sum(densities) / len(densities) if densities else 0.0

    def compare_images(self, image_names: List[str]) -> Dict:
        """Compare the label distributions of a set of images."""
        out = {}
        for name in image_names:
            path = self.labels_dir / f"{name}.png"
            if not path.exists():
                continue
            stats = self.extract_single(str(path))
            out[name] = stats.label_counts
        return out


def quick_extract(label_path: str) -> LabelStatistics:
    """Quick single-image extraction."""
    p = Path(label_path)
    extractor = LabelExtractor(str(p.parent))
    return extractor.extract_single(label_path)