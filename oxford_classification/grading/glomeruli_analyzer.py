"""
肾小球连通区域分析模块 - 提取肾小球数量、面积等特征
"""

import numpy as np
from scipy import ndimage
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ConnectedComponent:
    """连通区域信息"""
    label_id: int
    component_id: int
    pixel_count: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    area: float
    equivalent_diameter: float

    def to_dict(self) -> dict:
        return {
            'label_id': self.label_id,
            'component_id': self.component_id,
            'pixel_count': self.pixel_count,
            'centroid': self.centroid,
            'bbox': self.bbox,
            'area': self.area,
            'equivalent_diameter': self.equivalent_diameter,
        }


@dataclass
class GlomerulusAnalysis:
    """肾小球分析结果"""
    image_name: str
    image_shape: Tuple[int, int]
    total_glomeruli: int
    glomeruli_areas: List[int]
    glomeruli_diameters: List[float]
    glomeruli_centroids: List[Tuple[float, float]]
    normal_glomeruli: int
    sclerotic_glomeruli: int
    crescent_glomeruli: int
    sclerosis_pixels: int
    sclerosis_connected_regions: int
    focal_sclerosis_count: int
    global_sclerosis_count: int
    mesangial_lesion_pixels: int
    mesangial_cell_pixels: int
    mesangial_ratio: float
    other_components: Dict[int, List[ConnectedComponent]] = field(default_factory=dict)
    glomeruli_details: List[ConnectedComponent] = field(default_factory=list)

    def to_dict(self) -> dict:
        g = [c.to_dict() for c in self.glomeruli_details]
        return {
            'image_name': self.image_name,
            'image_shape': self.image_shape,
            'total_glomeruli': self.total_glomeruli,
            'glomeruli_areas': self.glomeruli_areas,
            'glomeruli_diameters': self.glomeruli_diameters,
            'glomeruli_centroids': self.glomeruli_centroids,
            'normal_glomeruli': self.normal_glomeruli,
            'sclerotic_glomeruli': self.sclerotic_glomeruli,
            'crescent_glomeruli': self.crescent_glomeruli,
            'sclerosis_pixels': self.sclerosis_pixels,
            'sclerosis_connected_regions': self.sclerosis_connected_regions,
            'focal_sclerosis_count': self.focal_sclerosis_count,
            'global_sclerosis_count': self.global_sclerosis_count,
            'mesangial_lesion_pixels': self.mesangial_lesion_pixels,
            'mesangial_cell_pixels': self.mesangial_cell_pixels,
            'mesangial_ratio': self.mesangial_ratio,
            'glomeruli_details': g,
        }

    def summary(self) -> str:
        return '\n'.join([
            f"图像: {self.image_name}",
            f"尺寸: {self.image_shape}",
            "==================================================",
            "【肾小球统计】",
            f"  总数: {self.total_glomeruli}",
            f"  正常: {self.normal_glomeruli}",
            f"  硬化: {self.sclerotic_glomeruli}",
            f"  新月体: {self.crescent_glomeruli}",
            "",
            "【硬化区域】",
            f"  像素数: {self.sclerosis_pixels:,}",
            f"  节段性硬化: {self.focal_sclerosis_count}",
            f"  全球性硬化: {self.global_sclerosis_count}",
            "",
            "【系膜区】",
            f"  系膜区病变像素: {self.mesangial_lesion_pixels:,}",
            f"  系膜细胞像素: {self.mesangial_cell_pixels:,}",
            f"  系膜/肾小球比例: {self.mesangial_ratio:.4f}",
        ])


class GlomeruliAnalyzer:
    """
    肾小球分析器 - 分析肾小球数量、大小、分布及关联病变
    """

    def __init__(
        self,
        min_glomerulus_size: int = 5000,
        sclerosis_threshold: float = 0.5,
        glomerulus_label: int = 6,
        sclerosis_label: int = 2,
        crescent_label: int = 1,
    ):
        """
        初始化肾小球分析器

        Args:
            min_glomerulus_size: 最小肾小球像素数 (过滤噪声)
            sclerosis_threshold: 硬化阈值 (肾小球内硬化比例超过此值视为全球硬化)
            glomerulus_label: 肾小球标签值
            sclerosis_label: 硬化标签值
            crescent_label: 新月体标签值
        """
        self.min_glomerulus_size = min_glomerulus_size
        self.sclerosis_threshold = sclerosis_threshold
        self.glomerulus_label = glomerulus_label
        self.sclerosis_label = sclerosis_label
        self.crescent_label = crescent_label

    def _find_connected_components(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        查找连通区域

        Args:
            binary_mask: 二值掩码

        Returns:
            Tuple: (标记后的数组, 连通区域数量)
        """
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(binary_mask, structure=structure)
        return labeled_array, num_features

    def _get_component_info(
        self,
        labeled_array: np.ndarray,
        component_id: int,
    ) -> Optional[ConnectedComponent]:
        """
        获取单个连通区域的信息

        Args:
            labeled_array: 标记后的数组
            component_id: 连通区域ID

        Returns:
            ConnectedComponent: 区域信息
        """
        mask = (labeled_array == component_id)
        pixel_count = int(np.sum(mask))
        if pixel_count < self.min_glomerulus_size:
            return None
        rows, cols = np.where(mask)
        centroid = (float(np.mean(rows)), float(np.mean(cols)))
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
        area = pixel_count
        equivalent_diameter = 2 * np.sqrt(area / np.pi)
        label_id = int(labeled_array[labeled_array == component_id][0]) if component_id > 0 else 0
        return ConnectedComponent(
            label_id=label_id,
            component_id=component_id,
            pixel_count=pixel_count,
            centroid=centroid,
            bbox=bbox,
            area=area,
            equivalent_diameter=equivalent_diameter,
        )

    def _classify_glomerulus(
        self,
        glomerulus_mask: np.ndarray,
        label_array: np.ndarray,
        glomerulus_component_id: int,
    ) -> str:
        """
        分类肾小球类型

        Args:
            glomerulus_mask: 肾小球掩码
            label_array: 完整标签数组
            glomerulus_component_id: 肾小球连通区域ID

        Returns:
            str: 'normal', 'sclerotic', 或 'crescent'
        """
        glomerulus_area = label_array[glomerulus_mask]
        has_sclerosis = np.any(glomerulus_area == self.sclerosis_label)
        has_crescent = np.any(glomerulus_area == self.crescent_label)
        if has_crescent:
            return 'crescent'
        if has_sclerosis:
            sclerosis_in_glomerulus = np.sum(glomerulus_area == self.sclerosis_label)
            total_area = len(glomerulus_area)
            sclerosis_ratio = sclerosis_in_glomerulus / total_area
            if sclerosis_ratio >= self.sclerosis_threshold:
                return 'sclerotic'
            return 'focal_sclerosis'
        return 'normal'

    def _calculate_sclerosis_overlap(
        self,
        glomerulus_mask: np.ndarray,
        sclerosis_mask: np.ndarray,
    ) -> Tuple[int, int, float]:
        """
        计算肾小球与硬化区域的重叠

        Args:
            glomerulus_mask: 肾小球掩码
            sclerosis_mask: 硬化掩码

        Returns:
            Tuple: (重叠像素数, 肾小球面积, 重叠比例)
        """
        overlap = glomerulus_mask & sclerosis_mask
        overlap_count = int(np.sum(overlap))
        glomerulus_area = int(np.sum(glomerulus_mask))
        overlap_ratio = overlap_count / glomerulus_area if glomerulus_area > 0 else 0
        return overlap_count, glomerulus_area, overlap_ratio

    def analyze_single(self, label_path: str) -> GlomerulusAnalysis:
        """
        分析单个标签图像

        Args:
            label_path: 标签图像路径

        Returns:
            GlomerulusAnalysis: 分析结果
        """
        label_img = Image.open(label_path)
        label_array = np.array(label_img)
        image_name = Path(label_path).stem
        image_shape = label_array.shape[:2]
        glomerulus_mask = (label_array == self.glomerulus_label)
        sclerosis_mask = (label_array == self.sclerosis_label)
        mesangial_lesion_mask = (label_array == 3)
        mesangial_cell_mask = (label_array == 5)

        labeled_glomeruli, num_glomeruli = self._find_connected_components(glomerulus_mask)
        glomeruli_details = []
        glomeruli_areas = []
        glomeruli_diameters = []
        glomeruli_centroids = []
        normal_count = 0
        sclerotic_count = 0
        crescent_count = 0
        for comp_id in range(1, num_glomeruli + 1):
            comp_info = self._get_component_info(labeled_glomeruli, comp_id)
            if comp_info is None:
                continue
            comp_mask = labeled_glomeruli == comp_id
            gtype = self._classify_glomerulus(comp_mask, label_array, comp_id)
            if gtype == 'normal':
                normal_count += 1
            elif gtype == 'sclerotic':
                sclerotic_count += 1
            elif gtype == 'crescent':
                crescent_count += 1
            glomeruli_details.append(comp_info)
            glomeruli_areas.append(comp_info.pixel_count)
            glomeruli_diameters.append(comp_info.equivalent_diameter)
            glomeruli_centroids.append(comp_info.centroid)

        labeled_sclerosis, num_sclerosis = self._find_connected_components(sclerosis_mask)
        focal_sclerosis = 0
        global_sclerosis = 0
        for comp_id in range(1, num_sclerosis + 1):
            sclerosis_comp_mask = (labeled_sclerosis == comp_id)
            overlap_with_glomeruli = sclerosis_comp_mask & glomerulus_mask
            if not np.any(overlap_with_glomeruli):
                continue
            overlap_pixels = int(np.sum(overlap_with_glomeruli))
            sclerosis_area = int(np.sum(sclerosis_comp_mask))
            overlap_ratio = overlap_pixels / sclerosis_area if sclerosis_area > 0 else 0
            if overlap_ratio >= 0.8:
                global_sclerosis += 1
            else:
                focal_sclerosis += 1

        mesangial_lesion_pixels = int(np.sum(mesangial_lesion_mask))
        mesangial_cell_pixels = int(np.sum(mesangial_cell_mask))
        total_glomerulus_pixels = int(np.sum(glomerulus_mask))
        mesangial_ratio = (mesangial_lesion_pixels + mesangial_cell_pixels) / total_glomerulus_pixels \
            if total_glomerulus_pixels > 0 else 0

        other_components = {}
        for label_id in (1, 3, 4, 5, 7):
            if label_id in (self.glomerulus_label, self.sclerosis_label):
                continue
            mask = (label_array == label_id)
            if not np.any(mask):
                continue
            labeled, num = self._find_connected_components(mask)
            components = []
            for comp_id in range(1, num + 1):
                comp_info = self._get_component_info(labeled, comp_id)
                if comp_info:
                    components.append(comp_info)
            if components:
                other_components[label_id] = components

        return GlomerulusAnalysis(
            image_name=image_name,
            image_shape=image_shape,
            total_glomeruli=len(glomeruli_details),
            glomeruli_areas=glomeruli_areas,
            glomeruli_diameters=glomeruli_diameters,
            glomeruli_centroids=glomeruli_centroids,
            normal_glomeruli=normal_count,
            sclerotic_glomeruli=sclerotic_count,
            crescent_glomeruli=crescent_count,
            sclerosis_pixels=int(np.sum(sclerosis_mask)),
            sclerosis_connected_regions=num_sclerosis,
            focal_sclerosis_count=focal_sclerosis,
            global_sclerosis_count=global_sclerosis,
            mesangial_lesion_pixels=mesangial_lesion_pixels,
            mesangial_cell_pixels=mesangial_cell_pixels,
            mesangial_ratio=mesangial_ratio,
            other_components=other_components,
            glomeruli_details=glomeruli_details,
        )

    def analyze_batch(self, label_paths: List[str]) -> Dict[str, GlomerulusAnalysis]:
        """
        批量分析多个标签图像

        Args:
            label_paths: 标签图像路径列表

        Returns:
            Dict[str, GlomerulusAnalysis]: 以图像名为键的分析结果
        """
        results = {}
        for path in label_paths:
            try:
                result = self.analyze_single(path)
                results[result.image_name] = result
            except Exception as e:
                print(f"分析 {path} 时出错: {e}")
        return results

    def get_glomerulus_statistics(self, results: Dict[str, GlomerulusAnalysis]) -> Dict:
        """
        获取所有图像的肾小球统计汇总

        Args:
            results: 分析结果字典

        Returns:
            Dict: 汇总统计
        """
        total_images = len(results)
        total_glomeruli = sum(r.total_glomeruli for r in results.values())
        total_normal = sum(r.normal_glomeruli for r in results.values())
        total_sclerotic = sum(r.sclerotic_glomeruli for r in results.values())
        total_crescent = sum(r.crescent_glomeruli for r in results.values())
        all_areas = []
        all_diameters = []
        for r in results.values():
            all_areas.extend(r.glomeruli_areas)
            all_diameters.extend(r.glomeruli_diameters)
        return {
            'total_images': total_images,
            'total_glomeruli': total_glomeruli,
            'mean_glomeruli_per_image': total_glomeruli / total_images if total_images > 0 else 0,
            'normal_glomeruli': total_normal,
            'sclerotic_glomeruli': total_sclerotic,
            'crescent_glomeruli': total_crescent,
            'mean_area': float(np.mean(all_areas)) if all_areas else 0,
            'std_area': float(np.std(all_areas)) if all_areas else 0,
            'mean_diameter': float(np.mean(all_diameters)) if all_diameters else 0,
            'std_diameter': float(np.std(all_diameters)) if all_diameters else 0,
        }


def quick_analyze(label_path: str) -> GlomerulusAnalysis:
    """
    快速分析单个标签图像

    Args:
        label_path: 标签图像路径

    Returns:
        GlomerulusAnalysis: 分析结果
    """
    analyzer = GlomeruliAnalyzer()
    return analyzer.analyze_single(label_path)