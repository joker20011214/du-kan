#!/usr/bin/env python3
"""
系膜增生智能分析工具
- 合并系膜区及其内部的系膜细胞
- 统计每个系膜区的系膜细胞数量
- 正常(≤3细胞): m0, 增生(>3细胞): m1
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os
import argparse
import json
from pathlib import Path

# 标签定义
LABEL_NAMES = {
    0: 'background',
    2: 'sclerosis',
    3: 'mesangial_lesion',   # 系膜区
    5: 'mesangial_cell',      # 系膜细胞
    6: 'glomerulus',
    7: 'endothelial_cell',
}

# 增生标签
MESANGIAL_HYPERPLASIA_LABELS = {
    'm0': 'normal',           # 正常: ≤3 细胞
    'm1': 'hyperplasia',      # 增生: >3 细胞
}

HYPERPLASIA_THRESHOLD = 3  # 系膜增生阈值


def analyze_mesangial_regions(label_img, dilation_iters=2, min_lesion_size=500, min_cell_size=50):
    """
    分析系膜区和系膜细胞的包含关系
    
    Args:
        label_img: 标签图像
        dilation_iters: 膨胀迭代次数
        min_lesion_size: 最小系膜区面积阈值
        min_cell_size: 最小系膜细胞面积阈值
    
    Returns:
        dict: 分析结果
    """
    lesion_mask = (label_img == 3).astype(np.int32)  # 系膜区
    cell_mask = (label_img == 5).astype(np.int32)   # 系膜细胞
    
    struct = np.ones((3, 3), np.int32)
    
    # 膨胀系膜区以包含相邻的系膜细胞
    dilated_lesion = ndimage.binary_dilation(lesion_mask, structure=struct, iterations=dilation_iters).astype(int)
    merged = np.maximum(dilated_lesion, cell_mask)
    merged_labeled, n_regions = ndimage.label(merged, structure=struct)
    
    # 统计每个合并区域
    regions_info = []
    for r in range(1, n_regions + 1):
        region_mask = merged_labeled == r
        lesion_in = np.sum(region_mask & lesion_mask)  # 系膜区像素
        cell_in_region = region_mask & cell_mask
        
        # 计算系膜细胞连通域数
        n_cell_regions = ndimage.label(cell_in_region, structure=struct)[1]
        
        # 判断是否为有效系膜区（有系膜区核心 且 有系膜细胞）
        if lesion_in >= min_lesion_size and n_cell_regions > 0:
            is_hyperplasia = n_cell_regions > HYPERPLASIA_THRESHOLD
            
            regions_info.append({
                'region_id': r,
                'mesangial_lesion_pixels': int(lesion_in),
                'mesangial_cell_count': n_cell_regions,
                'is_hyperplasia': is_hyperplasia,
                'label': 'm1' if is_hyperplasia else 'm0'
            })
    
    return regions_info


def create_labeled_output(regions_info, output_shape):
    """
    创建标记后的输出图像
    
    Args:
        regions_info: 系膜区分析结果
        output_shape: 输出图像形状
    
    Returns:
        numpy array: 标记后的图像
    """
    output = np.zeros(output_shape, dtype=np.uint8)
    
    for info in regions_info:
        label = 1 if info['is_hyperplasia'] else 0  # m1=1, m0=0
        # 存储每个区域的标签（实际应用中需要更复杂的掩膜映射）
    
    return output


def analyze_dataset(labels_dir, output_dir=None, dilation_iters=2, 
                   min_lesion_size=500, min_cell_size=50, first_n=None, image_list=None):
    """分析整个数据集"""
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    # 如果指定了图片列表，只分析这些图片
    if image_list:
        files = [f for f in files if f in image_list]
    elif first_n:
        files = files[:first_n]
    
    results = []
    all_cell_counts = []
    
    print("=" * 100)
    print("系膜增生分析报告")
    print("=" * 100)
    print(f"数据目录: {labels_dir}")
    print(f"图像数量: {len(files)}")
    print(f"增生阈值: 每个系膜区 > {HYPERPLASIA_THRESHOLD} 个系膜细胞")
    print("=" * 100)
    print()
    
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        regions = analyze_mesangial_regions(img, dilation_iters, min_lesion_size, min_cell_size)
        
        # 统计
        total_regions = len(regions)
        total_cells = sum(r['mesangial_cell_count'] for r in regions)
        normal_count = sum(1 for r in regions if not r['is_hyperplasia'])
        hyperplasia_count = sum(1 for r in regions if r['is_hyperplasia'])
        
        avg_cells = total_cells / total_regions if total_regions > 0 else 0
        hyperplasia_ratio = hyperplasia_count / total_regions * 100 if total_regions > 0 else 0
        
        # 收集细胞数分布
        for r in regions:
            all_cell_counts.append(r['mesangial_cell_count'])
        
        # 诊断分级
        if hyperplasia_ratio > 40:
            diagnosis = "【重度增生】"
        elif hyperplasia_ratio > 20:
            diagnosis = "【轻度增生】"
        elif hyperplasia_ratio > 0:
            diagnosis = "【基本正常】"
        else:
            diagnosis = "【完全正常】"
        
        result = {
            'image': f,
            'total_mesangial_regions': total_regions,
            'total_mesangial_cells': total_cells,
            'avg_cells_per_region': round(avg_cells, 2),
            'normal_regions_m0': normal_count,
            'hyperplasia_regions_m1': hyperplasia_count,
            'hyperplasia_ratio': round(hyperplasia_ratio, 1),
            'diagnosis': diagnosis,
            'regions_detail': [
                {
                    'region_id': r['region_id'],
                    'cell_count': r['mesangial_cell_count'],
                    'label': r['label'],
                    'is_hyperplasia': r['is_hyperplasia']
                }
                for r in regions
            ]
        }
        results.append(result)
        
        print(f"{f:<22} | 系膜区:{total_regions:>3} | 细胞:{total_cells:>4} | 均值:{avg_cells:>5.2f} | "
              f"m0(正常):{normal_count:>3} | m1(增生):{hyperplasia_count:>3} | 增生率:{hyperplasia_ratio:>5.1f}% {diagnosis}")
    
    # 汇总统计
    print()
    print("=" * 100)
    print("汇总统计")
    print("=" * 100)
    
    total_regions = sum(r['total_mesangial_regions'] for r in results)
    total_cells = sum(r['total_mesangial_cells'] for r in results)
    total_m0 = sum(r['normal_regions_m0'] for r in results)
    total_m1 = sum(r['hyperplasia_regions_m1'] for r in results)
    
    print(f"总图像数:    {len(files)}")
    print(f"总系膜区数:  {total_regions}")
    print(f"总系膜细胞:  {total_cells}")
    print(f"m0 (正常):  {total_m0} ({total_m0/total_regions*100:.1f}%)")
    print(f"m1 (增生):  {total_m1} ({total_m1/total_regions*100:.1f}%)")
    print(f"平均每图像系膜区数: {total_regions/len(files):.2f}")
    print(f"平均每系膜区细胞数: {total_cells/total_regions:.2f}" if total_regions > 0 else "N/A")
    
    # 细胞数分布
    print()
    print("每个系膜区系膜细胞数分布:")
    cell_distribution = {}
    for n in all_cell_counts:
        cell_distribution[n] = cell_distribution.get(n, 0) + 1
    
    for n in sorted(cell_distribution.keys()):
        count = cell_distribution[n]
        pct = count / len(all_cell_counts) * 100
        bar = '█' * int(pct / 2)
        print(f"  {n:>2} 细胞: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # 保存JSON结果
    output_data = {
        'summary': {
            'total_images': len(files),
            'total_mesangial_regions': total_regions,
            'total_mesangial_cells': total_cells,
            'm0_normal_count': total_m0,
            'm1_hyperplasia_count': total_m1,
            'm0_ratio': round(total_m0 / total_regions * 100, 1) if total_regions > 0 else 0,
            'm1_ratio': round(total_m1 / total_regions * 100, 1) if total_regions > 0 else 0,
            'hyperplasia_threshold': HYPERPLASIA_THRESHOLD,
            'cell_distribution': {str(k): v for k, v in sorted(cell_distribution.items())}
        },
        'images': results
    }
    
    if output_dir:
        output_file = os.path.join(output_dir, 'mesangial_hyperplasia_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    return output_data


def show_single_image_analysis(labels_dir, image_name, dilation_iters=2):
    """显示单张图像的详细分析"""
    img_path = os.path.join(labels_dir, image_name)
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return
    
    img = np.array(Image.open(img_path))
    regions = analyze_mesangial_regions(img, dilation_iters)
    
    print(f"\n{'='*80}")
    print(f"图像: {image_name}")
    print(f"{'='*80}")
    
    print(f"\n各系膜区详情:")
    print("-" * 60)
    print(f"{'区ID':>6} | {'系膜细胞数':>10} | {'标签':>6} | {'诊断':>12}")
    print("-" * 60)
    
    m0_count = 0
    m1_count = 0
    for r in sorted(regions, key=lambda x: x['mesangial_cell_count'], reverse=True):
        label = r['label']
        diagnosis = "增生" if r['is_hyperplasia'] else "正常"
        if label == 'm0':
            m0_count += 1
        else:
            m1_count += 1
        print(f"{r['region_id']:>6} | {r['mesangial_cell_count']:>10} | {label:>6} | {diagnosis:>12}")
    
    print("-" * 60)
    print(f"\n汇总:")
    print(f"  总系膜区数: {len(regions)}")
    print(f"  m0 (正常 ≤{HYPERPLASIA_THRESHOLD}细胞): {m0_count}")
    print(f"  m1 (增生 >{HYPERPLASIA_THRESHOLD}细胞): {m1_count}")
    print(f"  增生率: {m1_count/len(regions)*100:.1f}%" if regions else "N/A")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='系膜增生智能分析工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str, default='mesangial_analysis_output',
                       help='输出目录')
    parser.add_argument('--dilation', type=int, default=2,
                       help='膨胀迭代次数，默认2')
    parser.add_argument('--min_lesion_size', type=int, default=500,
                       help='最小系膜区面积阈值（像素），默认500')
    parser.add_argument('--min_cell_size', type=int, default=50,
                       help='最小系膜细胞面积阈值（像素），默认50')
    parser.add_argument('--first_n', type=int, default=None,
                       help='只分析前N张')
    parser.add_argument('--single', type=str, default=None,
                       help='分析单张图像')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                       help='分析指定的多张图像，如: --images KIDNEY_0001.png KIDNEY_0002.png')
    
    args = parser.parse_args()
    
    if args.single:
        show_single_image_analysis(args.labels_dir, args.single, args.dilation)
    elif args.images:
        analyze_dataset(args.labels_dir, args.output_dir, args.dilation,
                      args.min_lesion_size, args.min_cell_size, None, args.images)
    else:
        analyze_dataset(args.labels_dir, args.output_dir, args.dilation,
                      args.min_lesion_size, args.min_cell_size, args.first_n)
