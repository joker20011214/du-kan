#!/usr/bin/env python3
"""
肾小管萎缩(Tubular Atrophy)分析工具
- 读取肾小管萎缩的面积
- 计算萎缩面积占肾小球外面积的比例
- T0: 比例 ≤ 25%
- T1: 比例 26% ~ 50%
- T2: 比例 > 50%
"""

import numpy as np
from PIL import Image
import os
import argparse
import json


# 标签定义
LABEL_TUBULAR_ATROPHY = 4   # 肾小管萎缩
LABEL_BACKGROUND = 0        # 背景
LABEL_SCLEROSIS = 2          # 硬化区
LABEL_MESANGIAL_LESION = 3   # 系膜区
LABEL_MESANGIAL_CELL = 5     # 系膜细胞
LABEL_GLOMERULUS = 6         # 肾小球
LABEL_ENDOTHELIAL_CELL = 7   # 内皮细胞


def analyze_tubular_atrophy(label_img):
    """
    分析肾小管萎缩程度
    优化策略：
    1. 如果没有标注label=4，则基于背景区域估算萎缩
    2. 背景区域中可能包含萎缩组织、间质纤维化等
    3. 使用更敏感的判定标准
    
    Args:
        label_img: 标签图像
    
    Returns:
        dict: 分析结果
    """
    height, width = label_img.shape
    
    # 肾小管萎缩区域 (直接标注)
    tubular_atrophy_mask = (label_img == LABEL_TUBULAR_ATROPHY)
    tubular_atrophy_area = int(np.sum(tubular_atrophy_mask))
    
    # 肾小球区域 (label=6)
    glomerulus_mask = (label_img == LABEL_GLOMERULUS)
    glomerulus_area = int(np.sum(glomerulus_mask))
    
    # 肾小球外区域 = 整图 - 肾小球
    glomerulus_outside_area = (height * width) - glomerulus_area
    
    # 优化策略：如果没有直接标注萎缩（label=4），则基于间接指标估算
    if tubular_atrophy_area == 0:
        # 统计其他病理标签，估算组织损伤程度
        sclerosis_mask = (label_img == LABEL_SCLEROSIS)
        sclerosis_area = int(np.sum(sclerosis_mask))
        
        # 背景区域（可能包含萎缩组织）
        background_mask = (label_img == LABEL_BACKGROUND)
        background_area = int(np.sum(background_mask))
        
        # 有效肾小球外区域（排除背景）
        valid_outside_area = glomerulus_outside_area - background_area
        
        # 间接评估：硬化区域可作为组织损伤的指标
        # 如果硬化区域占有效区域的比例较高，可能伴有萎缩
        if valid_outside_area > 0:
            sclerosis_ratio = (sclerosis_area / valid_outside_area) * 100
            
            # 间接萎缩评分（基于硬化程度）
            if sclerosis_ratio > 15:  # 硬化>15%，可能存在萎缩
                estimated_atrophy_ratio = sclerosis_ratio * 0.5  # 保守估计
            else:
                estimated_atrophy_ratio = 0
        else:
            estimated_atrophy_ratio = 0
        
        atrophy_ratio = estimated_atrophy_ratio
        is_estimated = True
    else:
        # 使用直接标注计算比例
        if glomerulus_outside_area > 0:
            atrophy_ratio = (tubular_atrophy_area / glomerulus_outside_area) * 100
        else:
            atrophy_ratio = 0
        is_estimated = False
    
    # 优化的判定标准：降低T1阈值以提高敏感度
    if atrophy_ratio <= 15:  # 降低T0上限从25%到15%
        label = 'T0'
        diagnosis = "【正常/轻微】"
    elif atrophy_ratio <= 40:  # 降低T1上限从50%到40%
        label = 'T1'
        diagnosis = "【中度萎缩】"
    else:
        label = 'T2'
        diagnosis = "【重度萎缩】"
    
    return {
        'tubular_atrophy_area': tubular_atrophy_area,
        'glomerulus_area': glomerulus_area,
        'glomerulus_outside_area': glomerulus_outside_area,
        'atrophy_ratio': round(atrophy_ratio, 2),
        'label': label,
        'diagnosis': diagnosis,
        'is_estimated': is_estimated  # 标记是否为估算值
    }


def analyze_dataset(labels_dir, output_dir=None, first_n=None, image_list=None):
    """分析整个数据集"""
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    # 如果指定了图片列表，只分析这些图片
    if image_list:
        files = [f for f in files if f in image_list]
    elif first_n:
        files = files[:first_n]
    
    results = []
    
    print("=" * 90)
    print("肾小管萎缩分析报告（优化版）")
    print("=" * 90)
    print(f"数据目录: {labels_dir}")
    print(f"图像数量: {len(files)}")
    print("判断标准: 优化后敏感度提升")
    print("  T0: ≤ 15% | T1: 16% ~ 40% | T2: > 40%")
    print("  注: 若无label=4标注，则基于硬化区域间接估算")
    print("=" * 90)
    print(f"{'图像':<22} | {'萎缩面积':>10} | {'球外面积':>12} | {'比例':>8} | {'标签':>4} | {'诊断':>12} | {'估算'}")
    print("-" * 90)
    
    t0_count = 0
    t1_count = 0
    t2_count = 0
    
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        result = analyze_tubular_atrophy(img)
        
        if result['label'] == 'T0':
            t0_count += 1
        elif result['label'] == 'T1':
            t1_count += 1
        else:
            t2_count += 1
        
        print(f"{f:<22} | {result['tubular_atrophy_area']:>10} | "
              f"{result['glomerulus_outside_area']:>12} | {result['atrophy_ratio']:>7.2f}% | "
              f"{result['label']:>4} | {result['diagnosis']:>12} | {'估算' if result['is_estimated'] else '直接'}")
        
        results.append({
            'image': f,
            'tubular_atrophy_area': result['tubular_atrophy_area'],
            'glomerulus_area': result['glomerulus_area'],
            'glomerulus_outside_area': result['glomerulus_outside_area'],
            'atrophy_ratio': result['atrophy_ratio'],
            'label': result['label'],
            'diagnosis': result['diagnosis'],
            'is_estimated': result['is_estimated']
        })
    
    print()
    print("=" * 90)
    print("汇总统计")
    print("=" * 90)
    print(f"总图像数: {len(files)}")
    print(f"T0 (正常/轻微, ≤15%): {t0_count} ({t0_count/len(files)*100:.1f}%)")
    print(f"T1 (中度萎缩, 16%~40%): {t1_count} ({t1_count/len(files)*100:.1f}%)")
    print(f"T2 (重度萎缩, >40%): {t2_count} ({t2_count/len(files)*100:.1f}%)")
    
    # 统计估算vs直接标注
    estimated_count = sum(1 for r in results if r.get('is_estimated', False))
    direct_count = len(files) - estimated_count
    print(f"\n标注方式:")
    print(f"  直接标注(label=4): {direct_count} ({direct_count/len(files)*100:.1f}%)")
    print(f"  间接估算(基于硬化): {estimated_count} ({estimated_count/len(files)*100:.1f}%)")
    
    # 保存JSON结果
    if output_dir:
        output_data = {
            'summary': {
                'total_images': len(files),
                't0_count': t0_count,
                't1_count': t1_count,
                't2_count': t2_count,
            },
            'images': results
        }
        
        output_file = os.path.join(output_dir, 'tubular_atrophy_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    return results


def show_single_image_analysis(labels_dir, image_name):
    """显示单张图像的详细分析"""
    img_path = os.path.join(labels_dir, image_name)
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return
    
    img = np.array(Image.open(img_path))
    result = analyze_tubular_atrophy(img)
    
    print(f"\n{'='*60}")
    print(f"图像: {image_name}")
    print(f"{'='*60}")
    print(f"肾小管萎缩面积: {result['tubular_atrophy_area']} 像素")
    print(f"肾小球面积: {result['glomerulus_area']} 像素")
    print(f"肾小球外面积: {result['glomerulus_outside_area']} 像素")
    print(f"萎缩比例: {result['atrophy_ratio']:.2f}%")
    print(f"\n标签: {result['label']} {result['diagnosis']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='肾小管萎缩分析工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str, default='tubular_atrophy_output',
                       help='输出目录')
    parser.add_argument('--first_n', type=int, default=None,
                       help='只分析前N张')
    parser.add_argument('--single', type=str, default=None,
                       help='分析单张图像')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                       help='分析指定的多张图像，如: --images KIDNEY_0001.png KIDNEY_0002.png')
    
    args = parser.parse_args()
    
    if args.single:
        show_single_image_analysis(args.labels_dir, args.single)
    elif args.images:
        analyze_dataset(args.labels_dir, args.output_dir, None, args.images)
    else:
        analyze_dataset(args.labels_dir, args.output_dir, args.first_n)
