#!/usr/bin/env python3
"""
硬化与新月体分析工具
- 检测图像中是否存在硬化区(label 2)和新月体(label 1)
- 若存在硬化或新月体，标记为1；否则标记为0
"""

import numpy as np
from PIL import Image
import os
import argparse
import json


def analyze_sclerosis_crescent(label_img):
    """
    分析图像中的硬化区和新月体
    
    Args:
        label_img: 标签图像
    
    Returns:
        dict: 分析结果
    """
    # 标签定义
    LABEL_SCLEROSIS = 2   # 硬化区
    LABEL_CRESCENT = 1    # 新月体
    
    # 检测硬化区
    sclerosis_mask = (label_img == LABEL_SCLEROSIS)
    sclerosis_pixels = int(np.sum(sclerosis_mask))
    has_sclerosis = sclerosis_pixels > 0
    
    # 检测新月体
    crescent_mask = (label_img == LABEL_CRESCENT)
    crescent_pixels = int(np.sum(crescent_mask))
    has_crescent = crescent_pixels > 0
    
    # 判断: 存在硬化或新月体则为1，否则为0
    has_lesion = has_sclerosis or has_crescent
    label = 1 if has_lesion else 0
    
    return {
        'has_sclerosis': has_sclerosis,
        'sclerosis_pixels': sclerosis_pixels,
        'has_crescent': has_crescent,
        'crescent_pixels': crescent_pixels,
        'has_lesion': has_lesion,
        'label': label
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
    
    print("=" * 80)
    print("硬化与新月体分析报告")
    print("=" * 80)
    print(f"数据目录: {labels_dir}")
    print(f"图像数量: {len(files)}")
    print("=" * 80)
    print(f"{'图像':<22} | {'硬化区':>10} | {'新月体':>10} | {'标签':>4} | {'诊断':<10}")
    print("-" * 80)
    
    lesion_count = 0
    normal_count = 0
    
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        result = analyze_sclerosis_crescent(img)
        
        if result['has_lesion']:
            lesion_count += 1
            diagnosis = "【有病变】"
        else:
            normal_count += 1
            diagnosis = "【无病变】"
        
        sclerosis_str = f"{result['sclerosis_pixels']:>8}px" if result['has_sclerosis'] else "-"
        crescent_str = f"{result['crescent_pixels']:>8}px" if result['has_crescent'] else "-"
        
        print(f"{f:<22} | {sclerosis_str:>10} | {crescent_str:>10} | {result['label']:>4} | {diagnosis}")
        
        results.append({
            'image': f,
            'sclerosis_pixels': result['sclerosis_pixels'],
            'has_sclerosis': result['has_sclerosis'],
            'crescent_pixels': result['crescent_pixels'],
            'has_crescent': result['has_crescent'],
            'has_lesion': result['has_lesion'],
            'label': result['label'],
            'diagnosis': diagnosis
        })
    
    # 汇总统计
    print()
    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"总图像数: {len(files)}")
    print(f"有病变 (label=1): {lesion_count} ({lesion_count/len(files)*100:.1f}%)")
    print(f"无病变 (label=0): {normal_count} ({normal_count/len(files)*100:.1f}%)")
    
    # 保存JSON结果
    if output_dir:
        output_data = {
            'summary': {
                'total_images': len(files),
                'lesion_count': lesion_count,
                'normal_count': normal_count,
                'lesion_ratio': round(lesion_count / len(files), 4) if len(files) > 0 else 0
            },
            'images': results
        }
        
        output_file = os.path.join(output_dir, 'sclerosis_crescent_analysis.json')
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
    result = analyze_sclerosis_crescent(img)
    
    print(f"\n{'='*60}")
    print(f"图像: {image_name}")
    print(f"{'='*60}")
    print(f"硬化区 (label=2): {result['sclerosis_pixels']} 像素 | {'有' if result['has_sclerosis'] else '无'}")
    print(f"新月体 (label=1): {result['crescent_pixels']} 像素 | {'有' if result['has_crescent'] else '无'}")
    print(f"\n标签: {result['label']} ({'有病变' if result['has_lesion'] else '无病变'})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='硬化与新月体分析工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str, default='sclerosis_analysis_output',
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
