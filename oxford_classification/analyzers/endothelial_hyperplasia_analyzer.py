#!/usr/bin/env python3
"""
内皮增生智能分析工具
- 统计每个肾小球(glomerulus)内的内皮细胞数量
- 正常肾小球内皮细胞正常，异常时内皮细胞增生
- 正常标 e0, 增生标 e1
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os
import argparse
import json

# 标签定义
LABEL_NAMES = {
    0: 'background',
    2: 'sclerosis',        # 硬化区
    3: 'mesangial_lesion', # 系膜区
    5: 'mesangial_cell',   # 系膜细胞
    6: 'glomerulus',       # 肾小球
    7: 'endothelial_cell', # 内皮细胞
}

# 增生标签
ENDOTHELIAL_HYPERPLASIA_LABELS = {
    'e0': 'normal',        # 正常: 内皮细胞面积/肾小球面积 正常
    'e1': 'hyperplasia',   # 增生: 内皮细胞面积异常增多
}

# 增生阈值: 内皮细胞面积占肾小球面积的比例
HYPERPLASIA_RATIO_THRESHOLD = 0.015  # 1.5% (优化后：提高阈值以降低假阳性)


def analyze_endothelial_in_glomeruli(label_img, min_glomerulus_size=5000, 
                                     min_endothelial_size=100, min_ratio=0.01,
                                     dilation_iters=3):
    """
    分析每个肾小球内的内皮细胞
    
    Args:
        label_img: 标签图像
        min_glomerulus_size: 最小肾小球面积阈值
        min_endothelial_size: 最小内皮细胞面积阈值
        min_ratio: 最小内皮/肾小球比例阈值（低于此值认为无有效内皮）
        dilation_iters: 膨胀迭代次数
    
    Returns:
        list: 每个肾小球的分析结果
    """
    glomerulus_mask = (label_img == 6).astype(np.int32)  # 肾小球
    endothelial_mask = (label_img == 7).astype(np.int32)  # 内皮细胞
    
    struct = np.ones((3, 3), np.int32)
    
    # 膨胀内皮细胞以匹配肾小球边界
    dilated_endothelial = ndimage.binary_dilation(endothelial_mask, structure=struct, 
                                                iterations=dilation_iters).astype(int)
    
    # 标记肾小球连通域
    glomerulus_labeled, n_glomeruli = ndimage.label(glomerulus_mask, structure=struct)
    
    results = []
    for g_id in range(1, n_glomeruli + 1):
        g_mask = glomerulus_labeled == g_id
        g_area = np.sum(g_mask)  # 肾小球面积
        
        if g_area < min_glomerulus_size:
            continue
        
        # 统计该肾小球内的内皮细胞（膨胀后的内皮与肾小球求交）
        e_in_g = dilated_endothelial & g_mask
        e_area = np.sum(e_in_g)
        
        # 计算内皮/肾小球比例
        ratio = e_area / g_area if g_area > 0 else 0
        
        # 判断是否为有效内皮（面积达标）
        has_endothelial = e_area >= min_endothelial_size
        
        # 判断是否增生（基于比例阈值）
        is_hyperplasia = ratio > HYPERPLASIA_RATIO_THRESHOLD
        
        results.append({
            'glomerulus_id': g_id,
            'glomerulus_area': int(g_area),
            'endothelial_area': int(e_area),
            'endothelial_ratio': round(ratio, 4),
            'has_endothelial': bool(has_endothelial),
            'is_hyperplasia': bool(is_hyperplasia),
            'label': 'e1' if is_hyperplasia else 'e0'
        })
    
    return results


def analyze_dataset(labels_dir, output_dir=None, min_glomerulus_size=5000,
                   min_endothelial_size=100, min_ratio=0.01, first_n=None, image_list=None):
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
    all_ratios = []
    
    print("=" * 100)
    print("内皮增生分析报告")
    print("=" * 100)
    print(f"数据目录: {labels_dir}")
    print(f"图像数量: {len(files)}")
    print(f"增生阈值: 内皮细胞面积 / 肾小球面积 > {HYPERPLASIA_RATIO_THRESHOLD*100:.1f}%")
    print("=" * 100)
    print()
    
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        glomeruli = analyze_endothelial_in_glomeruli(img, min_glomerulus_size,
                                                     min_endothelial_size, min_ratio)
        
        # 统计
        total_glomeruli = len(glomeruli)
        total_e_area = sum(g['endothelial_area'] for g in glomeruli)
        total_g_area = sum(g['glomerulus_area'] for g in glomeruli)
        avg_ratio = total_e_area / total_g_area if total_g_area > 0 else 0
        
        normal_count = sum(1 for g in glomeruli if not g['is_hyperplasia'])
        hyperplasia_count = sum(1 for g in glomeruli if g['is_hyperplasia'])
        
        hyperplasia_ratio = hyperplasia_count / total_glomeruli * 100 if total_glomeruli > 0 else 0
        
        # 收集比例分布 (肾小球级别)
        for g in glomeruli:
            all_ratios.append(g['endothelial_ratio'])
        
        # 图像级别诊断：严格双条件判断
        # 条件1: 图像平均内皮比例 > 阈值
        # 条件2: 至少有增生的肾小球（内皮比例>阈值）
        image_is_hyperplasia = (avg_ratio > HYPERPLASIA_RATIO_THRESHOLD and 
                               hyperplasia_count > 0)
        image_label = 'e1' if image_is_hyperplasia else 'e0'
        
        # 诊断分级 (基于图像级别比例)
        if avg_ratio > 0.02:  # > 2%
            diagnosis = "【重度增生】"
        elif avg_ratio > HYPERPLASIA_RATIO_THRESHOLD:  # > 1.5%
            diagnosis = "【轻度增生】"
        else:
            diagnosis = "【正常】"
        
        result = {
            'image': f,
            'total_glomeruli': total_glomeruli,
            'total_endothelial_area': total_e_area,
            'total_glomerular_area': total_g_area,
            'avg_endothelial_ratio': round(avg_ratio, 4),
            'normal_glomeruli_e0': normal_count,
            'hyperplasia_glomeruli_e1': hyperplasia_count,
            'hyperplasia_ratio': round(hyperplasia_ratio, 1),
            'image_is_hyperplasia': bool(image_is_hyperplasia),
            'image_label': image_label,
            'diagnosis': diagnosis,
            'glomeruli_detail': [
                {
                    'glomerulus_id': g['glomerulus_id'],
                    'endothelial_area': g['endothelial_area'],
                    'glomerular_area': g['glomerulus_area'],
                    'ratio': g['endothelial_ratio'],
                    'label': g['label'],
                    'is_hyperplasia': bool(g['is_hyperplasia'])
                }
                for g in glomeruli
            ]
        }
        results.append(result)
        
        print(f"{f:<22} | 肾小球:{total_glomeruli:>3} | 内皮面积:{total_e_area:>8} | "
              f"平均比:{avg_ratio*100:>5.2f}% | 图像标签:{image_label} {diagnosis}")
    
    # 汇总统计
    print()
    print("=" * 100)
    print("汇总统计")
    print("=" * 100)
    
    total_g = sum(r['total_glomeruli'] for r in results)
    total_e = sum(r['total_endothelial_area'] for r in results)
    total_g_area = sum(r['total_glomerular_area'] for r in results)
    total_e0 = sum(r['normal_glomeruli_e0'] for r in results)
    total_e1 = sum(r['hyperplasia_glomeruli_e1'] for r in results)
    
    # 图像级别统计
    total_img_e0 = sum(1 for r in results if r['image_label'] == 'e0')
    total_img_e1 = sum(1 for r in results if r['image_label'] == 'e1')
    
    print(f"总图像数:      {len(files)}")
    print(f"总肾小球数:    {total_g}")
    print(f"总内皮面积:    {total_e}")
    print(f"总肾小球面积:  {total_g_area}")
    print(f"图像级 - e0 (正常): {total_img_e0} ({total_img_e0/len(files)*100:.1f}%)")
    print(f"图像级 - e1 (增生): {total_img_e1} ({total_img_e1/len(files)*100:.1f}%)")
    print(f"肾小球级 - e0: {total_e0} ({total_e0/total_g*100:.1f}%)" if total_g > 0 else "肾小球级 - e0: N/A")
    print(f"肾小球级 - e1: {total_e1} ({total_e1/total_g*100:.1f}%)" if total_g > 0 else "肾小球级 - e1: N/A")
    
    # 比例分布 (图像级别平均内皮比例)
    if all_ratios:
        print()
        print("图像平均内皮/肾小球比例分布:")
        ratio_bins = [(0, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1.0)]
        for low, high in ratio_bins:
            count = sum(1 for r in all_ratios if low <= r < high)
            pct = count / len(all_ratios) * 100
            bar = '█' * int(pct / 2)
            print(f"  {low*100:>5.2f}%-{high*100:>5.2f}%: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # 保存JSON结果
    output_data = {
        'summary': {
            'total_images': len(files),
            'total_glomeruli': total_g,
            'total_endothelial_area': total_e,
            'total_glomerular_area': total_g_area,
            'e0_normal_count': total_e0,
            'e1_hyperplasia_count': total_e1,
            'hyperplasia_threshold_ratio': HYPERPLASIA_RATIO_THRESHOLD,
        },
        'images': results
    }
    
    if output_dir:
        output_file = os.path.join(output_dir, 'endothelial_hyperplasia_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")
    
    return output_data


def show_single_image_analysis(labels_dir, image_name, min_glomerulus_size=5000,
                               min_endothelial_size=100, min_ratio=0.01):
    """显示单张图像的详细分析"""
    img_path = os.path.join(labels_dir, image_name)
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return
    
    img = np.array(Image.open(img_path))
    glomeruli = analyze_endothelial_in_glomeruli(img, min_glomerulus_size,
                                                 min_endothelial_size, min_ratio)
    
    print(f"\n{'='*80}")
    print(f"图像: {image_name}")
    print(f"{'='*80}")
    
    print(f"\n各肾小球内内皮细胞详情:")
    print("-" * 80)
    print(f"{'肾小球ID':>10} | {'肾小球面积':>12} | {'内皮面积':>10} | {'比例':>8} | {'标签':>6} | {'诊断':>12}")
    print("-" * 80)
    
    e0_count = 0
    e1_count = 0
    for g in sorted(glomeruli, key=lambda x: x['endothelial_ratio'], reverse=True):
        diagnosis = "增生" if g['is_hyperplasia'] else "正常"
        if g['label'] == 'e0':
            e0_count += 1
        else:
            e1_count += 1
        print(f"{g['glomerulus_id']:>10} | {g['glomerulus_area']:>12} | {g['endothelial_area']:>10} | "
              f"{g['endothelial_ratio']*100:>7.2f}% | {g['label']:>6} | {diagnosis:>12}")
    
    print("-" * 80)
    print(f"\n汇总:")
    print(f"  总肾小球数: {len(glomeruli)}")
    print(f"  e0 (正常, 比例≤{HYPERPLASIA_RATIO_THRESHOLD*100:.1f}%): {e0_count}")
    print(f"  e1 (增生, 比例>{HYPERPLASIA_RATIO_THRESHOLD*100:.1f}%): {e1_count}")
    print(f"  增生率: {e1_count/len(glomeruli)*100:.1f}%" if glomeruli else "N/A")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='内皮增生智能分析工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str, default='endothelial_analysis_output',
                       help='输出目录')
    parser.add_argument('--min_glomerulus_size', type=int, default=5000,
                       help='最小肾小球面积阈值（像素），默认5000')
    parser.add_argument('--min_endothelial_size', type=int, default=100,
                       help='最小内皮细胞面积阈值（像素），默认100')
    parser.add_argument('--min_ratio', type=float, default=0.01,
                       help='最小内皮/肾小球比例阈值，默认0.01 (1%)')
    parser.add_argument('--threshold', type=float, default=0.005,
                       help=f'增生阈值（内皮/肾小球比例），默认0.0125 (1.25%)')
    parser.add_argument('--first_n', type=int, default=None,
                       help='只分析前N张')
    parser.add_argument('--single', type=str, default=None,
                       help='分析单张图像')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                       help='分析指定的多张图像，如: --images KIDNEY_0001.png KIDNEY_0002.png')
    
    args = parser.parse_args()
    
    # 设置全局阈值
    HYPERPLASIA_RATIO_THRESHOLD = args.threshold
    
    if args.single:
        show_single_image_analysis(args.labels_dir, args.single, 
                                  args.min_glomerulus_size, 
                                  args.min_endothelial_size, args.min_ratio)
    elif args.images:
        analyze_dataset(args.labels_dir, args.output_dir, args.min_glomerulus_size,
                       args.min_endothelial_size, args.min_ratio, None, args.images)
    else:
        analyze_dataset(args.labels_dir, args.output_dir, args.min_glomerulus_size,
                       args.min_endothelial_size, args.min_ratio, args.first_n)
