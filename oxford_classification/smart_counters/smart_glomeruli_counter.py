#!/usr/bin/env python3
"""
肾小球智能统计工具
核心逻辑：以 glomerulus 为主体，合并所有内部结构（系膜区、系膜细胞、硬化、内皮细胞）
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os
import argparse
import json

# 肾小球组成部分（按病理优先级）
GLOMERULUS_PARTS = {
    6: ('glomerulus', 1.0),           # 肾小球 - 核心
    2: ('sclerosis', 0.9),            # 硬化
    3: ('mesangial_lesion', 0.8),     # 系膜区
    5: ('mesangial_cell', 0.7),       # 系膜细胞
    7: ('endothelial_cell', 0.6),      # 内皮细胞
}

def merge_glomeruli(label_img, dilation_iters=1, min_size=5000):
    """
    合并肾小球及其内部结构，统计数量
    
    Args:
        label_img: 标签图像
        dilation_iters: 膨胀迭代次数（用于连接相邻结构）
        min_size: 最小面积阈值
    
    Returns:
        dict: 统计结果
    """
    # 1. 创建完整 glomerulus 掩膜
    glomerulus_mask = np.isin(label_img, list(GLOMERULUS_PARTS.keys())).astype(np.int32)
    
    # 2. 膨胀glomerulus核心以连接相邻区域
    core_mask = (label_img == 6).astype(np.int32)
    struct = np.ones((3, 3), dtype=np.int32)  # 8连通
    dilated_core = ndimage.binary_dilation(core_mask, structure=struct, iterations=dilation_iters).astype(int)
    
    # 3. 合并
    merged_mask = np.maximum(glomerulus_mask, dilated_core)
    
    # 4. 连通域分析
    labeled, num_regions = ndimage.label(merged_mask, structure=struct)
    sizes = ndimage.sum(merged_mask, labeled, range(1, num_regions + 1))
    
    # 5. 过滤小区域
    large_regions = [s for s in sizes if s >= min_size]
    
    return {
        'num_regions': num_regions,
        'num_large_regions': len(large_regions),
        'total_pixels': int(sum(sizes)),
        'region_sizes': [int(s) for s in sorted(sizes, reverse=True)],
        'large_region_sizes': [int(s) for s in sorted(large_regions, reverse=True)]
    }

def analyze_dataset(labels_dir, output_file=None, dilation_iters=1, min_size=5000, first_n=None):
    """分析整个数据集"""
    files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    if first_n:
        files = files[:first_n]
    
    results = []
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        stats = merge_glomeruli(img, dilation_iters, min_size)
        
        result = {
            'image': f,
            'glomeruli_count': stats['num_large_regions'],
            'total_regions': stats['num_regions'],
            'total_pixels': stats['total_pixels'],
            'region_sizes': stats['large_region_sizes']
        }
        results.append(result)
        
        print(f"{f:<22} 肾小球数量: {stats['num_large_regions']:>2}  (原始区域: {stats['num_regions']:>3})")
    
    # 汇总
    total_glomeruli = sum(r['glomeruli_count'] for r in results)
    avg_glomeruli = total_glomeruli / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"分析完成: {len(files)} 张图像")
    print(f"参数: dilation={dilation_iters}, min_size={min_size}")
    print(f"肾小球总数: {total_glomeruli}")
    print(f"平均每张: {avg_glomeruli:.2f} 个")
    
    output_data = {
        'summary': {
            'total_images': len(files),
            'dilation_iterations': dilation_iters,
            'min_size_threshold': min_size,
            'total_glomeruli': total_glomeruli,
            'average_per_image': round(avg_glomeruli, 2)
        },
        'images': results
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_file}")
    
    return output_data

def show_structure_breakdown(labels_dir, image_name):
    """显示某张图像各结构的像素数"""
    img_path = os.path.join(labels_dir, image_name)
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return
    
    img = np.array(Image.open(img_path))
    
    print(f"\n{image_name} 各结构统计:")
    print("-" * 50)
    
    for label_id, (name, _) in GLOMERULUS_PARTS.items():
        count = np.sum(img == label_id)
        pct = count / (img.shape[0] * img.shape[1]) * 100
        print(f"  {name:20s}: {count:>12,} pixels ({pct:>5.2f}%)")
    
    # 合并后的统计
    merged = merge_glomeruli(img)
    print(f"\n  合并后:")
    print(f"  {'glomerulus(merged)':20s}: {merged['total_pixels']:>12,} pixels")
    print(f"  {'独立肾小球数':20s}: {merged['num_large_regions']:>12} 个")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='肾小球智能统计工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--dilation', type=int, default=1,
                       help='膨胀迭代次数，默认1')
    parser.add_argument('--min_size', type=int, default=5000,
                       help='最小面积阈值（像素），默认5000')
    parser.add_argument('--output', type=str, default='glomeruli_merged.json',
                       help='输出JSON文件')
    parser.add_argument('--first_n', type=int, default=None,
                       help='只分析前N张')
    parser.add_argument('--breakdown', type=str, default=None,
                       help='显示指定图像的详细结构')
    
    args = parser.parse_args()
    
    if args.breakdown:
        show_structure_breakdown(args.labels_dir, args.breakdown)
    else:
        analyze_dataset(args.labels_dir, args.output, args.dilation, args.min_size, args.first_n)
