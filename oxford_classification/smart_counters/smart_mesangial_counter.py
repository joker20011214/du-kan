#!/usr/bin/env python3
"""
系膜区智能统计工具
核心逻辑：以系膜区(mesangial_lesion)为主体，将系膜细胞(mesangial_cell)作为其内部结构，
统计每个系膜区包含几个系膜细胞
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import os
import argparse
import json

def analyze_mesangial_regions(label_img, dilation_iters=1, min_lesion_size=500, min_cell_size=50):
    """
    分析系膜区和系膜细胞的包含关系

    Args:
        label_img: 标签图像
        dilation_iters: 膨胀迭代次数（用于连接相邻结构）
        min_lesion_size: 最小系膜区面积阈值
        min_cell_size: 最小系膜细胞面积阈值

    Returns:
        dict: 统计结果
    """
    # 1. 获取系膜区和系膜细胞掩膜
    lesion_mask = (label_img == 3).astype(np.int32)  # 系膜区
    cell_mask = (label_img == 5).astype(np.int32)   # 系膜细胞

    # 2. 膨胀系膜区以包含相邻的系膜细胞
    struct = np.ones((3, 3), dtype=np.int32)
    dilated_lesion = ndimage.binary_dilation(lesion_mask, structure=struct, iterations=dilation_iters).astype(int)

    # 3. 合并系膜区和系膜细胞
    merged_mask = np.maximum(dilated_lesion, cell_mask)

    # 4. 连通域分析 - 得到合并后的系膜区
    labeled_merged, num_merged = ndimage.label(merged_mask, structure=struct)

    # 5. 统计每个合并区域内的系膜细胞数量
    cells_per_region = []
    total_cells = 0

    for region_id in range(1, num_merged + 1):
        region_coords = labeled_merged == region_id

        # 检查该区域是否包含系膜区核心（原始系膜区像素）
        lesion_in_region = np.sum(region_coords & (label_img == 3))

        if lesion_in_region >= min_lesion_size:
            # 计算该区域内的系膜细胞数量（使用原始系膜细胞掩膜）
            cells_in_region = np.sum(region_coords & (label_img == 5))
            num_cells = int(np.sum(
                ndimage.label(
                    region_coords & cell_mask,
                    structure=struct
                )[0]
            ))

            if num_cells > 0:
                cells_per_region.append(num_cells)
                total_cells += num_cells

    # 统计系膜细胞分布
    if cells_per_region:
        avg_cells = sum(cells_per_region) / len(cells_per_region)
        max_cells = max(cells_per_region)
        min_cells = min(cells_per_region)
    else:
        avg_cells = max_cells = min_cells = 0

    return {
        'num_lesions': len(cells_per_region),  # 有系膜细胞的系膜区数量
        'total_cells': total_cells,            # 总系膜细胞数
        'cells_per_lesion': cells_per_region,   # 每个系膜区的细胞数列表
        'avg_cells_per_lesion': round(avg_cells, 2),
        'max_cells_per_lesion': max_cells,
        'min_cells_per_lesion': min_cells,
    }


def analyze_dataset(labels_dir, output_file=None, dilation_iters=1, min_lesion_size=500, min_cell_size=50, first_n=None):
    """分析整个数据集"""
    files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    if first_n:
        files = files[:first_n]

    results = []
    for f in files:
        img = np.array(Image.open(os.path.join(labels_dir, f)))
        stats = analyze_mesangial_regions(img, dilation_iters, min_lesion_size, min_cell_size)

        result = {
            'image': f,
            'mesangial_regions': stats['num_lesions'],
            'total_cells': stats['total_cells'],
            'avg_cells_per_region': stats['avg_cells_per_lesion'],
            'max_cells_per_region': stats['max_cells_per_lesion'],
            'cells_per_region': stats['cells_per_lesion']
        }
        results.append(result)

        print(f"{f:<22} 系膜区: {stats['num_lesions']:>3}, 系膜细胞: {stats['total_cells']:>4}, "
              f"平均 {stats['avg_cells_per_lesion']:>4.2f}/区")

    # 汇总统计
    total_regions = sum(r['mesangial_regions'] for r in results)
    total_cells = sum(r['total_cells'] for r in results)
    avg_regions = total_regions / len(results) if results else 0
    avg_cells = total_cells / len(results) if results else 0

    # 收集所有 cells_per_region 分布
    all_cells_per_region = []
    for r in results:
        all_cells_per_region.extend(r['cells_per_region'])

    # 细胞数分布统计
    cell_distribution = {}
    for n in all_cells_per_region:
        cell_distribution[n] = cell_distribution.get(n, 0) + 1

    print(f"\n{'='*70}")
    print(f"分析完成: {len(files)} 张图像")
    print(f"参数: dilation={dilation_iters}, min_lesion_size={min_lesion_size}")
    print(f"\n系膜区总数: {total_regions}")
    print(f"平均每张图像: {avg_regions:.2f} 个系膜区")
    print(f"\n系膜细胞总数: {total_cells}")
    print(f"平均每张图像: {avg_cells:.2f} 个系膜细胞")
    print(f"\n每个系膜区的系膜细胞数分布:")
    for n in sorted(cell_distribution.keys()):
        count = cell_distribution[n]
        pct = count / len(all_cells_per_region) * 100 if all_cells_per_region else 0
        print(f"  {n} 个系膜细胞: {count:>4} 个系膜区 ({pct:>5.1f}%)")

    output_data = {
        'summary': {
            'total_images': len(files),
            'dilation_iterations': dilation_iters,
            'min_lesion_size': min_lesion_size,
            'total_mesangial_regions': total_regions,
            'total_mesangial_cells': total_cells,
            'avg_regions_per_image': round(avg_regions, 2),
            'avg_cells_per_image': round(avg_cells, 2),
            'cell_distribution': {str(k): v for k, v in sorted(cell_distribution.items())}
        },
        'images': results
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")

    return output_data


def show_structure_breakdown(labels_dir, image_name):
    """显示某张图像的系膜区结构"""
    img_path = os.path.join(labels_dir, image_name)
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return

    img = np.array(Image.open(img_path))

    print(f"\n{image_name} 系膜区分析:")
    print("-" * 50)

    # 基础统计
    lesion_count = np.sum(img == 3)
    cell_count = np.sum(img == 5)
    print(f"  系膜区像素: {lesion_count:>12,}")
    print(f"  系膜细胞像素: {cell_count:>12,}")

    # 详细分析
    stats = analyze_mesangial_regions(img)
    print(f"\n  有系膜细胞的系膜区数量: {stats['num_lesions']}")
    print(f"  系膜细胞总数: {stats['total_cells']}")
    print(f"  平均每区系膜细胞: {stats['avg_cells_per_lesion']}")

    if stats['cells_per_lesion']:
        print(f"\n  各系膜区系膜细胞数量:")
        for i, n in enumerate(stats['cells_per_lesion'], 1):
            print(f"    系膜区 {i}: {n} 个系膜细胞")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='系膜区智能统计工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--dilation', type=int, default=1,
                       help='膨胀迭代次数，默认1')
    parser.add_argument('--min_lesion_size', type=int, default=500,
                       help='最小系膜区面积阈值（像素），默认500')
    parser.add_argument('--min_cell_size', type=int, default=50,
                       help='最小系膜细胞面积阈值（像素），默认50')
    parser.add_argument('--output', type=str, default='mesangial_regions.json',
                       help='输出JSON文件')
    parser.add_argument('--first_n', type=int, default=None,
                       help='只分析前N张')
    parser.add_argument('--breakdown', type=str, default=None,
                       help='显示指定图像的详细结构')

    args = parser.parse_args()

    if args.breakdown:
        show_structure_breakdown(args.labels_dir, args.breakdown)
    else:
        analyze_dataset(args.labels_dir, args.output, args.dilation,
                      args.min_lesion_size, args.min_cell_size, args.first_n)
