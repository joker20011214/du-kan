#!/usr/bin/env python3
"""
牛津分级综合分析工具
整合四个分析模块的输出，生成完整的牛津分级报告

牛津分级标准:
- M (Mesangial hypercellularity): M0/M1
- E (Endocapillary hypercellularity): E0/E1  
- S (Segmental sclerosis): S0/S1
- T (Tubular atrophy): T0/T1/T2
"""

import os
import json
import argparse
from pathlib import Path


def load_analysis_results(output_dir):
    """加载各分析模块的结果"""
    results = {}
    
    # 加载系膜增生分析
    mesangial_file = os.path.join(output_dir, 'mesangial_hyperplasia_output', 'mesangial_hyperplasia_analysis.json')
    if os.path.exists(mesangial_file):
        with open(mesangial_file, 'r') as f:
            results['mesangial'] = json.load(f)
    else:
        mesangial_file = os.path.join(output_dir, 'mesangial_hyperplasia_output', 'mesangial_analysis.json')
        if os.path.exists(mesangial_file):
            with open(mesangial_file, 'r') as f:
                results['mesangial'] = json.load(f)
    
    # 加载内皮增生分析
    endothelial_file = os.path.join(output_dir, 'endothelial_analysis_output', 'endothelial_hyperplasia_analysis.json')
    if os.path.exists(endothelial_file):
        with open(endothelial_file, 'r') as f:
            results['endothelial'] = json.load(f)
    
    # 加载硬化与新月体分析
    sclerosis_file = os.path.join(output_dir, 'sclerosis_analysis_output', 'sclerosis_crescent_analysis.json')
    if os.path.exists(sclerosis_file):
        with open(sclerosis_file, 'r') as f:
            results['sclerosis'] = json.load(f)
    
    # 加载肾小管萎缩分析
    tubular_file = os.path.join(output_dir, 'tubular_atrophy_output', 'tubular_atrophy_analysis.json')
    if os.path.exists(tubular_file):
        with open(tubular_file, 'r') as f:
            results['tubular'] = json.load(f)
    
    return results


def generate_oxford_classification(results, image_list=None):
    """
    生成牛津分级
    
    Args:
        results: 各分析模块的结果
        image_list: 如果指定，则将这些图片作为整体输出一个汇总分级
    
    Returns:
        dict: 牛津分级结果
    """
    # 创建索引
    mesangial_dict = {img['image']: img for img in results.get('mesangial', {}).get('images', [])}
    endothelial_dict = {img['image']: img for img in results.get('endothelial', {}).get('images', [])}
    sclerosis_dict = {img['image']: img for img in results.get('sclerosis', {}).get('images', [])}
    tubular_dict = {img['image']: img for img in results.get('tubular', {}).get('images', [])}
    
    # 获取所有图像
    all_images = set(mesangial_dict.keys()) | set(endothelial_dict.keys()) | \
                 set(sclerosis_dict.keys()) | set(tubular_dict.keys())
    
    # 如果指定了图片列表，只分析这些图片
    if image_list:
        all_images = set(image_list) & all_images
    
    oxford_grades = []
    
    # 如果有指定的图片列表且多于1张，输出整体分级
    if image_list and len(image_list) > 1:
        target_images = [img for img in image_list if img in all_images]
        
        # 计算每张图片的单独分级
        for image_name in target_images:
            grade = {'image': image_name}
            
            # M
            if image_name in mesangial_dict:
                m_ratio = mesangial_dict[image_name].get('hyperplasia_ratio', 0)
                grade['M'] = 'M1' if m_ratio > 20 else 'M0'
                grade['mesangial_ratio'] = m_ratio
            else:
                grade['M'] = 'N/A'
                grade['mesangial_ratio'] = None
            
            # E
            if image_name in endothelial_dict:
                e_ratio = endothelial_dict[image_name].get('avg_endothelial_ratio', 0)
                grade['E'] = 'E1' if e_ratio > 0.0125 else 'E0'
                grade['endothelial_ratio'] = round(e_ratio * 100, 2)
            else:
                grade['E'] = 'N/A'
                grade['endothelial_ratio'] = None
            
            # S
            if image_name in sclerosis_dict:
                grade['S'] = 'S1' if sclerosis_dict[image_name]['has_sclerosis'] else 'S0'
                grade['sclerosis_pixels'] = sclerosis_dict[image_name]['sclerosis_pixels']
            else:
                grade['S'] = 'N/A'
                grade['sclerosis_pixels'] = None
            
            # T
            if image_name in tubular_dict:
                grade['T'] = tubular_dict[image_name]['label']
                grade['tubular_atrophy_ratio'] = tubular_dict[image_name]['atrophy_ratio']
            else:
                grade['T'] = 'N/A'
                grade['tubular_atrophy_ratio'] = None
            
            grade['oxford_score'] = f"{grade['M']}{grade['E']}{grade['S']}{grade['T']}"
            oxford_grades.append(grade)
        
        # 添加整体分级
        overall_grade = {'image': '整体分级', 'is_overall': True}
        
        # M: 任一M1则为M1
        has_m1 = any(g['M'] == 'M1' for g in oxford_grades)
        overall_grade['M'] = 'M1' if has_m1 else 'M0'
        
        # E: 任一E1则为E1
        has_e1 = any(g['E'] == 'E1' for g in oxford_grades)
        overall_grade['E'] = 'E1' if has_e1 else 'E0'
        
        # S: 任一S1则为S1
        has_s1 = any(g['S'] == 'S1' for g in oxford_grades)
        overall_grade['S'] = 'S1' if has_s1 else 'S0'
        
        # T: 取最大值 (T0=0, T1=1, T2=2)
        t_values = {'T0': 0, 'T1': 1, 'T2': 2}
        max_t = max((t_values.get(g['T'], 0) for g in oxford_grades), default=0)
        overall_grade['T'] = {v: k for k, v in t_values.items()}.get(max_t, 'T0')
        
        overall_grade['oxford_score'] = f"{overall_grade['M']}{overall_grade['E']}{overall_grade['S']}{overall_grade['T']}"
        overall_grade['images_count'] = len(oxford_grades)
        
        # 添加到最前面
        oxford_grades.insert(0, overall_grade)
        
        return oxford_grades
    
    # 单张图片或无指定列表：逐张输出
    for image_name in sorted(all_images):
        grade = {'image': image_name}
        
        # M
        if image_name in mesangial_dict:
            m_ratio = mesangial_dict[image_name].get('hyperplasia_ratio', 0)
            grade['M'] = 'M1' if m_ratio > 20 else 'M0'
            grade['mesangial_ratio'] = m_ratio
        else:
            grade['M'] = 'N/A'
            grade['mesangial_ratio'] = None
        
        # E
        if image_name in endothelial_dict:
            e_ratio = endothelial_dict[image_name].get('avg_endothelial_ratio', 0)
            grade['E'] = 'E1' if e_ratio > 0.0125 else 'E0'
            grade['endothelial_ratio'] = round(e_ratio * 100, 2)
        else:
            grade['E'] = 'N/A'
            grade['endothelial_ratio'] = None
        
        # S
        if image_name in sclerosis_dict:
            grade['S'] = 'S1' if sclerosis_dict[image_name]['has_sclerosis'] else 'S0'
            grade['sclerosis_pixels'] = sclerosis_dict[image_name]['sclerosis_pixels']
        else:
            grade['S'] = 'N/A'
            grade['sclerosis_pixels'] = None
        
        # T
        if image_name in tubular_dict:
            grade['T'] = tubular_dict[image_name]['label']
            grade['tubular_atrophy_ratio'] = tubular_dict[image_name]['atrophy_ratio']
        else:
            grade['T'] = 'N/A'
            grade['tubular_atrophy_ratio'] = None
        
        grade['oxford_score'] = f"{grade['M']}{grade['E']}{grade['S']}{grade['T']}"
        oxford_grades.append(grade)
    
    return oxford_grades


def print_oxford_report(oxford_grades, output_file=None):
    """打印牛津分级报告"""
    print("=" * 100)
    print("牛津分级 (Oxford Classification) 综合报告")
    print("=" * 100)
    print("M: 系膜增生 (M0<20%, M1≥20%) | E: 内皮增生 (E0<1.25%, E1≥1.25%)")
    print("S: 节段性硬化 (S0=无, S1=有)   | T: 肾小管萎缩 (T0≤25%, T1:26-50%, T2>50%)")
    print("=" * 100)
    
    # 检查是否有整体分级
    has_overall = any(g.get('is_overall', False) for g in oxford_grades)
    
    if has_overall:
        # 打印整体分级
        overall = next(g for g in oxford_grades if g.get('is_overall', False))
        print(f"\n【整体分级】(共 {overall['images_count']} 张图像)")
        print(f"  整体分级: {overall['oxford_score']}")
        print(f"  M={overall['M']}, E={overall['E']}, S={overall['S']}, T={overall['T']}")
        
        # 统计明细
        m0_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['M'] == 'M0')
        m1_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['M'] == 'M1')
        e0_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['E'] == 'E0')
        e1_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['E'] == 'E1')
        s0_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['S'] == 'S0')
        s1_count = sum(1 for g in oxford_grades if not g.get('is_overall', False) and g['S'] == 'S1')
        
        print(f"\n  各指标明细:")
        print(f"    M: M0={m0_count}, M1={m1_count}")
        print(f"    E: E0={e0_count}, E1={e1_count}")
        print(f"    S: S0={s0_count}, S1={s1_count}")
        print(f"    T: {overall['T']}")
        
        # 打印各图片明细
        print(f"\n{'='*100}")
        print("【各图像明细】")
        print(f"{'图像':<22} | {'M':>4} | {'E':>4} | {'S':>4} | {'T':>4} | {'分级':<10}")
        print("-" * 80)
        
        for g in oxford_grades:
            if not g.get('is_overall', False):
                print(f"{g['image']:<22} | {g['M']:>4} | {g['E']:>4} | {g['S']:>4} | {g['T']:>4} | {g['oxford_score']:<10}")
        
        n = 1  # 整体只有1个分级
    else:
        # 原有统计逻辑
        m0_count = sum(1 for g in oxford_grades if g['M'] == 'M0')
        m1_count = sum(1 for g in oxford_grades if g['M'] == 'M1')
        e0_count = sum(1 for g in oxford_grades if g['E'] == 'E0')
        e1_count = sum(1 for g in oxford_grades if g['E'] == 'E1')
        s0_count = sum(1 for g in oxford_grades if g['S'] == 'S0')
        s1_count = sum(1 for g in oxford_grades if g['S'] == 'S1')
        t0_count = sum(1 for g in oxford_grades if g['T'] == 'T0')
        t1_count = sum(1 for g in oxford_grades if g['T'] == 'T1')
        t2_count = sum(1 for g in oxford_grades if g['T'] == 'T2')
        
        n = len(oxford_grades)
        
        print(f"\n【汇总统计】(共 {n} 张图像)")
        print(f"  M0 (正常): {m0_count:>4} ({m0_count/n*100:>5.1f}%)")
        print(f"  M1 (增生): {m1_count:>4} ({m1_count/n*100:>5.1f}%)")
        print(f"  E0 (正常): {e0_count:>4} ({e0_count/n*100:>5.1f}%)")
        print(f"  E1 (增生): {e1_count:>4} ({e1_count/n*100:>5.1f}%)")
        print(f"  S0 (无硬化): {s0_count:>4} ({s0_count/n*100:>5.1f}%)")
        print(f"  S1 (有硬化): {s1_count:>4} ({s1_count/n*100:>5.1f}%)")
        print(f"  T0 (轻度): {t0_count:>4} ({t0_count/n*100:>5.1f}%)")
        print(f"  T1 (中度): {t1_count:>4} ({t1_count/n*100:>5.1f}%)")
        print(f"  T2 (重度): {t2_count:>4} ({t2_count/n*100:>5.1f}%)")
        
        # 详细列表
        print(f"\n{'='*100}")
        print("【详细分级】")
        print(f"{'图像':<22} | {'M':>4} | {'M%':>8} | {'E':>4} | {'E%':>8} | {'S':>4} | {'T':>4} | {'牛津分级':<10}")
        print("-" * 100)
        
        for g in oxford_grades:
            m_pct = f"{g['mesangial_ratio']:.2f}%" if g['mesangial_ratio'] is not None else "N/A"
            e_pct = f"{g['endothelial_ratio']:.2f}%" if g['endothelial_ratio'] is not None else "N/A"
            print(f"{g['image']:<22} | {g['M']:>4} | {m_pct:>8} | {g['E']:>4} | "
                  f"{e_pct:>8} | {g['S']:>4} | {g['T']:>4} | {g['oxford_score']:<10}")
    
    # 牛津分级分布
    print(f"\n{'='*100}")
    print("【牛津分级分布】")
    score_counts = {}
    for g in oxford_grades:
        score = g['oxford_score']
        score_counts[score] = score_counts.get(score, 0) + 1
    
    for score, count in sorted(score_counts.items(), key=lambda x: -x[1]):
        print(f"  {score:<10}: {count:>4} ({count/n*100:>5.1f}%)")
    
    # 保存结果
    if output_file:
        output_data = {
            'images': oxford_grades,
            'score_distribution': score_counts
        }
        if has_overall:
            output_data['overall_grade'] = overall['oxford_score']
            output_data['overall_details'] = overall
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {output_file}")


def run_all_analyzers(labels_dir, output_dir, image_list=None):
    """并行运行所有分析器"""
    import subprocess
    import concurrent.futures
    
    # 构建图片列表参数
    image_args = ''
    if image_list:
        image_args = ' --images ' + ' '.join(image_list)
    
    scripts = [
        ('mesangial', f'python3 mesangial_hyperplasia_analyzer.py --labels_dir {labels_dir} --output_dir {output_dir}/mesangial_hyperplasia_output{image_args}'),
        ('endothelial', f'python3 endothelial_hyperplasia_analyzer.py --labels_dir {labels_dir} --output_dir {output_dir}/endothelial_analysis_output{image_args}'),
        ('sclerosis', f'python3 sclerosis_crescent_analyzer.py --labels_dir {labels_dir} --output_dir {output_dir}/sclerosis_analysis_output{image_args}'),
        ('tubular', f'python3 tubular_atrophy_analyzer.py --labels_dir {labels_dir} --output_dir {output_dir}/tubular_atrophy_output{image_args}'),
    ]
    
    print("=" * 60)
    print("开始并行分析...")
    print("=" * 60)
    
    def run_script(name, cmd):
        print(f"\n>>> 运行 {name} 分析器...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return name, result.returncode, result.stdout, result.stderr
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_script, name, cmd) for name, cmd in scripts]
        for future in concurrent.futures.as_completed(futures):
            name, code, stdout, stderr = future.result()
            if code == 0:
                print(f"    {name}: ✓ 完成")
            else:
                print(f"    {name}: ✗ 失败")
                if stderr:
                    print(f"    错误: {stderr[:200]}")
    
    print("\n" + "=" * 60)
    print("所有分析完成!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='牛津分级综合分析工具')
    parser.add_argument('--labels_dir', type=str,
                       default='data/nnUNet_raw/Dataset800_Kidney_811Split/labelsTr',
                       help='标签目录')
    parser.add_argument('--output_dir', type=str, default='oxford_classification_output',
                       help='输出目录')
    parser.add_argument('--run', action='store_true',
                       help='运行所有分析器并生成报告')
    parser.add_argument('--skip_run', action='store_true',
                       help='跳过分析步骤，直接读取现有结果')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                       help='分析指定的多张图像，如: --images KIDNEY_0001.png KIDNEY_0002.png')
    parser.add_argument('--single', type=str, default=None,
                       help='分析单张图像')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.run or args.images or args.single:
        # 先运行所有分析器
        if args.single:
            run_all_analyzers(args.labels_dir, args.output_dir, [args.single])
        else:
            run_all_analyzers(args.labels_dir, args.output_dir, args.images)
    
    # 加载结果并生成报告
    print("\n正在加载分析结果...")
    results = load_analysis_results(args.output_dir)
    
    if not results:
        print("错误: 未找到分析结果，请先运行 --run 参数执行分析")
        exit(1)
    
    print(f"已加载: {list(results.keys())}")
    
    # 生成牛津分级
    oxford_grades = generate_oxford_classification(results, args.images)
    
    # 打印报告
    output_file = os.path.join(args.output_dir, 'oxford_classification.json')
    print_oxford_report(oxford_grades, output_file)
