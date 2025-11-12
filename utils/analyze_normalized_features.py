"""
基于归一化频率的特征区分度分析

计算每个特征的:
1. 归一化频率 (出现次数 / 样本数量)
2. 区分度倍数 (Flaky密度 / Non-flaky密度)
3. 特征分级 (独有、极强、强、中等)
"""
import re
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def tokenize_code(code: str) -> List[str]:
    """提取代码中的标识符"""
    identifiers = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-z][a-zA-Z0-9]*\b', code)
    
    java_keywords = {
        'public', 'private', 'protected', 'static', 'final', 'void', 'return',
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
        'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super',
        'class', 'interface', 'extends', 'implements', 'import', 'package',
        'int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short',
        'true', 'false', 'null', 'var', 'const', 'goto', 'assert', 'enum'
    }
    
    return [w for w in identifiers if len(w) >= 3 and w.lower() not in java_keywords]


def extract_api_calls(code: str) -> List[str]:
    """提取API调用（方法名）"""
    pattern = r'\b([a-z][a-zA-Z0-9]*)\s*\('
    return re.findall(pattern, code)


def extract_class_names(code: str) -> List[str]:
    """提取类名"""
    pattern = r'\b([A-Z][a-zA-Z0-9]*)\b'
    return re.findall(pattern, code)


def calculate_normalized_frequency(counter: Counter, sample_count: int, min_count: int = 3) -> Dict[str, float]:
    """
    计算归一化频率（密度）
    
    Args:
        counter: 词频统计
        sample_count: 样本总数
        min_count: 最小出现次数阈值
        
    Returns:
        {词: 归一化频率}
    """
    normalized = {}
    for word, count in counter.items():
        if count >= min_count:
            normalized[word] = count / sample_count
    return normalized


def calculate_discrimination_score(flaky_density: float, nonflaky_density: float) -> float:
    """
    计算区分度倍数
    
    Args:
        flaky_density: Flaky测试中的密度
        nonflaky_density: Non-flaky测试中的密度
        
    Returns:
        区分度倍数 (如果non-flaky为0则返回inf)
    """
    if nonflaky_density == 0:
        return float('inf')
    return flaky_density / nonflaky_density


def classify_feature(discrimination: float) -> str:
    """
    根据区分度对特征分级
    
    Args:
        discrimination: 区分度倍数
        
    Returns:
        特征等级
    """
    if discrimination == float('inf'):
        return 'unique'  # 独有特征
    elif discrimination >= 20:
        return 'very_strong'  # 极强区分特征 (20x+)
    elif discrimination >= 10:
        return 'strong'  # 强区分特征 (10-20x)
    elif discrimination >= 5:
        return 'moderate'  # 中等区分特征 (5-10x)
    else:
        return 'weak'  # 弱区分特征 (<5x)


def analyze_normalized_features(dataset_path: Path, output_dir: Path = None, min_count: int = 3):
    """
    分析归一化特征区分度
    
    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        min_count: 最小出现次数阈值
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'output' / 'facet_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("归一化特征区分度分析")
    print("=" * 80)
    print(f"最小出现次数阈值: {min_count}\n")
    
    # 加载数据集
    df = pd.read_csv(dataset_path)
    
    # 分离 flaky 和 non-flaky
    flaky_df = df[df['label'] != 'non-flaky'].copy()
    non_flaky_df = df[df['label'] == 'non-flaky'].copy()
    
    print(f"Flaky tests: {len(flaky_df)}")
    print(f"Non-flaky tests: {len(non_flaky_df)}\n")
    
    # 统计每个类别
    categories = flaky_df['label'].unique()
    
    # 存储所有分析结果
    all_analysis = {}
    
    # 先分析 non-flaky 作为基线
    print("分析 Non-flaky 基线...")
    non_flaky_codes = non_flaky_df['full_code'].dropna().tolist()
    
    nonflaky_identifiers = []
    nonflaky_methods = []
    nonflaky_classes = []
    
    for code in non_flaky_codes:
        code_str = str(code)
        nonflaky_identifiers.extend(tokenize_code(code_str))
        nonflaky_methods.extend(extract_api_calls(code_str))
        nonflaky_classes.extend(extract_class_names(code_str))
    
    nonflaky_id_counter = Counter(nonflaky_identifiers)
    nonflaky_method_counter = Counter(nonflaky_methods)
    nonflaky_class_counter = Counter(nonflaky_classes)
    
    nonflaky_id_density = calculate_normalized_frequency(nonflaky_id_counter, len(non_flaky_df), min_count)
    nonflaky_method_density = calculate_normalized_frequency(nonflaky_method_counter, len(non_flaky_df), min_count)
    nonflaky_class_density = calculate_normalized_frequency(nonflaky_class_counter, len(non_flaky_df), min_count)
    
    print(f"  Non-flaky 标识符: {len(nonflaky_id_density)}")
    print(f"  Non-flaky 方法: {len(nonflaky_method_density)}")
    print(f"  Non-flaky 类名: {len(nonflaky_class_density)}\n")
    
    # 分析每个 flaky 类别
    for category in categories:
        print(f"分析类别: {category}")
        category_df = flaky_df[flaky_df['label'] == category]
        sample_count = len(category_df)
        
        codes = category_df['full_code'].dropna().tolist()
        
        # 统计词频
        identifiers = []
        methods = []
        classes = []
        
        for code in codes:
            code_str = str(code)
            identifiers.extend(tokenize_code(code_str))
            methods.extend(extract_api_calls(code_str))
            classes.extend(extract_class_names(code_str))
        
        id_counter = Counter(identifiers)
        method_counter = Counter(methods)
        class_counter = Counter(classes)
        
        # 计算归一化频率
        id_density = calculate_normalized_frequency(id_counter, sample_count, min_count)
        method_density = calculate_normalized_frequency(method_counter, sample_count, min_count)
        class_density = calculate_normalized_frequency(class_counter, sample_count, min_count)
        
        # 计算区分度并分级
        features_by_level = {
            'unique': {'identifiers': [], 'methods': [], 'classes': []},
            'very_strong': {'identifiers': [], 'methods': [], 'classes': []},
            'strong': {'identifiers': [], 'methods': [], 'classes': []},
            'moderate': {'identifiers': [], 'methods': [], 'classes': []},
            'weak': {'identifiers': [], 'methods': [], 'classes': []}
        }
        
        # 分析标识符
        for word, flaky_dens in id_density.items():
            nonflaky_dens = nonflaky_id_density.get(word, 0)
            discrimination = calculate_discrimination_score(flaky_dens, nonflaky_dens)
            level = classify_feature(discrimination)
            
            features_by_level[level]['identifiers'].append({
                'feature': word,
                'flaky_density': flaky_dens,
                'nonflaky_density': nonflaky_dens,
                'discrimination': discrimination,
                'flaky_count': id_counter[word]
            })
        
        # 分析方法
        for method, flaky_dens in method_density.items():
            nonflaky_dens = nonflaky_method_density.get(method, 0)
            discrimination = calculate_discrimination_score(flaky_dens, nonflaky_dens)
            level = classify_feature(discrimination)
            
            features_by_level[level]['methods'].append({
                'feature': method,
                'flaky_density': flaky_dens,
                'nonflaky_density': nonflaky_dens,
                'discrimination': discrimination,
                'flaky_count': method_counter[method]
            })
        
        # 分析类名
        for cls, flaky_dens in class_density.items():
            nonflaky_dens = nonflaky_class_density.get(cls, 0)
            discrimination = calculate_discrimination_score(flaky_dens, nonflaky_dens)
            level = classify_feature(discrimination)
            
            features_by_level[level]['classes'].append({
                'feature': cls,
                'flaky_density': flaky_dens,
                'nonflaky_density': nonflaky_dens,
                'discrimination': discrimination,
                'flaky_count': class_counter[cls]
            })
        
        # 排序（按区分度降序）
        for level in features_by_level:
            for feature_type in ['identifiers', 'methods', 'classes']:
                features_by_level[level][feature_type].sort(
                    key=lambda x: (x['discrimination'], x['flaky_density']), 
                    reverse=True
                )
        
        all_analysis[category] = {
            'sample_count': sample_count,
            'features_by_level': features_by_level
        }
        
        # 打印统计
        print(f"  样本数: {sample_count}")
        for level in ['unique', 'very_strong', 'strong', 'moderate']:
            total = (len(features_by_level[level]['identifiers']) + 
                    len(features_by_level[level]['methods']) + 
                    len(features_by_level[level]['classes']))
            if total > 0:
                print(f"  {level}: {total} 个特征")
        print()
    
    # 保存JSON结果
    json_output = output_dir / 'normalized_feature_analysis.json'
    
    # 将 inf 转换为字符串以便JSON序列化
    def convert_inf(obj):
        if isinstance(obj, float) and obj == float('inf'):
            return 'inf'
        return obj
    
    json_data = json.loads(json.dumps(all_analysis, default=convert_inf))
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON 结果已保存: {json_output}")
    
    # 生成可读报告
    generate_readable_report(all_analysis, output_dir)
    
    # 生成特征查找表（用于实际判断）
    generate_feature_lookup_table(all_analysis, output_dir)
    
    print(f"\n✓ 分析完成！")


def generate_readable_report(all_analysis: Dict, output_dir: Path):
    """生成人类可读的报告"""
    report_path = output_dir / 'normalized_feature_report.txt'
    
    level_names = {
        'unique': '独有特征 (∞倍)',
        'very_strong': '极强区分特征 (20x+)',
        'strong': '强区分特征 (10-20x)',
        'moderate': '中等区分特征 (5-10x)'
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("归一化特征区分度分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        for category, analysis in all_analysis.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"类别: {category}\n")
            f.write("=" * 80 + "\n")
            f.write(f"样本数: {analysis['sample_count']}\n\n")
            
            features_by_level = analysis['features_by_level']
            
            for level in ['unique', 'very_strong', 'strong', 'moderate']:
                level_data = features_by_level[level]
                
                # 合并所有特征类型
                all_features = []
                for ftype in ['identifiers', 'methods', 'classes']:
                    for feat in level_data[ftype]:
                        all_features.append({
                            'type': ftype,
                            **feat
                        })
                
                if not all_features:
                    continue
                
                # 按区分度排序
                all_features.sort(key=lambda x: (x['discrimination'], x['flaky_density']), reverse=True)
                
                f.write("-" * 80 + "\n")
                f.write(f"{level_names[level]}\n")
                f.write("-" * 80 + "\n")
                
                for feat in all_features[:20]:  # 每个等级最多显示20个
                    disc_str = '∞' if feat['discrimination'] == float('inf') else f"{feat['discrimination']:.1f}x"
                    f.write(f"{feat['feature']:30s} [{feat['type'][:3]}] "
                           f"| Flaky: {feat['flaky_density']:.3f} "
                           f"| Non-flaky: {feat['nonflaky_density']:.3f} "
                           f"| {disc_str}\n")
                
                f.write("\n")
    
    print(f"✓ 可读报告已保存: {report_path}")


def generate_feature_lookup_table(all_analysis: Dict, output_dir: Path):
    """
    生成特征查找表（用于实际判断时快速查询）
    
    格式: {特征名: [(类别, 等级, 区分度, 密度), ...]}
    """
    lookup_table = defaultdict(list)
    
    for category, analysis in all_analysis.items():
        features_by_level = analysis['features_by_level']
        
        for level in ['unique', 'very_strong', 'strong', 'moderate']:
            level_data = features_by_level[level]
            
            for ftype in ['identifiers', 'methods', 'classes']:
                for feat in level_data[ftype]:
                    lookup_table[feat['feature']].append({
                        'category': category,
                        'level': level,
                        'type': ftype,
                        'discrimination': feat['discrimination'],
                        'flaky_density': feat['flaky_density'],
                        'nonflaky_density': feat['nonflaky_density']
                    })
    
    # 转换 inf 为字符串
    def convert_inf(obj):
        if isinstance(obj, float) and obj == float('inf'):
            return 'inf'
        return obj
    
    lookup_data = {k: v for k, v in lookup_table.items()}
    lookup_json = json.loads(json.dumps(lookup_data, default=convert_inf))
    
    # 保存查找表
    lookup_path = output_dir / 'feature_lookup_table.json'
    with open(lookup_path, 'w', encoding='utf-8') as f:
        json.dump(lookup_json, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 特征查找表已保存: {lookup_path}")
    
    # 生成统计摘要
    summary_path = output_dir / 'feature_statistics_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("特征统计摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总特征数: {len(lookup_table)}\n\n")
        
        # 统计每个类别的特征数
        category_stats = defaultdict(lambda: {'unique': 0, 'very_strong': 0, 'strong': 0, 'moderate': 0})
        
        for feature, occurrences in lookup_table.items():
            for occ in occurrences:
                category_stats[occ['category']][occ['level']] += 1
        
        f.write("各类别特征分布:\n")
        f.write("-" * 80 + "\n")
        for category, stats in category_stats.items():
            f.write(f"\n{category}:\n")
            f.write(f"  独有特征: {stats['unique']}\n")
            f.write(f"  极强区分: {stats['very_strong']}\n")
            f.write(f"  强区分: {stats['strong']}\n")
            f.write(f"  中等区分: {stats['moderate']}\n")
    
    print(f"✓ 统计摘要已保存: {summary_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='归一化特征区分度分析')
    parser.add_argument('--dataset', '-d',
                       default='dataset/FlakyLens_dataset_with_nonflaky_indented.csv',
                       help='数据集路径')
    parser.add_argument('--output', '-o',
                       default='output/facet_analysis',
                       help='输出目录')
    parser.add_argument('--min-count', '-m',
                       type=int,
                       default=3,
                       help='最小出现次数阈值')
    
    args = parser.parse_args()
    
    dataset_path = PROJECT_ROOT / args.dataset
    output_dir = PROJECT_ROOT / args.output
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    analyze_normalized_features(dataset_path, output_dir, args.min_count)


if __name__ == '__main__':
    main()
