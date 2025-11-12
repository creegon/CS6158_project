"""
基于数据驱动分析各类别的关键词
用于优化 Faceted API Matcher 的预定义标签

功能:
1. 按5个类别划分flaky tests
2. 统计每个类别中的词语频率
3. 提取高频关键词作为预定义标签
4. 生成可直接用于代码的配置
"""
import re
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def tokenize_code(code: str) -> List[str]:
    """
    将代码分词，提取有意义的标识符
    
    Args:
        code: 源代码字符串
        
    Returns:
        标识符列表
    """
    # 提取所有标识符（类名、方法名、变量名等）
    # 匹配 Java 标识符模式
    identifiers = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-z][a-zA-Z0-9]*\b', code)
    
    # 过滤掉常见的Java关键字和过短的词
    java_keywords = {
        'public', 'private', 'protected', 'static', 'final', 'void', 'return',
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
        'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super',
        'class', 'interface', 'extends', 'implements', 'import', 'package',
        'int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short',
        'true', 'false', 'null', 'var', 'const', 'goto', 'assert', 'enum'
    }
    
    # 过滤并返回
    filtered = [
        word for word in identifiers 
        if len(word) >= 3 and word.lower() not in java_keywords
    ]
    
    return filtered


def extract_api_calls(code: str) -> List[str]:
    """
    提取API调用（方法名）
    
    Args:
        code: 源代码字符串
        
    Returns:
        API调用列表
    """
    # 匹配方法调用模式: methodName(...) 或 obj.methodName(...)
    pattern = r'\b([a-z][a-zA-Z0-9]*)\s*\('
    methods = re.findall(pattern, code)
    return methods


def extract_class_names(code: str) -> List[str]:
    """
    提取类名
    
    Args:
        code: 源代码字符串
        
    Returns:
        类名列表
    """
    # 匹配大写开头的标识符（可能是类名）
    pattern = r'\b([A-Z][a-zA-Z0-9]*)\b'
    classes = re.findall(pattern, code)
    return classes


def analyze_category_keywords(dataset_path: Path, output_dir: Path = None):
    """
    分析每个类别的关键词频率
    
    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录，默认为 output/facet_analysis
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'output' / 'facet_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Flaky Test 类别关键词分析")
    print("=" * 60)
    
    # 加载数据集
    print(f"\n📂 加载数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"✓ 总记录数: {len(df)}")
    
    # 筛选出flaky tests（排除 non-flaky）
    flaky_df = df[df['label'] != 'non-flaky'].copy()
    print(f"✓ Flaky tests: {len(flaky_df)}")
    
    # 统计每个类别的数量（使用label列）
    category_counts = flaky_df['label'].value_counts()
    print(f"\n📊 Flaky类别分布:")
    for category, count in category_counts.items():
        print(f"  - {category}: {count}")
    
    # 筛选 non-flaky tests
    non_flaky_df = df[df['label'] == 'non-flaky'].copy()
    print(f"\n✓ Non-flaky tests: {len(non_flaky_df)}")
    
    # 为每个类别分析关键词
    category_analysis = {}
    
    for category in category_counts.index:
        print(f"\n{'=' * 60}")
        print(f"分析类别: {category}")
        print('=' * 60)
        
        # 获取该类别的所有测试代码（使用label列）
        category_df = flaky_df[flaky_df['label'] == category]
        codes = category_df['full_code'].tolist()
        
        # 统计词频
        all_identifiers = []
        all_methods = []
        all_classes = []
        
        for code in codes:
            if pd.isna(code):
                continue
            code_str = str(code)
            all_identifiers.extend(tokenize_code(code_str))
            all_methods.extend(extract_api_calls(code_str))
            all_classes.extend(extract_class_names(code_str))
        
        # 计数
        identifier_counter = Counter(all_identifiers)
        method_counter = Counter(all_methods)
        class_counter = Counter(all_classes)
        
        # 保存分析结果
        category_analysis[category] = {
            'sample_count': len(category_df),
            'top_identifiers': identifier_counter.most_common(50),
            'top_methods': method_counter.most_common(30),
            'top_classes': class_counter.most_common(30),
        }
        
        # 打印结果
        print(f"\n样本数: {len(category_df)}")
        
        print(f"\n🔤 Top 20 标识符:")
        for word, count in identifier_counter.most_common(20):
            print(f"  {word:30s} : {count:4d}")
        
        print(f"\n📞 Top 15 方法调用:")
        for method, count in method_counter.most_common(15):
            print(f"  {method:30s} : {count:4d}")
        
        print(f"\n🏷️  Top 15 类名:")
        for cls, count in class_counter.most_common(15):
            print(f"  {cls:30s} : {count:4d}")
    
    # 分析 non-flaky tests
    print(f"\n{'=' * 60}")
    print(f"分析类别: non-flaky")
    print('=' * 60)
    
    codes = non_flaky_df['full_code'].tolist()
    
    all_identifiers = []
    all_methods = []
    all_classes = []
    
    for code in codes:
        if pd.isna(code):
            continue
        code_str = str(code)
        all_identifiers.extend(tokenize_code(code_str))
        all_methods.extend(extract_api_calls(code_str))
        all_classes.extend(extract_class_names(code_str))
    
    identifier_counter = Counter(all_identifiers)
    method_counter = Counter(all_methods)
    class_counter = Counter(all_classes)
    
    # 保存 non-flaky 分析结果 (保存更多数据用于对比)
    category_analysis['non-flaky'] = {
        'sample_count': len(non_flaky_df),
        'top_identifiers': identifier_counter.most_common(100),
        'top_methods': method_counter.most_common(100),
        'top_classes': class_counter.most_common(100),
    }
    
    print(f"\n样本数: {len(non_flaky_df)}")
    
    print(f"\n🔤 Top 20 标识符:")
    for word, count in identifier_counter.most_common(20):
        print(f"  {word:30s} : {count:4d}")
    
    print(f"\n📞 Top 15 方法调用:")
    for method, count in method_counter.most_common(15):
        print(f"  {method:30s} : {count:4d}")
    
    print(f"\n🏷️  Top 15 类名:")
    for cls, count in class_counter.most_common(15):
        print(f"  {cls:30s} : {count:4d}")
    
    # 保存详细分析结果到JSON
    json_output = output_dir / 'category_keyword_analysis.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(category_analysis, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细分析结果已保存: {json_output}")
    
    # 生成推荐的预定义标签
    print(f"\n{'=' * 60}")
    print("生成推荐的预定义标签")
    print('=' * 60)
    
    recommendations = generate_facet_recommendations(category_analysis)
    
    # 保存推荐配置
    recommendations_output = output_dir / 'recommended_facets.py'
    with open(recommendations_output, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('基于数据分析生成的 Faceted API Matcher 预定义标签\n')
        f.write('自动生成时间: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('"""\n\n')
        f.write(recommendations)
    
    print(f"\n💾 推荐配置已保存: {recommendations_output}")
    
    # 生成可视化报告
    generate_report(category_analysis, output_dir)
    
    print(f"\n✓ 分析完成！所有结果保存在: {output_dir}")


def generate_facet_recommendations(category_analysis: Dict) -> str:
    """
    基于分析结果生成推荐的facet配置代码
    
    Args:
        category_analysis: 类别分析结果
        
    Returns:
        Python代码字符串
    """
    code_parts = []
    
    # 为每个类别生成推荐关键词
    for category, analysis in category_analysis.items():
        code_parts.append(f"# ========== {category} ==========")
        code_parts.append(f"# 样本数: {analysis['sample_count']}\n")
        
        # 提取高频方法名
        top_methods = [m for m, c in analysis['top_methods'][:20] if c >= 5]
        if top_methods:
            code_parts.append(f"{category.upper()}_METHODS = {{")
            for method in top_methods:
                code_parts.append(f"    '{method}',")
            code_parts.append("}\n")
        
        # 提取高频类名
        top_classes = [c for c, cnt in analysis['top_classes'][:20] if cnt >= 3]
        if top_classes:
            code_parts.append(f"{category.upper()}_CLASSES = {{")
            for cls in top_classes:
                code_parts.append(f"    '{cls}',")
            code_parts.append("}\n")
        
        code_parts.append("")
    
    return "\n".join(code_parts)


def generate_report(category_analysis: Dict, output_dir: Path):
    """
    生成文本分析报告
    
    Args:
        category_analysis: 类别分析结果
        output_dir: 输出目录
    """
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Flaky Test 类别关键词分析报告\n")
        f.write("生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n\n")
        
        for category, analysis in category_analysis.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"类别: {category}\n")
            f.write("=" * 80 + "\n")
            f.write(f"样本数: {analysis['sample_count']}\n\n")
            
            # 对于 non-flaky,显示更多数据
            is_non_flaky = (category == 'non-flaky')
            
            # 标识符
            id_count = 100 if is_non_flaky else 30
            f.write("-" * 40 + "\n")
            f.write(f"Top {id_count} 标识符\n")
            f.write("-" * 40 + "\n")
            for word, count in analysis['top_identifiers'][:id_count]:
                f.write(f"{word:30s} : {count:5d}\n")
            
            # 方法调用
            method_count = 100 if is_non_flaky else 20
            f.write("\n" + "-" * 40 + "\n")
            f.write(f"Top {method_count} 方法调用\n")
            f.write("-" * 40 + "\n")
            for method, count in analysis['top_methods'][:method_count]:
                f.write(f"{method:30s} : {count:5d}\n")
            
            # 类名
            class_count = 100 if is_non_flaky else 20
            f.write("\n" + "-" * 40 + "\n")
            f.write(f"Top {class_count} 类名\n")
            f.write("-" * 40 + "\n")
            for cls, count in analysis['top_classes'][:class_count]:
                f.write(f"{cls:30s} : {count:5d}\n")
            
            f.write("\n")
    
    print(f"📄 文本报告已保存: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析Flaky Test各类别的关键词频率')
    parser.add_argument('--dataset', '-d', 
                        default='dataset/FlakyLens_dataset_with_nonflaky_indented.csv',
                        help='数据集路径')
    parser.add_argument('--output', '-o',
                        default='output/facet_analysis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    dataset_path = PROJECT_ROOT / args.dataset
    output_dir = PROJECT_ROOT / args.output
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    analyze_category_keywords(dataset_path, output_dir)


if __name__ == '__main__':
    main()
