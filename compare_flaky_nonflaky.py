"""
对比分析 Flaky vs Non-Flaky Tests
识别 flaky tests 的独特特征
"""
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent


def load_analysis():
    """加载分析结果"""
    analysis_file = PROJECT_ROOT / 'output' / 'facet_analysis' / 'category_keyword_analysis.json'
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_enrichment(flaky_count, flaky_total, non_flaky_count, non_flaky_total, min_count=3):
    """
    计算关键词在 flaky 中的富集倍数
    
    Args:
        flaky_count: 在flaky中的出现次数
        flaky_total: flaky总样本数
        non_flaky_count: 在non-flaky中的出现次数
        non_flaky_total: non-flaky总样本数
        min_count: 最小出现次数
        
    Returns:
        富集倍数（越大表示越是flaky特征）
    """
    if flaky_count < min_count:
        return 0
    
    flaky_freq = flaky_count / flaky_total
    non_flaky_freq = non_flaky_count / non_flaky_total if non_flaky_count > 0 else 0.0001
    
    return flaky_freq / non_flaky_freq


def analyze_category_enrichment(analysis_data):
    """分析各类别相对于 non-flaky 的富集特征"""
    
    non_flaky_data = analysis_data.get('non-flaky', {})
    non_flaky_total = non_flaky_data['sample_count']
    
    # 构建 non-flaky 关键词字典
    non_flaky_methods = {m: c for m, c in non_flaky_data['top_methods']}
    non_flaky_classes = {c: cnt for c, cnt in non_flaky_data['top_classes']}
    
    results = {}
    
    for category, data in analysis_data.items():
        if category == 'non-flaky':
            continue
        
        category_total = data['sample_count']
        
        # 计算方法的富集度
        method_enrichment = []
        for method, count in data['top_methods']:
            non_flaky_count = non_flaky_methods.get(method, 0)
            enrichment = calculate_enrichment(
                count, category_total,
                non_flaky_count, non_flaky_total,
                min_count=5
            )
            if enrichment > 1.5:  # 至少1.5倍富集
                method_enrichment.append((method, count, enrichment))
        
        # 计算类名的富集度
        class_enrichment = []
        for cls, count in data['top_classes']:
            non_flaky_count = non_flaky_classes.get(cls, 0)
            enrichment = calculate_enrichment(
                count, category_total,
                non_flaky_count, non_flaky_total,
                min_count=3
            )
            if enrichment > 1.5:
                class_enrichment.append((cls, count, enrichment))
        
        # 按富集倍数排序
        method_enrichment.sort(key=lambda x: x[2], reverse=True)
        class_enrichment.sort(key=lambda x: x[2], reverse=True)
        
        results[category] = {
            'sample_count': category_total,
            'enriched_methods': method_enrichment[:15],
            'enriched_classes': class_enrichment[:15],
        }
    
    return results


def generate_comparison_report(enrichment_results):
    """生成对比分析报告"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("Flaky vs Non-Flaky 对比分析报告")
    lines.append("识别 Flaky Tests 的独特特征")
    lines.append("=" * 80)
    lines.append("")
    lines.append("说明：")
    lines.append("  - 富集倍数 = (关键词在Flaky中的频率) / (关键词在Non-Flaky中的频率)")
    lines.append("  - 富集倍数 > 1 表示该关键词在Flaky中更常见")
    lines.append("  - 富集倍数越大，越是Flaky的特征性指标")
    lines.append("")
    
    for category, data in enrichment_results.items():
        lines.append("\n" + "=" * 80)
        lines.append(f"类别: {category}")
        lines.append("=" * 80)
        lines.append(f"样本数: {data['sample_count']}")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("富集的方法调用（相比 Non-Flaky）")
        lines.append("-" * 80)
        lines.append(f"{'方法名':<30} {'出现次数':>10} {'富集倍数':>15}")
        lines.append("-" * 80)
        
        if data['enriched_methods']:
            for method, count, enrichment in data['enriched_methods']:
                lines.append(f"{method:<30} {count:>10} {enrichment:>15.2f}x")
        else:
            lines.append("（无显著富集的方法）")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("富集的类名（相比 Non-Flaky）")
        lines.append("-" * 80)
        lines.append(f"{'类名':<30} {'出现次数':>10} {'富集倍数':>15}")
        lines.append("-" * 80)
        
        if data['enriched_classes']:
            for cls, count, enrichment in data['enriched_classes']:
                lines.append(f"{cls:<30} {count:>10} {enrichment:>15.2f}x")
        else:
            lines.append("（无显著富集的类名）")
        
        lines.append("")
    
    return '\n'.join(lines)


def main():
    """主函数"""
    print("=" * 60)
    print("Flaky vs Non-Flaky 对比分析")
    print("=" * 60)
    
    # 加载分析数据
    print("\n📂 加载分析结果...")
    analysis_data = load_analysis()
    print(f"✓ 已加载 {len(analysis_data)} 个类别的数据")
    
    # 计算富集度
    print("\n🔬 计算富集特征...")
    enrichment_results = analyze_category_enrichment(analysis_data)
    print(f"✓ 已完成 {len(enrichment_results)} 个类别的分析")
    
    # 生成报告
    print("\n📝 生成对比报告...")
    report = generate_comparison_report(enrichment_results)
    
    # 保存报告
    output_dir = PROJECT_ROOT / 'output' / 'facet_analysis'
    report_file = output_dir / 'flaky_vs_nonflaky_comparison.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 报告已保存: {report_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("关键发现摘要")
    print("=" * 60)
    
    for category, data in enrichment_results.items():
        print(f"\n{category}:")
        if data['enriched_methods']:
            top_method = data['enriched_methods'][0]
            print(f"  最强方法特征: {top_method[0]} ({top_method[2]:.1f}x)")
        if data['enriched_classes']:
            top_class = data['enriched_classes'][0]
            print(f"  最强类名特征: {top_class[0]} ({top_class[2]:.1f}x)")
    
    print("\n" + "=" * 60)
    print("💡 下一步:")
    print("  1. 查看详细报告: output/facet_analysis/flaky_vs_nonflaky_comparison.txt")
    print("  2. 使用高富集度特征优化 Faceted API Matcher")
    print("  3. 考虑添加负面特征过滤（排除Non-Flaky常见但Flaky少见的特征）")
    print("=" * 60)


if __name__ == '__main__':
    main()
