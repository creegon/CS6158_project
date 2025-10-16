"""
评估报告生成器
生成详细的评估报告
"""
from typing import Dict
from pathlib import Path
import json
from utils.evaluation_utils import format_percentage


class EvaluationReport:
    """评估报告类"""
    
    def __init__(self, metrics: Dict):
        """
        初始化报告
        
        Args:
            metrics: 评估指标字典
        """
        self.metrics = metrics
    
    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "=" * 70)
        print("评估结果摘要")
        print("=" * 70)
        
        total = self.metrics['total_samples']
        overall_acc = self.metrics['overall_accuracy']
        
        print(f"\n📊 总体统计:")
        print(f"  样本总数: {total}")
        print(f"  总体准确率 (Overall Accuracy): {format_percentage(overall_acc)}")
        
        # Flaky检测指标
        flaky_metrics = self.metrics['flaky_detection']
        print(f"\n🔍 Flaky检测指标:")
        print(f"  准确率 (Accuracy): {format_percentage(flaky_metrics['accuracy'])}")
        print(f"  精确率 (Precision): {format_percentage(flaky_metrics['precision'])}")
        print(f"  召回率 (Recall): {format_percentage(flaky_metrics['recall'])}")
        print(f"  F1分数: {format_percentage(flaky_metrics['f1'])}")
        
        # 混淆矩阵
        cm = flaky_metrics['confusion_matrix']
        print(f"\n  混淆矩阵:")
        print(f"                预测Flaky  预测Non-Flaky")
        print(f"  实际Flaky      {cm['tp']:>6}      {cm['fn']:>6}")
        print(f"  实际Non-Flaky  {cm['fp']:>6}      {cm['tn']:>6}")
        
        # 类别分类指标
        category_metrics = self.metrics['category_classification']
        print(f"\n📋 类别分类指标:")
        print(f"  分类准确率: {format_percentage(category_metrics['accuracy'])}")
        
        print(f"\n  各类别详细指标:")
        print(f"  {'类别':<15} {'样本数':>8} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10}")
        print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        for category, stats in category_metrics['per_category'].items():
            print(f"  {category:<15} {stats['total']:>8} "
                  f"{format_percentage(stats['accuracy']):>10} "
                  f"{format_percentage(stats['precision']):>10} "
                  f"{format_percentage(stats['recall']):>10} "
                  f"{format_percentage(stats['f1']):>10}")
        
        print("=" * 70)
    
    def print_detailed(self):
        """打印详细报告"""
        self.print_summary()
        
        print("\n" + "=" * 70)
        print("详细分析")
        print("=" * 70)
        
        # 各类别的支持度（样本数）
        category_metrics = self.metrics['category_classification']['per_category']
        print(f"\n📊 类别分布:")
        total = self.metrics['total_samples']
        
        for category, stats in sorted(category_metrics.items(), 
                                      key=lambda x: x[1]['total'], 
                                      reverse=True):
            count = stats['total']
            percentage = count / total * 100 if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"  {category:<15} {count:>4} ({percentage:>5.2f}%) {bar}")
        
        # 性能分析
        print(f"\n📈 性能分析:")
        
        flaky_f1 = self.metrics['flaky_detection']['f1']
        category_acc = self.metrics['category_classification']['accuracy']
        
        if flaky_f1 >= 0.9:
            print(f"  ✅ Flaky检测性能优秀 (F1={format_percentage(flaky_f1)})")
        elif flaky_f1 >= 0.7:
            print(f"  ✓ Flaky检测性能良好 (F1={format_percentage(flaky_f1)})")
        else:
            print(f"  ⚠ Flaky检测性能需要改进 (F1={format_percentage(flaky_f1)})")
        
        if category_acc >= 0.8:
            print(f"  ✅ 类别分类性能优秀 (Acc={format_percentage(category_acc)})")
        elif category_acc >= 0.6:
            print(f"  ✓ 类别分类性能良好 (Acc={format_percentage(category_acc)})")
        else:
            print(f"  ⚠ 类别分类性能需要改进 (Acc={format_percentage(category_acc)})")
        
        # 错误案例分析
        if 'error_cases' in self.metrics and self.metrics['error_cases']:
            self._print_error_cases()
        
        print("=" * 70)
    
    def _print_error_cases(self):
        """打印错误案例详情"""
        error_cases = self.metrics['error_cases']
        
        print(f"\n❌ 错误案例详情 (共 {len(error_cases)} 个):")
        print("=" * 70)
        
        # 按错误类型分组
        flaky_errors = [e for e in error_cases if e['error_type'] in ['flaky', 'both']]
        category_errors = [e for e in error_cases if e['error_type'] in ['category', 'both']]
        
        if flaky_errors:
            print(f"\n🔴 Flaky判断错误 ({len(flaky_errors)} 个):")
            print(f"{'ID':<10} {'预测结果':<25} {'实际结果':<25}")
            print("-" * 70)
            for case in flaky_errors[:20]:  # 最多显示20个
                print(f"{case['id']:<10} {case['predicted']:<25} {case['actual']:<25}")
            if len(flaky_errors) > 20:
                print(f"... 还有 {len(flaky_errors) - 20} 个错误案例")
        
        if category_errors and category_errors != flaky_errors:
            print(f"\n🟡 类别判断错误 ({len(category_errors)} 个):")
            print(f"{'ID':<10} {'预测结果':<25} {'实际结果':<25}")
            print("-" * 70)
            for case in category_errors[:20]:  # 最多显示20个
                print(f"{case['id']:<10} {case['predicted']:<25} {case['actual']:<25}")
            if len(category_errors) > 20:
                print(f"... 还有 {len(category_errors) - 20} 个错误案例")
        
        # 统计错误类型分布
        error_type_counts = {}
        for case in error_cases:
            pred_type = case['predicted'].split(' - ')[1] if ' - ' in case['predicted'] else case['predicted']
            actual_type = case['actual'].split(' - ')[1] if ' - ' in case['actual'] else case['actual']
            key = f"{actual_type} → {pred_type}"
            error_type_counts[key] = error_type_counts.get(key, 0) + 1
        
        if error_type_counts:
            print(f"\n📊 错误类型分布:")
            for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type:<30} {count:>4} 个")
    
    def save_to_json(self, output_file: Path):
        """
        保存报告为JSON格式
        
        Args:
            output_file: 输出文件路径
        """
        from utils import save_json
        save_json(self.metrics, output_file)
    
    def save_to_text(self, output_file: Path):
        """
        保存报告为文本格式
        
        Args:
            output_file: 输出文件路径
        """
        import sys
        from io import StringIO
        
        # 捕获print输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        self.print_detailed()
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"✓ 文本报告已保存到: {output_file}")
