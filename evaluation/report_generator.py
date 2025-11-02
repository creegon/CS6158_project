"""
评估报告生成器
生成详细的评估报告
"""
from typing import Dict, List, Any
from pathlib import Path
import json
from utils.evaluation_utils import format_percentage


def check_context_availability(pred_item: Dict[str, Any]) -> tuple:
    """
    检查是否包含有效的上下文信息
    
    Args:
        pred_item: 预测项字典，包含input字段
        
    Returns:
        (has_context_window, has_calling_functions) 元组
    """
    input_text = pred_item.get('input', '')
    
    # 检查是否有上下文窗口信息
    has_context_window = False
    has_calling_functions = False
    
    if "该测试案例在原项目中的上下文为：" in input_text:
        # 提取上下文部分
        context_section = input_text.split("该测试案例在原项目中的上下文为：")[1]
        if "在原项目中，涉及到调用该测试案例的原文为：" in context_section:
            context_content = context_section.split("在原项目中，涉及到调用该测试案例的原文为：")[0]
        else:
            context_content = context_section
        
        # 检查是否是有效内容（不是错误信息）
        if "（无法获取上下文信息）" not in context_content and "文件路径:" in context_content:
            has_context_window = True
    
    if "在原项目中，涉及到调用该测试案例的原文为：" in input_text:
        # 提取调用函数部分
        calling_section = input_text.split("在原项目中，涉及到调用该测试案例的原文为：")[1]
        
        # 检查是否是有效内容（获取前200字符检查）
        calling_preview = calling_section[:200] if len(calling_section) > 200 else calling_section
        if "（无法获取调用信息）" not in calling_preview and "（未找到调用该测试方法的位置）" not in calling_preview:
            has_calling_functions = True
    
    return has_context_window, has_calling_functions


class EvaluationReport:
    """评估报告类"""
    
    def __init__(self, metrics: Dict, predictions: List[Dict] = None):
        """
        初始化报告
        
        Args:
            metrics: 评估指标字典
            predictions: 预测数据列表（包含input和output字段）
        """
        self.metrics = metrics
        self.predictions = predictions or []
    
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
        
        # 上下文信息影响分析
        if self.predictions:
            self._analyze_context_impact()
        
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
    
    def _analyze_context_impact(self):
        """分析上下文信息对预测准确率的影响"""
        if not self.predictions:
            return
        
        print("\n" + "="*70)
        print("📋 上下文信息影响分析")
        print("="*70)
        
        # 分类案例
        with_context_window = []
        without_context_window = []
        with_calling_info = []
        without_calling_info = []
        with_both = []
        with_neither = []
        
        # 获取错误案例ID集合
        error_ids = set()
        for error_case in self.metrics.get('error_cases', []):
            error_ids.add(error_case['id'])
        
        # 分类每个预测
        for pred in self.predictions:
            pred_id = pred.get('id')
            
            # 检查上下文可用性
            has_context, has_calling = check_context_availability(pred)
            
            # 检查预测是否正确
            is_correct = pred_id not in error_ids
            
            case_info = {
                'id': pred_id,
                'is_correct': is_correct
            }
            
            # 分类
            if has_context:
                with_context_window.append(case_info)
            else:
                without_context_window.append(case_info)
            
            if has_calling:
                with_calling_info.append(case_info)
            else:
                without_calling_info.append(case_info)
            
            if has_context and has_calling:
                with_both.append(case_info)
            elif not has_context and not has_calling:
                with_neither.append(case_info)
        
        # 计算准确率
        def calc_accuracy(cases):
            if not cases:
                return 0.0
            correct = sum(1 for c in cases if c['is_correct'])
            return correct / len(cases) * 100
        
        total = len(self.predictions)
        
        # 打印统计信息
        print(f"\n【上下文信息可用性】")
        print(f"  有上下文窗口: {len(with_context_window):>3} ({len(with_context_window)/total*100:5.1f}%)")
        print(f"  无上下文窗口: {len(without_context_window):>3} ({len(without_context_window)/total*100:5.1f}%)")
        print(f"  有调用信息:   {len(with_calling_info):>3} ({len(with_calling_info)/total*100:5.1f}%)")
        print(f"  无调用信息:   {len(without_calling_info):>3} ({len(without_calling_info)/total*100:5.1f}%)")
        print(f"  两者都有:     {len(with_both):>3} ({len(with_both)/total*100:5.1f}%)")
        print(f"  两者都无:     {len(with_neither):>3} ({len(with_neither)/total*100:5.1f}%)")
        
        # 计算准确率对比
        with_context_acc = calc_accuracy(with_context_window)
        without_context_acc = calc_accuracy(without_context_window)
        with_calling_acc = calc_accuracy(with_calling_info)
        without_calling_acc = calc_accuracy(without_calling_info)
        with_both_acc = calc_accuracy(with_both)
        with_neither_acc = calc_accuracy(with_neither)
        
        print(f"\n【准确率对比】")
        
        # 上下文窗口影响
        if with_context_window and without_context_window:
            diff_context = with_context_acc - without_context_acc
            print(f"  上下文窗口:")
            print(f"    有: {with_context_acc:5.2f}% ({len(with_context_window):>2}样本)")
            print(f"    无: {without_context_acc:5.2f}% ({len(without_context_window):>2}样本)")
            if abs(diff_context) >= 1.0:  # 只有差异大于1%时才显示
                status = "✅ 提升" if diff_context > 0 else "⚠️ 下降"
                print(f"    差异: {diff_context:+6.2f}% {status}")
        
        # 调用信息影响
        if with_calling_info and without_calling_info:
            diff_calling = with_calling_acc - without_calling_acc
            print(f"  调用信息:")
            print(f"    有: {with_calling_acc:5.2f}% ({len(with_calling_info):>2}样本)")
            print(f"    无: {without_calling_acc:5.2f}% ({len(without_calling_info):>2}样本)")
            if abs(diff_calling) >= 1.0:
                status = "✅ 提升" if diff_calling > 0 else "⚠️ 下降"
                print(f"    差异: {diff_calling:+6.2f}% {status}")
        
        # 组合效果
        if with_both and with_neither:
            diff_both = with_both_acc - with_neither_acc
            print(f"  组合效果:")
            print(f"    都有: {with_both_acc:5.2f}% ({len(with_both):>2}样本)")
            print(f"    都无: {with_neither_acc:5.2f}% ({len(with_neither):>2}样本)")
            if abs(diff_both) >= 1.0:
                status = "✅ 提升" if diff_both > 0 else "⚠️ 下降"
                print(f"    差异: {diff_both:+6.2f}% {status}")
        
        # 总结建议
        print(f"\n【分析结论】")
        if with_context_window and without_context_window:
            if with_context_acc > without_context_acc + 5:
                print("  ✅ 上下文窗口信息显著提升预测准确率")
            elif with_context_acc > without_context_acc:
                print("  ✓ 上下文窗口信息略微提升预测准确率")
            elif with_context_acc < without_context_acc - 5:
                print("  ⚠️ 上下文窗口信息反而降低准确率，需要优化提取质量")
            else:
                print("  ➖ 上下文窗口信息对准确率影响不明显")
        
        success_rate = len(with_context_window) / total * 100 if total > 0 else 0
        if success_rate < 50:
            print(f"  ⚠️ 上下文提取成功率较低({success_rate:.1f}%)，建议改进context_extractor")
        elif success_rate < 70:
            print(f"  ⚡ 上下文提取成功率中等({success_rate:.1f}%)，有优化空间")
        else:
            print(f"  ✅ 上下文提取成功率良好({success_rate:.1f}%)")
    
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
