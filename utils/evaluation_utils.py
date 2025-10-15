"""
评估工具函数
用于解析预测结果和计算评估指标
"""
import re
from typing import Dict, Tuple, Optional


def extract_answer(output_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从输出文本中提取答案
    
    Args:
        output_text: 模型输出的文本
        
    Returns:
        (是否Flaky, 类型) 的元组
        - 是否Flaky: '是' 或 '否'
        - 类型: 'Async', 'Conc', 'Time', 'UC', 'OD', 'Non-Flaky'
    """
    # 匹配多种可能的格式
    patterns = [
        r'答案[：:]\s*(是|否)\s*[-–—]\s*(\w+[-\w]*)',
        r'Answer[：:]\s*(Yes|No|是|否)\s*[-–—]\s*(\w+[-\w]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            is_flaky = match.group(1)
            category = match.group(2)
            
            # 标准化
            if is_flaky.lower() in ['yes', '是']:
                is_flaky = '是'
            elif is_flaky.lower() in ['no', '否']:
                is_flaky = '否'
            
            return is_flaky, category
    
    return None, None


def normalize_category(category: str) -> str:
    """
    标准化分类名称
    
    Args:
        category: 原始分类名称
        
    Returns:
        标准化后的分类名称
    """
    if not category:
        return 'Unknown'
    
    category_upper = category.upper()
    
    # 映射表
    category_map = {
        'ASYNC': 'Async',
        'ASYNC WAIT': 'Async',
        'ASYNC.': 'Async',
        'CONC': 'Conc',
        'CONC.': 'Conc',
        'CONCURRENCY': 'Conc',
        'TIME': 'Time',
        'TIME.': 'Time',
        'UC': 'UC',
        'UNORDERED COLLECTIONS': 'UC',
        'UNORDERED COLLECTION': 'UC',
        'OD': 'OD',
        'ORDER DEPENDENCY': 'OD',
        'TEST ORDER DEPENDENCY': 'OD',
        'NON-FLAKY': 'Non-Flaky',
        'NONFLAKY': 'Non-Flaky',
        'NON FLAKY': 'Non-Flaky',
    }
    
    return category_map.get(category_upper, category)


def calculate_metrics(predictions: Dict, ground_truths: Dict) -> Dict:
    """
    计算评估指标
    
    Args:
        predictions: 预测结果字典 {id: (is_flaky, category)}
        ground_truths: 真实标签字典 {id: (is_flaky, category)}
        
    Returns:
        包含各项指标的字典
    """
    # 初始化计数器
    total = 0
    correct_flaky = 0  # Flaky判断正确
    correct_category = 0  # 类型判断正确
    correct_both = 0  # 都正确
    
    # 按类别统计
    category_stats = {
        'Async': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
        'Conc': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
        'Time': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
        'UC': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
        'OD': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
        'Non-Flaky': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
    }
    
    # Flaky vs Non-Flaky统计
    flaky_confusion = {
        'tp': 0,  # 真正例：预测Flaky，实际Flaky
        'fp': 0,  # 假正例：预测Flaky，实际Non-Flaky
        'tn': 0,  # 真负例：预测Non-Flaky，实际Non-Flaky
        'fn': 0,  # 假负例：预测Non-Flaky，实际Flaky
    }
    
    # 收集错误案例
    error_cases = []
    
    # 遍历所有样本
    for test_id in ground_truths:
        if test_id not in predictions:
            continue
        
        total += 1
        pred_flaky, pred_category = predictions[test_id]
        true_flaky, true_category = ground_truths[test_id]
        
        # 标准化
        pred_category = normalize_category(pred_category)
        true_category = normalize_category(true_category)
        
        # Flaky判断
        if pred_flaky == true_flaky:
            correct_flaky += 1
        
        # 类型判断
        if pred_category == true_category:
            correct_category += 1
        
        # 都正确
        if pred_flaky == true_flaky and pred_category == true_category:
            correct_both += 1
        else:
            # 记录错误案例
            error_cases.append({
                'id': test_id,
                'predicted': f"{pred_flaky} - {pred_category}",
                'actual': f"{true_flaky} - {true_category}",
                'error_type': 'both' if (pred_flaky != true_flaky and pred_category != true_category) 
                             else ('flaky' if pred_flaky != true_flaky else 'category')
            })
        
        # Flaky混淆矩阵
        pred_is_flaky = (pred_flaky == '是')
        true_is_flaky = (true_flaky == '是')
        
        if pred_is_flaky and true_is_flaky:
            flaky_confusion['tp'] += 1
        elif pred_is_flaky and not true_is_flaky:
            flaky_confusion['fp'] += 1
        elif not pred_is_flaky and not true_is_flaky:
            flaky_confusion['tn'] += 1
        elif not pred_is_flaky and true_is_flaky:
            flaky_confusion['fn'] += 1
        
        # 按类别统计
        if true_category in category_stats:
            category_stats[true_category]['total'] += 1
            
            if pred_category == true_category:
                category_stats[true_category]['tp'] += 1
            else:
                category_stats[true_category]['fn'] += 1
                if pred_category in category_stats:
                    category_stats[pred_category]['fp'] += 1
    
    # 计算总体指标
    overall_acc = correct_both / total if total > 0 else 0
    flaky_acc = correct_flaky / total if total > 0 else 0
    category_acc = correct_category / total if total > 0 else 0
    
    # 计算Flaky检测的P/R/F1
    flaky_precision = flaky_confusion['tp'] / (flaky_confusion['tp'] + flaky_confusion['fp']) \
        if (flaky_confusion['tp'] + flaky_confusion['fp']) > 0 else 0
    flaky_recall = flaky_confusion['tp'] / (flaky_confusion['tp'] + flaky_confusion['fn']) \
        if (flaky_confusion['tp'] + flaky_confusion['fn']) > 0 else 0
    flaky_f1 = 2 * flaky_precision * flaky_recall / (flaky_precision + flaky_recall) \
        if (flaky_precision + flaky_recall) > 0 else 0
    
    # 计算各类别的P/R/F1
    category_metrics = {}
    for category, stats in category_stats.items():
        if stats['total'] == 0:
            continue
        
        precision = stats['tp'] / (stats['tp'] + stats['fp']) \
            if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) \
            if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0
        accuracy = stats['tp'] / stats['total'] if stats['total'] > 0 else 0
        
        category_metrics[category] = {
            'total': stats['total'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {
        'total_samples': total,
        'overall_accuracy': overall_acc,
        'flaky_detection': {
            'accuracy': flaky_acc,
            'precision': flaky_precision,
            'recall': flaky_recall,
            'f1': flaky_f1,
            'confusion_matrix': flaky_confusion
        },
        'category_classification': {
            'accuracy': category_acc,
            'per_category': category_metrics
        },
        'error_cases': error_cases  # 添加错误案例
    }


def format_percentage(value: float, decimal: int = 2) -> str:
    """
    格式化百分比
    
    Args:
        value: 0-1之间的值
        decimal: 小数位数
        
    Returns:
        格式化后的百分比字符串
    """
    return f"{value * 100:.{decimal}f}%"
