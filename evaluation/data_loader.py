"""
数据加载器
用于加载预测结果和真实标签
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union
from utils.evaluation_utils import extract_answer, normalize_category


def load_predictions_from_alpaca(json_file: Union[str, Path]) -> Dict[int, Tuple[str, str]]:
    """
    从Alpaca格式的JSON文件加载预测结果
    
    Args:
        json_file: Alpaca格式的JSON文件路径
        
    Returns:
        字典 {test_id: (is_flaky, category)}
    """
    predictions = {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        output = item.get('output', '')
        
        # 提取答案
        is_flaky, category = extract_answer(output)
        
        if is_flaky is None or category is None:
            print(f"⚠ 警告: 第 {i+1} 条记录无法提取答案")
            continue
        
        # 优先使用 id 字段，如果没有则使用索引
        test_id = item.get('id', i)
        predictions[test_id] = (is_flaky, category)
    
    return predictions


def load_ground_truth_from_csv(csv_file: Union[str, Path], 
                                label_column: str = 'label',
                                id_column: str = 'id') -> Dict[int, Tuple[str, str]]:
    """
    从CSV文件加载真实标签
    
    Args:
        csv_file: CSV文件路径
        label_column: 标签列名
        id_column: ID列名（默认为'id'）
        
    Returns:
        字典 {test_id: (is_flaky, category)}
    """
    df = pd.read_csv(csv_file)
    ground_truths = {}
    
    for idx, row in df.iterrows():
        # 获取标签
        label = str(row[label_column]).strip().lower()
        
        # 判断是否为Flaky
        if label == 'non-flaky' or label == 'nonflaky':
            is_flaky = '否'
            category = 'Non-Flaky'
        else:
            is_flaky = '是'
            # 标准化类别
            category = normalize_category(label)
        
        # 优先使用 id 列，如果没有则使用索引
        test_id = int(row[id_column]) if id_column and id_column in df.columns else idx
        ground_truths[test_id] = (is_flaky, category)
    
    return ground_truths


def align_predictions_and_labels(predictions: Dict, 
                                 ground_truths: Dict) -> Tuple[Dict, Dict]:
    """
    对齐预测结果和真实标签
    确保两者有相同的ID集合
    
    Args:
        predictions: 预测结果
        ground_truths: 真实标签
        
    Returns:
        (对齐后的预测结果, 对齐后的真实标签)
    """
    # 找到共同的ID
    common_ids = set(predictions.keys()) & set(ground_truths.keys())
    
    if len(common_ids) == 0:
        print("⚠ 警告: 预测结果和真实标签没有共同的ID")
        return {}, {}
    
    # 只保留共同的ID
    aligned_predictions = {id: predictions[id] for id in common_ids}
    aligned_ground_truths = {id: ground_truths[id] for id in common_ids}
    
    # 打印对齐信息
    print(f"✓ 数据对齐完成:")
    print(f"  预测结果: {len(predictions)} 条")
    print(f"  真实标签: {len(ground_truths)} 条")
    print(f"  对齐后: {len(common_ids)} 条")
    
    if len(common_ids) < len(predictions):
        print(f"  ⚠ {len(predictions) - len(common_ids)} 条预测结果未找到对应标签")
    if len(common_ids) < len(ground_truths):
        print(f"  ⚠ {len(ground_truths) - len(common_ids)} 条标签未找到对应预测")
    
    return aligned_predictions, aligned_ground_truths
