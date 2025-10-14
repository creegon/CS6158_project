"""
数据处理工具函数
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import random


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy和pandas数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'item'):  # 处理pandas的特殊类型
            return obj.item()
        elif hasattr(obj, '__dict__'):  # 处理自定义对象
            return str(obj)
        return super().default(obj)


def load_csv(file_path: Union[str, Path], encoding: str = 'utf-8') -> pd.DataFrame:
    """
    读取CSV文件
    
    Args:
        file_path: CSV文件路径
        encoding: 文件编码
        
    Returns:
        DataFrame对象
    """
    try:
        df = pd.read_csv(f"{file_path}", encoding=encoding)
        print(f"✓ 成功加载数据集: {len(df)} 条记录")
        print(f"  列名: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"✗ 加载CSV文件失败: {e}")
        raise


def sample_data(df: pd.DataFrame, 
                mode: str = 'all',
                n: int = 10,
                random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    从数据集中采样数据
    
    Args:
        df: 原始DataFrame
        mode: 采样模式 ['all', 'first', 'last', 'random']
        n: 采样数量
        random_seed: 随机种子（用于random模式）
        
    Returns:
        采样后的DataFrame
    """
    if mode == 'all':
        return df
    elif mode == 'first':
        return df.head(n)
    elif mode == 'last':
        return df.tail(n)
    elif mode == 'random':
        if random_seed is not None:
            random.seed(random_seed)
        indices = random.sample(range(len(df)), min(n, len(df)))
        return df.iloc[indices]
    else:
        raise ValueError(f"未知的采样模式: {mode}")


def convert_to_alpaca_format(row: pd.Series, 
                             reasoning: str,
                             code_column: str = 'code') -> Dict:
    """
    将数据转换为Alpaca格式
    
    Args:
        row: 数据行
        reasoning: 推理过程
        code_column: 代码列名
        
    Returns:
        Alpaca格式的字典
    """
    test_code = row.get(code_column, row.get('full_code', ''))
    
    alpaca_item = {
        "instruction": "请分析以下测试用例，判断它是否是一个Flaky Test（不稳定测试），并说明你的推理过程。",
        "input": f"测试代码：\n{test_code}",
        "output": reasoning
    }
    
    return alpaca_item


def save_json(data: Union[List, Dict], 
              file_path: Union[str, Path],
              encoding: str = 'utf-8',
              indent: int = 2) -> None:
    """
    保存JSON文件（自动处理numpy和pandas数据类型）
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
        encoding: 文件编码
        indent: 缩进空格数
    """
    def convert_to_serializable(obj):
        """递归转换不可序列化的对象"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'item'):  # 处理pandas的特殊类型（如Int64DType）
            try:
                return obj.item()
            except:
                return str(obj)
        elif pd.isna(obj):  # 处理NaN
            return None
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj
    
    try:
        # 预处理数据，转换所有不可序列化的对象
        serializable_data = convert_to_serializable(data)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=indent, cls=NumpyEncoder)
        print(f"✓ 数据已保存到: {file_path}")
    except Exception as e:
        print(f"✗ 保存JSON文件失败: {e}")
        raise


def load_json(file_path: Union[str, Path], 
              encoding: str = 'utf-8') -> Union[List, Dict]:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        encoding: 文件编码
        
    Returns:
        加载的数据
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        print(f"✓ 成功加载JSON: {file_path}")
        return data
    except Exception as e:
        print(f"✗ 加载JSON文件失败: {e}")
        raise


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    获取数据集统计信息（返回可JSON序列化的数据）
    
    Args:
        df: DataFrame对象
        
    Returns:
        统计信息字典
    """
    # 转换dtypes为字符串（避免不可序列化的类型）
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    
    # 转换missing_values为标准Python整数
    missing_values_dict = {col: int(count) for col, count in df.isnull().sum().to_dict().items()}
    
    stats = {
        "total_records": int(len(df)),
        "columns": df.columns.tolist(),
        "dtypes": dtypes_dict,
        "missing_values": missing_values_dict,
        "memory_usage": float(df.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
    }
    
    # 如果有label列，统计标签分布
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        # 转换为标准Python类型
        stats['label_distribution'] = {str(k): int(v) for k, v in label_counts.items()}
    
    return stats


def print_data_info(df: pd.DataFrame) -> None:
    """
    打印数据集详细信息
    
    Args:
        df: DataFrame对象
    """
    print("=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"记录数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"\n列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} ({df[col].dtype})")
    
    print(f"\n缺失值:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")
    
    if 'label' in df.columns:
        print(f"\n标签分布:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    print(f"\n前3条数据:")
    print(df.head(3).to_string())
    print("=" * 60)
