"""
数据加载和采样工具
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import random


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
