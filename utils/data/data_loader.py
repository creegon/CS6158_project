"""
数据加载和采样工具
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import random
import numpy as np


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
        random_seed: 随机种子（用于random模式，确保可复现）
        
    Returns:
        采样后的DataFrame
        
    Examples:
        >>> df = load_csv('dataset.csv')
        >>> # 获取前10条
        >>> sample_df = sample_data(df, mode='first', n=10)
        >>> # 随机采样10条（可复现）
        >>> sample_df = sample_data(df, mode='random', n=10, random_seed=42)
    """
    if mode == 'all':
        return df
    elif mode == 'first':
        return df.head(n)
    elif mode == 'last':
        return df.tail(n)
    elif mode == 'random':
        # 设置随机种子以确保可复现性
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 使用 pandas 的 sample 方法（基于 numpy.random）
        sample_size = min(n, len(df))
        return df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
    else:
        raise ValueError(f"未知的采样模式: {mode}，支持的模式: ['all', 'first', 'last', 'random']")
