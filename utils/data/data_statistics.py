"""
数据统计和信息展示工具
"""
import pandas as pd
from typing import Dict


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
