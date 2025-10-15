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


def split_dataset(df: pd.DataFrame,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 stratify_column: Optional[str] = None,
                 random_seed: int = 42,
                 shuffle: bool = True) -> Dict[str, pd.DataFrame]:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        df: 原始DataFrame
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.2）
        test_ratio: 测试集比例（默认0.1）
        stratify_column: 分层采样的列名（如'label'），None表示不分层
        random_seed: 随机种子（用于复现）
        shuffle: 是否在划分前打乱数据
        
    Returns:
        包含 'train', 'val', 'test' 的字典，值为对应的DataFrame
        
    Example:
        >>> splits = split_dataset(df, stratify_column='label', random_seed=42)
        >>> train_df = splits['train']
        >>> val_df = splits['val']
        >>> test_df = splits['test']
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1，当前为: {total_ratio}")
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 如果需要分层采样
    if stratify_column and stratify_column in df.columns:
        print(f"\n📊 使用分层采样，基于列: {stratify_column}")
        
        # 获取各类别
        categories = df[stratify_column].unique()
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        print(f"\n各类别划分情况:")
        print(f"{'类别':<20} {'总数':>8} {'训练集':>8} {'验证集':>8} {'测试集':>8}")
        print("-" * 60)
        
        for category in categories:
            # 获取该类别的所有数据
            cat_df = df[df[stratify_column] == category].copy()
            cat_size = len(cat_df)
            
            # 打乱数据
            if shuffle:
                cat_df = cat_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # 计算划分点
            train_size = int(cat_size * train_ratio)
            val_size = int(cat_size * val_ratio)
            
            # 划分
            train_cat = cat_df.iloc[:train_size]
            val_cat = cat_df.iloc[train_size:train_size + val_size]
            test_cat = cat_df.iloc[train_size + val_size:]
            
            train_dfs.append(train_cat)
            val_dfs.append(val_cat)
            test_dfs.append(test_cat)
            
            print(f"{str(category):<20} {cat_size:>8} {len(train_cat):>8} {len(val_cat):>8} {len(test_cat):>8}")
        
        # 合并所有类别
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # 再次打乱（可选）
        if shuffle:
            train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            val_df = val_df.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)
    
    else:
        print(f"\n📊 使用随机划分（不分层）")
        
        # 打乱数据
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # 计算划分点
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # 划分
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
    
    # 打印总体统计
    print(f"\n{'=' * 60}")
    print(f"划分结果:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  总计: {len(train_df) + len(val_df) + len(test_df)} 条")
    print(f"{'=' * 60}\n")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def save_split_datasets(splits: Dict[str, pd.DataFrame],
                       output_dir: Union[str, Path],
                       base_name: str = 'dataset',
                       format: str = 'csv') -> Dict[str, Path]:
    """
    保存划分后的数据集到文件
    
    Args:
        splits: split_dataset返回的字典
        output_dir: 输出目录
        base_name: 基础文件名
        format: 保存格式 ('csv' 或 'json')
        
    Returns:
        包含各数据集文件路径的字典
        
    Example:
        >>> splits = split_dataset(df)
        >>> files = save_split_datasets(splits, 'output/splits')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    for split_name, split_df in splits.items():
        if format == 'csv':
            file_path = output_dir / f"{base_name}_{split_name}.csv"
            split_df.to_csv(file_path, index=False, encoding='utf-8')
        elif format == 'json':
            file_path = output_dir / f"{base_name}_{split_name}.json"
            split_df.to_json(file_path, orient='records', force_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        saved_files[split_name] = file_path
        print(f"✓ 已保存 {split_name} 集到: {file_path}")
    
    return saved_files


def convert_to_alpaca_format(row: pd.Series, 
                            reasoning: str,
                            code_column: str = 'code',
                            include_id: bool = False,
                            system_prompt: str = None,
                            user_template: str = None) -> Dict:
    """
    将数据转换为Alpaca格式
    
    Args:
        row: 数据行
        reasoning: 推理过程
        code_column: 代码列名
        include_id: 是否包含ID字段（用于评估）
        system_prompt: 系统提示词（用作instruction）
        user_template: 用户提示词模板（用作input模板）
        
    Returns:
        Alpaca格式的字典
    """
    # 如果提供了 system_prompt 和 user_template，使用它们
    # 否则使用默认值
    if system_prompt is None:
        instruction = "请分析以下测试用例，判断它是否是一个Flaky Test（不稳定测试），并说明你的推理过程。"
    else:
        instruction = system_prompt
    
    if user_template is None:
        # 默认格式
        full_code = row.get(code_column, row.get('full_code', ''))
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        user_input = f"该测试代码所属project的名称为{project}，它的测试名称为{test_name}，完整代码为{full_code}。"
    else:
        # 使用模板格式化
        from utils.prompt_utils import format_prompt
        full_code = row.get(code_column, row.get('full_code', ''))
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        user_input = format_prompt(
            user_template,
            project=project,
            test_name=test_name,
            full_code=full_code
        )
    
    alpaca_item = {
        "instruction": instruction,
        "input": user_input,
        "output": reasoning
    }
    
    # 如果需要包含ID字段（用于评估）
    if include_id and 'id' in row:
        alpaca_item['id'] = int(row['id'])
    
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
