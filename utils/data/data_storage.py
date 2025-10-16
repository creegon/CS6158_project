"""
数据文件存储工具
包括保存数据集和加载JSON文件
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union


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


def save_kfold_datasets(folds: List[Dict[str, pd.DataFrame]],
                       output_dir: Union[str, Path],
                       base_name: str = 'fold',
                       format: str = 'csv') -> Dict[int, Dict[str, Path]]:
    """
    保存K折交叉验证数据集到文件
    
    Args:
        folds: create_project_wise_kfold_splits返回的列表
        output_dir: 输出目录
        base_name: 基础文件名
        format: 保存格式 ('csv' 或 'json')
        
    Returns:
        包含各折文件路径的字典
        
    Example:
        >>> folds = create_project_wise_kfold_splits(df, n_folds=4)
        >>> files = save_kfold_datasets(folds, 'output/kfold_splits')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    print(f"\n💾 保存K折数据集到: {output_dir}")
    
    for fold_idx, fold_data in enumerate(folds, 1):
        fold_files = {}
        
        for split_name in ['train', 'test']:
            split_df = fold_data[split_name]
            
            if format == 'csv':
                file_path = output_dir / f"{base_name}_{fold_idx}_{split_name}.csv"
                split_df.to_csv(file_path, index=False, encoding='utf-8')
            elif format == 'json':
                file_path = output_dir / f"{base_name}_{fold_idx}_{split_name}.json"
                split_df.to_json(file_path, orient='records', force_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            fold_files[split_name] = file_path
        
        # 额外保存项目列表
        projects_file = output_dir / f"{base_name}_{fold_idx}_projects.txt"
        with open(projects_file, 'w', encoding='utf-8') as f:
            f.write(f"训练集项目 ({len(fold_data['train_projects'])}):\n")
            for proj in sorted(fold_data['train_projects']):
                f.write(f"  {proj}\n")
            f.write(f"\n测试集项目 ({len(fold_data['test_projects'])}):\n")
            for proj in sorted(fold_data['test_projects']):
                f.write(f"  {proj}\n")
        
        fold_files['projects'] = projects_file
        saved_files[fold_idx] = fold_files
        
        print(f"  ✓ Fold {fold_idx}: {len(fold_data['train'])} 训练样本, {len(fold_data['test'])} 测试样本")
    
    print(f"✓ 所有数据集保存完成!\n")
    
    return saved_files


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
