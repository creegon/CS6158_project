"""
配置管理工具
用于保存和加载实验配置
"""
import json
from pathlib import Path
from typing import Dict, Optional, List


CONFIG_DIR = Path(__file__).parent.parent / 'configs'
CONFIG_DIR.mkdir(exist_ok=True)


def save_config(config: Dict, name: str) -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        name: 配置名称
        
    Returns:
        是否保存成功
    """
    try:
        config_file = CONFIG_DIR / f"{name}.json"
        
        # 转换Path对象为字符串
        config_to_save = {}
        for key, value in config.items():
            if isinstance(value, Path):
                config_to_save[key] = str(value)
            else:
                config_to_save[key] = value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 配置已保存: {config_file}")
        return True
    
    except Exception as e:
        print(f"✗ 保存配置失败: {e}")
        return False


def load_config(name: str) -> Optional[Dict]:
    """
    从文件加载配置
    
    Args:
        name: 配置名称
        
    Returns:
        配置字典，如果不存在则返回None
    """
    try:
        config_file = CONFIG_DIR / f"{name}.json"
        
        if not config_file.exists():
            print(f"✗ 配置文件不存在: {name}")
            return None
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 转换字符串路径为Path对象
        if 'test_dataset' in config:
            config['test_dataset'] = Path(config['test_dataset'])
        if 'train_dataset' in config and config['train_dataset']:
            config['train_dataset'] = Path(config['train_dataset'])
        
        return config
    
    except Exception as e:
        print(f"✗ 加载配置失败: {e}")
        return None


def list_saved_configs() -> List[str]:
    """
    列出所有保存的配置
    
    Returns:
        配置名称列表
    """
    if not CONFIG_DIR.exists():
        return []
    
    configs = []
    for config_file in CONFIG_DIR.glob('*.json'):
        configs.append(config_file.stem)
    
    return sorted(configs)


def delete_config(name: str) -> bool:
    """
    删除配置文件
    
    Args:
        name: 配置名称
        
    Returns:
        是否删除成功
    """
    try:
        config_file = CONFIG_DIR / f"{name}.json"
        
        if not config_file.exists():
            print(f"✗ 配置文件不存在: {name}")
            return False
        
        config_file.unlink()
        print(f"✓ 配置已删除: {name}")
        return True
    
    except Exception as e:
        print(f"✗ 删除配置失败: {e}")
        return False


def display_config(config: Dict) -> None:
    """
    显示配置内容
    
    Args:
        config: 配置字典
    """
    print("\n配置详情:")
    print("=" * 60)
    
    # 任务类型
    print(f"任务类型: {config.get('task_type', 'Unknown')}")
    
    # 数据集
    test_dataset = config.get('test_dataset', 'Unknown')
    if isinstance(test_dataset, Path):
        test_dataset = test_dataset.name
    print(f"测试集: {test_dataset}")
    
    train_dataset = config.get('train_dataset')
    if train_dataset:
        if isinstance(train_dataset, Path):
            train_dataset = train_dataset.name
        print(f"训练集: {train_dataset}")
    
    # API匹配
    if config.get('use_api_matching'):
        print(f"API匹配: 开启 (Top-{config.get('top_k_shots', 3)} few-shots)")
    else:
        print("API匹配: 关闭")
    
    # 测试模式
    mode = config.get('mode', 'Unknown')
    test_size = config.get('test_size')
    print(f"测试模式: {mode}")
    print(f"数据量: {test_size if test_size else '全部'}")
    
    # 并行配置
    print(f"并行线程: {config.get('parallel_workers', 1)}")
    print(f"批次大小: {config.get('batch_size', 5)}")
    
    print("=" * 60)
