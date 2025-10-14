"""
Prompt处理工具函数
"""
from pathlib import Path
from typing import Dict, Optional


def load_prompt(prompt_name: str, prompts_dir: Optional[Path] = None) -> str:
    """
    从文件加载prompt模板
    
    Args:
        prompt_name: prompt文件名（不含.txt扩展名）
        prompts_dir: prompts目录路径
        
    Returns:
        prompt内容
    """
    if prompts_dir is None:
        from config import PROMPTS_DIR
        prompts_dir = PROMPTS_DIR
    
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到prompt文件: {prompt_file}")
    except Exception as e:
        raise Exception(f"读取prompt文件失败: {e}")


def format_prompt(template: str, **kwargs) -> str:
    """
    格式化prompt模板
    
    Args:
        template: prompt模板字符串
        **kwargs: 要填充的变量
        
    Returns:
        格式化后的prompt
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"缺少必需的变量: {e}")


def load_and_format_prompt(prompt_name: str, 
                           prompts_dir: Optional[Path] = None,
                           **kwargs) -> str:
    """
    加载并格式化prompt
    
    Args:
        prompt_name: prompt文件名
        prompts_dir: prompts目录路径
        **kwargs: 要填充的变量
        
    Returns:
        格式化后的prompt
    """
    template = load_prompt(prompt_name, prompts_dir)
    return format_prompt(template, **kwargs)


def save_prompt(prompt_name: str, 
                content: str,
                prompts_dir: Optional[Path] = None) -> None:
    """
    保存prompt到文件
    
    Args:
        prompt_name: prompt文件名（不含.txt扩展名）
        content: prompt内容
        prompts_dir: prompts目录路径
    """
    if prompts_dir is None:
        from config import PROMPTS_DIR
        prompts_dir = PROMPTS_DIR
    
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    try:
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Prompt已保存到: {prompt_file}")
    except Exception as e:
        raise Exception(f"保存prompt文件失败: {e}")
