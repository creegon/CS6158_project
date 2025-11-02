"""
数据格式转换工具
包括Alpaca格式转换
"""
import pandas as pd
from typing import Dict


def convert_to_alpaca_format(row: pd.Series, 
                            reasoning: str,
                            code_column: str = 'code',
                            include_id: bool = False,
                            system_prompt: str = None,
                            user_template: str = None,
                            user_prompt: str = None,  # 新增：直接传入已生成的prompt
                            few_shot_examples: list = None) -> Dict:
    """
    将数据转换为Alpaca格式
    
    Args:
        row: 数据行
        reasoning: 推理过程
        code_column: 代码列名
        include_id: 是否包含ID字段（用于评估）
        system_prompt: 系统提示词（用作instruction）
        user_template: 用户提示词模板（用作input模板）
        user_prompt: 已生成的用户提示词（如果提供则直接使用，优先级高于user_template）
        few_shot_examples: Few-shot样本列表（用于debug）
        
    Returns:
        Alpaca格式的字典
    """
    # 如果提供了 system_prompt，使用它；否则使用默认值
    if system_prompt is None:
        instruction = "请分析以下测试用例，判断它是否是一个Flaky Test（不稳定测试），并说明你的推理过程。"
    else:
        instruction = system_prompt
    
    # 优先使用已生成的 user_prompt
    if user_prompt is not None:
        user_input = user_prompt
    elif user_template is None:
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
        
        # 如果user_prompt已经包含few_shots，直接使用；否则设置为空
        few_shots = row.get('few_shots', '（未提供few-shot案例）')
        
        user_input = format_prompt(
            user_template,
            project=project,
            test_name=test_name,
            full_code=full_code,
            few_shots=few_shots
        )
    
    alpaca_item = {
        "instruction": instruction,
        "input": user_input,
        "output": reasoning
    }
    
    # 如果需要包含ID字段（用于评估）
    if include_id and 'id' in row:
        alpaca_item['id'] = int(row['id'])
        
        # 如果提供了few-shot examples，也记录下来（用于debug）
        if few_shot_examples:
            alpaca_item['few_shot_examples'] = few_shot_examples
    
    return alpaca_item
