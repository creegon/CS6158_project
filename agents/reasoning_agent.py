"""
ReasoningAgent - 推理指引Agent
用于生成结构化的推理指引，帮助判断Agent避免常见的逻辑陷阱
"""
from typing import Optional
import pandas as pd

from agents.base_agent import BaseAgent
from utils import load_prompt, format_prompt


class ReasoningAgent(BaseAgent):
    """
    推理指引Agent
    
    作为双Agent推理链的第一步，负责：
    1. 分析测试代码结构
    2. 识别潜在风险点
    3. 提出关键推理问题
    4. 警示常见陷阱
    5. 建议推理路径
    
    目标: 引导后续的判断Agent进行因果推理，而非简单的模式匹配
    """
    
    def __init__(self, **kwargs):
        """
        初始化ReasoningAgent
        
        Args:
            **kwargs: 传递给BaseAgent的参数
        """
        super().__init__(**kwargs)
        
        # 加载推理指引专用的prompt模板
        self.system_prompt = load_prompt('reasoning_guide_system')
        self.user_template = load_prompt('reasoning_guide_user')
        
        # 加载辩护专用的prompt模板
        self.defense_system_prompt = load_prompt('reasoning_defense_system')
        self.defense_user_template = load_prompt('reasoning_defense_user')
        
        print("✓ ReasoningAgent已初始化")
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个推理指引专家，负责分析测试代码并提供结构化的推理框架。"
    
    def generate_reasoning_guide(self,
                                project: str,
                                test_name: str,
                                full_code: str) -> Optional[str]:
        """
        生成推理指引
        
        Args:
            project: 项目名称
            test_name: 测试名称
            full_code: 完整测试代码
            
        Returns:
            推理指引文本，失败返回None
        """
        # 构建推理指引的prompt
        guide_prompt = format_prompt(
            self.user_template,
            project=project,
            test_name=test_name,
            full_code=full_code
        )
        
        # 调用API获取推理指引
        try:
            reasoning_guide = self.call_api(
                guide_prompt,
                system_prompt=self.system_prompt
            )
            return reasoning_guide
        except Exception as e:
            print(f"⚠ 生成推理指引失败: {e}")
            return None
    
    def generate_from_row(self, row: pd.Series, code_column: str = 'code') -> Optional[str]:
        """
        从DataFrame行生成推理指引（便捷方法）
        
        Args:
            row: 数据行
            code_column: 代码列名
            
        Returns:
            推理指引文本，失败返回None
        """
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        full_code = row.get(code_column, row.get('full_code', ''))
        
        return self.generate_reasoning_guide(project, test_name, full_code)
    
    def run(self, project: str, test_name: str, full_code: str, **kwargs) -> Optional[str]:
        """
        执行推理指引生成任务（实现BaseAgent的抽象方法）
        
        Args:
            project: 项目名称
            test_name: 测试名称
            full_code: 完整测试代码
            **kwargs: 其他参数（未使用）
            
        Returns:
            推理指引文本，失败返回None
        """
        return self.generate_reasoning_guide(project, test_name, full_code)

    def defend_analysis(self,
                       project: str,
                       test_name: str,
                       full_code: str,
                       reasoning_guide: str,
                       critique: str) -> Optional[str]:
        """
        针对质疑进行辩护
        """
        prompt = format_prompt(
            self.defense_user_template,
            project=project,
            test_name=test_name,
            full_code=full_code,
            reasoning_guide=reasoning_guide,
            critique=critique
        )
        
        try:
            defense = self.call_api(
                prompt,
                system_prompt=self.defense_system_prompt
            )
            return defense
        except Exception as e:
            print(f"⚠ 辩护生成失败: {e}")
            return None
