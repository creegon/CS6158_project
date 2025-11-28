"""
ChallengerAgent - 杠精Agent
负责对ReasoningAgent的分析进行挑剔和质疑
"""
from typing import Optional
from agents.base_agent import BaseAgent
from utils import load_prompt, format_prompt

class ChallengerAgent(BaseAgent):
    """
    杠精Agent
    
    角色：多疑的代码审计员
    任务：攻击ReasoningAgent生成的推理指引，寻找潜在漏洞
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = load_prompt('challenger_system')
        self.user_template = load_prompt('challenger_user')
        print("✓ ChallengerAgent已初始化 (准备抬杠)")
        
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个极其挑剔、多疑的代码审计专家。"

    def run(self, *args, **kwargs):
        """
        执行Agent的主要任务
        """
        return self.challenge(*args, **kwargs)

    def challenge(self, 
                 project: str, 
                 test_name: str, 
                 full_code: str, 
                 reasoning_guide: str) -> Optional[str]:
        """
        生成质疑
        """
        prompt = format_prompt(
            self.user_template,
            project=project,
            test_name=test_name,
            full_code=full_code,
            reasoning_guide=reasoning_guide
        )
        
        try:
            critique = self.call_api(
                prompt,
                system_prompt=self.system_prompt
            )
            return critique
        except Exception as e:
            print(f"⚠ 质疑生成失败: {e}")
            return None
