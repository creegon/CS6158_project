"""
BaseAgent - 所有Agent的基类
"""
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    CURRENT_PROVIDER,
    get_api_config,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_RETRIES
)


class BaseAgent(ABC):
    """
    Agent基类，封装通用的API调用逻辑
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 max_retries: Optional[int] = None,
                 system_prompt: Optional[str] = None,
                 provider: Optional[str] = None):
        """
        初始化Agent
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            system_prompt: 系统提示词
            provider: API提供商 (deepseek/siliconflow)
        """
        # 如果指定了provider，使用对应的配置
        if provider:
            provider_api_key, provider_base_url, provider_model = get_api_config(provider)
            self.api_key = api_key or provider_api_key
            self.base_url = base_url or provider_base_url
            self.model = model or provider_model
            self.provider = provider
        else:
            # 使用当前配置的提供商
            provider_api_key, provider_base_url, provider_model = get_api_config(CURRENT_PROVIDER)
            self.api_key = api_key or provider_api_key
            self.base_url = base_url or provider_base_url
            self.model = model or provider_model
            self.provider = CURRENT_PROVIDER
            
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.max_retries = max_retries or DEFAULT_MAX_RETRIES
        self.system_prompt = system_prompt or self.get_default_system_prompt()
        
        # 初始化客户端
        if self.provider == "gemini":
            if genai is None:
                raise ImportError("请安装 google-genai 库以使用 Gemini 模型: pip install google-genai")
            self.client = genai.Client(api_key=self.api_key)
        else:
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        
        # 统计信息
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "provider": self.provider,
            "model": self.model
        }
    
    @abstractmethod
    def get_default_system_prompt(self) -> str:
        """
        获取默认的系统提示词（子类必须实现）
        
        Returns:
            系统提示词
        """
        pass
    
    def call_api(self, 
                 user_prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> Optional[str]:
        """
        调用API生成响应
        
        Args:
            user_prompt: 用户提示词
            system_prompt: 系统提示词（可选，默认使用实例的system_prompt）
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）
            
        Returns:
            生成的响应内容，失败返回None
        """
        system_prompt = system_prompt or self.system_prompt
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        for attempt in range(self.max_retries):
            try:
                self.stats["total_calls"] += 1
                
                if self.provider == "gemini":
                    # Gemini API调用
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=8192  # 设为Gemini最大值
                    )
                    
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=config
                    )
                    
                    content = response.text
                    
                    # 更新统计
                    self.stats["successful_calls"] += 1
                    if hasattr(response, 'usage_metadata'):
                        self.stats["total_tokens"] += response.usage_metadata.total_token_count
                        
                else:
                    # OpenAI API调用
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    content = response.choices[0].message.content
                    
                    # 更新统计
                    self.stats["successful_calls"] += 1
                    if hasattr(response, 'usage'):
                        self.stats["total_tokens"] += response.usage.total_tokens
                
                return content
                
            except Exception as e:
                print(f"⚠ API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                self.stats["failed_calls"] += 1
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    print(f"✗ API调用最终失败")
                    return None
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """
        重置统计信息
        """
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "provider": self.provider,
            "model": self.model
        }
    
    def print_stats(self) -> None:
        """
        打印统计信息
        """
        print("\n" + "=" * 60)
        print("📊 统计信息")
        print("=" * 60)
        print(f"🔧 提供商: {self.stats.get('provider', 'unknown')}")
        print(f"🤖 模型: {self.stats.get('model', 'unknown')}")
        print(f"📞 API调用总次数: {self.stats['total_calls']}")
        print(f"✅ 成功: {self.stats['successful_calls']}")
        print(f"❌ 失败: {self.stats['failed_calls']}")
        print(f"🎯 成功率: {self.stats['successful_calls'] / max(self.stats['total_calls'], 1) * 100:.2f}%")
        print(f"💰 总Token使用量: {self.stats['total_tokens']}")
        print("=" * 60 + "\n")
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """
        执行Agent的主要任务（子类必须实现）
        """
        pass
