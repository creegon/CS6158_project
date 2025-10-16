"""
API提供商管理工具
"""
import os
from pathlib import Path
from typing import Tuple, Optional


def get_env_file_path() -> Path:
    """获取.env文件路径"""
    return Path(__file__).parent.parent / '.env'


def switch_provider(provider: str) -> bool:
    """
    切换API提供商
    
    Args:
        provider: 提供商名称 (deepseek/siliconflow)
        
    Returns:
        是否切换成功
    """
    valid_providers = ['deepseek', 'siliconflow']
    provider = provider.lower()
    
    if provider not in valid_providers:
        print(f"❌ 无效的提供商: {provider}")
        print(f"   可选: {', '.join(valid_providers)}")
        return False
    
    env_file = get_env_file_path()
    
    if not env_file.exists():
        print("❌ 找不到 .env 文件")
        print("   提示: 复制 .env.example 为 .env")
        return False
    
    # 读取现有内容
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 更新 CURRENT_PROVIDER
    found = False
    for i, line in enumerate(lines):
        if line.startswith('CURRENT_PROVIDER='):
            lines[i] = f'CURRENT_PROVIDER={provider}\n'
            found = True
            break
    
    if not found:
        lines.append(f'\nCURRENT_PROVIDER={provider}\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ API提供商已切换为: {provider.upper()}")
    
    # 显示对应的默认模型
    from config import get_api_config
    api_key, base_url, default_model = get_api_config(provider)
    
    print(f"\n📌 当前配置:")
    print(f"   提供商: {provider}")
    print(f"   默认模型: {default_model}")
    print(f"   API URL: {base_url}")
    print(f"   API密钥: {'已配置 ✓' if api_key else '未配置 ✗'}")
    
    if not api_key:
        print(f"\n⚠️  警告: 请在 .env 文件中配置 {provider.upper()}_API_KEY")
    
    return True


def get_current_config() -> Tuple[str, str, str, str, bool]:
    """
    获取当前API配置
    
    Returns:
        (provider, model, base_url, api_key_status, has_key) 的元组
    """
    from config import CURRENT_PROVIDER, get_api_config
    
    api_key, base_url, default_model = get_api_config()
    has_key = bool(api_key)
    api_key_status = '已配置 ✓' if has_key else '未配置 ✗'
    
    return CURRENT_PROVIDER, default_model, base_url, api_key_status, has_key


def show_current_config() -> None:
    """显示当前配置"""
    from config import CURRENT_PROVIDER, get_api_config, SUPPORTED_MODELS
    
    api_key, base_url, default_model = get_api_config()
    
    print("\n" + "=" * 60)
    print("当前API配置")
    print("=" * 60)
    print(f"提供商: {CURRENT_PROVIDER}")
    print(f"默认模型: {default_model}")
    print(f"API URL: {base_url}")
    print(f"API密钥: {'已配置 ✓' if api_key else '未配置 ✗'}")
    
    print(f"\n支持的模型:")
    models = SUPPORTED_MODELS.get(CURRENT_PROVIDER, [])
    for model in models:
        print(f"  • {model}")
    
    print("=" * 60)


def list_providers() -> list:
    """
    列出所有可用的提供商
    
    Returns:
        提供商名称列表
    """
    return ['deepseek', 'siliconflow']


def get_supported_models(provider: Optional[str] = None) -> list:
    """
    获取指定提供商支持的模型列表
    
    Args:
        provider: 提供商名称，None表示当前提供商
        
    Returns:
        模型名称列表
    """
    from config import SUPPORTED_MODELS, CURRENT_PROVIDER
    
    provider = provider or CURRENT_PROVIDER
    return SUPPORTED_MODELS.get(provider, [])


def validate_provider_config(provider: str) -> Tuple[bool, str]:
    """
    验证提供商配置是否完整
    
    Args:
        provider: 提供商名称
        
    Returns:
        (是否有效, 错误信息) 的元组
    """
    from config import get_api_config
    
    try:
        api_key, base_url, default_model = get_api_config(provider)
        
        if not api_key:
            return False, f"{provider.upper()}_API_KEY 未配置"
        
        if not base_url:
            return False, f"{provider.upper()}_BASE_URL 未配置"
        
        return True, "配置完整"
        
    except ValueError as e:
        return False, str(e)


def show_all_models() -> None:
    """显示所有支持的模型"""
    from config import SUPPORTED_MODELS
    
    print("\n📋 所有支持的模型:")
    
    for provider, models in SUPPORTED_MODELS.items():
        print(f"\n🔧 {provider.upper()}:")
        for model in models:
            print(f"   • {model}")
