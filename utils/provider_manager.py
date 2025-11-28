"""
API提供商和模型管理工具
"""
import os
from pathlib import Path
from typing import Tuple, Optional, List


def get_env_file_path() -> Path:
    """获取.env文件路径"""
    return Path(__file__).parent.parent / '.env'


def load_external_models(config_file: Optional[str] = None) -> dict:
    """
    从外部TOML文件加载模型配置
    
    Args:
        config_file: TOML配置文件路径，默认从MaiBot读取
        
    Returns:
        模型配置字典
    """
    if config_file is None:
        # 默认路径
        config_file = r"D:\MaiBot_mutsumi\modules\MaiBot\config\model_config.toml"
    
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"⚠️ 外部配置文件不存在: {config_file}")
        return {}
    
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli
        except ImportError:
            print("⚠️ 需要安装 tomli 库: pip install tomli")
            return {}
    
    try:
        with open(config_path, 'rb') as f:
            config = tomli.load(f)
        
        # 提取模型列表
        models = {}
        if 'models' in config:
            for model_data in config['models']:
                provider = model_data.get('provider', 'unknown').lower()
                model_name = model_data.get('model_name', '')
                
                if provider not in models:
                    models[provider] = []
                
                if model_name and model_name not in models[provider]:
                    models[provider].append(model_name)
        
        return models
    except Exception as e:
        print(f"❌ 读取外部配置文件失败: {e}")
        return {}


def get_available_models(provider: Optional[str] = None) -> List[str]:
    """
    获取可用模型列表（优先从外部配置读取）
    
    Args:
        provider: 提供商名称，None表示当前提供商
        
    Returns:
        模型名称列表
    """
    from config import SUPPORTED_MODELS, CURRENT_PROVIDER
    
    provider = provider or CURRENT_PROVIDER
    provider = provider.lower()
    
    # 尝试从外部配置加载
    external_models = load_external_models()
    if external_models and provider in external_models:
        return external_models[provider]
    
    # 回退到内置配置
    return SUPPORTED_MODELS.get(provider, [])


def switch_provider(provider: str, model: Optional[str] = None) -> bool:
    """
    切换API提供商和/或模型
    
    Args:
        provider: 提供商名称 (deepseek/siliconflow)
        model: 可选，指定要切换的模型名称
        
    Returns:
        是否切换成功
    """
    valid_providers = ['deepseek', 'siliconflow', 'gemini']
    provider = provider.lower()
    
    if provider not in valid_providers:
        print(f"❌ 无效的提供商: {provider}")
        print(f"   可选: {', '.join(valid_providers)}")
        return False
    
    # 如果指定了模型，验证模型是否可用
    if model:
        available_models = get_available_models(provider)
        if model not in available_models:
            print(f"❌ 模型 '{model}' 在 {provider} 中不可用")
            print(f"\n可用模型:")
            for m in available_models[:10]:  # 显示前10个
                print(f"  • {m}")
            if len(available_models) > 10:
                print(f"  ... 还有 {len(available_models) - 10} 个模型")
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
    provider_found = False
    model_found = False
    
    for i, line in enumerate(lines):
        if line.startswith('CURRENT_PROVIDER='):
            lines[i] = f'CURRENT_PROVIDER={provider}\n'
            provider_found = True
        elif model and line.startswith('CURRENT_MODEL='):
            lines[i] = f'CURRENT_MODEL={model}\n'
            model_found = True
    
    if not provider_found:
        lines.append(f'\nCURRENT_PROVIDER={provider}\n')
    
    if model and not model_found:
        lines.append(f'CURRENT_MODEL={model}\n')
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ API提供商已切换为: {provider.upper()}")
    if model:
        print(f"✅ 模型已切换为: {model}")
    
    # 显示对应的配置
    from config import get_api_config
    api_key, base_url, default_model = get_api_config(provider)
    
    # 如果没有指定模型，使用默认模型
    if not model:
        model = default_model
    
    print(f"\n📌 当前配置:")
    print(f"   提供商: {provider}")
    print(f"   模型: {model}")
    print(f"   API URL: {base_url}")
    print(f"   API密钥: {'已配置 ✓' if api_key else '未配置 ✗'}")
    
    if not api_key:
        print(f"\n⚠️  警告: 请在 .env 文件中配置 {provider.upper()}_API_KEY")
    
    return True


def switch_model(model: str, provider: Optional[str] = None) -> bool:
    """
    切换模型（不改变提供商）
    
    Args:
        model: 模型名称
        provider: 可选，指定提供商，None表示当前提供商
        
    Returns:
        是否切换成功
    """
    from config import CURRENT_PROVIDER
    
    provider = provider or CURRENT_PROVIDER
    return switch_provider(provider, model)


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
    from config import CURRENT_PROVIDER, get_api_config
    
    api_key, base_url, default_model = get_api_config()
    
    # 检查是否有自定义模型配置
    env_file = get_env_file_path()
    custom_model = None
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('CURRENT_MODEL='):
                    custom_model = line.split('=', 1)[1].strip()
                    break
    
    current_model = custom_model or default_model
    
    print("\n" + "=" * 70)
    print("当前API配置")
    print("=" * 70)
    print(f"提供商: {CURRENT_PROVIDER}")
    print(f"模型: {current_model}")
    print(f"API URL: {base_url}")
    print(f"API密钥: {'已配置 ✓' if api_key else '未配置 ✗'}")
    
    print(f"\n支持的模型 (前10个):")
    models = get_available_models(CURRENT_PROVIDER)
    for i, model in enumerate(models[:10], 1):
        marker = " 👉" if model == current_model else ""
        print(f"  {i:2}. {model}{marker}")
    
    if len(models) > 10:
        print(f"  ... 还有 {len(models) - 10} 个模型 (使用 show_all_models() 查看全部)")
    
    print("=" * 70)


def show_all_models(provider: Optional[str] = None, search: Optional[str] = None) -> None:
    """
    显示所有支持的模型
    
    Args:
        provider: 指定提供商，None表示显示所有
        search: 搜索关键词，只显示包含该关键词的模型
    """
    from config import CURRENT_PROVIDER
    
    print("\n📋 可用模型列表:")
    
    if provider:
        providers = [provider.lower()]
    else:
        providers = ['deepseek', 'siliconflow', 'gemini']
    
    for prov in providers:
        models = get_available_models(prov)
        
        # 应用搜索过滤
        if search:
            models = [m for m in models if search.lower() in m.lower()]
        
        if not models:
            continue
        
        current_marker = " (当前)" if prov == CURRENT_PROVIDER.lower() else ""
        print(f"\n🔧 {prov.upper()}{current_marker} - {len(models)} 个模型:")
        
        for i, model in enumerate(models, 1):
            print(f"  {i:3}. {model}")


def list_models_by_family(provider: Optional[str] = None) -> None:
    """
    按模型系列分组显示
    
    Args:
        provider: 提供商名称，None表示当前提供商
    """
    from config import CURRENT_PROVIDER
    
    provider = provider or CURRENT_PROVIDER
    models = get_available_models(provider)
    
    # 按系列分组
    families = {}
    for model in models:
        # 提取系列名（第一个/或-之前的部分）
        if '/' in model:
            family = model.split('/')[0]
        elif '-' in model:
            family = model.split('-')[0]
        else:
            family = 'Other'
        
        if family not in families:
            families[family] = []
        families[family].append(model)
    
    print(f"\n📚 {provider.upper()} 模型系列:")
    print("=" * 70)
    
    for family in sorted(families.keys()):
        print(f"\n【{family}】 ({len(families[family])} 个)")
        for model in families[family]:
            print(f"  • {model}")


def list_providers() -> list:
    """
    列出所有可用的提供商
    
    Returns:
        提供商名称列表
    """
    return ['deepseek', 'siliconflow', 'gemini']


def get_supported_models(provider: Optional[str] = None) -> list:
    """
    获取指定提供商支持的模型列表（兼容性函数）
    
    Args:
        provider: 提供商名称，None表示当前提供商
        
    Returns:
        模型名称列表
    """
    return get_available_models(provider)


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
        
        # Gemini 不需要 base_url
        if provider.lower() != 'gemini' and not base_url:
            return False, f"{provider.upper()}_BASE_URL 未配置"
        
        return True, "配置完整"
        
    except ValueError as e:
        return False, str(e)

