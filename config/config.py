"""
配置文件 - 管理API密钥和其他全局配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载.env文件
def load_env():
    """从.env文件加载环境变量"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip()

# 加载环境变量
load_env()

# API配置 - 支持多个提供商
# DeepSeek配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# SiliconFlow配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# Gemini配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAgDhx4kCTx4fmKs_Of69rX3DkkHairV7c")

# 当前使用的提供商（可选：deepseek, siliconflow, gemini）
CURRENT_PROVIDER = os.getenv("CURRENT_PROVIDER", "deepseek")

# 根据提供商选择API配置
def get_api_config(provider: str = None):
    """
    获取指定提供商的API配置
    
    Args:
        provider: 提供商名称 (deepseek/siliconflow/gemini)
        
    Returns:
        (api_key, base_url, default_model) 的元组
    """
    provider = provider or CURRENT_PROVIDER
    
    # 默认模型配置
    default_models = {
        "deepseek": "deepseek-chat",
        "siliconflow": "Qwen/Qwen2.5-7B-Instruct",
        "gemini": "gemini-2.0-flash-exp"
    }
    
    if provider.lower() == "deepseek":
        # 优先从环境变量读取自定义模型,否则使用默认模型
        model = os.getenv("CURRENT_MODEL", default_models["deepseek"])
        return DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, model
    elif provider.lower() == "siliconflow":
        # 优先从环境变量读取自定义模型,否则使用默认模型
        model = os.getenv("CURRENT_MODEL", default_models["siliconflow"])
        return SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, model
    elif provider.lower() == "gemini":
        model = os.getenv("CURRENT_MODEL", default_models["gemini"])
        return GEMINI_API_KEY, None, model
    else:
        raise ValueError(f"不支持的提供商: {provider}")

# 获取当前配置
CURRENT_API_KEY, CURRENT_BASE_URL, CURRENT_MODEL = get_api_config()

# 检查API密钥是否配置
if not CURRENT_API_KEY:
    print(f"⚠ 警告: 未找到 {CURRENT_PROVIDER.upper()}_API_KEY，请在.env文件中配置")
    print("   提示: 复制.env.example为.env并填入你的API密钥")

# 支持的模型列表
SUPPORTED_MODELS = {
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder"
    ],
    "siliconflow": [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "gemini": [
        "gemini-2.5-flash-preview-09-2025"
    ]
}


def get_provider_models(provider: str) -> list:
    """
    获取指定提供商支持的模型列表
    
    Args:
        provider: 提供商名称
        
    Returns:
        模型名称列表
    """
    return SUPPORTED_MODELS.get(provider.lower(), [])


# 模型配置（向后兼容）
DEFAULT_MODEL = CURRENT_MODEL
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8192  # 设为最大值以支持长输出
DEFAULT_MAX_RETRIES = 3

# 数据集路径
DATASET_PATH = PROJECT_ROOT / "dataset" / "FlakyLens_dataset_with_nonflaky_indented.csv"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"

# Prompt路径
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# API限流配置
API_BATCH_SIZE = 10  # 每批次处理的数据量
API_BATCH_DELAY = 1  # 批次间延迟（秒）
CHECKPOINT_INTERVAL = 50  # 保存检查点的间隔

# 特征提示配置
FEATURE_HINT_MODE = os.getenv("FEATURE_HINT_MODE", "global-highest")  # 可选: "category-wise" 或 "global-highest"
FEATURE_HINT_MAX_PER_LEVEL = int(os.getenv("FEATURE_HINT_MAX_PER_LEVEL", "0"))  # category-wise模式下每个级别最多保留N个特征，0表示不限制

# 特征提示模式说明:
# - "category-wise": 每个类别独立选择最高级别,每个级别最多保留FEATURE_HINT_MAX_PER_LEVEL个特征
#   例如: async有very_strong,concurrency有strong,OD有moderate → 都会输出
# - "global-highest": 所有类别中只选择全局最高级别的特征
#   例如: async有very_strong,concurrency有strong,OD有moderate → 只输出very_strong的特征

# 日志配置
LOG_LEVEL = "INFO"
