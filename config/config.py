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

# 当前使用的提供商（可选：deepseek, siliconflow）
CURRENT_PROVIDER = os.getenv("CURRENT_PROVIDER", "deepseek")

# 根据提供商选择API配置
def get_api_config(provider: str = None):
    """
    获取指定提供商的API配置
    
    Args:
        provider: 提供商名称 (deepseek/siliconflow)
        
    Returns:
        (api_key, base_url, default_model) 的元组
    """
    provider = provider or CURRENT_PROVIDER
    
    if provider.lower() == "deepseek":
        return DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, "deepseek-chat"
    elif provider.lower() == "siliconflow":
        return SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, "Qwen/Qwen2.5-7B-Instruct"
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
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "THUDM/glm-4-9b-chat",
        "01-ai/Yi-1.5-9B-Chat-16K",
        "deepseek-ai/DeepSeek-V2.5"
    ]
}


# 模型配置（向后兼容）
DEFAULT_MODEL = CURRENT_MODEL
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
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

# 日志配置
LOG_LEVEL = "INFO"
