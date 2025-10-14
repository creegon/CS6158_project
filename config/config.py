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

# API配置 - 从环境变量读取，如果没有则使用默认值
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 检查API密钥是否配置
if not DEEPSEEK_API_KEY:
    print("⚠ 警告: 未找到DEEPSEEK_API_KEY，请在.env文件中配置")
    print("   提示: 复制.env.example为.env并填入你的API密钥")


# 模型配置
DEFAULT_MODEL = "deepseek-chat"
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
