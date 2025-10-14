"""
配置文件 - 管理API密钥和其他全局配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# API配置
DEEPSEEK_API_KEY = "sk-b7a22a2706bd4c40919bbf86b2490712"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 模型配置
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MAX_RETRIES = 3

# 数据集路径
DATASET_PATH = PROJECT_ROOT / "FlakyLens_dataset_with_nonflaky_indented.csv"

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
