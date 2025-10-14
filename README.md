# Flaky Test分析系统

这是一个模块化的Flaky Test分析系统，支持数据蒸馏、数据讲解和多Agent协作等功能。

## 📁 项目结构

```
CS6158 project/
├── config/                      # 配置文件
│   ├── __init__.py
│   └── config.py               # API密钥、路径等配置
│
├── prompts/                     # Prompt模板
│   ├── distillation_system.txt # 蒸馏系统提示词
│   ├── distillation_user.txt   # 蒸馏用户提示词
│   ├── explainer_system.txt    # 讲解系统提示词
│   └── explainer_user.txt      # 讲解用户提示词
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── data_utils.py           # 数据处理工具
│   └── prompt_utils.py         # Prompt处理工具
│
├── agents/                      # Agent模块
│   ├── __init__.py
│   ├── base_agent.py           # Agent基类
│   ├── distillation_agent.py   # 数据蒸馏Agent
│   ├── data_explainer_agent.py # 数据讲解Agent
│   └── multi_agent.py          # 多Agent协作框架
│
├── examples/                    # 使用示例
│   ├── distillation_example.py
│   ├── data_explainer_example.py
│   └── multi_agent_example.py
│
├── output/                      # 输出目录
│   └── (生成的文件)
│
└── FlakyLens_dataset_with_nonflaky_indented.csv  # 数据集
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas openai tqdm
```

### 2. 配置API密钥

编辑 `config/config.py` 文件，设置你的DeepSeek API密钥：

```python
DEEPSEEK_API_KEY = "your-api-key-here"
```

### 3. 运行示例

#### 数据蒸馏（推荐先用测试模式）

```python
from agents import DistillationAgent

# 创建Agent（测试模式，只处理最后10条）
agent = DistillationAgent(test_mode='last', test_size=10)

# 运行蒸馏任务
result = agent.run(output_name='test_distillation')
```

#### 数据讲解

```python
from agents import DataExplainerAgent

# 创建Agent
agent = DataExplainerAgent(sample_size=20)

# 运行分析任务
result = agent.run(output_name='dataset_analysis')
```

## 📚 核心模块说明

### 1. Config模块 (`config/`)

管理API密钥、路径和全局配置参数。

**主要配置项:**
- `DEEPSEEK_API_KEY`: API密钥
- `DATASET_PATH`: 数据集路径
- `OUTPUT_DIR`: 输出目录
- `DEFAULT_MODEL`: 默认模型
- `DEFAULT_TEMPERATURE`: 默认温度参数
- `API_BATCH_SIZE`: 批次大小
- `CHECKPOINT_INTERVAL`: 检查点保存间隔

### 2. Utils模块 (`utils/`)

提供通用的工具函数。

**数据处理工具 (`data_utils.py`):**
- `load_csv()`: 读取CSV文件
- `sample_data()`: 数据采样
- `convert_to_alpaca_format()`: 转换为Alpaca格式
- `save_json()`: 保存JSON文件
- `get_data_statistics()`: 获取数据统计信息
- `print_data_info()`: 打印数据信息

**Prompt工具 (`prompt_utils.py`):**
- `load_prompt()`: 加载prompt模板
- `format_prompt()`: 格式化prompt
- `save_prompt()`: 保存prompt

### 3. Agents模块 (`agents/`)

核心Agent实现。

#### BaseAgent (`base_agent.py`)

所有Agent的基类，封装了API调用、统计信息等通用功能。

**主要方法:**
- `call_api()`: 调用API生成响应
- `get_stats()`: 获取统计信息
- `print_stats()`: 打印统计信息
- `run()`: 执行任务（子类实现）

#### DistillationAgent (`distillation_agent.py`)

数据蒸馏Agent，用于生成包含推理过程的训练数据集。

**主要参数:**
- `test_mode`: 测试模式 ('all', 'first', 'last', 'random')
- `test_size`: 测试时使用的数据量
- `batch_size`: 批次大小
- `checkpoint_interval`: 检查点保存间隔

**使用示例:**
```python
agent = DistillationAgent(
    test_mode='random',
    test_size=10,
    temperature=0.7,
    batch_size=10
)
result = agent.run(output_name='my_dataset')
```

#### DataExplainerAgent (`data_explainer_agent.py`)

数据讲解Agent，随机抽取数据样本并生成详细的解读报告。

**主要参数:**
- `sample_size`: 采样数量
- `random_seed`: 随机种子
- `code_column`: 代码列名
- `label_column`: 标签列名

**使用示例:**
```python
agent = DataExplainerAgent(sample_size=20)
result = agent.run(output_name='analysis')
```

#### MultiAgent协作框架 (`multi_agent.py`)

支持多个Agent协作完成复杂任务。

**协调器类型:**
- `SequentialCoordinator`: 顺序执行
- `ParallelCoordinator`: 并行执行（待实现）
- `PipelineCoordinator`: 流水线执行（待实现）

**使用示例:**
```python
from agents import SequentialCoordinator, DistillationAgent, DataExplainerAgent

coordinator = SequentialCoordinator()
coordinator.add_agent(DataExplainerAgent(), name="Explainer")
coordinator.add_agent(DistillationAgent(test_mode='first', test_size=5), name="Distiller")

tasks = [
    {'agent_index': 0, 'description': '分析数据', 'params': {}},
    {'agent_index': 1, 'description': '数据蒸馏', 'params': {}}
]

results = coordinator.execute(tasks)
```

## 🎯 使用场景

### 场景1: 快速测试

```python
# 使用测试模式快速验证流程
agent = DistillationAgent(test_mode='last', test_size=5)
result = agent.run()
```

### 场景2: 数据集全量处理

```python
# 处理完整数据集
agent = DistillationAgent(test_mode='all')
result = agent.run(output_name='full_dataset')
```

### 场景3: 数据集分析

```python
# 分析数据集特征
agent = DataExplainerAgent(sample_size=30)
result = agent.run()
```

### 场景4: 自定义参数

```python
# 自定义各种参数
agent = DistillationAgent(
    test_mode='random',
    test_size=100,
    temperature=0.8,
    max_tokens=2000,
    batch_size=10,
    batch_delay=1,
    checkpoint_interval=50
)
result = agent.run()
```

## 📝 Prompt管理

Prompt模板存储在 `prompts/` 目录下，每个场景使用独立的txt文件：

- `distillation_system.txt`: 蒸馏任务的系统提示词
- `distillation_user.txt`: 蒸馏任务的用户提示词模板
- `explainer_system.txt`: 讲解任务的系统提示词
- `explainer_user.txt`: 讲解任务的用户提示词模板

可以直接编辑这些文件来更新prompt，无需修改代码。

## 🔧 高级功能

### 检查点恢复

系统会自动保存检查点，如果中断可以从 `output/temp_checkpoint.json` 恢复。

### 统计信息

每个Agent都会记录统计信息：
- API调用次数
- 成功/失败次数
- 总Token使用量
- 耗时

调用 `agent.print_stats()` 查看详细统计。

### 多Agent协作

使用 `SequentialCoordinator` 可以协调多个Agent按顺序执行任务。

## 🚧 待实现功能

1. **ParallelCoordinator**: 并行执行多个Agent
2. **PipelineCoordinator**: 流水线式Agent协作
3. **更多Agent类型**: 如分类Agent、评估Agent等

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！
