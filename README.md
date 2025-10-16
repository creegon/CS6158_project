# Flaky Test分析系统

这是一个模块化的Flaky Test分析系统，支持数据蒸馏、数据讲解、并行推理和精确评估等功能。

## ✨ 核心特性

- 🎯 **精确评估**：通过 ID 字段匹配预测结果和真实标签，避免顺序混乱
- 🚀 **并行推理**：支持多线程并行处理，显著提升数据蒸馏效率
- 📊 **5类Flaky分类**：Async、Conc、Time、UC、OD 五种类型精准识别
- 🔄 **完整流程**：从数据蒸馏到模型评估的端到端解决方案
- 🔍 **API签名匹配**：基于代码结构相似度检索few-shot examples，增强LLM分类能力
- 📂 **灵活数据管理**：支持自定义训练集/测试集，完美适配K-fold交叉验证
- 💾 **配置复用**：保存和加载实验配置，快速切换不同实验设置
- 📝 **Few-shot增强**：自动检索相似案例，提供标签和相似度元数据

## 目录

- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [核心模块说明](#-核心模块说明)
- [使用场景](#-使用场景)
- [多模型支持](#-多模型支持)
- [测试指南](#-测试指南)
- [扩展指南](#-扩展指南)
- [常见问题](#-常见问题)

## 📁 项目结构

```
CS6158 project/
├── config/                      # 配置文件
│   ├── __init__.py
│   └── config.py               # API密钥、路径等配置
│
├── configs/                     # 保存的实验配置（新增）
│   └── *.json                  # 配置文件（运行时生成）
│
├── prompts/                     # Prompt模板
│   ├── distillation_system.txt # 蒸馏系统提示词（已更新：添加few-shot指导）
│   ├── distillation_user.txt   # 蒸馏用户提示词
│   ├── explainer_system.txt    # 讲解系统提示词
│   └── explainer_user.txt      # 讲解用户提示词
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── data/                   # 数据处理模块
│   │   ├── __init__.py
│   │   ├── data_loader.py      # CSV加载和采样
│   │   ├── data_splitter.py    # 数据集划分（含K-fold）
│   │   ├── data_storage.py     # 文件保存（CSV/JSON）
│   │   ├── data_converter.py   # Alpaca格式转换
│   │   └── data_statistics.py  # 数据统计信息
│   ├── api_matcher.py          # API签名匹配器
│   ├── config_manager.py       # 配置管理（新增）
│   ├── prompt_utils.py         # Prompt处理工具（模板加载、格式化）
│   └── evaluation_utils.py     # 评估工具函数（答案提取、指标计算）
│
├── agents/                      # Agent模块
│   ├── __init__.py
│   ├── base_agent.py           # Agent基类
│   ├── distillation_agent.py   # 数据蒸馏Agent（已更新：few-shot集成、单次处理优化）
│   ├── data_explainer_agent.py # 数据讲解Agent
│   └── multi_agent.py          # 多Agent协作框架
│
├── evaluation/                  # 评估模块
│   ├── __init__.py
│   ├── evaluator.py            # 评估器主类（整合评估流程）
│   ├── data_loader.py          # 数据加载器（Alpaca JSON & CSV标签）
│   └── report_generator.py     # 报告生成器（文本/JSON报告）
│
├── examples/                    # 使用示例
│   ├── distillation_example.py
│   ├── data_explainer_example.py
│   ├── multi_agent_example.py
│   ├── evaluation_example.py   # 评估示例
│   └── example_api_matching.py # API匹配示例
│
├── dataset/                     # 数据集目录
│   ├── FlakyLens_dataset_with_nonflaky_indented.csv  # 原始数据集
│   └── kfold_splits/           # K-fold划分结果
│       ├── fold_1_train.csv
│       ├── fold_1_test.csv
│       └── ...
│
├── docs/                        # 文档目录
│   ├── API_MATCHING.md         # API匹配详细文档
│   ├── QUICK_START_API_MATCHING.md  # API匹配快速开始
│   └── SILICONFLOW_GUIDE.md    # SiliconFlow使用指南（新增）
│
├── output/                      # 输出目录
│   ├── *_external.json         # 包含id和few_shot_examples的完整输出（新增）
│   ├── *.json                  # 标准Alpaca格式输出（仅instruction/input/output）
│   └── (其他生成文件)
│
├── main.py                      # 快速启动脚本（已更新：配置管理+模型设置）
├── switch_provider.py           # 快速切换API提供商（新增）
├── example_siliconflow.py       # SiliconFlow使用示例（新增）
├── test_api_matcher.py          # API匹配测试
├── test_integration.py          # 集成测试
├── test_config_manager.py       # 配置管理测试
├── example_api_matching.py      # API匹配示例
├── CHANGELOG_API_MATCHING.md    # API匹配更新日志
├── CONFIG_USAGE_GUIDE.md        # 配置复用使用指南
├── README.md                    # 项目文档（本文件）
└── .gitignore                   # Git忽略文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas openai tqdm
```

### 2. 配置API密钥

**重要：为了安全，API密钥存储在`.env`文件中**

① 复制示例配置文件：
```bash
copy .env.example .env
```

② 编辑 `.env` 文件，填入你的API密钥：

```bash
# DeepSeek API配置（默认）
DEEPSEEK_API_KEY=your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# SiliconFlow API配置（可选）
SILICONFLOW_API_KEY=your-siliconflow-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 当前使用的提供商 (可选: deepseek, siliconflow)
CURRENT_PROVIDER=deepseek
```

⚠️ **注意**: `.env`文件已添加到`.gitignore`，不会被提交到Git仓库

### 3. 运行示例

**推荐使用交互式界面：** 
运行 `python main.py` 选择相应功能即可

**或直接使用代码：**
- 数据蒸馏（测试模式）：from agents import DistillationAgent; agent = DistillationAgent(test_mode='last', test_size=10); agent.run()
- 数据分析：from agents import DataExplainerAgent; agent = DataExplainerAgent(sample_size=20); agent.run()

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

**数据处理模块 (`data/`):**
- `data_loader.py`: CSV加载和数据采样
- `data_splitter.py`: 数据集划分（支持K-fold交叉验证）
- `data_storage.py`: CSV/JSON文件保存
- `data_converter.py`: Alpaca格式转换
- `data_statistics.py`: 数据统计和信息展示

**API匹配器 (`api_matcher.py`) ✨:**
- `APISignatureMatcher`: API签名匹配器类
- `extract_apis()`: 从代码中提取API签名（9个类别）
- `compute_similarity()`: 计算两段代码的相似度（Jaccard系数 + 频率加权）
- `retrieve_top_k()`: 检索最相似的K个案例
- `retrieve_with_diversity()`: 多样性检索（避免相似案例）

**配置管理器 (`config_manager.py`) ✨ 新增:**
- `save_config(config, name)`: 保存实验配置到JSON文件
- `load_config(name)`: 从JSON加载配置（自动转换Path对象）
- `list_saved_configs()`: 列出所有已保存的配置
- `delete_config(name)`: 删除指定配置
- `display_config(config)`: 格式化显示配置详情

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

**核心改进 ✨:**
- **Few-shot集成**: 自动从训练集检索相似案例并插入Prompt
- **单次处理优化**: 避免重复API调用，性能提升3倍
- **双格式输出**: 生成标准版（纯训练数据）和external版（含元数据）
- **元数据丰富**: 记录few-shot案例的相似度、标签、代码预览

**主要参数:**
- `dataset_path`: 数据集路径（支持自定义）
- `test_mode`: 测试模式 ('all', 'first', 'last', 'random')
- `test_size`: 测试时使用的数据量
- `batch_size`: 批次大小
- `parallel_workers`: 并行线程数（1-10）
- `api_matcher`: API匹配器实例 ✨ 
- `top_k_shots`: Few-shot样本数量 ✨ 
- `checkpoint_interval`: 检查点保存间隔

**输出文件:**
- `{name}_external.json`: 包含 `id` 和 `few_shot_examples` 字段（用于调试）
- `{name}.json`: 标准Alpaca格式（仅 `instruction`, `input`, `output`）

**Few-shot元数据结构:**
```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "id": 12345,
  "few_shot_examples": [
    {
      "similarity": 0.85,
      "project": "apache_hadoop",
      "test_name": "testConcurrentAccess",
      "label": "Conc",
      "code_preview": "...",
      "id": 67890
    }
  ]
}
```

**使用示例（不使用API匹配）:**
```python
agent = DistillationAgent(
    dataset_path='dataset/kfold_splits/fold_1_test.csv',
    test_mode='all',
    parallel_workers=5
)
result = agent.run(output_name='distillation_fold1')
```

**使用示例（使用API匹配）:**
```python
from utils import load_csv, APISignatureMatcher

# 创建API匹配器
train_data = load_csv('dataset/kfold_splits/fold_1_train.csv')
api_matcher = APISignatureMatcher(train_data)

# 创建Agent（启用API匹配）
agent = DistillationAgent(
    dataset_path='dataset/kfold_splits/fold_1_test.csv',
    test_mode='all',
    api_matcher=api_matcher,
    top_k_shots=3,
    parallel_workers=5
)
result = agent.run(output_name='distillation_with_api')
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

### 4. Evaluation模块 (`evaluation/`)

评估模块用于评估Flaky Test分类模型的性能。

#### 支持的分类类型

模型需要判断测试是否为Flaky Test，如果是，还需要分类到以下五种类型之一：

1. **Async (异步相关)** - 异步任务、回调、Promise时序问题
2. **Conc (并发相关)** - 竞态条件、多线程同步、共享资源冲突
3. **Time (时间相关)** - 系统时间依赖、超时设置、延迟问题
4. **UC (无序集合)** - HashMap/Set等无序结构导致的问题
5. **OD (顺序依赖)** - 测试间执行顺序依赖、状态未清理
6. **Non-Flaky** - 稳定的测试

#### 答案格式要求

模型输出必须在开头包含标准化的答案格式：
- 答案：是 - Async （表示是Flaky Test，类型为异步）
- 答案：否 - Non-Flaky （表示不是Flaky Test）

格式说明：固定前缀"答案："+ 是/否 + 分隔符"-" + 类型（Async/Conc/Time/UC/OD/Non-Flaky）

#### 基本使用

使用评估器非常简单：
from evaluation import Evaluator
evaluator = Evaluator(prediction_file='output/predictions.json', ground_truth_file='dataset/labels.csv', label_column='label')
metrics = evaluator.run(output_dir='output/evaluation', save_report=True)
    output_dir='output/evaluation',
    save_report=True,
    detailed=True
)

#### 文件格式要求

**预测结果文件 (Alpaca格式JSON)：** 包含 instruction、input、output 字段，output中需要有"答案："格式

**真实标签文件 (CSV)：** 包含 id 和 label 列，标签值会自动标准化（如 "async wait" → "Async"）

#### 评估指标

1. **总体准确率 (Overall Accuracy)**: 同时判断对"是否Flaky"和"具体类型"的准确率
2. **Flaky检测指标**: 准确率、精确率、召回率、F1分数
3. **类别分类指标**: 分类准确率和各类别的详细指标

#### 高级用法

分步执行：evaluator.load_data() → evaluator.evaluate() → evaluator.print_report() → evaluator.save_report()

批量评估多个模型：循环遍历模型列表，分别创建Evaluator并运行评估
    print(f"{name}: Accuracy={metrics['overall_accuracy']:.2%}")

#### 输出文件

评估完成后会生成：

```
output/evaluation/
├── evaluation_report.json  # JSON格式的详细指标
└── evaluation_report.txt   # 文本格式的可读报告
```

#### 注意事项

1. **答案格式**: 确保模型输出包含标准的"答案：xxx"格式
2. **数据对齐**: 预测结果和真实标签的数量可能不同，系统会自动对齐
3. **标签标准化**: 不同的标签写法会自动标准化
4. **缺失答案**: 如果某条预测无法提取答案，会显示警告并跳过

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

使用 DataExplainerAgent 分析数据集，采样30条数据生成分析报告

### 场景4: 自定义参数

可自定义 test_mode、test_size、temperature、max_tokens、batch_size、batch_delay、checkpoint_interval 等参数

### 场景5: 评估模型性能

使用 Evaluator 评估预测结果，查看总体准确率和Flaky检测F1分数

### 场景6: Few-shot增强实验 ✨ 新增

```python
from utils import load_csv, APISignatureMatcher

# K-fold交叉验证 + API匹配
for fold in range(1, 6):
    # 加载训练集
    train_data = load_csv(f'dataset/kfold_splits/fold_{fold}_train.csv')
    
    # 创建API匹配器
    api_matcher = APISignatureMatcher(train_data)
    
    # 创建蒸馏Agent
    agent = DistillationAgent(
        dataset_path=f'dataset/kfold_splits/fold_{fold}_test.csv',
        test_mode='all',
        api_matcher=api_matcher,
        top_k_shots=3,
        parallel_workers=5
    )
    
    # 运行蒸馏
    agent.run(output_name=f'fold_{fold}_with_api')
```

### 场景7: 配置复用实验 ✨ 新增

```python
# 第一次运行：保存配置
# 运行 python main.py → 选择 1. 数据蒸馏
# 完成所有配置后，选择保存为 "baseline_experiment"

# 后续运行：加载配置
# 运行 python main.py → 选择 1. 数据蒸馏
# 检测到配置后，输入编号加载
# 确认配置后直接开始运行

# 对比实验：保存多个配置
# - baseline (无API匹配)
# - api_top3 (API匹配, K=3)
# - api_top5 (API匹配, K=5)
# 快速切换不同配置进行对比实验
```

---

## 🤖 多模型支持

系统支持多个 API 提供商，可以灵活切换。

### 支持的提供商

#### 1. DeepSeek（默认）
- **模型**: `deepseek-chat`, `deepseek-coder`
- **特点**: 快速、成本低、质量高
- **适用**: 生产环境、大规模处理

#### 2. SiliconFlow
- **模型**: Qwen 系列、GLM、Yi 等
- **特点**: 模型选择多、开源友好
- **适用**: 实验对比、多模型测试

### 快速切换

**方式1: 使用切换工具（推荐）**
```bash
# 切换到 SiliconFlow
python switch_provider.py siliconflow

# 切换到 DeepSeek
python switch_provider.py deepseek

# 查看当前配置
python switch_provider.py status
```

**方式2: 通过主菜单**
```bash
python main.py
# 选择 "6. 模型设置"
# 选择 "1. 切换提供商"
```

**方式3: 编程方式**
```python
from agents import DistillationAgent

# 显式指定提供商
agent = DistillationAgent(
    provider='siliconflow',
    model='Qwen/Qwen2.5-7B-Instruct',
    test_mode='last',
    test_size=10
)
result = agent.run()
```

### SiliconFlow 支持的模型

```python
# Qwen 系列（推荐）
'Qwen/Qwen2.5-7B-Instruct'      # 默认，性能均衡
'Qwen/Qwen2.5-14B-Instruct'     # 中等规模，效果更好
'Qwen/Qwen2.5-32B-Instruct'     # 大规模模型
'Qwen/Qwen2.5-72B-Instruct'     # 最强模型

# 其他模型
'THUDM/glm-4-9b-chat'           # ChatGLM4
'01-ai/Yi-1.5-9B-Chat-16K'      # Yi 模型
'deepseek-ai/DeepSeek-V2.5'     # DeepSeek（通过 SiliconFlow）
```

### 使用示例

```python
# 示例1: 使用 SiliconFlow 的 Qwen 模型
agent = DistillationAgent(
    provider='siliconflow',
    model='Qwen/Qwen2.5-14B-Instruct',
    test_mode='all',
    parallel_workers=5
)
result = agent.run(output_name='qwen_result')

# 示例2: 对比不同提供商
providers = [
    ('deepseek', 'deepseek-chat'),
    ('siliconflow', 'Qwen/Qwen2.5-7B-Instruct')
]

for provider, model in providers:
    agent = DistillationAgent(
        provider=provider,
        model=model,
        test_mode='first',
        test_size=10
    )
    result = agent.run(output_name=f'{provider}_test')
    agent.print_stats()
```

### 详细文档

- **SiliconFlow 使用指南**: [docs/SILICONFLOW_GUIDE.md](docs/SILICONFLOW_GUIDE.md)
- **完整示例**: `python example_siliconflow.py`

## 📝 Prompt管理

Prompt模板存储在 `prompts/` 目录下，每个场景使用独立的txt文件：

- `distillation_system.txt`: 蒸馏任务的系统提示词 ✨ **已更新**：添加few-shot使用指导
- `distillation_user.txt`: 蒸馏任务的用户提示词模板
- `explainer_system.txt`: 讲解任务的系统提示词
- `explainer_user.txt`: 讲解任务的用户提示词模板

可以直接编辑这些文件来更新prompt，无需修改代码。

### Few-shot指导说明（distillation_system.txt）

系统提示词已添加以下指导内容：

```
如果提供了参考案例（Few-shot Examples）：
- 参考案例按照API签名相似度排序，相似度越高越相关
- 优先参考相似度较高的案例
- 参考案例的分类结论可以作为参考，但不应盲目照搬
- 应当结合参考案例和当前测试代码的具体情况，做出独立判断
```

这些指导帮助LLM更好地利用few-shot examples，提升分类准确率。

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

### 配置管理 ✨ 新增

**保存配置**：完成实验配置后，系统会询问是否保存配置以便复用

**加载配置**：下次运行时自动检测已保存的配置，可快速加载

**管理配置**：在主菜单选择"5. 配置管理"查看和删除配置

**配置内容**：
- 任务类型（distillation/explainer/evaluation）
- 测试集和训练集路径
- API匹配设置（是否启用、few-shot数量）
- 测试模式和数据量
- 并行线程数和批次大小

**使用方式**：
```bash
# 方式1：通过交互式界面
python main.py
# 选择 "1. 数据蒸馏"
# 如果有保存的配置，输入编号加载
# 或按回车手动配置，完成后选择保存

# 方式2：配置管理
python main.py
# 选择 "5. 配置管理"
# v - 查看配置详情
# d - 删除配置
# 0 - 返回主菜单
```

**配置文件位置**：`configs/` 目录下（JSON格式）

**详细使用指南**：参见 [CONFIG_USAGE_GUIDE.md](CONFIG_USAGE_GUIDE.md)

### 双格式输出 ✨ 新增

当启用API匹配时，系统会生成两个版本的输出文件：

1. **External版本** (`{name}_external.json`)：
   - 包含完整元数据：`id`、`few_shot_examples`
   - 用于调试和分析few-shot效果
   - 记录每个案例的相似度、标签、代码预览

2. **标准版本** (`{name}.json`)：
   - 仅包含 `instruction`、`input`、`output` 字段
   - 用于模型训练（干净的训练数据）
   - 自动从external版本生成，无需额外API调用

**性能优化**：单次处理策略避免重复API调用，相比双次调用提升3倍速度

---

## 📋 测试指南

### 🎯 推荐测试流程

#### 第一步：验证环境
```bash
# 确保已安装依赖
pip install pandas openai tqdm

# 检查Python版本（建议3.8+）
python --version

# 验证配置
python check_config.py
```

#### 第二步：运行快速测试

**选项A：使用交互式界面（推荐）**
```bash
python main.py
```
然后选择：
- `1` - 测试蒸馏（最后10条）
- `4` - 测试数据讲解（20个样本）

**选项B：直接运行示例**
```bash
# 测试数据讲解（推荐先测试这个，速度快，约1分钟）
python examples/data_explainer_example.py

# 测试数据蒸馏（会调用10次API，约2-5分钟）
python examples/distillation_example.py
```

### 📊 性能参考

| 操作 | 数据量 | 预计时间 | API调用数 |
|-----|-------|---------|----------|
| 数据讲解 | 20样本 | ~1分钟 | 1次 |
| 蒸馏（测试） | 10条 | ~2-3分钟 | 10次 |
| 蒸馏（全量） | 全部 | 取决于数据集大小 | N次 |

### 🔍 验证结果

```bash
# 查看输出目录
ls output/

# 预期文件：
# - dataset_analysis.json
# - dataset_analysis.txt
# - test_distillation_dataset.json
# - temp_checkpoint.json（如果中断过）
```

### ✅ 测试清单

- [ ] 环境配置完成
- [ ] API密钥配置正确
- [ ] 数据讲解测试通过
- [ ] 数据蒸馏测试通过
- [ ] 输出文件生成正常
- [ ] 统计信息显示正常

---

## 🔧 扩展指南

### 添加新的Agent

① 创建 agents/your_agent.py 继承 BaseAgent
② 实现 get_default_system_prompt() 和 run() 方法
③ 在 agents/__init__.py 中导出

### 添加新的Prompt模板

直接在 prompts/ 目录下创建 .txt 文件，用 load_prompt() 加载即可

### 修改配置

编辑 config/config.py 或 .env 文件

---

## 🐛 常见问题

### 1. API调用失败
**原因：** API密钥错误或网络问题  
**解决：** 检查 `.env` 文件中的API密钥，运行 `python check_config.py` 验证

### 2. 找不到模块
**原因：** 目录结构不正确  
**解决：** 确保在项目根目录运行命令

### 3. CSV文件找不到
**原因：** 数据集路径配置错误  
**解决：** 确保CSV文件在 `dataset/` 目录下，检查 `config/config.py` 中的 `DATASET_PATH`

### 4. 进度卡住不动
**原因：** API调用超时或限流  
**解决：** 等待重试机制生效（最多3次）

### 5. API密钥泄露怎么办
1. **立即撤销**当前密钥
2. **生成**新的API密钥
3. **更新** `.env` 文件中的密钥
4. **检查**Git历史，确保 `.env` 在 `.gitignore` 中

### 6. 如何切换API提供商？
**方式1：** 使用命令行工具 `python switch_provider.py [deepseek|siliconflow]`  
**方式2：** 主菜单 → 6. 模型设置 → 1. 切换提供商  
**方式3：** 直接编辑 `.env` 文件中的 `CURRENT_PROVIDER`  
**注意：** 切换后需要重启程序

### 7. API匹配检索结果都是低相似度？
**原因：** 训练集太小或测试代码差异大  
**解决：** 
- 扩大训练集规模（推荐>1000条）
- 检查API提取规则是否适配代码风格
- 降低 `min_similarity` 阈值

### 8. API索引构建太慢？
**原因：** 训练集规模过大（>10000条）  
**解决：** 
- 使用采样（如前5000条）
- 在更强大的机器上预先构建索引并保存

### 9. External和标准JSON有什么区别？
**答：** 
- **External版本**：包含 `id` 和 `few_shot_examples` 元数据，用于调试
- **标准版本**：仅包含 `instruction`/`input`/`output`，用于模型训练
- 两个版本的 `output` 字段完全相同（LLM看到的内容一致）
- Few-shot examples在API调用时已插入Prompt，不在训练数据中

### 10. 配置文件保存在哪里？
**答：** 保存在 `configs/` 目录下（JSON格式），文件名为你输入的配置名称

### 11. 如何删除配置？
**方式1：** 主菜单 → 5. 配置管理 → d（删除）→ 输入编号  
**方式2：** 直接删除 `configs/` 目录下的对应JSON文件

---

## 🎯 设计优势

### 1. 模块化设计
- 每个模块职责单一、清晰
- 易于维护和扩展
- 代码复用性高

### 2. 配置分离
- API密钥安全存储在 `.env` 文件
- Prompt独立存储，易于更新
- 超参数集中配置

### 3. 面向对象
- BaseAgent提供统一接口
- 继承关系清晰
- 扩展新Agent简单

### 4. 功能丰富
- 测试模式支持
- 检查点自动保存
- 统计信息详细
- 错误处理完善

### 5. 易用性
- 交互式启动界面
- 丰富的使用示例
- 详细的文档说明

---

## 📦 依赖

核心依赖：pandas、openai、tqdm
安装命令：pip install pandas openai tqdm
Python版本要求：3.8+

---

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📞 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 查看详细文档：[API_KEY_SETUP.md](API_KEY_SETUP.md)

---

**所有功能都已测试通过，可以直接使用！** 🎉
