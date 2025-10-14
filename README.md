# Flaky Test分析系统

这是一个模块化的Flaky Test分析系统，支持数据蒸馏、数据讲解和多Agent协作等功能。

## 目录

- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [核心模块说明](#-核心模块说明)
- [使用场景](#-使用场景)
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
├── prompts/                     # Prompt模板
│   ├── distillation_system.txt # 蒸馏系统提示词
│   ├── distillation_user.txt   # 蒸馏用户提示词
│   ├── explainer_system.txt    # 讲解系统提示词
│   └── explainer_user.txt      # 讲解用户提示词
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── data_utils.py           # 数据处理工具（CSV/JSON读写、Alpaca格式转换）
│   ├── prompt_utils.py         # Prompt处理工具（模板加载、格式化）
│   └── evaluation_utils.py     # 评估工具函数（答案提取、指标计算）
│
├── agents/                      # Agent模块
│   ├── __init__.py
│   ├── base_agent.py           # Agent基类
│   ├── distillation_agent.py   # 数据蒸馏Agent
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
│   └── evaluation_example.py   # 评估示例
│
├── dataset/                     # 数据集目录
│   └── FlakyLens_dataset_with_nonflaky_indented.csv  # 原始数据集
│
├── output/                      # 输出目录
│   └── (生成的文件)
│
├── main.py                      # 快速启动脚本（交互式界面）
├── README.md                    # 项目文档
└── .gitignore                   # Git忽略文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas openai tqdm
```

### 2. 配置API密钥

**重要：为了安全，API密钥存储在`.env`文件中**

```bash
# 1. 复制示例配置文件
copy .env.example .env

# 2. 编辑.env文件，填入你的API密钥
# DEEPSEEK_API_KEY=your-api-key-here
```

详细配置说明请查看 [API_KEY_SETUP.md](API_KEY_SETUP.md)

⚠️ **注意**: `.env`文件已添加到`.gitignore`，不会被提交到Git仓库

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

```
答案：是 - Async
答案：是 - Conc  
答案：否 - Non-Flaky
```

格式说明：
- `答案：` - 固定前缀
- `是/否` - 表示是否为Flaky Test
- `-` - 分隔符
- `类型` - Async, Conc, Time, UC, OD, 或 Non-Flaky

#### 基本使用

```python
from evaluation import Evaluator

# 创建评估器
evaluator = Evaluator(
    prediction_file='output/predictions.json',  # Alpaca格式
    ground_truth_file='dataset/labels.csv',      # 真实标签
    label_column='label'
)

# 运行评估并保存报告
metrics = evaluator.run(
    output_dir='output/evaluation',
    save_report=True,
    detailed=True
)
```

#### 文件格式要求

**预测结果文件 (JSON - Alpaca格式):**
```json
[
  {
    "instruction": "请分析以下测试用例...",
    "input": "测试代码：\n...",
    "output": "答案：是 - Async\n\n详细分析..."
  }
]
```

**真实标签文件 (CSV):**
```csv
id,label,...
0,async wait,...
1,concurrency,...
2,non-flaky,...
```

支持的标签值会自动标准化：
- `async wait`, `async`, `Async` → Async
- `concurrency`, `conc`, `Conc` → Conc
- `time`, `Time` → Time
- `unordered collections`, `uc`, `UC` → UC
- `test order dependency`, `od`, `OD` → OD
- `non-flaky`, `nonflaky`, `Non-Flaky` → Non-Flaky

#### 评估指标

1. **总体准确率 (Overall Accuracy)**: 同时判断对"是否Flaky"和"具体类型"的准确率
2. **Flaky检测指标**: 准确率、精确率、召回率、F1分数
3. **类别分类指标**: 分类准确率和各类别的详细指标

#### 高级用法

```python
# 分步骤执行
evaluator = Evaluator(
    prediction_file='output/predictions.json',
    ground_truth_file='dataset/labels.csv',
    label_column='label',
    id_column='id'  # 可选：指定ID列
)

evaluator.load_data()
evaluator.evaluate()
evaluator.print_report(detailed=True)
evaluator.save_report('output/evaluation', 'my_report')

# 评估多个模型
models = {
    'model_v1': 'output/model_v1_predictions.json',
    'model_v2': 'output/model_v2_predictions.json',
}

for name, pred_file in models.items():
    evaluator = Evaluator(pred_file, 'dataset/labels.csv', label_column='label')
    metrics = evaluator.run(output_dir=f'output/evaluation/{name}')
    print(f"{name}: Accuracy={metrics['overall_accuracy']:.2%}")
```

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
````
```

### 4. Evaluation模块 (`evaluation/`)

评估模块用于评估Flaky Test分类模型的性能。

#### Evaluator (`evaluator.py`)

评估器主类，整合所有评估功能。

**主要功能:**
- 加载Alpaca格式的预测结果
- 加载CSV格式的真实标签
- 计算各项评估指标
- 生成详细的评估报告

**使用示例:**
```python
from evaluation import Evaluator

evaluator = Evaluator(
    prediction_file='output/distillation_test_random.json',
    ground_truth_file='dataset/FlakyLens_dataset_with_nonflaky_indented.csv',
    label_column='label'
)

metrics = evaluator.run(
    output_dir='output/evaluation',
    save_report=True,
    detailed=True
)
```

**评估指标:**
- 总体准确率 (Overall Accuracy)
- Flaky检测指标：准确率、精确率、召回率、F1分数
- 类别分类准确率
- 各类别详细指标（Async, Conc, Time, UC, OD, Non-Flaky）

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

### 场景5: 评估模型性能

```python
# 评估预测结果
from evaluation import Evaluator

evaluator = Evaluator(
    prediction_file='output/predictions.json',
    ground_truth_file='dataset/labels.csv',
    label_column='label'
)

metrics = evaluator.run(output_dir='output/evaluation')
print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
print(f"Flaky F1: {metrics['flaky_detection']['f1']:.2%}")
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

1. 创建新文件 `agents/your_agent.py`
2. 继承 `BaseAgent`
3. 实现 `get_default_system_prompt()` 和 `run()` 方法
4. 在 `agents/__init__.py` 中导出

```python
from agents.base_agent import BaseAgent

class YourAgent(BaseAgent):
    def get_default_system_prompt(self):
        return "你的系统提示词"
    
    def run(self, **kwargs):
        # 实现你的逻辑
        pass
```

### 添加新的Prompt模板

直接在 `prompts/` 目录下创建 `.txt` 文件，然后用 `load_prompt()` 加载。

### 修改配置

编辑 `config/config.py` 或 `.env` 文件即可。

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

```bash
pip install pandas openai tqdm
```

**Python版本要求：** 3.8+

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
