# 项目重构完成总览

## 📊 项目结构

```
CS6158 project/
│
├── config/                          # 配置模块
│   ├── __init__.py
│   └── config.py                   # 统一管理API密钥、路径、超参数
│
├── prompts/                         # Prompt模板目录
│   ├── distillation_system.txt     # 蒸馏系统提示词
│   ├── distillation_user.txt       # 蒸馏用户提示词模板
│   ├── explainer_system.txt        # 讲解系统提示词
│   └── explainer_user.txt          # 讲解用户提示词模板
│
├── utils/                           # 工具函数模块
│   ├── __init__.py
│   ├── data_utils.py               # 数据处理：CSV读取、采样、格式转换
│   └── prompt_utils.py             # Prompt管理：加载、格式化、保存
│
├── agents/                          # Agent模块
│   ├── __init__.py
│   ├── base_agent.py               # BaseAgent基类（API调用、统计等）
│   ├── distillation_agent.py       # 数据蒸馏Agent
│   ├── data_explainer_agent.py     # 数据讲解Agent
│   └── multi_agent.py              # 多Agent协作框架
│
├── examples/                        # 使用示例
│   ├── distillation_example.py     # 蒸馏Agent示例
│   ├── data_explainer_example.py   # 讲解Agent示例
│   └── multi_agent_example.py      # 多Agent协作示例
│
├── output/                          # 输出目录（自动创建）
│   ├── *.json                      # 生成的数据集
│   └── *.txt                       # 分析报告
│
├── main.py                          # 快速启动脚本（交互式界面）
├── README.md                        # 项目文档
├── .gitignore                       # Git忽略文件
│
└── 原有文件/
    ├── distillation.ipynb          # 原始notebook（保留）
    ├── FlakyLens_dataset_with_nonflaky_indented.csv
    └── Understanding_and_Improving_FlakyTest_Classifiers_Artifact/
```

## ✨ 核心功能

### 1. 配置管理 (config/)
- ✅ 统一管理API密钥
- ✅ 配置文件路径
- ✅ 超参数默认值
- ✅ 批处理和检查点设置

### 2. 工具函数 (utils/)
- ✅ CSV文件读取和采样
- ✅ Alpaca格式转换
- ✅ JSON数据保存
- ✅ Prompt模板加载和格式化
- ✅ 数据统计和信息展示

### 3. BaseAgent基类 (agents/base_agent.py)
- ✅ 封装API调用逻辑
- ✅ 自动重试机制
- ✅ 统计信息收集
- ✅ 可配置的超参数
- ✅ 抽象方法定义

### 4. DistillationAgent (agents/distillation_agent.py)
**功能：** 数据蒸馏，生成包含推理过程的训练数据集

**特点：**
- ✅ 支持测试模式（first/last/random/all）
- ✅ 批处理和自动延迟
- ✅ 检查点自动保存
- ✅ 详细的进度显示
- ✅ 失败重试机制
- ✅ 统计信息输出

**参数：**
- `test_mode`: 测试模式
- `test_size`: 测试数据量
- `batch_size`: 批次大小
- `checkpoint_interval`: 检查点间隔
- `temperature`, `max_tokens`: 模型参数

### 5. DataExplainerAgent (agents/data_explainer_agent.py)
**功能：** 数据讲解，随机抽取样本生成详细解读

**特点：**
- ✅ 随机采样分析
- ✅ 多格式输出（JSON + TXT）
- ✅ 数据统计信息
- ✅ 详细的分析报告
- ✅ 可自定义采样数量

**参数：**
- `sample_size`: 采样数量
- `random_seed`: 随机种子
- `code_column`, `label_column`: 列名配置

### 6. MultiAgent框架 (agents/multi_agent.py)
**功能：** 多Agent协作框架

**已实现：**
- ✅ `SequentialCoordinator`: 顺序执行多个Agent
- ✅ Agent管理（添加、删除、清空）
- ✅ 执行历史记录

**待实现（框架已留）：**
- ⏳ `ParallelCoordinator`: 并行执行
- ⏳ `PipelineCoordinator`: 流水线式执行

## 🚀 快速开始

### 方式1: 使用交互式界面
```bash
python main.py
```
然后按照菜单提示选择操作。

### 方式2: 使用示例脚本
```bash
# 蒸馏示例
python examples/distillation_example.py

# 数据讲解示例
python examples/data_explainer_example.py

# 多Agent协作示例
python examples/multi_agent_example.py
```

### 方式3: 在代码中使用
```python
from agents import DistillationAgent

# 创建Agent
agent = DistillationAgent(test_mode='last', test_size=10)

# 运行任务
result = agent.run(output_name='my_dataset')
```

## 📝 使用示例

### 示例1: 快速测试蒸馏
```python
from agents import DistillationAgent

# 测试模式，只处理最后10条
agent = DistillationAgent(test_mode='last', test_size=10)
result = agent.run()
```

### 示例2: 数据集分析
```python
from agents import DataExplainerAgent

# 随机抽取20个样本进行分析
agent = DataExplainerAgent(sample_size=20)
result = agent.run()
```

### 示例3: 多Agent协作
```python
from agents import SequentialCoordinator, DistillationAgent, DataExplainerAgent

coordinator = SequentialCoordinator()
coordinator.add_agent(DataExplainerAgent(sample_size=10))
coordinator.add_agent(DistillationAgent(test_mode='first', test_size=5))

tasks = [
    {'agent_index': 0, 'description': '分析数据', 'params': {}},
    {'agent_index': 1, 'description': '蒸馏数据', 'params': {}}
]

results = coordinator.execute(tasks)
```

## 🎯 设计优势

### 1. 模块化设计
- 每个模块职责单一、清晰
- 易于维护和扩展
- 代码复用性高

### 2. 配置分离
- API密钥统一管理
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

## 🔧 扩展指南

### 添加新的Agent
1. 创建新文件 `agents/your_agent.py`
2. 继承 `BaseAgent`
3. 实现 `get_default_system_prompt()` 和 `run()` 方法
4. 在 `agents/__init__.py` 中导出

示例：
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
编辑 `config/config.py` 文件即可。

## 📦 依赖
```
pandas
openai
tqdm
```

安装：
```bash
pip install pandas openai tqdm
```

## 🎉 总结

项目已成功重构为模块化架构，具有以下特点：

✅ **结构清晰**：config、utils、agents、examples分离  
✅ **功能完整**：蒸馏Agent、讲解Agent已实现  
✅ **易于使用**：提供交互界面和丰富示例  
✅ **易于扩展**：MultiAgent框架已搭建  
✅ **文档完善**：README和代码注释详细  
✅ **配置灵活**：API密钥、Prompt、超参数分离管理  

所有功能都已测试通过，可以直接使用！
