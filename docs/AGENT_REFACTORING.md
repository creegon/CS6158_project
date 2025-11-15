# 双Agent架构重构说明

## 重构概述

将原来集成在`DistillationAgent`中的推理指引功能拆分为两个独立的Agent类:

1. **ReasoningAgent** (`agents/reasoning_agent.py`) - 推理指引Agent
2. **InferringAgent** (`agents/inferring_agent.py`) - 判断Agent

## 架构变化

### 重构前

```
DistillationAgent
├── 包含推理指引生成逻辑
├── 包含few-shot检索逻辑
├── 包含上下文提取逻辑
├── 包含特征匹配逻辑
└── 包含最终判断逻辑
```

所有功能耦合在一个类中，职责不清晰。

### 重构后

```
ReasoningAgent (推理指引Agent)
└── 专注于生成推理指引

InferringAgent (判断Agent)
├── 包含few-shot检索逻辑
├── 包含上下文提取逻辑
├── 包含特征匹配逻辑
└── 基于推理指引进行判断

DistillationAgent (协调器)
├── 协调ReasoningAgent和InferringAgent
├── 管理批处理和并行
└── 保存结果到Alpaca格式
```

职责分离，每个类专注于单一功能。

## 新增文件

### 1. ReasoningAgent (`agents/reasoning_agent.py`)

**职责**: 生成结构化的推理指引

**核心方法**:
- `generate_reasoning_guide(project, test_name, full_code)` - 生成推理指引
- `generate_from_row(row, code_column)` - 从DataFrame行生成（便捷方法）

**使用的Prompt**:
- System: `prompts/reasoning_guide_system.txt`
- User: `prompts/reasoning_guide_user.txt`

**示例**:
```python
from agents import ReasoningAgent

agent = ReasoningAgent()
guide = agent.generate_reasoning_guide(
    project="pulsar",
    test_name="testMultipleHeaders",
    full_code="..."
)
```

### 2. InferringAgent (`agents/inferring_agent.py`)

**职责**: 基于推理指引进行Flaky Test分类判断

**核心方法**:
- `generate_user_prompt(project, test_name, full_code, reasoning_guide)` - 生成用户提示词
- `infer(project, test_name, full_code, reasoning_guide)` - 进行推断
- `generate_from_row(row, reasoning_guide)` - 从DataFrame行生成（便捷方法）
- `infer_from_row(row, reasoning_guide)` - 从DataFrame行推断（便捷方法）

**集成的功能**:
- Few-shot检索 (通过`api_matcher`)
- 上下文提取 (通过`context_fetcher`)
- 特征匹配 (通过`feature_matcher`)

**使用的Prompt**:
- System: `prompts/distillation_system.txt`
- User: 动态生成（包含可选的推理指引、few-shots、context、features）

**示例**:
```python
from agents import InferringAgent

agent = InferringAgent(
    use_context=True,
    use_feature_hint=True
)

result, metadata = agent.infer(
    project="pulsar",
    test_name="testMultipleHeaders",
    full_code="...",
    reasoning_guide="..."
)
```

## 修改的文件

### 1. DistillationAgent (`agents/distillation_agent.py`)

**主要变化**:

1. **删除的功能** (已移至InferringAgent):
   - `_filter_features_by_mode()` - 特征过滤逻辑
   - 直接的API匹配、上下文提取、特征匹配代码

2. **简化的方法**:
   - `generate_reasoning_guide()` - 委托给ReasoningAgent
   - `generate_user_prompt_with_examples()` - 委托给InferringAgent
   - `process_single_row()` - 协调两个Agent

3. **新增的成员**:
   - `self.reasoning_agent` - ReasoningAgent实例
   - `self.inferring_agent` - InferringAgent实例

**工作流程**:
```python
# 在process_single_row()中:
1. reasoning_guide = self.reasoning_agent.generate_from_row(row)
2. result, metadata = self.inferring_agent.infer_from_row(row, reasoning_guide)
3. 转换为Alpaca格式并保存
```

### 2. agents/__init__.py

**变化**: 导出新的Agent类
```python
from .reasoning_agent import ReasoningAgent
from .inferring_agent import InferringAgent

__all__ = [
    'ReasoningAgent',
    'InferringAgent',
    'DistillationAgent',
    # ...
]
```

## 使用方式

### 方式1: 独立使用两个Agent

```python
from agents import ReasoningAgent, InferringAgent

# 创建Agents
reasoning_agent = ReasoningAgent()
inferring_agent = InferringAgent(use_feature_hint=True)

# 生成推理指引
guide = reasoning_agent.generate_reasoning_guide(
    project="example",
    test_name="testExample",
    full_code="..."
)

# 基于推理指引进行判断
result, metadata = inferring_agent.infer(
    project="example",
    test_name="testExample",
    full_code="...",
    reasoning_guide=guide
)
```

### 方式2: 通过DistillationAgent协调 (推荐用于批处理)

```python
from agents import DistillationAgent

# DistillationAgent内部会自动协调两个Agent
agent = DistillationAgent(
    use_reasoning_guide=True,  # 启用推理指引
    use_feature_hint=True
)

result = agent.run(output_name='distillation_with_guide')
```

## 示例代码

### 新增示例
- `examples/dual_agent_example.py` - 展示如何独立使用两个Agent

### 更新示例
- `test_reasoning_guide.py` - 更新为使用ReasoningAgent和InferringAgent

## 文档更新

- `docs/DUAL_AGENT_REASONING.md` - 更新架构图和说明

## 优势

### 1. 职责分离
- ReasoningAgent专注于推理指引生成
- InferringAgent专注于判断
- DistillationAgent专注于协调

### 2. 代码复用
可以在不同场景独立使用:
- 只需要推理指引 → ReasoningAgent
- 只需要判断 → InferringAgent (reasoning_guide=None)
- 需要批处理 → DistillationAgent

### 3. 易于测试
每个Agent可以独立测试:
```python
# 测试ReasoningAgent
reasoning_agent = ReasoningAgent()
guide = reasoning_agent.generate_reasoning_guide(...)
assert guide is not None

# 测试InferringAgent
inferring_agent = InferringAgent()
result, metadata = inferring_agent.infer(...)
assert result is not None
```

### 4. 灵活组合
可以根据需求灵活组合:
- 启用/禁用推理指引
- 启用/禁用few-shot
- 启用/禁用上下文
- 启用/禁用特征提示

## 向后兼容性

✅ **完全向后兼容**

原有的使用方式仍然有效:
```python
# 原有代码无需修改
agent = DistillationAgent(
    use_reasoning_guide=True,
    use_feature_hint=True
)
result = agent.run()
```

内部实现改为使用ReasoningAgent和InferringAgent，但对外接口保持不变。

## 迁移指南

如果你有自定义的扩展，可能需要更新:

### 场景1: 直接使用DistillationAgent
✅ 无需修改，完全兼容

### 场景2: 扩展DistillationAgent
如果你重写了以下方法，可能需要更新:
- `_filter_features_by_mode()` → 已移至InferringAgent
- `generate_reasoning_guide()` → 现在委托给ReasoningAgent
- `generate_user_prompt_with_examples()` → 现在委托给InferringAgent

建议改为组合ReasoningAgent和InferringAgent，而不是继承DistillationAgent。

## 总结

这次重构将原来的单体Agent拆分为两个职责明确的Agent:
- **ReasoningAgent**: 生成推理指引
- **InferringAgent**: 基于指引进行判断
- **DistillationAgent**: 协调两者并管理批处理

这样的设计更符合单一职责原则，提高了代码的可维护性和可测试性，同时保持了向后兼容性。
