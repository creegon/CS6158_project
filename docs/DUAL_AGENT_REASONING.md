# 双Agent推理链系统

## 概述

双Agent推理链系统通过引入一个**推理指引Agent**,帮助**判断Agent**避免常见的逻辑错误和判断陷阱,从而提高Flaky Test分类的准确性。

## 问题背景

### 模型常犯的错误

在分析ID 98317的案例时,我们发现模型犯了以下错误:

**错误判断**: 将一个Non-Flaky测试误判为UC Flaky

**错误原因**:
1. **表面模式匹配**: 看到`HashMap` + `entrySet()` + 索引比较就套用UC模式
2. **过度依赖统计特征**: `entrySet`在concurrency类别有27.3x倍率
3. **忽略关键事实**: 两次迭代是对同一个HashMap实例在同一次运行中进行的
4. **混淆时间维度**: 没有区分"跨运行的不确定性"和"单次运行内的确定性"

### 根本问题

模型使用**启发式模式匹配**而非**因果推理**:

❌ **错误思路**:
```
HashMap + entrySet() + 索引比较 
→ 看起来像UC类型的Flaky Test
→ 判断为UC
```

✓ **正确思路**:
```
代码逻辑是什么？
→ 同一个HashMap迭代两次
→ 两次迭代之间没有修改
→ Java保证同一对象的迭代顺序在同一次运行中一致
→ 不会出现顺序不一致的情况
→ Non-Flaky
```

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  输入: 测试代码                                          │
└─────────────────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│  ReasoningAgent - 推理指引Agent                          │
│  (agents/reasoning_agent.py)                           │
│                                                         │
│  任务:                                                  │
│  • 分析代码结构                                         │
│  • 识别潜在风险点                                       │
│  • 提出关键推理问题                                     │
│  • 警示常见陷阱                                         │
│  • 建议推理路径                                         │
└─────────────────────────────────────────────────────────┘
                     │
                     ↓ (推理指引文本)
                     │
┌─────────────────────────────────────────────────────────┐
│  InferringAgent - 判断Agent                             │
│  (agents/inferring_agent.py)                           │
│                                                         │
│  输入:                                                  │
│  • 测试代码                                             │
│  • ReasoningAgent的推理指引                             │
│  • Few-shot examples (可选)                            │
│  • 上下文信息 (可选)                                    │
│  • 特征词频 (可选)                                      │
│                                                         │
│  任务:                                                  │
│  • 按照推理指引的路径分析                               │
│  • 避免指引中警示的陷阱                                 │
│  • 给出最终判断                                         │
└─────────────────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│  DistillationAgent - 协调器                             │
│  (agents/distillation_agent.py)                        │
│                                                         │
│  • 协调ReasoningAgent和InferringAgent                   │
│  • 管理批处理和并行                                     │
│  • 保存结果到Alpaca格式                                 │
└─────────────────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│  输出: 分类结果 + 推理过程                               │
└─────────────────────────────────────────────────────────┘
```

### ReasoningAgent (推理指引Agent)

**文件位置**: `agents/reasoning_agent.py`

**System Prompt** (`prompts/reasoning_guide_system.txt`):

核心职责:
- ✓ 引导深度推理而非简单模式匹配
- ✓ 质疑表面特征
- ✓ 追求因果推理
- ✓ 强调反证验证
- ✓ 区分时间维度

**输出结构**:
```
【核心逻辑】
<一句话描述测试在做什么>

【关键操作时序】
<列出主要操作的先后顺序>

【潜在风险点识别】
<列出可能导致不稳定的因素>

【关键推理问题】
<针对风险点提出3-5个问题>

【反证验证】
<思考：这个测试在什么情况下会失败？>

【陷阱警示】
<提醒可能的误判方向>

【推荐推理路径】
<提供分步推理建议>
```

### InferringAgent (判断Agent)

**文件位置**: `agents/inferring_agent.py`

**修改的System Prompt** (`prompts/distillation_system.txt`):

在`<think>`部分增加:
```
**必须遵循推理指引中的建议路径**，重点关注：
1. 核心逻辑理解 - 这个测试真正在验证什么？
2. 不确定性来源识别 - 哪些因素可能导致结果不一致？
3. 时间维度区分 - 是单次运行内的问题还是跨运行的问题？
4. 因果机制验证 - 不确定性的具体触发条件是什么？
5. 反证检验 - 在什么具体场景下这个测试会失败？

**避免常见陷阱：**
- ❌ 不要仅凭关键词（HashMap、thread等）就下结论
- ❌ 不要混淆统计相关性和因果关系
- ❌ 不要忽略同一运行内的确定性
- ✓ 要寻找具体的失败场景和触发机制
- ✓ 要区分表面特征和实质逻辑
```

## 使用方法

### 方式1: 独立使用两个Agent

```python
from agents import ReasoningAgent, InferringAgent

# 创建ReasoningAgent
reasoning_agent = ReasoningAgent()

# 创建InferringAgent
inferring_agent = InferringAgent(
    use_context=True,
    use_feature_hint=True
)

# 第一步: 生成推理指引
reasoning_guide = reasoning_agent.generate_reasoning_guide(
    project="pulsar",
    test_name="testMultipleHeaders",
    full_code="..."
)

# 第二步: 基于推理指引进行判断
result, metadata = inferring_agent.infer(
    project="pulsar",
    test_name="testMultipleHeaders",
    full_code="...",
    reasoning_guide=reasoning_guide
)

print(result)  # 判断结果
print(metadata)  # few-shot examples, context, features等
```

### 方式2: 通过DistillationAgent协调

DistillationAgent内部会自动协调ReasoningAgent和InferringAgent。

**命令行方式**:

```python
from agents import DistillationAgent

agent = DistillationAgent(
    test_mode='random',
    test_size=100,
    use_reasoning_guide=True,  # 启用推理指引
    use_feature_hint=True,
    # ... 其他参数
)

result = agent.run(output_name='distillation_with_guide')
```

**交互式方式(main.py)**:

```bash
python main.py
```

在Step 3.7中选择是否启用推理指引:
```
【Step 3.7/7】推理指引(双Agent推理链)
提示: 启用后将使用第一个Agent生成推理指引,帮助第二个Agent避免常见陷阱
      这会增加API调用次数和时间,但可能提高判断准确性
是否启用推理指引？(y/n, 默认n): y
```

### 输出文件命名

启用推理指引后,输出文件名会包含`guide`标识:

```
distillation_fold_1_test_random_100samples_feature_global_guide_p10_20251113_135332_external.json
                                                        ^^^^^ 
                                                        推理指引标识
```

## 效果示例

### 案例: testMultipleHeaders

**传统方式**:
- Prompt长度: ~500字符
- 判断: ❌ UC Flaky (错误)
- 原因: 看到HashMap + entrySet() → 模式匹配 → UC

**双Agent方式**:
- Prompt长度: ~1500字符 (增加200%)
- 推理指引提醒:
  - 两次迭代是对同一个HashMap
  - 迭代之间没有修改
  - Java保证单次运行内迭代顺序一致
  - 反证: 找不到失败场景
- 判断: ✓ Non-Flaky (正确)

## 性能开销

### API调用
- **传统方式**: 1次API调用/样本
- **双Agent方式**: 2次API调用/样本 (100%增加)

### 处理时间
- **传统方式**: T秒/样本
- **双Agent方式**: ~2T秒/样本

### Token消耗
- **推理指引Agent**: ~500-1000 tokens/样本
- **判断Agent**: 增加~1000-1500 tokens/样本
- **总增加**: ~1500-2500 tokens/样本

## 权衡考虑

### 优势 ✓
1. **提高准确性**: 避免常见的逻辑陷阱
2. **增强可解释性**: 推理指引可作为调试依据
3. **系统性思考**: 强制模型进行结构化推理
4. **质量保证**: 双重验证机制

### 劣势 ✗
1. **成本翻倍**: API调用和token消耗
2. **时间增加**: 处理时间翻倍
3. **复杂度**: 系统复杂度提高
4. **不保证**: 仍可能出错,只是概率降低

## 适用场景

### 推荐使用
- ✓ 高价值数据集(小样本、难例)
- ✓ 需要高准确性的场景
- ✓ 成本不敏感的情况
- ✓ 用于验证/质量检查

### 不推荐使用
- ✗ 大规模数据集(成本过高)
- ✗ 时间敏感场景
- ✗ 成本受限环境
- ✗ 简单明确的案例

## 配置选项

### 环境变量

```bash
# .env 文件
USE_REASONING_GUIDE=true
```

### 代码配置

```python
agent = DistillationAgent(
    use_reasoning_guide=True,  # 启用推理指引
    # ... 其他配置
)
```

## 未来改进方向

1. **选择性启用**: 
   - 只对模型不确定的案例启用
   - 基于confidence score决定

2. **缓存机制**:
   - 相似代码复用推理指引
   - 减少重复调用

3. **推理质量评估**:
   - 评估推理指引的有效性
   - 自动优化prompt

4. **混合策略**:
   - 结合规则引擎预筛选
   - 只对边界案例使用双Agent

## 总结

双Agent推理链系统通过引入专门的推理指引Agent,帮助判断Agent避免常见的启发式陷阱,从表面的模式匹配转向深度的因果推理。虽然成本翻倍,但对于提高分类准确性,特别是在难例上,具有显著价值。

适合在以下场景使用:
- 需要高质量标注的小规模数据集
- 难例分析和错误诊断
- 模型验证和质量保证
- 对成本不敏感的研究场景
