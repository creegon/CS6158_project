# Flaky Test Analyzer

为LLM构建高质量的Java Flaky Test分类信息的工具。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 运行演示
python main.py --demo

# 分析单个测试
python main.py --code "@Test public void test() { Thread.sleep(1000); }"

# 分析数据集
python main.py --dataset data.xlsx --output results.json

# 生成LLM提示
python main.py --file test.java --format llm_prompt
```

## 六种Flaky类型

| Category | Label | 典型特征 |
|----------|-------|----------|
| 0 | async wait | Thread.sleep, await, CountDownLatch, Future.get |
| 1 | concurrency | AtomicInteger, synchronized, ExecutorService |
| 2 | time | System.currentTimeMillis, Date, nanoTime |
| 3 | unordered collections | HashMap/HashSet迭代, JSON字段顺序 |
| 4 | test order dependency | static字段, 文件系统, Configuration |
| 5 | non-flaky | 确定性测试 |

## 核心功能

### 1. 分析测试代码
```python
from src.analyzer import analyze_test

result = analyze_test(code)
print(result['prediction']['label'])  # 预测类别
print(result['nondeterministic_operations'])  # 非确定性操作
print(result['variable_analysis']['tainted_variables'])  # 受影响变量
```

### 2. 生成LLM输入
```python
from src.analyzer import create_llm_input

prompt = create_llm_input(code, "llm_prompt")
# 直接发送给LLM进行分类
```

### 3. 批量处理数据集
```python
from batch_processor import process_dataset

process_dataset("data.xlsx", "./output")
# 生成:
# - structured_analysis.json (结构化分析)
# - evaluation_report.txt (评估报告)
# - llm_prompts/ (LLM提示)
```

## 输出格式

### JSON格式
```json
{
  "prediction": {
    "category": 0,
    "label": "async wait",
    "confidence": 0.93
  },
  "nondeterministic_operations": {
    "by_category": {
      "async wait": [
        {"pattern": "Thread.sleep(", "line": 5, "confidence": 0.9}
      ]
    }
  },
  "variable_analysis": {
    "tainted_variables": {"result": ["async wait"]}
  },
  "assertion_analysis": {
    "high_risk_count": 1
  }
}
```

### LLM Prompt格式
生成包含完整分析上下文的提示，包括:
- 测试代码
- 检测到的非确定性操作及位置
- 变量污染追踪
- 断言风险分析
- 各类别置信度分数

## 项目结构

```
flaky_analyzer/
├── main.py              # CLI入口
├── batch_processor.py   # 批量处理
├── examples.py          # 使用示例
├── src/
│   ├── analyzer.py      # 主分析器
│   ├── ast_analyzer.py  # AST解析
│   ├── nondeterminism_detector.py  # 非确定性检测
│   └── llm_formatter.py # 输出格式化
└── config/
    └── flaky_patterns.py  # 模式配置
```

## 扩展模式

在 `config/flaky_patterns.py` 中添加新的检测模式:

```python
# 添加新的async wait模式
async_wait_patterns.append(r"myCustomAwait\s*\(")

# 调整权重
PATTERN_WEIGHTS[FlakyCategory.ASYNC_WAIT]["myCustomAwait"] = 0.85
```
