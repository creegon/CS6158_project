# Feature Hints 配置说明

## 配置选项

在 `config/config.py` 中有两个配置选项控制特征提示的行为:

### 1. FEATURE_HINT_MODE (特征提示模式)

```python
FEATURE_HINT_MODE = "global-highest"  # 或 "category-wise"
```

#### 模式说明:

**`global-highest` (全局最高级别模式) - 推荐用于分类任务**
- 在所有类别中找到最高优先级的级别
- 只输出该级别的特征,忽略其他较低级别
- **优点**: 只显示最强信号,避免噪音干扰模型判断
- **适用场景**: 当希望给模型提供最明确的分类线索时

**示例:**
```
async wait: very_strong级别 (executorService 71.7x)
concurrency: very_strong级别 (scheduled 40.9x)
time: strong级别 (timeout 15.3x)
OD: moderate级别 (Override 7.3x)

输出: 只保留 very_strong 级别的特征
→ executorService (async wait)
→ scheduled (concurrency)
```

---

**`category-wise` (按类别分组模式)**
- 每个类别独立选择自己的最高级别
- 所有类别都会被保留
- **优点**: 提供更全面的特征信息
- **适用场景**: 当需要查看所有可能的分类线索时

**示例:**
```
async wait: very_strong级别 (executorService 71.7x)
concurrency: very_strong级别 (scheduled 40.9x)
time: strong级别 (timeout 15.3x)
OD: moderate级别 (Override 7.3x)

输出: 每个类别保留各自的最高级别
→ executorService (async wait, very_strong)
→ scheduled (concurrency, very_strong)
→ timeout (time, strong)
→ Override (OD, moderate)
```

### 2. FEATURE_HINT_MAX_PER_LEVEL (每级别最大特征数)

```python
FEATURE_HINT_MAX_PER_LEVEL = 3  # 0表示不限制
```

**仅在 `category-wise` 模式下生效**

- `0`: 输出该级别的所有特征(不限制)
- `> 0`: 每个级别最多保留N个特征(按discrimination排序)

**示例 (FEATURE_HINT_MAX_PER_LEVEL = 3):**
```
async wait有10个very_strong特征
→ 只保留discrimination最高的前3个
```

## 使用建议

### 场景1: 提高分类准确性 (推荐)
```python
FEATURE_HINT_MODE = "global-highest"
FEATURE_HINT_MAX_PER_LEVEL = 0  # 此参数不生效
```
只显示最强信号,帮助模型做出更准确的判断。

### 场景2: 探索性分析
```python
FEATURE_HINT_MODE = "category-wise"
FEATURE_HINT_MAX_PER_LEVEL = 5
```
查看所有类别的特征,但限制每类特征数量避免信息过载。

### 场景3: 完整特征视图
```python
FEATURE_HINT_MODE = "category-wise"
FEATURE_HINT_MAX_PER_LEVEL = 0
```
输出所有匹配的特征,用于详细分析。

## 环境变量配置

也可以通过环境变量设置:

```bash
# .env 文件
FEATURE_HINT_MODE=global-highest
FEATURE_HINT_MAX_PER_LEVEL=0
```

或在命令行中:

```bash
# Windows PowerShell
$env:FEATURE_HINT_MODE="global-highest"
$env:FEATURE_HINT_MAX_PER_LEVEL="3"

# Linux/Mac
export FEATURE_HINT_MODE=global-highest
export FEATURE_HINT_MAX_PER_LEVEL=3
```

## 级别优先级

从高到低:
1. **unique** (∞x): 该特征只在某一类flaky中出现
2. **very_strong** (≥20x): 极强区分度
3. **strong** (10-20x): 强区分度
4. **moderate** (5-10x): 中等区分度

## 实际效果对比

### 测试案例: apache_samza/TestMonitorService.monitor

**原始匹配结果:**
```
async wait: scheduled (40.9x, very_strong)
concurrency: executorService (71.7x, very_strong)
time: executorService (50.3x, very_strong)
OD: Override (7.3x, moderate) × 2
```

**global-highest模式输出:**
```
【executorService】: 它的类别是【concurrency】，它的词频倍率是【71.7x】
【executorService】: 它的类别是【time】，它的词频倍率是【50.3x】
【scheduled】: 它的类别是【async wait】，它的词频倍率是【40.9x】
```

**category-wise模式输出:**
```
【scheduled】: 它的类别是【async wait】，它的词频倍率是【40.9x】
【executorService】: 它的类别是【concurrency】，它的词频倍率是【71.7x】
【executorService】: 它的类别是【time】，它的词频倍率是【50.3x】
【Override】: 它的类别是【test order dependency】，它的词频倍率是【7.3x】
【Override】: 它的类别是【test order dependency】，它的词频倍率是【7.3x】
```

## 性能影响

两种模式对处理速度的影响可忽略不计,主要区别在于:
- **global-highest**: 生成的prompt更短,模型推理更快
- **category-wise**: prompt更长,但提供更多上下文信息

## 更新日志

- **2025-11-13**: 新增配置选项,支持两种过滤模式
- **2025-11-12**: 初始实现,仅支持category-wise模式
