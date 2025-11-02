# Faceted Search 实施指南

## 📋 概述

基于 RubikSQL 的混合检索思想，我们实现了 **Faceted Search（多维标签过滤）** 作为 Few-shot 检索的第一层增强。

---

## 🎯 为什么选择 Faceted Search？

### 对比其他方案

| 方案 | 实施难度 | 效果提升 | 可解释性 | 推荐优先级 |
|------|---------|---------|---------|-----------|
| **Faceted Search** ✅ | ⭐⭐ (低) | ⭐⭐⭐⭐ (高) | ⭐⭐⭐⭐⭐ (最强) | **1️⃣ 最高** |
| Multi-vector | ⭐⭐⭐ (中) | ⭐⭐⭐⭐⭐ (最高) | ⭐⭐ (低) | **2️⃣ 中期** |
| Graph-based | ⭐⭐⭐⭐ (高) | ⭐⭐⭐ (中) | ⭐⭐⭐ (中) | 3️⃣ 长期 |
| Agentic Hybrid | ⭐⭐⭐⭐⭐ (最高) | ⭐⭐⭐⭐ (高) | ⭐⭐⭐⭐ (高) | 4️⃣ 未来 |

### 核心优势

1. **实施成本低** - 基于现有 API 提取规则扩展
2. **效果立竿见影** - 解决"API 重叠但场景不同"的问题
3. **可解释性强** - 能清楚看到匹配维度（并发/Mock/I/O 等）
4. **已有基础** - 你的代码已识别多个维度

---

## 🏗️ 核心架构

### 1. CodeFacets 数据结构

```python
@dataclass
class CodeFacets:
    # 9 大维度标签
    has_concurrency: bool      # 是否包含并发
    concurrency_types: Set     # 并发类型 (Thread, Lock, ...)
    
    has_mock: bool             # 是否使用 Mock
    mock_frameworks: Set       # Mock 框架 (Mockito, PowerMock, ...)
    
    has_timing: bool           # 是否时间相关
    timing_apis: Set           # 时间 API (sleep, TimeUnit, ...)
    
    has_io: bool               # 是否 I/O 操作
    io_types: Set              # I/O 类型 (File, Stream, ...)
    
    has_database: bool         # 是否数据库操作
    db_types: Set              # 数据库类型 (Connection, Statement, ...)
    
    has_exception: bool        # 是否异常处理
    exception_types: Set       # 异常类型 (NPE, IOException, ...)
    
    assert_types: Set          # 断言类型 (assertEquals, verify, ...)
    collection_types: Set      # 集合类型 (List, Map, ...)
    test_annotations: Set      # 测试注解 (@Test, @Before, ...)
```

### 2. 相似度计算策略

```python
# 混合相似度 = Facet 相似度 + API 相似度
final_score = facet_weight * facet_score + api_weight * api_score

# 默认权重: 30% Facet + 70% API
facet_weight = 0.3
api_weight = 0.7
```

### 3. 三种检索模式

#### 模式 1: 软过滤（默认）
```python
results = matcher.retrieve_top_k(
    test_code,
    top_k=3,
    facet_weight=0.3,
    api_weight=0.7,
    min_similarity=0.0,
    require_facet_match=False  # 不强制 Facet 匹配
)
```
- 综合考虑 Facet 和 API 相似度
- 适合大部分场景

#### 模式 2: 硬过滤
```python
results = matcher.retrieve_top_k(
    test_code,
    top_k=3,
    require_facet_match=True  # 强制 Facet 匹配度 >= 0.3
)
```
- 只检索 Facet 匹配的案例
- 适合有明确场景特征的测试（如纯并发测试）

#### 模式 3: 多样性检索
```python
results = matcher.retrieve_with_diversity(
    test_code,
    top_k=3,
    diversity_threshold=0.3  # 案例间相似度要 < 0.3
)
```
- 避免检索相似的重复案例
- 提供更丰富的参考信息

---

## 📝 使用示例

### 快速开始

```python
from utils import FacetedAPISignatureMatcher

# 1. 构建匹配器
matcher = FacetedAPISignatureMatcher(
    train_data=train_df,
    code_column='full_code'
)

# 2. 检索 few-shot examples
results = matcher.retrieve_top_k(
    test_code=test_code,
    top_k=3,
    facet_weight=0.3,
    api_weight=0.7
)

# 3. 查看结果
for idx, similarity, row in results:
    print(f"相似度: {similarity:.3f}")
    print(f"项目: {row['project']}")
    print(f"标签: {row['category']}")
```

### 运行演示

```bash
cd examples
python faceted_search_example.py
```

演示内容：
1. Facet 提取功能
2. 原版 vs Faceted 版本对比
3. 硬过滤模式演示
4. 多样性检索演示

---

## 🔧 集成到 DistillationAgent

### 步骤 1: 修改初始化参数

在 `agents/distillation_agent.py` 中：

```python
# 导入
from utils.faceted_api_matcher import FacetedAPISignatureMatcher

# 初始化时支持选择匹配器类型
def __init__(self, 
             ...,
             api_matcher=None,
             use_faceted_search=True,  # 新增参数
             facet_weight=0.3,         # 新增参数
             ...):
    
    if use_faceted_search and api_matcher:
        # 如果已经是 FacetedAPISignatureMatcher，直接使用
        if isinstance(api_matcher, FacetedAPISignatureMatcher):
            self.api_matcher = api_matcher
        else:
            # 否则提示用户
            print("⚠ 建议使用 FacetedAPISignatureMatcher 以获得更好的检索效果")
            self.api_matcher = api_matcher
    else:
        self.api_matcher = api_matcher
    
    self.facet_weight = facet_weight
```

### 步骤 2: 修改检索调用

在 `generate_user_prompt_with_examples` 方法中：

```python
# 检索最相似的案例
if isinstance(self.api_matcher, FacetedAPISignatureMatcher):
    # 使用 Faceted 检索
    similar_cases = self.api_matcher.retrieve_top_k(
        full_code,
        top_k=self.top_k_shots,
        facet_weight=self.facet_weight,
        api_weight=1 - self.facet_weight,
        min_similarity=0.1
    )
else:
    # 使用原版检索
    similar_cases = self.api_matcher.retrieve_top_k(
        full_code,
        top_k=self.top_k_shots,
        min_similarity=0.1
    )
```

### 步骤 3: 更新 main.py 交互界面

```python
# 在数据蒸馏配置中添加选项
print("\n【API匹配配置】")
use_faceted = input("使用 Faceted Search？(y/n, 默认y): ").strip().lower()
use_faceted = use_faceted != 'n'

if use_faceted:
    facet_weight = input("Facet 权重 (0-1, 默认0.3): ").strip()
    facet_weight = float(facet_weight) if facet_weight else 0.3
    
    # 使用 FacetedAPISignatureMatcher
    from utils.faceted_api_matcher import FacetedAPISignatureMatcher
    api_matcher = FacetedAPISignatureMatcher(train_data, code_column='full_code')
else:
    # 使用原版
    from utils.api_matcher import APISignatureMatcher
    api_matcher = APISignatureMatcher(train_data, code_column='full_code')
```

---

## 📊 实验评估

### 评估维度

1. **准确率提升**
   - 对比原版 API Matcher 和 Faceted 版本的分类准确率
   - 在不同 Flaky 类型上的表现差异

2. **Few-shot 质量**
   - 检索到的案例与测试样本的场景一致性
   - 不同标签（并发/Mock/I/O）的匹配准确性

3. **参数敏感性**
   - 不同 `facet_weight` (0.1, 0.2, 0.3, 0.4, 0.5) 的影响
   - 硬过滤 vs 软过滤的效果对比

4. **多样性分析**
   - 标准检索 vs 多样性检索的案例重复度
   - 多样性对 LLM 推理的影响

### 实验脚本模板

```python
# 对比实验
results = {}

# 1. Baseline (无 API 匹配)
results['baseline'] = run_distillation(api_matcher=None)

# 2. 原版 API Matcher
results['original'] = run_distillation(
    api_matcher=APISignatureMatcher(train_data)
)

# 3. Faceted (不同权重)
for weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
    results[f'faceted_{weight}'] = run_distillation(
        api_matcher=FacetedAPISignatureMatcher(train_data),
        facet_weight=weight
    )

# 4. Faceted + 硬过滤
results['faceted_hard'] = run_distillation(
    api_matcher=FacetedAPISignatureMatcher(train_data),
    require_facet_match=True
)

# 5. Faceted + 多样性
results['faceted_diverse'] = run_distillation(
    api_matcher=FacetedAPISignatureMatcher(train_data),
    use_diversity=True
)

# 评估
evaluate_all(results)
```

---

## 🚀 下一步：Multi-vector 混合检索

### 实施计划（中期）

1. **引入 Code Embedding**
   ```python
   from transformers import AutoTokenizer, AutoModel
   
   # 使用 CodeBERT 或 UniXcoder
   model = AutoModel.from_pretrained("microsoft/codebert-base")
   tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
   
   # 生成 embedding
   code_embedding = get_code_embedding(code, model, tokenizer)
   ```

2. **混合相似度计算**
   ```python
   # 三重混合
   final_score = (
       0.3 * facet_score +      # Facet 匹配
       0.4 * jaccard_score +    # API 结构
       0.3 * embedding_score    # 语义相似度
   )
   ```

3. **MMR 重排序**
   ```python
   # Maximal Marginal Relevance
   def mmr_rerank(candidates, lambda_param=0.7):
       selected = []
       while len(selected) < k:
           scores = []
           for c in candidates:
               relevance = similarity(query, c)
               diversity = max(similarity(c, s) for s in selected)
               score = lambda_param * relevance - (1 - lambda_param) * diversity
               scores.append(score)
           selected.append(candidates[argmax(scores)])
       return selected
   ```

---

## 📈 预期效果

基于 RubikSQL 的实验结果，Faceted Search 预期能带来：

1. **准确率提升**: +3-5% (在有明确 Facet 特征的测试上)
2. **检索精准度**: +10-15% (场景相关性)
3. **可解释性**: 显著提升（能看到匹配维度）
4. **鲁棒性**: 对不同类型测试的适应性更强

---

## 🔍 调试建议

### 1. 查看 Facet 分布

```python
stats = matcher.get_statistics()
print(stats['facet_distribution'])

# 输出示例:
# {
#   'has_concurrency': 1250 (24.8%),
#   'has_mock': 3200 (63.5%),
#   'has_timing': 890 (17.7%),
#   ...
# }
```

### 2. 分析检索质量

```python
# 保存检索结果时包含 Facet 信息
example_info = {
    'similarity': float(similarity),
    'facet_score': float(facet_score),
    'api_score': float(api_score),
    'facets': {
        'concurrency': case_facets.has_concurrency,
        'mock': case_facets.has_mock,
        'timing': case_facets.has_timing,
    }
}
```

### 3. 可视化 Facet 匹配

```python
import matplotlib.pyplot as plt

# 绘制 Facet 匹配热力图
facet_dims = ['并发', 'Mock', '时间', 'I/O', '数据库']
test_facets = [1, 0, 1, 0, 0]
case1_facets = [1, 0, 1, 0, 0]  # 高匹配
case2_facets = [0, 1, 0, 1, 0]  # 低匹配

# 绘制对比图
```

---

## ✅ 检查清单

实施 Faceted Search 前的准备：

- [ ] 阅读 `faceted_api_matcher.py` 代码
- [ ] 运行 `faceted_search_example.py` 演示
- [ ] 理解 CodeFacets 数据结构
- [ ] 理解三种检索模式的差异
- [ ] 在小数据集上测试效果
- [ ] 集成到 DistillationAgent
- [ ] 更新 main.py 交互界面
- [ ] 运行完整实验并对比结果
- [ ] 分析不同参数的影响
- [ ] 准备下一步的 Multi-vector 实施

---

## 📚 参考资料

1. **RubikSQL 论文** - Faceted Search (§5.1.2)
2. **原版 API Matcher** - `utils/api_matcher.py`
3. **Faceted 实现** - `utils/faceted_api_matcher.py`
4. **使用示例** - `examples/faceted_search_example.py`

---

## 💬 常见问题

### Q1: Faceted Search 适合所有测试吗？

**A**: 不一定。对于有明确场景特征（并发/Mock/I/O）的测试效果显著，对于简单的单元测试可能提升有限。

### Q2: facet_weight 如何选择？

**A**: 建议从 0.3 开始，根据实验结果调整：
- 如果检索案例场景不匹配 → 提高 facet_weight (0.4-0.5)
- 如果检索案例 API 不匹配 → 降低 facet_weight (0.1-0.2)

### Q3: 硬过滤会不会检索不到案例？

**A**: 可能。如果训练集中某类 Facet 的案例很少，硬过滤可能导致检索结果不足。建议先检查 `get_statistics()` 中的 Facet 分布。

### Q4: 能否同时用 Faceted + Multi-vector？

**A**: 可以！这正是下一步的计划。先实施 Faceted Search，积累经验后再加入 embedding 检索。

---

**实施优先级建议**: ⭐⭐⭐⭐⭐ **强烈推荐立即实施！**
