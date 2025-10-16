# API签名匹配索引

## 功能概述

API签名匹配索引是一个用于从训练集中检索最相似测试案例的工具，作为few-shot examples来增强LLM的分类能力。

### 核心思想

- **问题**: LLM在分类Flaky Test时缺乏领域知识和具体案例参考
- **解决方案**: 从训练集（知识库）中检索API签名相似的历史案例作为few-shot examples
- **优势**: 
  - 提供具体的分类参考
  - 基于代码结构相似度（而非语义相似度）
  - 快速高效（无需向量数据库）

## 工作原理

### Step 1: API提取

从Java测试代码中提取关键API调用：

```python
apis = extract_apis(code)
# 返回: ['@Test', 'Thread.sleep', 'assertEquals', ...]
```

**提取规则**:
- 测试注解: `@Test`, `@Before`, `@After`, `@Mock`等
- 方法调用: `object.method()`
- 断言API: `assertEquals`, `assertNull`, `verify`等
- 并发关键字: `Thread`, `ExecutorService`, `synchronized`等
- 时间API: `Thread.sleep`, `TimeUnit`, `System.currentTimeMillis`等
- 集合类型: `List`, `Set`, `Map`, `ArrayList`等
- I/O操作: `InputStream`, `FileReader`, `BufferedWriter`等
- Mock框架: `Mockito`, `PowerMock`
- 数据库: `Connection`, `PreparedStatement`, `ResultSet`等

### Step 2: 相似度计算

使用Jaccard相似度 + 频率加权:

```python
similarity = compute_similarity(test_apis, train_apis)
# Jaccard: |交集| / |并集|
# 频率权重: 考虑API出现次数
# 最终分数: 0.6 * jaccard + 0.4 * freq_weight
```

### Step 3: Top-K检索

返回相似度最高的K个训练样本:

```python
similar_cases = api_matcher.retrieve_top_k(test_code, top_k=3)
# 返回: [(idx, similarity, data_row), ...]
```

### Step 4: Few-shot Prompt增强

将检索到的案例插入到LLM的Prompt中：

```
参考案例（根据API签名相似度检索）：
============================================================

【案例 1】(相似度: 0.85)
项目: apache_hadoop
分类: 2 (Concurrency)
代码: ...

【案例 2】(相似度: 0.72)
项目: spring_spring-framework
分类: 2 (Concurrency)
代码: ...

待分析的测试代码:
...
```

## 使用方法

### 方法1: 通过main.py交互式使用

```bash
python main.py
```

选择 `1. 数据蒸馏`，然后按提示操作：

```
【Step 1/5】选择测试集
  1. 主数据集
  2. K-Fold: Fold 1 Train
  3. K-Fold: Fold 1 Test
  ...
选择 (1-10): 3

【Step 2/5】选择训练集（用于API匹配，可选）
是否使用API匹配？(y/n, 默认n): y

请选择训练集（用作知识库）:
  0. (不使用)
  1. 主数据集
  2. K-Fold: Fold 1 Train
  ...
选择 (0-10): 2

正在加载训练集并构建API索引...
✓ API索引构建完成:
  - 训练样本数: 6430
  - 唯一API数: 1250
  - 平均API数/样本: 18.3

请输入few-shot样本数 (默认3): 3

【Step 3/5】测试模式
1. 最后N条
2. 前N条
3. 随机N条
4. 全部数据
选择模式 (1-4, 默认1): 1

请输入数据量 (默认10): 10

【Step 4/5】并行配置
请输入并行线程数 (1-10，默认1): 1
请输入批次大小 (默认5): 5

【Step 5/5】配置确认
============================================================
测试集: fold_1_test.csv
训练集: fold_1_train.csv
API匹配: 开启 (Top-3 few-shots)
测试模式: last
数据量: 10
并行线程: 1
批次大小: 5
============================================================

确认开始？(y/n): y
```

### 方法2: 编程方式使用

```python
from utils import load_csv, APISignatureMatcher
from agents import DistillationAgent

# 1. 加载训练集
train_data = load_csv('dataset/kfold_splits/fold_1_train.csv')

# 2. 创建API匹配器
api_matcher = APISignatureMatcher(train_data, code_column='full_code')

# 3. 查看统计信息
stats = api_matcher.get_statistics()
print(f"唯一API数: {stats['total_unique_apis']}")
print(f"最常见API: {stats['most_common_apis'][:5]}")

# 4. 创建蒸馏Agent（启用API匹配）
agent = DistillationAgent(
    dataset_path='dataset/kfold_splits/fold_1_test.csv',
    test_mode='all',
    api_matcher=api_matcher,
    top_k_shots=3,
    parallel_workers=1
)

# 5. 运行蒸馏
result = agent.run(output_name='distillation_with_api_matching')
```

### 方法3: 单独使用API匹配器

```python
from utils import load_csv, APISignatureMatcher

# 加载数据
data = load_csv('dataset/FlakyLens_dataset_with_nonflaky_indented.csv')
train_data = data.head(1000)

# 创建匹配器
matcher = APISignatureMatcher(train_data)

# 检索相似案例
test_code = data.iloc[1500]['full_code']
similar_cases = matcher.retrieve_top_k(test_code, top_k=3)

for idx, similarity, row in similar_cases:
    print(f"相似度: {similarity:.3f}")
    print(f"项目: {row['project']}")
    print(f"类别: {row['category']}")
```

## API参考

### APISignatureMatcher类

```python
class APISignatureMatcher:
    def __init__(self, train_data: pd.DataFrame, code_column: str = 'full_code')
    
    @staticmethod
    def extract_apis(code: str) -> List[str]
    
    def compute_similarity(self, test_apis: List[str], train_apis: List[str]) -> float
    
    def retrieve_top_k(self, test_code: str, top_k: int = 3, 
                      min_similarity: float = 0.0) -> List[Tuple[int, float, pd.Series]]
    
    def retrieve_with_diversity(self, test_code: str, top_k: int = 3,
                                diversity_threshold: float = 0.3) -> List[...]
    
    def batch_retrieve(self, test_codes: List[str], top_k: int = 3,
                      use_diversity: bool = False) -> List[List[...]]
    
    def get_statistics(self) -> Dict
```

### DistillationAgent新增参数

```python
DistillationAgent(
    ...,
    api_matcher=None,        # API匹配器实例
    top_k_shots: int = 3,    # Few-shot数量
)
```

## 性能优化

### 预处理优化

- **预先提取API**: 训练集API在初始化时提取并缓存
- **避免重复计算**: 相似度计算使用集合操作，O(n)复杂度

### 内存优化

- **按需加载**: 只在需要时加载训练集
- **增量处理**: 支持批量检索而非一次性加载所有结果

### 速度对比

在8574条数据集上的性能测试：

- **API索引构建**: ~2秒 (100条) / ~20秒 (1000条)
- **单次检索**: ~0.01秒 (Top-3)
- **批量检索**: ~0.5秒 (50条测试样本)

## 高级特性

### 多样性检索

避免检索到彼此相似的案例：

```python
diverse_cases = matcher.retrieve_with_diversity(
    test_code, 
    top_k=5,
    diversity_threshold=0.3  # 候选案例之间相似度应<0.3
)
```

### 批量检索

一次性检索多个测试样本：

```python
results = matcher.batch_retrieve(
    test_codes=[code1, code2, code3],
    top_k=3,
    use_diversity=True
)
```

### 最小相似度过滤

只返回相似度高于阈值的案例：

```python
filtered_cases = matcher.retrieve_top_k(
    test_code,
    top_k=10,
    min_similarity=0.2  # 只返回相似度>=0.2的案例
)
```

## 实验建议

### Few-shot数量

- **3个**: 推荐，平衡质量和成本
- **1-2个**: 轻量级，适合快速测试
- **5-10个**: 高质量，但prompt较长

### 训练集选择

- **K-fold场景**: 使用对应fold的训练集
- **全量场景**: 使用整个主数据集
- **项目级**: 可以选择同项目的历史案例

### 相似度阈值

- **0.0**: 无限制，总是返回Top-K
- **0.1-0.2**: 过滤低质量案例
- **0.3+**: 只保留高度相似的案例

## 测试脚本

```bash
# 测试API匹配功能
python test_api_matcher.py

# 查看使用示例
python example_api_matching.py
```

## 未来改进

- [ ] 支持更多语言（Python, JavaScript等）
- [ ] 增加语义相似度（结合Embedding）
- [ ] 支持自定义API提取规则
- [ ] 添加缓存机制提升检索速度
- [ ] 支持增量更新训练集

## 常见问题

**Q: API匹配会增加多少时间？**  
A: 索引构建时间约为每1000条样本20秒，检索时间可忽略不计（<0.01秒/次）。

**Q: 可以不使用API匹配吗？**  
A: 可以，在main.py中选择"不使用"即可，Agent会正常工作。

**Q: Few-shot数量越多越好吗？**  
A: 不一定。3-5个通常足够，太多会导致prompt过长且成本增加。

**Q: 相似度为0是什么意思？**  
A: 表示两段代码没有共同的API签名，可能是完全不同类型的测试。

**Q: 如何判断API匹配是否有效？**  
A: 查看检索结果的相似度分数，通常>0.3表示有一定参考价值。
