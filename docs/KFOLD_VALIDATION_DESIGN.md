# K折交叉验证设计说明

## 📋 目录
- [概述](#概述)
- [设计动机](#设计动机)
- [核心设计原则](#核心设计原则)
- [实现细节](#实现细节)
- [数据结构](#数据结构)
- [使用示例](#使用示例)
- [优缺点分析](#优缺点分析)
- [最佳实践](#最佳实践)

---

## 概述

本项目实现了一个**项目级独立的K折交叉验证（Project-wise K-Fold Cross Validation）**数据集划分方案，专门用于 Flaky Test 分类任务。该设计确保同一项目的测试用例不会同时出现在训练集和测试集中，从而提供更真实的模型性能评估。

### 关键特性
- ✅ **项目级隔离**：训练集和测试集项目完全不重叠
- ✅ **类别平衡**：保证每个测试集包含足够的各类别样本
- ✅ **智能分配**：针对稀有类别采用优先分配策略
- ✅ **可复现性**：使用随机种子确保结果可重复
- ✅ **验证机制**：自动检查约束条件并给出建议

---

## 设计动机

### 为什么需要项目级K折交叉验证？

#### 1. **避免数据泄露（Data Leakage）**

**问题场景**：
```
传统的随机划分方式：
项目A: [test1, test2, test3, test4, test5]
      ↓
训练集: [test1, test2, test3]  ← 来自项目A
测试集: [test4, test5]         ← 也来自项目A

⚠️ 风险：同一项目的测试用例通常有相似的代码风格、依赖库、测试模式
        模型可能学到项目特有的特征，导致过拟合
```

**我们的解决方案**：
```
项目级划分：
训练集项目: [Project A, Project B, Project C]
测试集项目: [Project D]  ← 完全不同的项目

✓ 优势：模型必须学习通用的 Flaky Test 特征
       而不是记住特定项目的模式
```

#### 2. **真实场景模拟**

在实际应用中，模型通常需要对**新项目**的测试用例进行分类。项目级划分更接近这种场景：

```
实际应用场景：
已知项目: [A, B, C, D, E]  ← 用于训练
新项目: [F]                ← 需要预测

项目级K折验证：
Fold 1: 训练[A,B,C,D], 测试[E]  ← 模拟在E上应用
Fold 2: 训练[A,B,C,E], 测试[D]  ← 模拟在D上应用
Fold 3: 训练[A,B,D,E], 测试[C]  ← 模拟在C上应用
Fold 4: 训练[A,C,D,E], 测试[B]  ← 模拟在B上应用
```

#### 3. **类别不平衡问题**

Flaky Test 数据集通常有以下特点：
```
类别分布不均：
- Non-Flaky: 500 个样本（占 50%）
- Order-Dependent: 300 个样本（占 30%）
- Async Wait: 150 个样本（占 15%）
- Concurrency: 50 个样本（占 5%）  ← 稀有类别

⚠️ 挑战：如何确保每个测试集都包含足够的稀有类别样本？
```

**我们的策略**：
1. 识别稀有类别（样本数 < 总样本数的 1%）
2. 优先均匀分配包含稀有类别的"关键项目"
3. 对剩余项目使用贪心算法平衡分配

---

## 核心设计原则

### 1. 项目级隔离（Project-wise Disjoint）

**严格约束**：
```python
∀ fold_i, fold_j where i ≠ j:
    train_projects(fold_i) ∩ test_projects(fold_j) = ∅
```

用人话说：任意两个不同折的训练集项目和测试集项目完全不重叠。

**验证方法**：
```python
test_proj_set = set(test_df['project'].unique())
train_proj_set = set(train_df['project'].unique())
overlap = test_proj_set & train_proj_set
assert len(overlap) == 0, f"发现重叠项目: {overlap}"
```

### 2. 类别平衡约束

**软约束**（尽力满足）：
```python
∀ fold_i, ∀ category_c:
    test_samples(fold_i, category_c) ≥ min_samples_per_category
```

默认 `min_samples_per_category = 4`，即每个测试集中每个类别至少有 4 个样本。

**为什么是 4？**
- 统计学上的最小样本数（可以计算基本的准确率、召回率）
- 能够观察到该类别的基本分布特征
- 对于 4 折交叉验证，意味着该类别至少需要 16 个样本

### 3. 样本数平衡

**目标**：
```python
target_size_per_fold = total_samples / n_folds

∀ fold_i:
    minimize |actual_size(fold_i) - target_size_per_fold|
```

使每个折的样本数尽量接近平均值，避免某个折过大或过小。

---

## 实现细节

### 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│ 第一阶段：数据分析                                              │
├─────────────────────────────────────────────────────────────┤
│ 1. 统计总样本数、项目数、类别数                                 │
│ 2. 分析各类别分布和项目分布                                     │
│ 3. 识别稀有类别（样本数 < 总样本数 1%）                         │
│ 4. 检查是否能满足最小样本数约束                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第二阶段：识别关键项目                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. 找出所有包含稀有类别的项目 → "关键项目"                       │
│ 2. 对每个稀有类别，按该类别的样本数对项目排序                    │
│ 3. 记录关键项目集合 critical_projects                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第三阶段：Round-Robin 分配关键项目                               │
├─────────────────────────────────────────────────────────────┤
│ 1. 将关键项目按总样本数降序排序                                 │
│ 2. 使用轮询方式分配到各折：                                     │
│    fold_0 ← project_1                                       │
│    fold_1 ← project_2                                       │
│    fold_2 ← project_3                                       │
│    fold_3 ← project_4                                       │
│    fold_0 ← project_5  (循环)                                │
│    ...                                                      │
│ 3. 确保每折都能获得各种稀有类别的样本                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第四阶段：贪心分配剩余项目                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. 对剩余项目按样本数降序排序（先分配大项目）                    │
│ 2. 对每个项目：                                                │
│    a) 计算加入各折后的"不平衡度"：                              │
│       score = size_imbalance * 0.1 + category_imbalance    │
│    b) 选择得分最低（最平衡）的折                               │
│    c) 特殊情况：如果某折的稀有类别数量不足，给予负分优先分配      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第五阶段：生成数据集并验证                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. 为每一折生成训练集和测试集                                   │
│ 2. 验证项目不重叠约束                                           │
│ 3. 验证类别平衡约束                                             │
│ 4. 输出统计信息和警告                                           │
└─────────────────────────────────────────────────────────────┘
```

### 核心代码解析

#### 1. 稀有类别识别

```python
# 定义稀有类别：样本数少于总样本数 1% 的类别
rare_threshold = len(df) * 0.01
rare_categories = [
    cat for cat in category_counts.index 
    if category_counts[cat] < rare_threshold
]

# 示例：总样本数 1000
# → rare_threshold = 10
# → 样本数 < 10 的类别被标记为稀有类别
```

#### 2. 不平衡度计算

```python
for i, fold in enumerate(folds):
    # 样本数不平衡度
    target_size = len(df) / n_folds
    size_imbalance = abs(new_size - target_size)
    
    # 类别不平衡度
    category_imbalance = 0
    for cat in category_counts.index:
        target_cat_count = category_counts[cat] / n_folds
        category_imbalance += abs(new_cat_count - target_cat_count)
    
    # 如果是稀有类别且不足最小值，优先分配
    if cat in rare_categories and current_count < min_samples_per_category:
        category_imbalance -= 100  # 很大的负分
    
    # 总分：类别平衡权重更高
    score = size_imbalance * 0.1 + category_imbalance
```

**权重设计思路**：
- `size_imbalance * 0.1`：样本数平衡的权重较低（10%）
- `category_imbalance * 1.0`：类别平衡的权重较高（90%）
- 原因：类别平衡比样本数平衡更重要，因为我们需要每个类别都有足够的测试样本

#### 3. 训练集和测试集生成

```python
for i, test_fold in enumerate(folds):
    # 测试集：当前折的所有项目
    test_projects = test_fold['projects']
    test_df = pd.concat([
        project_info[proj]['df'] for proj in test_projects
    ], ignore_index=True)
    
    # 训练集：其他折的所有项目
    train_projects = []
    for j, fold in enumerate(folds):
        if j != i:
            train_projects.extend(fold['projects'])
    train_df = pd.concat([
        project_info[proj]['df'] for proj in train_projects
    ], ignore_index=True)
    
    # 验证项目不重叠
    assert len(set(test_projects) & set(train_projects)) == 0
```

---

## 数据结构

### 输入数据格式

CSV 文件包含以下字段：

| 字段 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `id` | int | 样本唯一标识 | 1, 2, 3, ... |
| `project` | str | 项目名称 | "neo4j_neo4j", "apache_spark" |
| `test_name` | str | 测试方法名 | "testRetryLogic" |
| `full_code` | str | 完整测试代码 | "@Test\npublic void ..." |
| `label` | int | Flaky 标签 | 0=Non-Flaky, 1=Flaky |
| `category` | int | 细分类别 | 0~5 |

**类别定义**：
```
0: Non-Flaky          - 非 Flaky 测试
1: Order-Dependent    - 依赖执行顺序
2: Async Wait         - 异步等待问题
3: Concurrency        - 并发问题
4: Test Order         - 测试顺序敏感
5: Randomness         - 随机性问题
```

### 输出数据结构

#### K折数据集列表

```python
folds = [
    {
        'train': pd.DataFrame,      # 训练集 DataFrame
        'test': pd.DataFrame,       # 测试集 DataFrame
        'train_projects': List[str], # 训练集项目列表
        'test_projects': List[str]   # 测试集项目列表
    },
    ... (重复 n_folds 次)
]
```

#### 文件命名规范

```
dataset/kfold_splits/
├── fold_1_train.csv    # 第1折训练集
├── fold_1_test.csv     # 第1折测试集
├── fold_2_train.csv    # 第2折训练集
├── fold_2_test.csv     # 第2折测试集
├── fold_3_train.csv
├── fold_3_test.csv
├── fold_4_train.csv
└── fold_4_test.csv
```

**实际项目使用的数据集**：
```
Understanding_and_Improving_FlakyTest_Classifiers_Artifact/
└── src/
    └── FlakyLens_Categorization_PerProject-Data/
        ├── train_set_1.csv    # Fold 1 训练集
        ├── test_set_1.csv     # Fold 1 测试集
        ├── train_set_2.csv    # Fold 2 训练集
        ├── test_set_2.csv     # Fold 2 测试集
        ├── train_set_3.csv    # Fold 3 训练集
        ├── test_set_3.csv     # Fold 3 测试集
        ├── train_set_4.csv    # Fold 4 训练集
        └── test_set_4.csv     # Fold 4 测试集
```

---

## 使用示例

### 基础使用

```python
from utils import create_project_wise_kfold_splits, save_kfold_datasets
import pandas as pd

# 1. 加载原始数据集
df = pd.read_csv('dataset/flaky_tests.csv')

# 2. 创建4折交叉验证数据集
folds = create_project_wise_kfold_splits(
    df,
    project_column='project',      # 项目名称列
    category_column='category',    # 类别列
    n_folds=4,                     # 4折
    min_samples_per_category=4,    # 每个测试集每类至少4个样本
    random_seed=42                 # 随机种子
)

# 3. 保存到文件
file_paths = save_kfold_datasets(
    folds,
    output_dir='dataset/kfold_splits',
    base_name='fold'
)

print(f"已保存 {len(file_paths)} 个文件")
```

### 训练循环示例

```python
from agents.few_shot_agent import FewShotAgent
from utils import load_csv

# 遍历所有折进行训练和评估
results = []

for fold_idx in range(1, 5):  # 4折
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}")
    print(f"{'='*60}")
    
    # 加载训练数据作为 Few-Shot 示例
    train_data = load_csv(f'dataset/kfold_splits/fold_{fold_idx}_train.csv')
    
    # 创建 Agent
    agent = FewShotAgent(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        few_shot_examples=train_data[:10],  # 使用前10个作为示例
        provider='siliconflow'
    )
    
    # 在测试集上评估
    result = agent.run(
        dataset_path=f'dataset/kfold_splits/fold_{fold_idx}_test.csv',
        output_name=f'results/fold_{fold_idx}',
        test_mode='all'
    )
    
    results.append({
        'fold': fold_idx,
        'accuracy': result['accuracy'],
        'f1_score': result['f1_score']
    })
    
    agent.print_stats()

# 计算平均性能
avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
avg_f1 = sum(r['f1_score'] for r in results) / len(results)

print(f"\n{'='*60}")
print(f"K折交叉验证结果汇总")
print(f"{'='*60}")
print(f"平均准确率: {avg_accuracy:.2%}")
print(f"平均F1分数: {avg_f1:.4f}")
```

### 自定义参数示例

```python
# 针对数据量较小的情况：减少最小样本数要求
folds = create_project_wise_kfold_splits(
    df,
    n_folds=3,                      # 减少折数
    min_samples_per_category=2,     # 降低最小样本数
    random_seed=2024
)

# 针对类别极度不平衡的情况：可能需要接受约束违反
folds = create_project_wise_kfold_splits(
    df,
    n_folds=5,
    min_samples_per_category=1,     # 进一步降低要求
    random_seed=42
)
```

---

## 优缺点分析

### ✅ 优点

#### 1. **避免过拟合**
- 项目级隔离防止模型记住特定项目的特征
- 强制模型学习通用的 Flaky Test 模式
- 提供更真实的泛化性能评估

#### 2. **真实场景模拟**
- 模拟模型应用于新项目的场景
- 测试集性能更能反映实际应用效果
- 有助于发现模型的真实局限性

#### 3. **类别平衡保证**
- 智能分配策略确保稀有类别有足够样本
- 每个折都能全面评估各类别性能
- 避免某些折缺少特定类别导致评估不完整

#### 4. **可复现性**
- 使用随机种子确保结果可重复
- 便于不同模型之间的公平比较
- 有利于学术研究的可验证性

#### 5. **自动验证**
- 内置约束检查和警告机制
- 自动检测项目重叠问题
- 提供详细的统计信息和建议

### ⚠️ 缺点与限制

#### 1. **样本利用率降低**
```
传统随机划分：
- 训练集: 75% 样本
- 测试集: 25% 样本

项目级划分（假设项目大小不均）：
- 最坏情况：某个大项目占 40% 样本
  → 作为测试集时，训练集只有 60% 样本
```

**影响**：训练集可能比随机划分时更小。

**缓解方法**：
- 增加折数（如 5 折或 10 折）
- 使用 Leave-One-Project-Out 策略（项目数多时）

#### 2. **可能无法满足类别平衡约束**
```
极端情况：
- 某个稀有类别只在 1 个项目中出现
- 该项目只有 5 个该类别样本
- 4 折划分，平均每折 1.25 个样本
→ 无法满足 min_samples_per_category=4 的要求
```

**影响**：某些折的某些类别样本数不足。

**缓解方法**：
- 降低 `min_samples_per_category` 参数
- 减少折数
- 接受约束违反（算法会给出警告）

#### 3. **类别分布不均**
```
示例：
Fold 1: 类别0=100, 类别1=20, 类别2=5
Fold 2: 类别0=80,  类别1=30, 类别2=10
Fold 3: 类别0=90,  类别1=25, 类别2=8
Fold 4: 类别0=70,  类别1=35, 类别2=12
```

由于项目级约束，无法像随机划分那样完美平衡每个类别。

**影响**：不同折的性能可能差异较大。

**缓解方法**：
- 使用加权平均计算总体性能
- 报告性能的标准差
- 使用更多折数平滑差异

#### 4. **计算复杂度较高**
```
时间复杂度：O(n_projects * n_folds * n_categories)
空间复杂度：O(n_folds * n_samples)
```

对于大规模数据集，项目分配算法可能较慢。

**缓解方法**：
- 使用缓存保存划分结果
- 只在必要时重新划分

---

## 最佳实践

### 1. 选择合适的折数

```python
# 项目数量与折数的对应关系
if n_projects < 10:
    n_folds = 3  # 项目少时用3折
elif n_projects < 30:
    n_folds = 4  # 中等规模用4折
elif n_projects < 50:
    n_folds = 5  # 较大规模用5折
else:
    # 项目很多时可以考虑 Leave-One-Project-Out
    n_folds = min(10, n_projects // 5)
```

### 2. 调整最小样本数要求

```python
# 根据稀有类别的样本数决定
min_category_count = min(category_counts.values())
recommended_min = max(1, min_category_count // n_folds)

print(f"推荐的 min_samples_per_category: {recommended_min}")

folds = create_project_wise_kfold_splits(
    df,
    n_folds=4,
    min_samples_per_category=recommended_min
)
```

### 3. 处理警告信息

```python
# 如果出现约束违反警告：
# ⚠️ Fold 2, 类别 5: 只有 2 个样本 (< 4)

# 选项1: 降低要求
folds = create_project_wise_kfold_splits(df, min_samples_per_category=2)

# 选项2: 减少折数
folds = create_project_wise_kfold_splits(df, n_folds=3)

# 选项3: 接受违反（在报告中说明）
# 评估时对该类别的结果谨慎解读
```

### 4. 验证划分质量

```python
def validate_kfold_splits(folds, df, project_column='project'):
    """验证K折划分的质量"""
    print(f"\n{'='*60}")
    print("K折划分质量验证")
    print(f"{'='*60}")
    
    # 1. 检查项目不重叠
    for i, fold in enumerate(folds):
        train_projects = set(fold['train'][project_column].unique())
        test_projects = set(fold['test'][project_column].unique())
        overlap = train_projects & test_projects
        
        if overlap:
            print(f"❌ Fold {i+1}: 发现重叠项目 {overlap}")
        else:
            print(f"✓ Fold {i+1}: 项目完全不重叠")
    
    # 2. 检查样本覆盖
    all_train_samples = set()
    all_test_samples = set()
    
    for fold in folds:
        all_train_samples.update(fold['train']['id'].values)
        all_test_samples.update(fold['test']['id'].values)
    
    total_samples = len(df)
    covered = len(all_train_samples | all_test_samples)
    
    print(f"\n样本覆盖率: {covered}/{total_samples} ({covered/total_samples*100:.1f}%)")
    
    # 3. 计算类别分布方差
    category_distributions = []
    for fold in folds:
        dist = fold['test']['category'].value_counts(normalize=True).to_dict()
        category_distributions.append(dist)
    
    print(f"\n类别分布一致性: ", end='')
    # 这里可以计算方差等指标
    print("需要进一步分析")
    
    print(f"{'='*60}\n")

# 使用
validate_kfold_splits(folds, df)
```

### 5. 报告结果时的注意事项

在学术论文或技术报告中，应该清楚说明：

```markdown
## 实验设置

我们使用项目级4折交叉验证（Project-wise 4-Fold Cross Validation）评估模型性能。

**划分策略**：
- 确保训练集和测试集的项目完全不重叠
- 每个测试集包含约 25% 的样本
- 采用智能分配策略平衡类别分布
- 使用随机种子 42 确保可复现性

**数据统计**：
| Fold | 训练样本 | 测试样本 | 训练项目 | 测试项目 |
|------|---------|---------|---------|---------|
| 1    | 750     | 250     | 15      | 5       |
| 2    | 780     | 220     | 16      | 4       |
| 3    | 730     | 270     | 14      | 6       |
| 4    | 760     | 240     | 15      | 5       |

**性能指标**：
- 准确率: 85.3% ± 2.1%
- F1 分数: 0.823 ± 0.015
- 各折性能详见附录 A
```

### 6. 保存和版本管理

```python
# 保存划分方案和统计信息
import json

metadata = {
    'creation_date': '2025-01-22',
    'n_folds': 4,
    'random_seed': 42,
    'min_samples_per_category': 4,
    'folds': []
}

for i, fold in enumerate(folds):
    fold_meta = {
        'fold_id': i + 1,
        'train_samples': len(fold['train']),
        'test_samples': len(fold['test']),
        'train_projects': fold['train_projects'],
        'test_projects': fold['test_projects'],
        'train_category_dist': fold['train']['category'].value_counts().to_dict(),
        'test_category_dist': fold['test']['category'].value_counts().to_dict()
    }
    metadata['folds'].append(fold_meta)

# 保存元数据
with open('dataset/kfold_splits/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ 元数据已保存到 metadata.json")
```

---

## 相关文档

- **数据处理工具**: `utils/data/data_splitter.py`
- **数据存储工具**: `utils/data/data_storage.py`
- **使用示例**: `README.md` 第 5.3 节
- **API 文档**: `docs/API_REFERENCE.md`

---

## 更新日志

| 日期 | 版本 | 变更说明 |
|-----|------|---------|
| 2025-01-22 | 1.0.0 | 初始版本，完整的K折交叉验证设计文档 |

---

## 常见问题

### Q1: 为什么不使用 sklearn 的 KFold？

**A**: sklearn 的 `KFold` 是基于样本索引的随机划分，无法保证项目级隔离。我们需要自定义实现来满足项目级约束。

### Q2: 如果数据集项目数少于折数怎么办？

**A**: 两种方案：
1. 减少折数（如 3 折或 2 折）
2. 使用 Leave-One-Project-Out（LOPO）交叉验证，即项目数 = 折数

### Q3: 为什么有时无法满足类别平衡约束？

**A**: 当稀有类别只集中在少数项目中时，受项目级约束限制，无法将这些样本均匀分配到各折。可以：
- 降低 `min_samples_per_category`
- 减少折数
- 接受约束违反并在结果中说明

### Q4: 如何选择随机种子？

**A**: 
- 固定种子（如 42）用于可复现性
- 不同种子可以测试结果的稳定性
- 可以尝试多个种子（如 42, 2024, 12345）并报告平均结果

### Q5: 项目级划分会不会太严格？

**A**: 确实更严格，但更符合实际应用场景。如果希望宽松一些：
1. 可以按"子项目"或"模块"划分而不是整个项目
2. 允许同一组织的不同项目出现在训练集和测试集中
3. 使用混合策略：大部分保持项目级隔离，小部分允许重叠

### Q6: 如何处理新项目的预测？

**A**: K折交叉验证已经模拟了这个场景！每一折都是用一部分项目训练，在另一部分项目上测试。
实际应用时：
```python
# 训练最终模型：使用所有已知项目
final_model = train_on_all_projects(all_train_data)

# 预测新项目
new_project_predictions = final_model.predict(new_project_tests)
```

---

**文档作者**: GitHub Copilot  
**最后更新**: 2025年1月22日  
**适用版本**: v1.0.0+
