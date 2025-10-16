# Few-shot Examples记录功能

## 功能说明

在使用API匹配进行数据蒸馏时，系统会自动记录检索到的few-shot examples到输出JSON中（仅在`include_id=True`时）。

## 输出格式

### 标准输出文件（不带ID）
```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

### 带ID的输出文件（用于评估和Debug）
```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "id": 100,
  "few_shot_examples": [
    {
      "index": 0,
      "similarity": 0.85,
      "project": "apache_hadoop",
      "test_name": "testConcurrency",
      "category": 2,
      "code_preview": "@Test public void test() { ... }",
      "id": 1
    },
    {
      "index": 1,
      "similarity": 0.72,
      "project": "spring_framework",
      "test_name": "testAsync",
      "category": 2,
      "code_preview": "@Test public void test() { ... }",
      "id": 2
    }
  ]
}
```

## Few-shot Examples字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | int | 在训练集中的索引位置 |
| `similarity` | float | API签名相似度（0-1之间） |
| `project` | string | 案例所属项目 |
| `test_name` | string | 测试方法名称 |
| `category` | int | Flaky分类（0=非flaky, 1-5=不同flaky类型） |
| `code_preview` | string | 代码预览（前200字符） |
| `id` | int | 案例ID（如果有） |

## 使用场景

### 1. Debug分析
查看LLM参考了哪些案例进行分类：
```python
import json

with open('output/distillation_with_id.json', 'r') as f:
    data = json.load(f)

for item in data:
    if 'few_shot_examples' in item:
        print(f"样本ID: {item['id']}")
        print("参考案例:")
        for ex in item['few_shot_examples']:
            print(f"  - {ex['test_name']} (相似度={ex['similarity']:.2f}, 类别={ex['category']})")
```

### 2. 质量评估
分析few-shot质量与预测准确率的关系：
```python
# 检查相似度分布
similarities = []
for item in data:
    if 'few_shot_examples' in item:
        for ex in item['few_shot_examples']:
            similarities.append(ex['similarity'])

print(f"平均相似度: {np.mean(similarities):.3f}")
print(f"最高相似度: {np.max(similarities):.3f}")
print(f"最低相似度: {np.min(similarities):.3f}")
```

### 3. 案例多样性分析
检查检索到的案例是否足够多样：
```python
# 统计案例的项目分布
projects = []
categories = []
for item in data:
    if 'few_shot_examples' in item:
        for ex in item['few_shot_examples']:
            projects.append(ex['project'])
            categories.append(ex['category'])

from collections import Counter
print("项目分布:", Counter(projects).most_common(10))
print("类别分布:", Counter(categories))
```

### 4. 错误分析
当预测错误时，检查是否是few-shot质量问题：
```python
# 找出预测错误且相似度低的案例
for item in data:
    if 'few_shot_examples' in item:
        avg_sim = np.mean([ex['similarity'] for ex in item['few_shot_examples']])
        if avg_sim < 0.3:  # 平均相似度很低
            print(f"⚠️ 样本ID {item['id']} 的few-shot质量较低 (平均相似度={avg_sim:.2f})")
```

## 配置选项

### 启用few-shot记录
```python
agent = DistillationAgent(
    dataset_path='test.csv',
    api_matcher=api_matcher,
    top_k_shots=3  # 检索3个案例
)

# run()方法会自动生成两个文件：
# - distillation_xxx.json (标准格式)
# - distillation_xxx_with_id.json (包含ID和few-shot)
result = agent.run(output_name='distillation_with_api')
```

### 不使用API匹配
```python
agent = DistillationAgent(
    dataset_path='test.csv',
    api_matcher=None  # 不使用API匹配
)

# 输出的with_id文件不会包含few_shot_examples字段
result = agent.run(output_name='distillation_no_api')
```

## 存储开销

Few-shot examples会增加JSON文件大小：

| Few-shot数量 | 单条记录增加 | 10000条记录增加 |
|-------------|------------|----------------|
| 0（不使用）| 0 KB | 0 MB |
| 3个 | ~0.5 KB | ~5 MB |
| 5个 | ~0.8 KB | ~8 MB |
| 10个 | ~1.5 KB | ~15 MB |

**建议**: 
- 生产环境：使用3-5个few-shot（平衡质量和存储）
- Debug模式：可增加到10个获取更多信息

## 隐私和安全

Few-shot examples包含训练集的代码片段（前200字符），请注意：
1. 不要在公开场合分享包含敏感代码的JSON文件
2. 如果训练集包含私有代码，确保输出文件的访问权限
3. 可以考虑在保存时截断`code_preview`或完全移除

## 示例分析脚本

```python
"""
分析few-shot examples的质量和分布
"""
import json
import numpy as np
from collections import Counter

def analyze_few_shot_quality(json_file):
    """分析few-shot质量"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计
    total_samples = len(data)
    samples_with_few_shot = sum(1 for item in data if 'few_shot_examples' in item)
    
    print(f"总样本数: {total_samples}")
    print(f"使用few-shot的样本数: {samples_with_few_shot}")
    print(f"使用率: {samples_with_few_shot/total_samples*100:.1f}%")
    
    if samples_with_few_shot == 0:
        return
    
    # 相似度分析
    all_similarities = []
    for item in data:
        if 'few_shot_examples' in item:
            for ex in item['few_shot_examples']:
                all_similarities.append(ex['similarity'])
    
    print(f"\n相似度统计:")
    print(f"  平均: {np.mean(all_similarities):.3f}")
    print(f"  中位数: {np.median(all_similarities):.3f}")
    print(f"  标准差: {np.std(all_similarities):.3f}")
    print(f"  最小: {np.min(all_similarities):.3f}")
    print(f"  最大: {np.max(all_similarities):.3f}")
    
    # 类别分布
    categories = []
    for item in data:
        if 'few_shot_examples' in item:
            for ex in item['few_shot_examples']:
                categories.append(ex['category'])
    
    print(f"\n类别分布:")
    for cat, count in Counter(categories).most_common():
        print(f"  类别 {cat}: {count} ({count/len(categories)*100:.1f}%)")

if __name__ == '__main__':
    analyze_few_shot_quality('output/distillation_with_api_with_id.json')
```

## 常见问题

**Q: 为什么只在with_id文件中记录few-shot？**  
A: 为了保持标准Alpaca格式的纯净，few-shot信息仅用于debug和分析，不影响模型训练。

**Q: 可以禁用few-shot记录吗？**  
A: 可以，只需不使用API匹配（`api_matcher=None`）或设置`top_k_shots=0`。

**Q: Few-shot记录会影响性能吗？**  
A: 几乎不影响，记录过程只增加<1ms的序列化时间。

**Q: 可以自定义code_preview的长度吗？**  
A: 目前固定为200字符，可在`distillation_agent.py`中修改。
