# 快速测试指南

## 🎯 推荐测试流程

### 第一步：验证环境
```bash
# 确保已安装依赖
pip install pandas openai tqdm

# 检查Python版本（建议3.8+）
python --version
```

### 第二步：配置API密钥
编辑 `config/config.py`，确认API密钥正确：
```python
DEEPSEEK_API_KEY = "sk-b7a22a2706bd4c40919bbf86b2490712"
```

### 第三步：运行快速测试

#### 选项A：使用交互式界面（推荐）
```bash
python main.py
```
然后选择：
- `1` - 测试蒸馏（最后10条）
- `4` - 测试数据讲解（20个样本）

#### 选项B：直接运行示例
```bash
# 测试数据讲解（推荐先测试这个，速度快）
python examples/data_explainer_example.py

# 测试数据蒸馏（会调用10次API）
python examples/distillation_example.py
```

## 📊 各功能测试说明

### 1. 数据讲解测试（约1分钟）
```bash
python examples/data_explainer_example.py
```

**预期输出：**
- 数据集信息打印
- 进度条显示
- 生成 `output/dataset_analysis.json` 和 `.txt`
- 显示分析报告预览

### 2. 数据蒸馏测试（约2-5分钟）
```bash
python examples/distillation_example.py
```

**预期输出：**
- 处理进度条
- 每10条数据暂停1秒
- 生成 `output/test_distillation_dataset.json`
- 显示成功/失败统计

### 3. 多Agent协作测试（约3-6分钟）
```bash
python examples/multi_agent_example.py
```

**预期输出：**
- 顺序执行两个任务
- 先分析数据，后蒸馏
- 显示执行摘要

## 🔍 验证结果

### 检查输出文件
```bash
# 查看输出目录
ls output/

# 预期文件：
# - dataset_analysis.json
# - dataset_analysis.txt
# - test_distillation_dataset.json
# - temp_checkpoint.json（如果中断过）
```

### 查看JSON内容
```python
import json

# 查看蒸馏结果
with open('output/test_distillation_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"数据条数: {len(data)}")
    print(f"第一条: {data[0]}")
```

## 🐛 常见问题

### 1. API调用失败
**原因：** API密钥错误或网络问题  
**解决：** 检查 `config/config.py` 中的API密钥

### 2. 找不到模块
**原因：** 目录结构不正确  
**解决：** 确保在项目根目录运行命令

### 3. CSV文件找不到
**原因：** 数据集路径配置错误  
**解决：** 检查 `config/config.py` 中的 `DATASET_PATH`

### 4. 进度卡住不动
**原因：** API调用超时或限流  
**解决：** 等待重试机制生效（最多3次）

## 💡 测试技巧

### 1. 最小化测试
使用最小数据量快速验证：
```python
agent = DistillationAgent(test_mode='first', test_size=1)
result = agent.run()
```

### 2. 查看详细日志
Agent会打印详细的执行日志，注意观察：
- ✓ 成功标记
- ✗ 失败标记
- ⚠ 警告信息

### 3. 检查统计信息
每个Agent运行后都会显示统计信息：
```python
agent.print_stats()
```

### 4. 使用检查点恢复
如果中断，可以从 `output/temp_checkpoint.json` 查看已处理的数据。

## 🚀 进阶测试

### 自定义参数测试
```python
from agents import DistillationAgent

agent = DistillationAgent(
    test_mode='random',
    test_size=5,
    temperature=0.8,
    batch_size=2,
    batch_delay=0.5
)

result = agent.run(output_name='custom_test')
```

### 不同采样模式测试
```python
# 测试所有采样模式
modes = ['first', 'last', 'random']
for mode in modes:
    agent = DistillationAgent(test_mode=mode, test_size=3)
    result = agent.run(output_name=f'test_{mode}')
```

## 📈 性能参考

| 操作 | 数据量 | 预计时间 | API调用数 |
|-----|-------|---------|----------|
| 数据讲解 | 20样本 | ~1分钟 | 1次 |
| 蒸馏（测试） | 10条 | ~2-3分钟 | 10次 |
| 蒸馏（全量） | 全部 | 取决于数据集大小 | N次 |

## ✅ 测试清单

- [ ] 环境配置完成
- [ ] API密钥配置正确
- [ ] 数据讲解测试通过
- [ ] 数据蒸馏测试通过
- [ ] 输出文件生成正常
- [ ] 统计信息显示正常
- [ ] 多Agent协作测试（可选）

## 📞 问题反馈

如遇到问题，请检查：
1. Python版本（建议3.8+）
2. 依赖包是否安装完整
3. API密钥是否有效
4. 数据集文件是否存在
5. 输出目录是否有写权限

---

**祝测试顺利！** 🎉
