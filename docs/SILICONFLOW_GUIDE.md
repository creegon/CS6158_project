# SiliconFlow 集成指南

## 📋 概述

SiliconFlow 是一个提供多种开源大模型 API 服务的平台，包括 Qwen、GLM、Yi 等模型。本系统已集成 SiliconFlow，可以轻松切换使用。

## 🚀 快速开始

### 1. 配置 API 密钥

编辑 `.env` 文件，添加 SiliconFlow API 密钥：

```bash
# SiliconFlow API配置
SILICONFLOW_API_KEY=your-siliconflow-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 设置为当前提供商
CURRENT_PROVIDER=siliconflow
```

### 2. 切换到 SiliconFlow

**方式1: 使用切换工具**
```bash
python switch_provider.py siliconflow
```

**方式2: 通过主菜单**
```bash
python main.py
# 选择 "6. 模型设置"
# 选择 "1. 切换提供商"
# 选择 "2. SiliconFlow"
```

**方式3: 直接编辑 .env**
```bash
# 修改 .env 文件中的 CURRENT_PROVIDER
CURRENT_PROVIDER=siliconflow
```

### 3. 使用 SiliconFlow

重启程序后，所有 Agent 将自动使用 SiliconFlow API。

## 🤖 支持的模型

SiliconFlow 提供以下模型：

### Qwen 系列（推荐）
- `Qwen/Qwen2.5-7B-Instruct` - 默认模型，性能均衡
- `Qwen/Qwen2.5-14B-Instruct` - 中等规模，效果更好
- `Qwen/Qwen2.5-32B-Instruct` - 大规模模型
- `Qwen/Qwen2.5-72B-Instruct` - 最强模型

### 其他模型
- `THUDM/glm-4-9b-chat` - ChatGLM4
- `01-ai/Yi-1.5-9B-Chat-16K` - Yi 模型
- `deepseek-ai/DeepSeek-V2.5` - DeepSeek（通过 SiliconFlow）

## 💡 使用示例

### 基本使用

```python
from agents import DistillationAgent

# 方式1: 使用全局配置（推荐）
# 先通过 switch_provider.py 或主菜单切换到 siliconflow
agent = DistillationAgent(
    test_mode='last',
    test_size=10
)
result = agent.run()

# 方式2: 显式指定提供商
agent = DistillationAgent(
    test_mode='last',
    test_size=10,
    provider='siliconflow',  # 显式指定
    model='Qwen/Qwen2.5-7B-Instruct'  # 可选：指定具体模型
)
result = agent.run()
```

### 带 API 匹配

```python
from agents import DistillationAgent
from utils import load_csv, APISignatureMatcher

# 加载训练集
train_data = load_csv('dataset/kfold_splits/fold_1_train.csv')
api_matcher = APISignatureMatcher(train_data)

# 使用 SiliconFlow + API 匹配
agent = DistillationAgent(
    dataset_path='dataset/kfold_splits/fold_1_test.csv',
    test_mode='all',
    provider='siliconflow',
    model='Qwen/Qwen2.5-14B-Instruct',  # 使用14B模型
    api_matcher=api_matcher,
    top_k_shots=3,
    parallel_workers=5
)
result = agent.run(output_name='siliconflow_with_api')
```

### 切换模型

```python
# 测试不同规模的模型
models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-32B-Instruct'
]

for model in models:
    agent = DistillationAgent(
        test_mode='first',
        test_size=5,
        provider='siliconflow',
        model=model
    )
    
    print(f"\n测试模型: {model}")
    result = agent.run(output_name=f'test_{model.split("/")[1]}')
    agent.print_stats()
```

## 🔧 高级配置

### 自定义参数

```python
agent = DistillationAgent(
    provider='siliconflow',
    model='Qwen/Qwen2.5-14B-Instruct',
    temperature=0.7,      # 温度参数
    max_tokens=2000,      # 最大token数
    max_retries=3,        # 最大重试次数
    parallel_workers=5    # 并行线程数
)
```

### 混合使用多个提供商

```python
# Agent 1: 使用 DeepSeek
agent1 = DistillationAgent(
    provider='deepseek',
    test_mode='first',
    test_size=10
)

# Agent 2: 使用 SiliconFlow
agent2 = DistillationAgent(
    provider='siliconflow',
    model='Qwen/Qwen2.5-7B-Instruct',
    test_mode='last',
    test_size=10
)

# 分别运行
result1 = agent1.run(output_name='deepseek_result')
result2 = agent2.run(output_name='siliconflow_result')
```

## 📊 性能对比

### DeepSeek vs SiliconFlow (Qwen)

运行对比测试：
```bash
python example_siliconflow.py compare
```

**预期差异：**

| 特性 | DeepSeek | SiliconFlow (Qwen) |
|------|----------|-------------------|
| 速度 | 快 | 中等 |
| 成本 | 较低 | 中等 |
| 模型选择 | 2个 | 7个 |
| 质量 | 高 | 取决于模型 |

## 🛠️ 故障排除

### 1. API 密钥错误

**错误信息：** `401 Unauthorized`

**解决方案：**
- 检查 `.env` 中的 `SILICONFLOW_API_KEY` 是否正确
- 确认密钥是否已激活
- 检查是否有余额

### 2. 模型不可用

**错误信息：** `Model not found`

**解决方案：**
- 查看支持的模型列表：`python switch_provider.py status`
- 使用正确的模型名称（区分大小写）
- 确认该模型在 SiliconFlow 平台上可用

### 3. 切换后仍使用旧提供商

**解决方案：**
- 确认已重启程序
- 检查 `.env` 文件中的 `CURRENT_PROVIDER` 设置
- 使用 `python switch_provider.py status` 查看当前配置

### 4. API 调用失败

**解决方案：**
- 检查网络连接
- 确认 API URL 正确：`https://api.siliconflow.cn/v1`
- 查看是否触及速率限制
- 检查日志输出的具体错误信息

## 📝 常见问题

### Q1: 如何获取 SiliconFlow API 密钥？

访问 [SiliconFlow 官网](https://siliconflow.cn/) 注册账号并创建 API 密钥。

### Q2: 哪个模型最适合 Flaky Test 分类？

**推荐：**
- 快速测试：`Qwen/Qwen2.5-7B-Instruct`
- 生产环境：`Qwen/Qwen2.5-14B-Instruct`
- 最佳效果：`Qwen/Qwen2.5-72B-Instruct`

### Q3: 可以同时使用 DeepSeek 和 SiliconFlow 吗？

可以！通过 `provider` 参数显式指定：

```python
# 使用 DeepSeek
agent1 = DistillationAgent(provider='deepseek', ...)

# 使用 SiliconFlow
agent2 = DistillationAgent(provider='siliconflow', ...)
```

### Q4: SiliconFlow 的成本如何？

SiliconFlow 采用按需付费，不同模型价格不同。查看官网了解最新定价。

### Q5: 如何切换回 DeepSeek？

```bash
python switch_provider.py deepseek
```

然后重启程序。

## 🔗 相关资源

- [SiliconFlow 官网](https://siliconflow.cn/)
- [API 文档](https://docs.siliconflow.cn/)
- [Qwen 模型介绍](https://github.com/QwenLM/Qwen)
- [本项目 README](../README.md)

## ✨ 快速命令参考

```bash
# 查看当前配置
python switch_provider.py status

# 切换到 SiliconFlow
python switch_provider.py siliconflow

# 切换到 DeepSeek
python switch_provider.py deepseek

# 运行 SiliconFlow 示例
python example_siliconflow.py basic

# 测试 API 匹配
python example_siliconflow.py api

# 对比两个提供商
python example_siliconflow.py compare
```

---

**更新日期：** 2025-10-16  
**版本：** 1.0
