# Proposal

hl2595

by325



## Problem

针对特定数据集FlakyBench做优化

在实际软件开发中，flaky tests（非确定性测试）会导致 CI/CD 管道频繁报假警、增加调试成本

已有方法在早期数据集上表现良好，但存在数据泄漏、过拟合和不真实分布的问题。

如何在真实分布（绝大多数 non-flaky，少数 flaky）下，准确识别不同类型的 flaky tests？

现有的模型微调大部分集中在诸如codeBERT等小模型上面，有借鉴和学习意义，但难以放在实际生产和应用场景中（正确率远远不够）

## Challenges

类别十分不均衡，Flaky tests 只占不到 4%，导致模型易偏向预测 non-flaky

现有模型过度依赖特定模式的词法token，导致鲁棒性不足

不同flaky类型触发机制差异很大，难以统一建模
FlakeBench 只有 8574 个样本，样本量偏小

注意要按项目维度划分数据集，即Project-wise split 

## Approach

针对类别不均衡：

1. 手动修改并自定义权重loss，为少类别的几项任务制定更高的比重（比如LLaMA-Factory的话，就是修改`LLaMA-Factory/src/llamafactory/train/sft/trainer.py`下的`compute_loss`方法）
2. 借助大模型扩展+人工审核的方式，在训练阶段制造一部分的合成数据进而缩小比例差距

微调：

1. 直接使用开源GPT类模型+LoRA

Prompt Engineering：

1. few shots
2. step by step thinking, CoT
3. markdown format

框架优化：

1. majority vote
2. project-wise cross-validation 进行严谨评估
3. 集成多个模型，分块处理对应任务

## Evaluation

主要指标：macro-F1（各类平均）

辅助指标：Accuracy、per-class F1（特别关注少数类 Concurrency, Time）

**划分**：FlakeBench 的 project-wise 4-fold split（50% train / 20% val / 30% test）

**鲁棒性测试**：在 dead code 注入、变量改名、注释扰动等情况下，观察性能下降幅度

**对比基线**：主要为该paper提出的FlakyLens (65.79% macro-F1)，Flakify, FlakyCat，以及零样本大模型

**成功标准**：能在 macro-F1 上比 FlakyLens 有所提升

## Timeline and Task Assignments



## Deliverables