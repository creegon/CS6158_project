# 文档索引

本目录包含了关于 Flaky Test 分类模型的分析和改进文档。

## 📋 文档列表

### 核心分析文档

1. **[DIAGNOSIS_SUMMARY.md](./DIAGNOSIS_SUMMARY.md)** ⭐ 推荐从这里开始
   - 模型问题诊断总结
   - 核心问题概览
   - 典型错误案例
   - 快速了解模型现状

2. **[MODEL_ERROR_ANALYSIS.md](./MODEL_ERROR_ANALYSIS.md)** 
   - 深入的错误案例分析
   - 每类错误的详细解释
   - 错误模式统计
   - 根本原因剖析

3. **[IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md)**
   - 具体的改进方案
   - 优先级排序（P0/P1/P2）
   - 实施步骤
   - 预期效果

### 技术文档

4. **[API_MATCHING.md](./API_MATCHING.md)**
   - API签名匹配系统
   - Few-shot样本检索
   - 使用方法和示例

5. **[FEW_SHOT_RECORDING.md](./FEW_SHOT_RECORDING.md)**
   - Few-shot examples记录
   - 训练数据追踪
   - 样本质量分析

6. **[QUICK_START_API_MATCHING.md](./QUICK_START_API_MATCHING.md)**
   - API匹配快速入门
   - 配置和使用示例

7. **[SILICONFLOW_GUIDE.md](./SILICONFLOW_GUIDE.md)**
   - SiliconFlow API使用指南
   - 模型配置

## 🎯 阅读建议

### 如果你想...

**快速了解模型问题**
→ 阅读 [DIAGNOSIS_SUMMARY.md](./DIAGNOSIS_SUMMARY.md)

**深入理解错误原因**
→ 阅读 [MODEL_ERROR_ANALYSIS.md](./MODEL_ERROR_ANALYSIS.md)

**制定改进计划**
→ 阅读 [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md)

**使用API匹配功能**
→ 阅读 [API_MATCHING.md](./API_MATCHING.md) 或 [QUICK_START_API_MATCHING.md](./QUICK_START_API_MATCHING.md)

**配置不同的API提供商**
→ 阅读 [SILICONFLOW_GUIDE.md](./SILICONFLOW_GUIDE.md)

## 📊 关键数据

### 当前模型性能（基于100个样本）

| 指标 | 数值 | 评价 |
|------|------|------|
| 总体准确率 | 84.00% | ✅ 良好 |
| Flaky检测精确率 | 6.25% | ❌ 严重问题 |
| Flaky检测召回率 | 100.00% | ✅ 优秀 |
| Flaky检测F1 | 11.76% | ❌ 严重问题 |

### 主要问题

1. **假阳性率过高** (93.75%)
   - 预测16个Flaky，其中15个是错的
   
2. **特征识别表面化**
   - 看到异步 → 判Async Flaky (53%误判)
   - 看到多线程 → 判Conc Flaky (20%误判)
   - 看到时间 → 判Time Flaky (20%误判)

3. **语境理解不足**
   - 混淆"测试并发"和"有并发问题"
   - 忽略Mock对象和同步机制

## 🚀 改进优先级

### P0 - 立即实施
- [ ] 优化System Prompt
- [ ] 调整评估指标

### P1 - 短期（2周）
- [ ] 平衡训练数据
- [ ] 改进Few-Shot Examples

### P2 - 中期（1月）
- [ ] 多阶段推理
- [ ] 规则辅助系统

## 📈 预期改进

| 阶段 | 精确率 | 召回率 | F1 |
|------|--------|--------|-----|
| 当前 | 6.25% | 100% | 11.76% |
| P0后 | >30% | >70% | >40% |
| P1后 | >50% | >70% | >60% |
| P2后 | >70% | >80% | >75% |

## 📝 更新记录

- **2025-10-17**: 创建诊断文档
  - 完成100个样本的评估
  - 识别核心问题
  - 制定改进计划

---

**维护者**: CS6158 Project Team  
**最后更新**: 2025-10-17
