# Flaky Test 分类器改进方案

基于错误案例分析，提出以下具体改进方案。

---

## 方案1: 优化System Prompt（立即可实施）

### 当前问题
- 模型对"测试并发代码"和"测试有并发问题"混淆
- 看到异步操作就判断为Async Flaky

### 改进建议

在 `prompts/distillation_system.txt` 中添加：

```
**重要判断原则：**

1. **区分测试意图与测试问题**
   - ✅ "测试并发代码的功能" ≠ "测试本身有并发问题"
   - ✅ "包含异步操作" ≠ "有异步时序问题"
   - ❌ 只有当测试**缺乏适当的同步/等待机制**时才是Flaky

2. **识别保护机制**
   在判断前，必须检查是否存在以下保护机制：
   - Mock对象（FakeClock, MockExecutor等）
   - 明确的等待/同步机制（await, CountDownLatch, synchronized）
   - 测试框架的超时保护
   - 确定性的执行顺序控制
   
   **如果存在这些机制，通常不是Flaky Test**

3. **时间相关测试的判断标准**
   - ✅ 使用固定sleep等待不确定事件 → Time Flaky
   - ✅ 比较绝对时间值（依赖系统时间） → Time Flaky
   - ❌ 比较时间大小关系（A <= B）→ 通常不是Flaky
   - ❌ 使用Mock时间源 → 不是Flaky

4. **网络/IO相关测试的判断标准**
   - ✅ 真实网络请求 + 固定超时 → 可能Flaky
   - ❌ 本地服务器 + 适当等待机制 → 通常不是Flaky
   - ❌ Mock网络响应 → 不是Flaky

**负面案例（不是Flaky Test的情况）：**

示例1：并发测试但使用Mock时间
```java
@Test
public void testResourcePool() {
    FakeClock clock = new FakeClock();  // ← Mock时间源
    ResourcePool pool = new ResourcePool(clock);
    // 测试并发行为，但时间是确定的
}
```
判断：否 - Non-Flaky（使用了Mock，消除了不确定性）

示例2：网络测试但有适当同步
```java
@Test
public void testHttpServer() {
    server.start();
    server.waitUntilReady();  // ← 明确等待
    String response = client.get(url);
    assertEquals("OK", response);
}
```
判断：否 - Non-Flaky（有明确的同步机制）

示例3：时间比较的合理性
```java
@Test
public void testTimeService() {
    long time1 = service.getTime();
    long time2 = System.currentTimeMillis();
    assertTrue(time1 <= time2);  // ← 合理的大小关系
}
```
判断：否 - Non-Flaky（断言方向合理，time1应该<=time2）
```

---

## 方案2: 改进Few-Shot Examples（中期）

### 策略
添加"易混淆"的负面案例（看起来像但不是Flaky）

### 示例

**案例1：并发测试 vs 并发问题**
```
项目：neo4j
测试名：ResourcePoolTest.testConcurrentAccess
代码：
@Test
public void testConcurrentAccess() throws Exception {
    FakeClock clock = new FakeClock();
    ResourcePool<Something> pool = new ResourcePool(clock);
    
    ExecutorService executor = Executors.newFixedThreadPool(10);
    List<Future<?>> futures = new ArrayList<>();
    
    for (int i = 0; i < 100; i++) {
        futures.add(executor.submit(() -> {
            Something resource = pool.acquire();
            pool.release(resource);
        }));
    }
    
    for (Future<?> future : futures) {
        future.get();  // 等待所有任务完成
    }
    
    assertEquals(POOL_SIZE, pool.size());
}

分析：
1. 代码特征：多线程、资源池、并发访问
2. 关键保护机制：
   - 使用FakeClock消除时间不确定性
   - future.get()确保所有任务完成
   - 明确的断言条件
3. 测试意图：测试资源池在并发场景下的正确性

答案：否 - Non-Flaky

原因：虽然测试并发代码，但使用了确定性的时间控制和明确的同步机制，
测试结果是可预测和可重复的。这是一个设计良好的并发功能测试，而不是
一个有并发问题的测试。
```

**案例2：异步操作 vs 异步问题**
```
项目：apache_hadoop
测试名：TestNfs3HttpServer.testHttpServer
代码：
@Test
public void testHttpServer() throws Exception {
    Nfs3 nfs = new Nfs3(conf);
    nfs.startServiceInternal(false);
    
    RpcProgramNfs3 nfsd = (RpcProgramNfs3) nfs.getRpcProgram();
    Nfs3HttpServer infoServer = nfsd.getInfoServer();
    
    String urlRoot = infoServer.getServerURI().toString();
    String pageContents = DFSTestUtil.urlGet(new URL(urlRoot + "/jmx"));
    
    assertTrue(pageContents.contains("\"name\":\"Nfs3Metrics\""));
}

分析：
1. 代码特征：启动HTTP服务、网络请求
2. 关键保护机制：
   - startServiceInternal()是同步启动
   - getServerURI()在服务就绪后才返回
   - DFSTestUtil.urlGet()内部有重试机制
3. 测试意图：验证HTTP服务器正确启动和响应

答案：否 - Non-Flaky

原因：虽然涉及异步服务启动和网络请求，但框架提供了适当的同步机制，
确保服务就绪后才进行请求。这是一个标准的集成测试，不是Flaky Test。
```

---

## 方案3: 训练数据平衡（重要）

### 当前问题
```
Non-Flaky: 99%
Flaky: 1%
```

### 解决方案

#### 3.1 欠采样 + 过采样组合
```python
# 在训练数据准备阶段
def balance_dataset(df, target_ratio=0.3, random_seed=42):
    """
    平衡数据集
    
    Args:
        target_ratio: Flaky样本的目标比例
    """
    flaky_samples = df[df['label'] != 'non-flaky']
    non_flaky_samples = df[df['label'] == 'non-flaky']
    
    # 计算需要的样本数
    n_flaky = len(flaky_samples)
    n_non_flaky_target = int(n_flaky / target_ratio * (1 - target_ratio))
    
    # 欠采样Non-Flaky
    non_flaky_sampled = non_flaky_samples.sample(
        n=n_non_flaky_target, 
        random_state=random_seed
    )
    
    # 合并
    balanced_df = pd.concat([flaky_samples, non_flaky_sampled])
    
    return balanced_df.sample(frac=1, random_state=random_seed)
```

#### 3.2 分层采样策略
```python
def stratified_split(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    按类别分层划分数据集
    """
    from sklearn.model_selection import train_test_split
    
    # 确保每个类别都有样本
    train_data, temp_data = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],  # 保持类别比例
        random_state=42
    )
    
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio/(val_ratio + test_ratio),
        stratify=temp_data['label'],
        random_state=42
    )
    
    return train_data, val_data, test_data
```

---

## 方案4: 多阶段推理（高级）

### 概念
将判断过程分为多个阶段，逐步缩小范围

### Prompt设计

```
请按照以下步骤分析测试用例：

【阶段1：风险识别】
识别测试中可能导致不稳定的因素：
- [ ] 异步操作（回调、Promise、异步方法）
- [ ] 并发操作（多线程、共享资源）
- [ ] 时间依赖（sleep、系统时间）
- [ ] 无序集合（HashMap、Set的迭代）
- [ ] 外部依赖（网络、文件系统）

【阶段2：保护机制检查】
检查是否存在缓解措施：
- [ ] Mock/Fake对象（隔离不确定性）
- [ ] 明确的等待/同步（await、锁、栅栏）
- [ ] 重试机制
- [ ] 确定性的执行控制

【阶段3：综合判断】
基于阶段1和阶段2的分析：
- 如果有风险因素但无保护机制 → 可能是Flaky
- 如果有风险因素且有保护机制 → 检查保护是否充分
- 如果无风险因素 → 不是Flaky

【阶段4：最终结论】
给出明确的判断和理由。
```

---

## 方案5: 引入规则辅助（混合方法）

### 概念
结合规则和模型，提高准确性

### 实现

```python
class FlakyDetector:
    """混合Flaky Test检测器"""
    
    def __init__(self, model, rules):
        self.model = model
        self.rules = rules
    
    def detect(self, test_code):
        # 1. 规则预筛选
        rule_result = self.rules.check(test_code)
        
        if rule_result == "DEFINITELY_NOT_FLAKY":
            # 如果规则确定不是Flaky，直接返回
            return ("否", "Non-Flaky", "规则判断")
        
        if rule_result == "DEFINITELY_FLAKY":
            # 如果规则确定是Flaky，用模型判断类型
            category = self.model.classify_type(test_code)
            return ("是", category, "规则+模型")
        
        # 2. 不确定的情况，使用模型
        return self.model.predict(test_code)

class FlakyRules:
    """Flaky Test判断规则"""
    
    def check(self, test_code):
        # 强Non-Flaky指标
        if self._has_mock_time(test_code):
            return "DEFINITELY_NOT_FLAKY"
        
        if self._has_deterministic_executor(test_code):
            return "DEFINITELY_NOT_FLAKY"
        
        # 强Flaky指标
        if self._has_bare_sleep_without_retry(test_code):
            return "DEFINITELY_FLAKY"
        
        return "UNCERTAIN"
    
    def _has_mock_time(self, code):
        patterns = ['FakeClock', 'MockClock', 'TestClock']
        return any(p in code for p in patterns)
    
    def _has_deterministic_executor(self, code):
        patterns = ['newSingleThreadExecutor', 'directExecutor']
        return any(p in code for p in patterns)
    
    def _has_bare_sleep_without_retry(self, code):
        import re
        # Thread.sleep但没有重试机制
        has_sleep = bool(re.search(r'Thread\.sleep|Thread\.SLEEP', code))
        has_retry = bool(re.search(r'retry|Retry|await|Await', code))
        return has_sleep and not has_retry
```

---

## 实施优先级

### P0 - 立即实施（本周）
1. ✅ 优化System Prompt（添加判断原则和负面案例）
2. ✅ 调整评估指标（关注精确率和召回率，不只是准确率）

### P1 - 短期（2周内）
3. 🔄 数据平衡策略（欠采样Non-Flaky）
4. 🔄 改进Few-Shot Examples（添加易混淆案例）

### P2 - 中期（1月内）
5. 📋 多阶段推理Prompt
6. 📋 规则辅助系统

### P3 - 长期（研究方向）
7. 🔬 代码语义理解（AST分析）
8. 🔬 知识库构建（测试模式库）

---

## 预期效果

### 当前指标
- 精确率: 6.25%
- 召回率: 100%
- F1分数: 11.76%

### 目标指标（实施P0+P1后）
- 精确率: >50%
- 召回率: >70%
- F1分数: >60%

### 长期目标（实施P0+P1+P2后）
- 精确率: >70%
- 召回率: >80%
- F1分数: >75%
