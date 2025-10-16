"""
API匹配使用示例

演示如何在数据蒸馏中使用API签名匹配来检索few-shot examples
"""
from pathlib import Path
from utils import load_csv, APISignatureMatcher
from agents import DistillationAgent

def example_with_api_matching():
    """演示使用API匹配的数据蒸馏"""
    print("=" * 60)
    print("示例：使用API匹配的数据蒸馏")
    print("=" * 60)
    
    # 1. 加载数据集
    dataset_path = Path(__file__).parent / 'dataset' / 'FlakyLens_dataset_with_nonflaky_indented.csv'
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    print("\n【Step 1】加载数据集")
    data = load_csv(dataset_path)
    print(f"✓ 数据集大小: {len(data)}")
    
    # 2. 划分训练集和测试集
    print("\n【Step 2】划分数据集")
    train_data = data.head(100)  # 前100条作为训练集（知识库）
    test_data = data.iloc[100:110]  # 接下来10条作为测试集
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(test_data)} 条")
    
    # 保存测试集到临时文件
    test_file = Path(__file__).parent / 'output' / 'temp_test_set.csv'
    test_file.parent.mkdir(exist_ok=True)
    test_data.to_csv(test_file, index=False)
    print(f"✓ 测试集保存到: {test_file.name}")
    
    # 3. 创建API匹配器
    print("\n【Step 3】构建API匹配索引")
    api_matcher = APISignatureMatcher(train_data, code_column='full_code')
    
    stats = api_matcher.get_statistics()
    print(f"✓ API索引构建完成:")
    print(f"  - 训练样本数: {stats['total_train_samples']}")
    print(f"  - 唯一API数: {stats['total_unique_apis']}")
    print(f"  - 平均API数/样本: {stats['avg_apis_per_sample']:.1f}")
    
    # 4. 演示检索过程
    print("\n【Step 4】演示API匹配检索")
    sample_code = test_data.iloc[0]['full_code']
    sample_project = test_data.iloc[0]['project']
    
    print(f"\n待分析的测试代码:")
    print(f"  项目: {sample_project}")
    print(f"  代码长度: {len(sample_code)} 字符")
    print(f"  代码预览: {sample_code[:200]}...")
    
    print("\n检索Top-3相似案例...")
    similar_cases = api_matcher.retrieve_top_k(sample_code, top_k=3)
    
    print(f"\n检索结果:")
    for i, (idx, similarity, row) in enumerate(similar_cases, 1):
        print(f"\n  案例 {i}:")
        print(f"    相似度: {similarity:.3f}")
        print(f"    项目: {row['project']}")
        print(f"    类别: {row['category']}")
        print(f"    代码预览: {row['full_code'][:150]}...")
    
    # 5. 使用API匹配运行蒸馏（演示模式，不实际调用API）
    print("\n【Step 5】配置蒸馏Agent")
    print("\n配置:")
    print(f"  测试集: {test_file.name}")
    print(f"  训练集: {len(train_data)} 条（用于API匹配）")
    print(f"  Few-shot数量: 3")
    print(f"  测试模式: all（处理全部10条）")
    
    print("\n✓ 配置完成！")
    print("\n如要实际运行蒸馏，可使用以下代码:")
    print("-" * 60)
    print("""
agent = DistillationAgent(
    dataset_path=test_file,
    test_mode='all',
    api_matcher=api_matcher,
    top_k_shots=3,
    parallel_workers=1
)

result = agent.run(output_name='distillation_with_api_matching')
    """)
    print("-" * 60)
    

def example_without_api_matching():
    """演示不使用API匹配的数据蒸馏（对照组）"""
    print("\n\n" + "=" * 60)
    print("对照：不使用API匹配的数据蒸馏")
    print("=" * 60)
    
    print("\n配置:")
    print("  测试集: FlakyLens_dataset_with_nonflaky_indented.csv")
    print("  测试模式: last")
    print("  数据量: 10")
    print("  API匹配: 关闭")
    
    print("\n✓ 这是标准的数据蒸馏流程，不使用few-shot examples")
    print("\n如要实际运行，可使用以下代码:")
    print("-" * 60)
    print("""
agent = DistillationAgent(
    test_mode='last',
    test_size=10,
    parallel_workers=1
)

result = agent.run(output_name='distillation_without_api_matching')
    """)
    print("-" * 60)


def compare_prompts():
    """比较有无API匹配的Prompt差异"""
    print("\n\n" + "=" * 60)
    print("Prompt对比")
    print("=" * 60)
    
    print("\n【不使用API匹配的Prompt】")
    print("-" * 60)
    print("""
项目: netty_netty
测试名称: testTimeout
代码:
@Test
public void testTimeout() {
    ...
}

请分析这个测试是否为Flaky Test，并给出分类和理由。
    """)
    
    print("\n【使用API匹配的Prompt（包含few-shot examples）】")
    print("-" * 60)
    print("""
参考案例（根据API签名相似度检索）：
============================================================

【案例 1】(相似度: 0.85)
项目: apache_hadoop
分类: 2 (Concurrency)
代码:
@Test
public void testConcurrency() {
    Thread.sleep(1000);
    ...
}
------------------------------------------------------------

【案例 2】(相似度: 0.72)
项目: spring_spring-framework
分类: 2 (Concurrency)
代码:
@Test
public void testThreadSafety() {
    ExecutorService executor = ...
    ...
}
------------------------------------------------------------

【案例 3】(相似度: 0.68)
项目: netty_netty
分类: 0 (Non-flaky)
代码:
@Test
public void testSimple() {
    ...
}
------------------------------------------------------------

待分析的测试代码:
项目: netty_netty
测试名称: testTimeout
代码:
@Test
public void testTimeout() {
    ...
}

请参考上述案例，分析这个测试是否为Flaky Test，并给出分类和理由。
    """)
    print("-" * 60)
    
    print("\n💡 对比说明:")
    print("  - 使用API匹配后，LLM可以参考相似的历史案例")
    print("  - Few-shot examples提供了具体的分类参考")
    print("  - 相似度分数帮助LLM判断参考价值")


if __name__ == '__main__':
    example_with_api_matching()
    example_without_api_matching()
    compare_prompts()
    
    print("\n\n" + "=" * 60)
    print("✓ 示例演示完成")
    print("=" * 60)
    print("\n💡 提示:")
    print("  1. 在 main.py 中选择 '1. 数据蒸馏'")
    print("  2. 按提示选择训练集和测试集")
    print("  3. 选择是否使用API匹配")
    print("  4. 配置few-shot数量（推荐3-5个）")
    print("  5. 开始蒸馏！")
