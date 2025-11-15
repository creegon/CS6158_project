"""
双Agent推理链示例
演示如何独立使用ReasoningAgent和InferringAgent
"""
from agents import ReasoningAgent, InferringAgent

# 示例测试代码
test_code = """
@Test
public void testMultipleHeaders() {
    Map<String, String> headers = new HashMap<>();
    headers.put("Content-Type", "application/json");
    headers.put("Authorization", "Bearer token");
    
    List<String> firstIteration = new ArrayList<>();
    for (Map.Entry<String, String> entry : headers.entrySet()) {
        firstIteration.add(entry.getKey());
    }
    
    List<String> secondIteration = new ArrayList<>();
    for (Map.Entry<String, String> entry : headers.entrySet()) {
        secondIteration.add(entry.getKey());
    }
    
    for (int i = 0; i < firstIteration.size(); i++) {
        assertEquals(firstIteration.get(i), secondIteration.get(i));
    }
}
"""

def main():
    print("=" * 80)
    print("双Agent推理链示例")
    print("=" * 80)
    
    # 创建ReasoningAgent
    print("\n【步骤1】创建ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    
    # 创建InferringAgent
    print("【步骤2】创建InferringAgent...")
    inferring_agent = InferringAgent(
        use_context=False,  # 禁用上下文（因为示例不在external_projects中）
        use_feature_hint=True  # 启用特征提示
    )
    
    print("\n" + "=" * 80)
    print("第一步: ReasoningAgent生成推理指引")
    print("=" * 80)
    
    # 生成推理指引
    reasoning_guide = reasoning_agent.generate_reasoning_guide(
        project="example",
        test_name="testMultipleHeaders",
        full_code=test_code
    )
    
    if reasoning_guide:
        print("\n【推理指引内容】:")
        print(reasoning_guide)
        print("\n" + "-" * 80)
    else:
        print("\n⚠️ 推理指引生成失败")
        return
    
    print("\n" + "=" * 80)
    print("第二步: InferringAgent基于推理指引进行判断")
    print("=" * 80)
    
    # 进行推断
    result, metadata = inferring_agent.infer(
        project="example",
        test_name="testMultipleHeaders",
        full_code=test_code,
        reasoning_guide=reasoning_guide
    )
    
    if result:
        print("\n【判断结果】:")
        print(result)
        print("\n" + "-" * 80)
        
        print("\n【元数据】:")
        if metadata.get('few_shot_examples'):
            print(f"  Few-shot examples: {len(metadata['few_shot_examples'])}个")
        if metadata.get('external_context'):
            print(f"  外部上下文: ✓")
        if metadata.get('feature_hints'):
            print(f"  特征提示: {len(metadata['feature_hints'])}个特征")
            for feat in metadata['feature_hints'][:3]:  # 只显示前3个
                disc_str = '∞' if feat['discrimination'] == float('inf') else f"{feat['discrimination']:.1f}x"
                print(f"    - {feat['feature']} ({feat['category']}, {feat['level']}, {disc_str})")
    else:
        print("\n⚠️ 判断失败")
    
    print("\n" + "=" * 80)
    print("API统计信息")
    print("=" * 80)
    
    # 打印两个Agent的统计信息
    print("\n【ReasoningAgent统计】:")
    reasoning_agent.print_stats()
    
    print("\n【InferringAgent统计】:")
    inferring_agent.print_stats()
    
    # 计算总成本
    total_stats = {
        'total_calls': reasoning_agent.get_stats()['total_calls'] + inferring_agent.get_stats()['total_calls'],
        'total_tokens': reasoning_agent.get_stats()['total_tokens'] + inferring_agent.get_stats()['total_tokens']
    }
    
    print("\n【总计】:")
    print(f"  总API调用: {total_stats['total_calls']}")
    print(f"  总Token消耗: {total_stats['total_tokens']}")
    
    print("\n" + "=" * 80)
    print("双Agent vs 单Agent对比")
    print("=" * 80)
    print("\n传统单Agent方式:")
    print("  - API调用: 1次")
    print("  - 可能出错: 模式匹配陷阱 (HashMap → UC)")
    print("\n双Agent推理链方式:")
    print("  - API调用: 2次 (ReasoningAgent + InferringAgent)")
    print("  - 优势: 结构化推理，避免常见陷阱")
    print("  - 成本: API调用和Token消耗翻倍")
    print("\n推荐场景:")
    print("  ✓ 高价值数据集")
    print("  ✓ 需要高准确性")
    print("  ✓ 难例分析")
    print("  ✗ 大规模数据集（成本考虑）")


if __name__ == "__main__":
    main()
