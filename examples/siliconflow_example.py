"""
使用 SiliconFlow API 的示例
"""
from agents import DistillationAgent
from utils import load_csv, APISignatureMatcher

def test_siliconflow_basic():
    """测试基本的 SiliconFlow API 调用"""
    print("=" * 60)
    print("测试 SiliconFlow API - 基本调用")
    print("=" * 60)
    
    # 创建使用 SiliconFlow 的 Agent
    agent = DistillationAgent(
        test_mode='last',
        test_size=3,
        provider='siliconflow',  # 指定使用 SiliconFlow
        model='Qwen/Qwen2.5-7B-Instruct'  # 可选：指定具体模型
    )
    
    print(f"\n✅ Agent 配置:")
    print(f"   提供商: {agent.provider}")
    print(f"   模型: {agent.model}")
    print(f"   API URL: {agent.base_url}")
    
    # 运行蒸馏
    result = agent.run(output_name='siliconflow_test')
    
    # 显示统计
    agent.print_stats()
    
    return result


def test_siliconflow_with_api_matching():
    """测试 SiliconFlow + API 匹配"""
    print("\n" + "=" * 60)
    print("测试 SiliconFlow API - 带 API 匹配")
    print("=" * 60)
    
    # 加载训练集
    train_data = load_csv('dataset/kfold_splits/fold_1_train.csv')
    
    # 创建 API 匹配器
    api_matcher = APISignatureMatcher(train_data)
    print(f"✅ API 匹配器已创建，知识库大小: {len(train_data)}")
    
    # 创建使用 SiliconFlow 的 Agent（带 API 匹配）
    agent = DistillationAgent(
        dataset_path='dataset/kfold_splits/fold_1_test.csv',
        test_mode='first',
        test_size=5,
        provider='siliconflow',
        model='Qwen/Qwen2.5-7B-Instruct',
        api_matcher=api_matcher,
        top_k_shots=3,
        parallel_workers=3
    )
    
    print(f"\n✅ Agent 配置:")
    print(f"   提供商: {agent.provider}")
    print(f"   模型: {agent.model}")
    print(f"   API 匹配: 已启用 (Top-3)")
    print(f"   并行线程: 3")
    
    # 运行蒸馏
    result = agent.run(output_name='siliconflow_with_api')
    
    # 显示统计
    agent.print_stats()
    
    return result


def compare_providers():
    """对比不同提供商的效果"""
    print("\n" + "=" * 60)
    print("对比 DeepSeek vs SiliconFlow")
    print("=" * 60)
    
    test_cases = [
        ('DeepSeek', 'deepseek', 'deepseek-chat'),
        ('SiliconFlow', 'siliconflow', 'Qwen/Qwen2.5-7B-Instruct')
    ]
    
    results = {}
    
    for name, provider, model in test_cases:
        print(f"\n{'='*60}")
        print(f"测试 {name}")
        print(f"{'='*60}")
        
        agent = DistillationAgent(
            test_mode='first',
            test_size=2,
            provider=provider,
            model=model
        )
        
        result = agent.run(output_name=f'{provider}_comparison')
        results[name] = agent.get_stats()
        
        print(f"\n{name} 统计:")
        print(f"  成功率: {results[name]['successful_calls']}/{results[name]['total_calls']}")
        print(f"  Token使用: {results[name]['total_tokens']}")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  总调用: {stats['total_calls']}")
        print(f"  成功: {stats['successful_calls']}")
        print(f"  失败: {stats['failed_calls']}")
        print(f"  Token: {stats['total_tokens']}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'basic':
            test_siliconflow_basic()
        elif mode == 'api':
            test_siliconflow_with_api_matching()
        elif mode == 'compare':
            compare_providers()
        else:
            print("用法:")
            print("  python example_siliconflow.py basic     # 基本测试")
            print("  python example_siliconflow.py api       # API匹配测试")
            print("  python example_siliconflow.py compare   # 对比测试")
    else:
        # 默认运行基本测试
        print("运行基本测试...\n")
        test_siliconflow_basic()
        
        print("\n\n" + "=" * 60)
        print("其他测试选项:")
        print("  python example_siliconflow.py api       # 测试 API 匹配")
        print("  python example_siliconflow.py compare   # 对比不同提供商")
        print("=" * 60)
