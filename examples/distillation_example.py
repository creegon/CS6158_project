"""
DistillationAgent使用示例
演示如何使用DistillationAgent进行数据蒸馏
"""
from agents import DistillationAgent


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 80)
    print("示例1: 基本使用 - 处理全部数据")
    print("=" * 80)
    
    # 创建Agent实例
    agent = DistillationAgent()
    
    # 运行蒸馏任务
    result = agent.run(output_name='full_distillation_dataset')
    
    print(f"\n处理结果:")
    print(f"  成功: {result['success_count']} 条")
    print(f"  失败: {result['failed_count']} 条")
    print(f"  输出文件: {result['output_file']}")


def example_test_mode():
    """测试模式示例"""
    print("\n" + "=" * 80)
    print("示例2: 测试模式 - 只处理最后10条数据")
    print("=" * 80)
    
    # 创建Agent实例（测试模式）
    agent = DistillationAgent(
        test_mode='last',  # 可选: 'first', 'last', 'random'
        test_size=10
    )
    
    # 运行蒸馏任务
    result = agent.run(output_name='test_distillation_dataset')
    
    print(f"\n处理结果:")
    print(f"  成功: {result['success_count']} 条")
    print(f"  耗时: {result['elapsed_time']:.2f} 秒")


def example_custom_parameters():
    """自定义参数示例"""
    print("\n" + "=" * 80)
    print("示例3: 自定义参数")
    print("=" * 80)
    
    # 创建Agent实例（自定义参数）
    agent = DistillationAgent(
        test_mode='random',
        test_size=5,
        random_seed=42,
        temperature=0.8,  # 调整温度参数
        max_tokens=1500,  # 调整最大token数
        batch_size=5,     # 批次大小
        batch_delay=2,    # 批次延迟（秒）
        checkpoint_interval=25  # 检查点间隔
    )
    
    # 运行蒸馏任务
    result = agent.run(output_name='custom_distillation_dataset')
    
    # 打印统计信息
    agent.print_stats()


def example_different_dataset():
    """使用不同数据集示例"""
    print("\n" + "=" * 80)
    print("示例4: 使用不同的数据集")
    print("=" * 80)
    
    # 创建Agent实例
    agent = DistillationAgent(test_mode='first', test_size=3)
    
    # 指定不同的数据集路径
    result = agent.run(
        dataset_path='path/to/your/dataset.csv',
        output_name='another_distillation_dataset'
    )


if __name__ == '__main__':
    # 运行示例（选择一个运行）
    
    # example_basic_usage()  # 基本使用（处理全部数据，慎用！）
    example_test_mode()      # 测试模式（推荐先用这个）
    # example_custom_parameters()  # 自定义参数
    # example_different_dataset()  # 使用不同数据集
