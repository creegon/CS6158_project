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


def example_parallel_inference():
    """并行推理示例"""
    print("\n" + "=" * 80)
    print("示例5: 并行推理 - 加速数据处理")
    print("=" * 80)
    
    # 创建Agent实例（使用3个并行线程）
    agent = DistillationAgent(
        test_mode='random',
        test_size=20,
        parallel_workers=3,  # 并行线程数
        batch_size=5,
        batch_delay=0.5,  # 并行时可以减少延迟
        random_seed=42
    )
    
    # 运行蒸馏任务
    result = agent.run(output_name='parallel_distillation_dataset')
    
    print(f"\n处理结果:")
    print(f"  成功: {result['success_count']} 条")
    print(f"  失败: {result['failed_count']} 条")
    print(f"  耗时: {result['elapsed_time']:.2f} 秒")
    print(f"  平均速度: {result['success_count'] / result['elapsed_time']:.2f} 条/秒")
    print(f"  输出文件: {result['output_file']}")


def example_compare_serial_vs_parallel():
    """对比串行与并行性能"""
    print("\n" + "=" * 80)
    print("示例6: 性能对比 - 串行 vs 并行")
    print("=" * 80)
    
    import time
    
    test_size = 15
    random_seed = 42
    
    # 串行处理
    print("\n测试1: 串行处理")
    agent_serial = DistillationAgent(
        test_mode='random',
        test_size=test_size,
        parallel_workers=1,  # 串行
        random_seed=random_seed
    )
    result_serial = agent_serial.run(output_name='serial_test')
    
    # 并行处理
    print("\n测试2: 并行处理（5线程）")
    agent_parallel = DistillationAgent(
        test_mode='random',
        test_size=test_size,
        parallel_workers=5,  # 5个线程并行
        random_seed=random_seed,
        batch_delay=0.3
    )
    result_parallel = agent_parallel.run(output_name='parallel_test')
    
    # 对比结果
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    print(f"串行处理:")
    print(f"  耗时: {result_serial['elapsed_time']:.2f} 秒")
    print(f"  速度: {result_serial['success_count'] / result_serial['elapsed_time']:.2f} 条/秒")
    print(f"\n并行处理:")
    print(f"  耗时: {result_parallel['elapsed_time']:.2f} 秒")
    print(f"  速度: {result_parallel['success_count'] / result_parallel['elapsed_time']:.2f} 条/秒")
    print(f"\n提速: {result_serial['elapsed_time'] / result_parallel['elapsed_time']:.2f}x")
    print("=" * 80)


if __name__ == '__main__':
    # 运行示例（选择一个运行）
    
    # example_basic_usage()  # 基本使用（处理全部数据，慎用！）
    # example_test_mode()      # 测试模式（推荐先用这个）
    # example_custom_parameters()  # 自定义参数
    # example_different_dataset()  # 使用不同数据集
    example_parallel_inference()  # 并行推理（推荐！）
    # example_compare_serial_vs_parallel()  # 性能对比
