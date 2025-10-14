"""
DataExplainerAgent使用示例
演示如何使用DataExplainerAgent进行数据解读
"""
from agents import DataExplainerAgent


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 80)
    print("示例1: 基本使用 - 分析数据集")
    print("=" * 80)
    
    # 创建Agent实例
    agent = DataExplainerAgent()
    
    # 运行数据讲解任务
    result = agent.run(output_name='dataset_analysis')
    
    if result['success']:
        print(f"\n分析成功!")
        print(f"  JSON文件: {result['json_file']}")
        print(f"  文本文件: {result['txt_file']}")


def example_custom_sample_size():
    """自定义采样数量示例"""
    print("\n" + "=" * 80)
    print("示例2: 自定义采样数量")
    print("=" * 80)
    
    # 创建Agent实例（采样30条数据）
    agent = DataExplainerAgent(
        sample_size=30,
        random_seed=123
    )
    
    # 运行任务
    result = agent.run(output_name='dataset_analysis_30samples')
    
    # 打印统计信息
    if result['success']:
        stats = result['statistics']
        print(f"\n数据集统计:")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  列数: {len(stats['columns'])}")
        if 'label_distribution' in stats:
            print(f"  标签分布: {stats['label_distribution']}")


def example_custom_parameters():
    """自定义参数示例"""
    print("\n" + "=" * 80)
    print("示例3: 自定义参数")
    print("=" * 80)
    
    # 创建Agent实例（自定义参数）
    agent = DataExplainerAgent(
        sample_size=15,
        random_seed=42,
        temperature=0.6,  # 降低温度，输出更确定
        max_tokens=3000,  # 增加最大token数
        code_column='code',
        label_column='label'
    )
    
    # 运行任务
    result = agent.run(output_name='detailed_analysis')
    
    # 打印API统计
    agent.print_stats()


def example_different_dataset():
    """使用不同数据集示例"""
    print("\n" + "=" * 80)
    print("示例4: 分析不同的数据集")
    print("=" * 80)
    
    # 创建Agent实例
    agent = DataExplainerAgent(sample_size=10)
    
    # 指定不同的数据集路径
    result = agent.run(
        dataset_path='path/to/your/dataset.csv',
        output_name='another_dataset_analysis'
    )


if __name__ == '__main__':
    # 运行示例（选择一个运行）
    
    example_basic_usage()  # 基本使用（推荐）
    # example_custom_sample_size()  # 自定义采样数量
    # example_custom_parameters()  # 自定义参数
    # example_different_dataset()  # 使用不同数据集
