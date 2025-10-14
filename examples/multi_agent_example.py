"""
MultiAgent协作示例
演示如何使用多Agent协作完成复杂任务
"""
from agents import (
    DistillationAgent,
    DataExplainerAgent,
    SequentialCoordinator
)


def example_sequential_execution():
    """顺序执行示例 - 先分析数据，再进行蒸馏"""
    print("\n" + "=" * 80)
    print("示例: 顺序执行多Agent任务")
    print("=" * 80)
    
    # 创建协调器
    coordinator = SequentialCoordinator()
    
    # 创建并添加Agents
    explainer = DataExplainerAgent(sample_size=10)
    distiller = DistillationAgent(test_mode='first', test_size=5)
    
    coordinator.add_agent(explainer, name="DataExplainer")
    coordinator.add_agent(distiller, name="Distiller")
    
    # 定义任务序列
    tasks = [
        {
            'agent_index': 0,  # DataExplainer
            'description': '分析数据集',
            'params': {
                'output_name': 'multi_agent_analysis'
            }
        },
        {
            'agent_index': 1,  # Distiller
            'description': '数据蒸馏',
            'params': {
                'output_name': 'multi_agent_distillation'
            }
        }
    ]
    
    # 执行任务
    results = coordinator.execute(tasks)
    
    # 打印摘要
    coordinator.print_summary()
    
    return results


def example_pipeline_framework():
    """流水线框架示例（待实现）"""
    print("\n" + "=" * 80)
    print("示例: 流水线协作（框架）")
    print("=" * 80)
    print("提示: 可以实现前一个Agent的输出作为下一个Agent的输入")
    print("例如: 数据分析 -> 根据分析结果调整蒸馏参数 -> 执行蒸馏")


if __name__ == '__main__':
    # 运行示例
    
    example_sequential_execution()  # 顺序执行
    # example_pipeline_framework()  # 流水线框架
