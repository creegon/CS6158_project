"""
快速启动脚本
提供交互式界面来运行不同的Agent任务
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import DistillationAgent, DataExplainerAgent


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print("Flaky Test分析系统 - 快速启动")
    print("=" * 60)
    print("1. 数据蒸馏（测试模式 - 最后10条）")
    print("2. 数据蒸馏（测试模式 - 前10条）")
    print("3. 数据蒸馏（测试模式 - 随机10条）")
    print("4. 数据讲解（20个样本）")
    print("5. 数据讲解（自定义样本数）")
    print("6. 退出")
    print("=" * 60)


def run_distillation_test(mode='last'):
    """运行数据蒸馏测试"""
    print(f"\n🚀 启动数据蒸馏Agent（测试模式: {mode}）...")
    
    agent = DistillationAgent(
        test_mode=mode,
        test_size=10,
        batch_size=5,
        batch_delay=1
    )
    
    output_name = f'distillation_test_{mode}'
    result = agent.run(output_name=output_name)
    
    print(f"\n✓ 蒸馏完成!")
    print(f"  成功: {result['success_count']} 条")
    print(f"  失败: {result['failed_count']} 条")
    print(f"  输出: {result['output_file']}")


def run_data_explainer(sample_size=20):
    """运行数据讲解"""
    print(f"\n🚀 启动数据讲解Agent（样本数: {sample_size}）...")
    
    agent = DataExplainerAgent(
        sample_size=sample_size,
        random_seed=42
    )
    
    result = agent.run(output_name='dataset_analysis')
    
    if result['success']:
        print(f"\n✓ 分析完成!")
        print(f"  JSON: {result['json_file']}")
        print(f"  文本: {result['txt_file']}")


def main():
    """主函数"""
    while True:
        print_menu()
        choice = input("\n请选择操作 (1-6): ").strip()
        
        if choice == '1':
            run_distillation_test(mode='last')
        elif choice == '2':
            run_distillation_test(mode='first')
        elif choice == '3':
            run_distillation_test(mode='random')
        elif choice == '4':
            run_data_explainer(sample_size=20)
        elif choice == '5':
            try:
                size = int(input("请输入样本数: ").strip())
                run_data_explainer(sample_size=size)
            except ValueError:
                print("✗ 无效的数字")
        elif choice == '6':
            print("\n👋 再见!")
            break
        else:
            print("\n✗ 无效的选择，请重试")
        
        input("\n按回车键继续...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已中断，再见!")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
