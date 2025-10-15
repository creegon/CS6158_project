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
from evaluation import Evaluator
from utils import load_csv, split_dataset, save_split_datasets
from config import DATASET_PATH, OUTPUT_DIR


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print("Flaky Test分析系统 - 快速启动")
    print("=" * 60)
    print("1. 数据蒸馏（测试模式 - 最后10条，单线程）")
    print("2. 数据蒸馏（测试模式 - 前10条，单线程）")
    print("3. 数据蒸馏（测试模式 - 随机10条，单线程）")
    print("4. 数据蒸馏（测试模式 - 并行推理）")
    print("5. 数据讲解（20个样本）")
    print("6. 数据讲解（自定义样本数）")
    print("7. 评估预测结果")
    print("8. 数据集划分（训练集/验证集/测试集）")
    print("9. 退出")
    print("=" * 60)


def run_distillation_test(mode='last', parallel_workers=1):
    """运行数据蒸馏测试"""
    worker_text = f"{parallel_workers}线程并行" if parallel_workers > 1 else "单线程"
    print(f"\n🚀 启动数据蒸馏Agent（测试模式: {mode}，{worker_text}）...")
    
    agent = DistillationAgent(
        test_mode=mode,
        test_size=10,
        batch_size=5,
        batch_delay=1,
        parallel_workers=parallel_workers
    )
    
    output_name = f'distillation_test_{mode}{"_parallel" if parallel_workers > 1 else ""}'
    result = agent.run(output_name=output_name)
    
    print(f"\n✓ 蒸馏完成!")
    print(f"  成功: {result['success_count']} 条")
    print(f"  失败: {result['failed_count']} 条")
    print(f"  输出: {result['output_file']}")


def run_distillation_parallel():
    """运行并行数据蒸馏"""
    print("\n" + "=" * 60)
    print("并行数据蒸馏配置")
    print("=" * 60)
    
    try:
        # 选择测试模式
        print("\n测试模式:")
        print("1. 最后N条")
        print("2. 前N条")
        print("3. 随机N条")
        mode_choice = input("选择模式 (1-3): ").strip()
        
        mode_map = {'1': 'last', '2': 'first', '3': 'random'}
        mode = mode_map.get(mode_choice, 'last')
        
        # 输入数据量
        test_size = int(input("请输入测试数据量 (默认10): ").strip() or "10")
        
        # 输入并行线程数
        parallel_workers = int(input("请输入并行线程数 (1-10，推荐3-5): ").strip() or "3")
        parallel_workers = max(1, min(10, parallel_workers))  # 限制在1-10之间
        
        print(f"\n配置:")
        print(f"  模式: {mode}")
        print(f"  数据量: {test_size}")
        print(f"  并行线程: {parallel_workers}")
        
        confirm = input("\n确认开始？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        # 创建Agent并运行
        agent = DistillationAgent(
            test_mode=mode,
            test_size=test_size,
            batch_size=5,
            batch_delay=0.5 if parallel_workers > 1 else 1,
            parallel_workers=parallel_workers
        )
        
        output_name = f'distillation_{mode}_{test_size}samples_parallel{parallel_workers}'
        result = agent.run(output_name=output_name)
        
        print(f"\n✓ 蒸馏完成!")
        print(f"  成功: {result['success_count']} 条")
        print(f"  失败: {result['failed_count']} 条")
        print(f"  耗时: {result['elapsed_time']:.2f} 秒")
        print(f"  输出: {result['output_file']}")
        
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


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


def run_evaluation():
    """运行评估任务"""
    print("\n" + "=" * 60)
    print("评估预测结果")
    print("=" * 60)
    
    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    
    # 列出output目录中的所有JSON文件（优先显示带_with_id的文件）
    json_files = list(output_dir.glob('*_with_id.json'))
    if not json_files:
        # 如果没有带_with_id的文件，则显示所有JSON文件
        json_files = list(output_dir.glob('*.json'))
    
    if not json_files:
        print("\n✗ output目录中没有找到JSON文件")
        print("提示: 请先运行数据蒸馏任务生成预测结果（建议使用带_with_id的文件进行评估）")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件:")
    for i, file in enumerate(json_files, 1):
        # 标注哪些是带ID的文件
        marker = " ✓ (推荐)" if "_with_id" in file.name else ""
        print(f"  {i}. {file.name}{marker}")
    
    try:
        # 选择要评估的文件
        file_choice = input("\n请选择要评估的文件编号: ").strip()
        file_idx = int(file_choice) - 1
        
        if file_idx < 0 or file_idx >= len(json_files):
            print("✗ 无效的文件编号")
            return
        
        prediction_file = json_files[file_idx]
        print(f"\n选择的文件: {prediction_file.name}")
        
        # 检查是否使用带ID的文件
        if "_with_id" not in prediction_file.name:
            print("⚠ 警告: 该文件不包含ID字段，评估可能不准确")
            confirm = input("是否继续？(y/n): ").strip().lower()
            if confirm != 'y':
                print("已取消")
                return
        
        # 使用默认的ground truth文件
        ground_truth_file = Path(__file__).parent / 'dataset' / 'FlakyLens_dataset_with_nonflaky_indented.csv'
        
        if not ground_truth_file.exists():
            print(f"✗ 未找到ground truth文件: {ground_truth_file}")
            return
        
        # 创建评估器（通过ID字段匹配）
        print("\n🚀 开始评估...")
        print("📌 评估方式: 通过ID字段匹配预测结果和真实标签")
        evaluator = Evaluator(
            prediction_file=prediction_file,
            ground_truth_file=ground_truth_file,
            label_column='label',
            id_column='id'
        )
        
        # 运行评估
        eval_output_dir = output_dir / 'evaluation'
        metrics = evaluator.run(
            output_dir=eval_output_dir,
            save_report=True,
            detailed=True
        )
        
        print(f"\n✓ 评估完成!")
        print(f"  报告已保存到: {eval_output_dir}")
        
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()


def run_dataset_split():
    """运行数据集划分任务"""
    print("\n" + "=" * 60)
    print("数据集划分 - 训练集/验证集/测试集")
    print("=" * 60)
    
    try:
        # 输入划分比例
        print("\n请输入划分比例（默认 7:2:1）")
        train_ratio = float(input("  训练集比例 (0-1, 默认0.7): ").strip() or "0.7")
        val_ratio = float(input("  验证集比例 (0-1, 默认0.2): ").strip() or "0.2")
        test_ratio = float(input("  测试集比例 (0-1, 默认0.1): ").strip() or "0.1")
        
        # 验证比例总和
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            print(f"✗ 比例总和必须为 1，当前为: {total}")
            return
        
        # 是否使用分层采样
        use_stratify = input("\n是否使用分层采样（基于label列）？(y/n, 默认y): ").strip().lower()
        stratify_column = 'label' if use_stratify != 'n' else None
        
        # 随机种子
        random_seed = int(input("随机种子 (默认42): ").strip() or "42")
        
        # 输出格式
        print("\n输出格式:")
        print("1. CSV")
        print("2. JSON")
        print("3. 两者都保存")
        format_choice = input("选择格式 (1-3, 默认1): ").strip() or "1"
        
        print(f"\n配置:")
        print(f"  训练集: {train_ratio*100:.1f}%")
        print(f"  验证集: {val_ratio*100:.1f}%")
        print(f"  测试集: {test_ratio*100:.1f}%")
        print(f"  分层采样: {'是 (基于label)' if stratify_column else '否'}")
        print(f"  随机种子: {random_seed}")
        
        confirm = input("\n确认开始划分？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        # 加载数据集
        print(f"\n📂 加载数据集: {DATASET_PATH}")
        df = load_csv(DATASET_PATH)
        
        # 划分数据集
        print("\n🔀 开始划分数据集...")
        splits = split_dataset(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_column=stratify_column,
            random_seed=random_seed,
            shuffle=True
        )
        
        # 保存划分后的数据集
        output_dir = OUTPUT_DIR / 'splits'
        
        if format_choice in ['1', '3']:
            print("\n💾 保存CSV格式...")
            csv_files = save_split_datasets(
                splits,
                output_dir=output_dir,
                base_name='flaky_dataset',
                format='csv'
            )
        
        if format_choice in ['2', '3']:
            print("\n💾 保存JSON格式...")
            json_files = save_split_datasets(
                splits,
                output_dir=output_dir,
                base_name='flaky_dataset',
                format='json'
            )
        
        print(f"\n✓ 数据集划分完成!")
        print(f"  输出目录: {output_dir}")
        
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 划分失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    while True:
        print_menu()
        choice = input("\n请选择操作 (1-9): ").strip()
        
        if choice == '1':
            run_distillation_test(mode='last', parallel_workers=1)
        elif choice == '2':
            run_distillation_test(mode='first', parallel_workers=1)
        elif choice == '3':
            run_distillation_test(mode='random', parallel_workers=1)
        elif choice == '4':
            run_distillation_parallel()
        elif choice == '5':
            run_data_explainer(sample_size=20)
        elif choice == '6':
            try:
                size = int(input("请输入样本数: ").strip())
                run_data_explainer(sample_size=size)
            except ValueError:
                print("✗ 无效的数字")
        elif choice == '7':
            run_evaluation()
        elif choice == '8':
            run_dataset_split()
        elif choice == '9':
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
