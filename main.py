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
from utils import (load_csv, split_dataset, save_split_datasets, 
                   create_project_wise_kfold_splits, save_kfold_datasets,
                   APISignatureMatcher, save_config, load_config, 
                   list_saved_configs, delete_config, display_config,
                   switch_provider, get_current_config, show_current_config,
                   list_providers, get_supported_models, show_all_models)
from config import DATASET_PATH, OUTPUT_DIR


def list_available_datasets():
    """列出可用的数据集文件"""
    from pathlib import Path
    
    dataset_dir = Path(__file__).parent / 'dataset'
    
    # 收集所有CSV文件
    datasets = []
    
    # 1. 主数据集
    main_dataset = dataset_dir / 'FlakyLens_dataset_with_nonflaky_indented.csv'
    if main_dataset.exists():
        datasets.append(('主数据集', main_dataset))
    
    # 2. K-fold划分
    kfold_dir = dataset_dir / 'kfold_splits'
    if kfold_dir.exists():
        for fold_file in sorted(kfold_dir.glob('*.csv')):
            fold_name = fold_file.stem.replace('_', ' ').title()
            datasets.append((f'K-Fold: {fold_name}', fold_file))
    
    # 3. 其他划分
    for csv_file in dataset_dir.glob('*.csv'):
        if csv_file != main_dataset:
            datasets.append((csv_file.stem, csv_file))
    
    return datasets


def select_dataset(prompt="请选择数据集", allow_none=False):
    """
    交互式选择数据集
    
    Args:
        prompt: 提示信息
        allow_none: 是否允许不选择（返回None）
        
    Returns:
        选中的数据集路径，或None
    """
    datasets = list_available_datasets()
    
    if not datasets:
        print("✗ 未找到可用的数据集文件")
        return None
    
    print(f"\n{prompt}:")
    if allow_none:
        print("  0. (不使用)")
    
    for i, (name, path) in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    
    try:
        choice = input(f"\n选择 ({0 if allow_none else 1}-{len(datasets)}): ").strip()
        if not choice:
            return None if allow_none else datasets[0][1]
        
        idx = int(choice)
        
        if idx == 0 and allow_none:
            return None
        
        if idx < 1 or idx > len(datasets):
            print("✗ 无效的选择")
            return None
        
        return datasets[idx - 1][1]
    
    except ValueError:
        print("✗ 输入无效")
        return None


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print("Flaky Test分析系统 - 快速启动")
    print("=" * 60)
    print("1. 数据蒸馏")
    print("2. 数据讲解")
    print("3. 评估预测结果")
    print("4. 数据集划分")
    print("5. 配置管理")
    print("6. 模型设置")
    print("7. 退出")
    print("=" * 60)


def run_distillation():
    """运行数据蒸馏（支持自定义训练集/测试集和API匹配）"""
    print("\n" + "=" * 60)
    print("数据蒸馏配置")
    print("=" * 60)
    
    # 检查是否有保存的配置
    saved_configs = list_saved_configs()
    use_saved_config = False
    config_to_save = {}
    
    if saved_configs:
        print("\n💾 发现已保存的配置:")
        for i, config_name in enumerate(saved_configs, 1):
            print(f"  {i}. {config_name}")
        print(f"  0. 新建配置")
        
        choice = input("\n选择配置 (0-{}，默认0): ".format(len(saved_configs))).strip() or "0"
        
        if choice != '0':
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(saved_configs):
                    config_name = saved_configs[idx]
                    config = load_config(config_name)
                    
                    if config and config.get('task_type') == 'distillation':
                        display_config(config)
                        confirm = input("\n使用此配置？(y/n): ").strip().lower()
                        if confirm == 'y':
                            use_saved_config = True
                            # 从配置中提取参数
                            test_dataset = config['test_dataset']
                            train_dataset = config.get('train_dataset')
                            use_api_matching = config.get('use_api_matching', False)
                            top_k_shots = config.get('top_k_shots', 3)
                            mode = config.get('mode', 'random')
                            test_size = config.get('test_size', 10)
                            parallel_workers = config.get('parallel_workers', 5)
                            batch_size = config.get('batch_size', 5)
                    else:
                        print("✗ 配置类型不匹配")
            except (ValueError, IndexError):
                print("✗ 无效选择")
    
    if not use_saved_config:
        # 原有的配置流程
        try:
            # Step 1: 选择测试集
            print("\n【Step 1/5】选择测试集")
            test_dataset = select_dataset("请选择测试集")
            if not test_dataset:
                print("已取消")
                return
            print(f"✓ 测试集: {test_dataset.name}")
            
            # Step 2: 选择训练集（可选，用于API匹配）
            print("\n【Step 2/5】选择训练集（用于API匹配，可选）")
            print("提示: 如果选择训练集，将使用API签名匹配来检索few-shot examples")
            use_api_matching = input("是否使用API匹配？(y/n, 默认n): ").strip().lower() == 'y'
            
            train_dataset = None
            top_k_shots = 3
            
            if use_api_matching:
                train_dataset = select_dataset("请选择训练集（用作知识库）", allow_none=True)
                if train_dataset:
                    print(f"✓ 训练集: {train_dataset.name}")
                    
                    # 设置few-shot数量
                    top_k_shots = int(input("\n请输入few-shot样本数 (默认3): ").strip() or "3")
                    top_k_shots = max(1, min(10, top_k_shots))
                else:
                    print("✓ 跳过API匹配")
                    use_api_matching = False
            
            # Step 3: 选择测试模式
            print("\n【Step 3/5】测试模式")
            print("1. 最后N条")
            print("2. 前N条")
            print("3. 随机N条")
            print("4. 全部数据")
            mode_choice = input("选择模式 (1-4, 默认3): ").strip() or "3"
            
            mode_map = {'1': 'last', '2': 'first', '3': 'random', '4': 'all'}
            mode = mode_map.get(mode_choice, 'last')
            
            # 输入数据量（如果不是全部）
            if mode != 'all':
                test_size = int(input("请输入数据量 (默认10): ").strip() or "10")
            else:
                test_size = None
                print("将处理全部数据")
            
            # Step 4: 并行配置
            print("\n【Step 4/5】并行配置")
            parallel_workers = int(input("请输入并行线程数 (1-10，默认5): ").strip() or "5")
            parallel_workers = max(1, min(10, parallel_workers))
            
            batch_size = int(input("请输入批次大小 (默认5): ").strip() or "5")
            
            # 保存配置以备后用
            config_to_save = {
                'task_type': 'distillation',
                'test_dataset': test_dataset,
                'train_dataset': train_dataset,
                'use_api_matching': use_api_matching,
                'top_k_shots': top_k_shots,
                'mode': mode,
                'test_size': test_size,
                'parallel_workers': parallel_workers,
                'batch_size': batch_size
            }
            
        except ValueError as e:
            print(f"✗ 输入错误: {e}")
            return
        except Exception as e:
            print(f"✗ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            return
    
    try:
        # Step 5: 确认配置
        print("\n【Step 5/5】配置确认")
        print("=" * 60)
        print(f"测试集: {test_dataset.name}")
        if use_api_matching and train_dataset:
            print(f"训练集: {train_dataset.name}")
            print(f"API匹配: 开启 (Top-{top_k_shots} few-shots)")
        else:
            print("API匹配: 关闭")
        print(f"测试模式: {mode}")
        print(f"数据量: {test_size if test_size else '全部'}")
        print(f"并行线程: {parallel_workers}")
        print(f"批次大小: {batch_size}")
        print("=" * 60)
        
        confirm = input("\n确认开始？(y/n): ").strip().lower() or "y"
        if confirm != 'y':
            print("已取消")
            return
        
        # 如果是新配置，询问是否保存
        if not use_saved_config and config_to_save:
            save_choice = input("\n💾 是否保存此配置供下次使用？(y/n): ").strip().lower()
            if save_choice == 'y':
                config_name = input("请输入配置名称: ").strip()
                if config_name:
                    save_config(config_to_save, config_name)
        
        # 加载训练数据并创建API匹配器（如果需要）
        api_matcher = None
        train_data = None
        
        if use_api_matching and train_dataset:
            print("\n正在加载训练集并构建API索引...")
            train_data = load_csv(train_dataset)
            api_matcher = APISignatureMatcher(train_data, code_column='full_code')
            
            # 显示统计信息
            stats = api_matcher.get_statistics()
            print(f"✓ API索引构建完成:")
            print(f"  - 训练样本数: {stats['total_train_samples']}")
            print(f"  - 唯一API数: {stats['total_unique_apis']}")
            print(f"  - 平均API数/样本: {stats['avg_apis_per_sample']:.1f}")
            print(f"  - 最常见API: {', '.join([api for api, _ in stats['most_common_apis'][:5]])}")
        
        # 创建Agent并运行
        print("\n🚀 开始数据蒸馏...")
        
        agent = DistillationAgent(
            dataset_path=str(test_dataset),
            test_mode=mode,
            test_size=test_size,
            batch_size=batch_size,
            batch_delay=0.5 if parallel_workers > 1 else 1,
            parallel_workers=parallel_workers,
            api_matcher=api_matcher,
            top_k_shots=top_k_shots if use_api_matching else 0
        )
        
        # 构建输出文件名
        output_name_parts = [
            'distillation',
            test_dataset.stem,
            mode,
            f'{test_size if test_size else "all"}samples'
        ]
        if use_api_matching:
            output_name_parts.append(f'api_top{top_k_shots}')
        output_name_parts.append(f'p{parallel_workers}')
        
        output_name = '_'.join(output_name_parts)
        result = agent.run(output_name=output_name)
        
        print(f"\n✓ 蒸馏完成!")
        print(f"  成功: {result['success_count']} 条")
        print(f"  失败: {result['failed_count']} 条")
        print(f"  耗时: {result.get('elapsed_time', 0):.2f} 秒")
        print(f"  输出: {result['output_file']}")
        
        if use_api_matching:
            print(f"\n📊 API匹配统计:")
            print(f"  - 使用训练集: {train_dataset.name}")
            print(f"  - Few-shot数量: {top_k_shots}")
            print(f"  - 知识库大小: {len(train_data)}")
        
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def run_data_explainer():
    """运行数据讲解（统一配置）"""
    print("\n" + "=" * 60)
    print("数据讲解配置")
    print("=" * 60)
    
    try:
        # 输入样本数
        sample_size = int(input("\n请输入要分析的样本数 (默认20): ").strip() or "20")
        
        # 随机种子
        random_seed = int(input("请输入随机种子 (默认42): ").strip() or "42")
        
        print(f"\n配置:")
        print(f"  样本数: {sample_size}")
        print(f"  随机种子: {random_seed}")
        
        confirm = input("\n确认开始？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        print(f"\n🚀 启动数据讲解Agent...")
        
        agent = DataExplainerAgent(
            sample_size=sample_size,
            random_seed=random_seed
        )
        
        result = agent.run(output_name='dataset_analysis')
        
        if result['success']:
            print(f"\n✓ 分析完成!")
            print(f"  JSON: {result['json_file']}")
            print(f"  文本: {result['txt_file']}")
    
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


def run_evaluation():
    """运行评估任务"""
    print("\n" + "=" * 60)
    print("评估预测结果")
    print("=" * 60)
    
    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    
    # 列出output目录中的所有JSON文件（优先显示带_external的文件）
    json_files = list(output_dir.glob('*_external.json'))
    if not json_files:
        # 如果没有带_external的文件，则显示所有JSON文件
        json_files = list(output_dir.glob('*.json'))
    
    if not json_files:
        print("\n✗ output目录中没有找到JSON文件")
        print("提示: 请先运行数据蒸馏任务生成预测结果（建议使用带_external的文件进行评估）")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件:")
    for i, file in enumerate(json_files, 1):
        # 标注哪些是带额外信息的文件
        marker = " ✓ (推荐)" if "_external" in file.name else ""
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
        
        # 检查是否使用带额外信息的文件
        if "_external" not in prediction_file.name:
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
    """运行数据集划分（统一配置）"""
    print("\n" + "=" * 60)
    print("数据集划分")
    print("=" * 60)
    
    try:
        # 选择划分类型
        print("\n划分类型:")
        print("1. 训练集/验证集/测试集划分")
        print("2. K折交叉验证（项目级独立）")
        split_type = input("选择类型 (1-2, 默认1): ").strip() or "1"
        
        if split_type == '1':
            # 训练集/验证集/测试集划分
            print("\n" + "=" * 60)
            print("训练集/验证集/测试集划分")
            print("=" * 60)
            
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
            
            print(f"\n配置:")
            print(f"  训练集: {train_ratio*100:.1f}%")
            print(f"  验证集: {val_ratio*100:.1f}%")
            print(f"  测试集: {test_ratio*100:.1f}%")
            print(f"  分层采样: {'是 (基于label)' if stratify_column else '否'}")
            print(f"  随机种子: {random_seed}")
            
        elif split_type == '2':
            # K折交叉验证
            print("\n" + "=" * 60)
            print("K折交叉验证（项目级独立）")
            print("=" * 60)
            print("特点:")
            print("  ✓ 同一项目的测试不会同时出现在训练和测试集中")
            print("  ✓ 类别平衡约束（每个测试集至少包含每种类别的最小样本数）")
            print("=" * 60)
            
            # 输入参数
            n_folds = int(input("\n折数 (默认4): ").strip() or "4")
            min_samples = int(input("每个测试集中每类的最小样本数 (默认4): ").strip() or "4")
            random_seed = int(input("随机种子 (默认42): ").strip() or "42")
            
            print(f"\n配置:")
            print(f"  折数: {n_folds}")
            print(f"  每类最小样本数: {min_samples}")
            print(f"  随机种子: {random_seed}")
        
        else:
            print("✗ 无效的选择")
            return
        
        # 输出格式（两种类型都需要）
        print("\n输出格式:")
        print("1. CSV")
        print("2. JSON")
        print("3. 两者都保存")
        format_choice = input("选择格式 (1-3, 默认1): ").strip() or "1"
        
        confirm = input("\n确认开始划分？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        # 加载数据集
        print(f"\n📂 加载数据集: {DATASET_PATH}")
        df = load_csv(DATASET_PATH)
        
        if split_type == '1':
            # 执行训练集/验证集/测试集划分
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
                save_split_datasets(splits, output_dir=output_dir, base_name='flaky_dataset', format='csv')
            
            if format_choice in ['2', '3']:
                print("\n💾 保存JSON格式...")
                save_split_datasets(splits, output_dir=output_dir, base_name='flaky_dataset', format='json')
            
            print(f"\n✓ 数据集划分完成!")
            print(f"  输出目录: {output_dir}")
        
        elif split_type == '2':
            # 执行K折交叉验证划分
            folds = create_project_wise_kfold_splits(
                df,
                project_column='project',
                category_column='category',
                n_folds=n_folds,
                min_samples_per_category=min_samples,
                random_seed=random_seed
            )
            
            # 保存K折数据集
            output_dir = OUTPUT_DIR / 'kfold_splits'
            
            if format_choice in ['1', '3']:
                print("\n💾 保存CSV格式...")
                save_kfold_datasets(folds, output_dir=output_dir, base_name='fold', format='csv')
            
            if format_choice in ['2', '3']:
                print("\n💾 保存JSON格式...")
                save_kfold_datasets(folds, output_dir=output_dir, base_name='fold', format='json')
            
            print(f"\n✓ K折交叉验证数据集划分完成!")
            print(f"  输出目录: {output_dir}")
            print(f"  共生成 {n_folds} 折数据")
            print(f"\n📝 每折包含:")
            print(f"  - fold_X_train.csv/json: 训练集")
            print(f"  - fold_X_test.csv/json: 测试集")
            print(f"  - fold_X_projects.txt: 项目列表")
        
    except ValueError as e:
        print(f"✗ 输入错误: {e}")
    except Exception as e:
        print(f"✗ 划分失败: {e}")
        import traceback
        traceback.print_exc()


        traceback.print_exc()


def run_config_manager():
    """配置管理"""
    print("\n" + "=" * 60)
    print("配置管理")
    print("=" * 60)
    
    while True:
        saved_configs = list_saved_configs()
        
        if not saved_configs:
            print("\n📝 当前没有保存的配置")
            print("\n提示: 在数据蒸馏或其他任务完成配置后，可以选择保存配置供下次使用")
            return
        
        print(f"\n💾 已保存的配置 (共{len(saved_configs)}个):")
        for i, config_name in enumerate(saved_configs, 1):
            print(f"  {i}. {config_name}")
        
        print("\n操作:")
        print("  v. 查看配置")
        print("  d. 删除配置")
        print("  0. 返回主菜单")
        
        choice = input("\n选择操作: ").strip().lower()
        
        if choice == '0':
            break
        elif choice == 'v':
            idx_input = input("请输入要查看的配置编号: ").strip()
            try:
                idx = int(idx_input) - 1
                if 0 <= idx < len(saved_configs):
                    config_name = saved_configs[idx]
                    config = load_config(config_name)
                    if config:
                        print(f"\n📄 配置: {config_name}")
                        display_config(config)
                else:
                    print("✗ 无效的编号")
            except ValueError:
                print("✗ 请输入数字")
        
        elif choice == 'd':
            idx_input = input("请输入要删除的配置编号: ").strip()
            try:
                idx = int(idx_input) - 1
                if 0 <= idx < len(saved_configs):
                    config_name = saved_configs[idx]
                    confirm = input(f"确认删除配置 '{config_name}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        delete_config(config_name)
                else:
                    print("✗ 无效的编号")
            except ValueError:
                print("✗ 请输入数字")
        
        else:
            print("✗ 无效的操作")


def run_model_settings():
    """模型设置"""
    print("\n" + "=" * 60)
    print("模型设置")
    print("=" * 60)
    
    # 显示当前配置
    provider, model, base_url, api_key_status, has_key = get_current_config()
    print(f"\n📌 当前配置:")
    print(f"   提供商: {provider}")
    print(f"   模型: {model}")
    print(f"   API URL: {base_url}")
    print(f"   API密钥: {api_key_status}")
    
    print("\n" + "-" * 60)
    print("可用操作:")
    print("  1. 切换提供商")
    print("  2. 查看当前提供商支持的模型")
    print("  3. 查看所有支持的模型")
    print("  0. 返回主菜单")
    print("-" * 60)
    
    choice = input("\n请选择操作: ").strip()
    
    if choice == '1':
        # 切换提供商
        providers = list_providers()
        print("\n📋 可用提供商:")
        for i, p in enumerate(providers, 1):
            print(f"  {i}. {p.upper()}")
        
        try:
            provider_idx = int(input(f"\n请选择提供商 (1-{len(providers)}): ").strip())
            
            if 1 <= provider_idx <= len(providers):
                new_provider = providers[provider_idx - 1]
                if switch_provider(new_provider):
                    print("⚠️  请重启程序以使更改生效")
            else:
                print("✗ 无效的选择")
        except ValueError:
            print("✗ 请输入数字")
    
    elif choice == '2':
        # 查看当前提供商支持的模型
        models = get_supported_models()
        print(f"\n📋 {provider.upper()} 支持的模型:")
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        print("\n💡 提示: 可以在创建Agent时通过 model 参数使用指定模型")
        print(f"   示例: DistillationAgent(model='{models[0] if models else 'model-name'}')")
    
    elif choice == '3':
        # 查看所有支持的模型
        show_all_models()
    
    elif choice == '0':
        return
    else:
        print("✗ 无效的操作")


def main():
    """主函数"""
    while True:
        print_menu()
        choice = input("\n请选择操作 (1-7): ").strip()
        
        if choice == '1':
            run_distillation()
        elif choice == '2':
            run_data_explainer()
        elif choice == '3':
            run_evaluation()
        elif choice == '4':
            run_dataset_split()
        elif choice == '5':
            run_config_manager()
        elif choice == '6':
            run_model_settings()
        elif choice == '7':
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
