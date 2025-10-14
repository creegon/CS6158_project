"""
评估模块使用示例
演示如何使用Evaluator评估预测结果
"""
from evaluation import Evaluator


def example_basic_evaluation():
    """基本评估示例"""
    print("\n" + "=" * 70)
    print("示例: 基本评估")
    print("=" * 70)
    
    # 创建评估器
    evaluator = Evaluator(
        prediction_file='output/distillation_test_random.json',  # Alpaca格式的预测结果
        ground_truth_file='dataset/FlakyLens_dataset_with_nonflaky_indented.csv',  # 真实标签
        label_column='label'  # CSV中的标签列名
    )
    
    # 运行评估
    metrics = evaluator.run(
        output_dir='output/evaluation',
        save_report=True,
        detailed=True
    )
    
    return metrics


def example_custom_evaluation():
    """自定义评估示例"""
    print("\n" + "=" * 70)
    print("示例: 自定义评估流程")
    print("=" * 70)
    
    # 创建评估器
    evaluator = Evaluator(
        prediction_file='output/your_predictions.json',
        ground_truth_file='dataset/your_labels.csv',
        label_column='label',
        id_column='id'  # 如果CSV有ID列
    )
    
    # 分步骤执行
    print("\n步骤1: 加载数据")
    evaluator.load_data()
    
    print("\n步骤2: 执行评估")
    evaluator.evaluate()
    
    print("\n步骤3: 打印报告")
    evaluator.print_report(detailed=False)  # 只打印摘要
    
    print("\n步骤4: 保存报告")
    evaluator.save_report('output/evaluation', 'custom_report')
    
    return evaluator.metrics


def example_multiple_files():
    """评估多个预测文件"""
    print("\n" + "=" * 70)
    print("示例: 评估多个模型")
    print("=" * 70)
    
    prediction_files = [
        'output/model_v1_predictions.json',
        'output/model_v2_predictions.json',
        'output/model_v3_predictions.json'
    ]
    
    ground_truth_file = 'dataset/FlakyLens_dataset_with_nonflaky_indented.csv'
    
    results = {}
    
    for pred_file in prediction_files:
        model_name = pred_file.split('/')[-1].replace('_predictions.json', '')
        print(f"\n评估模型: {model_name}")
        print("-" * 70)
        
        evaluator = Evaluator(
            prediction_file=pred_file,
            ground_truth_file=ground_truth_file,
            label_column='label'
        )
        
        metrics = evaluator.run(
            output_dir=f'output/evaluation/{model_name}',
            save_report=True,
            detailed=False
        )
        
        results[model_name] = metrics
    
    # 比较结果
    print("\n" + "=" * 70)
    print("模型比较")
    print("=" * 70)
    print(f"{'模型':<20} {'总体准确率':>12} {'Flaky F1':>12} {'类别准确率':>12}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        overall_acc = metrics['overall_accuracy'] * 100
        flaky_f1 = metrics['flaky_detection']['f1'] * 100
        category_acc = metrics['category_classification']['accuracy'] * 100
        
        print(f"{model_name:<20} {overall_acc:>11.2f}% {flaky_f1:>11.2f}% {category_acc:>11.2f}%")
    
    return results


if __name__ == '__main__':
    # 运行基本评估示例
    example_basic_evaluation()
    
    # 其他示例（取消注释使用）
    # example_custom_evaluation()
    # example_multiple_files()
