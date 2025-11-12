"""
评估器主类
整合所有评估功能
"""
from datetime import datetime
from pathlib import Path
from typing import Union, Optional
from evaluation.data_loader import (
    load_predictions_from_alpaca,
    load_ground_truth_from_csv,
    align_predictions_and_labels
)
from utils.evaluation_utils import calculate_metrics
from evaluation.report_generator import EvaluationReport


class Evaluator:
    """
    评估器
    用于评估Flaky Test分类模型的性能
    """
    
    def __init__(self,
                 prediction_file: Union[str, Path],
                 ground_truth_file: Union[str, Path],
                 label_column: str = 'label',
                 id_column: str = 'id'):
        """
        初始化评估器
        
        Args:
            prediction_file: Alpaca格式的预测结果JSON文件（应包含id字段）
            ground_truth_file: 真实标签CSV文件
            label_column: CSV中的标签列名
            id_column: CSV中的ID列名（默认为'id'）
        """
        self.prediction_file = Path(prediction_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.label_column = label_column
        self.id_column = id_column
        
        self.predictions = None
        self.ground_truths = None
        self.raw_predictions = None  # 保存原始预测数据（用于上下文分析）
        self.metrics = None
        self.report = None
    
    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 70)
        print("加载数据")
        print("=" * 70)
        
        print(f"\n📂 加载预测结果: {self.prediction_file.name}")
        
        # 加载原始数据（用于上下文分析）
        import json
        with open(self.prediction_file, 'r', encoding='utf-8') as f:
            self.raw_predictions = json.load(f)
        
        # 解析预测结果
        self.predictions = load_predictions_from_alpaca(self.prediction_file)
        print(f"   ✓ 加载了 {len(self.predictions)} 条预测结果")
        
        print(f"\n📂 加载真实标签: {self.ground_truth_file.name}")
        self.ground_truths = load_ground_truth_from_csv(
            self.ground_truth_file,
            label_column=self.label_column,
            id_column=self.id_column
        )
        print(f"   ✓ 加载了 {len(self.ground_truths)} 条真实标签")
        
        # 对齐数据
        print(f"\n🔄 对齐数据...")
        self.predictions, self.ground_truths = align_predictions_and_labels(
            self.predictions,
            self.ground_truths
        )
    
    def evaluate(self):
        """执行评估"""
        if self.predictions is None or self.ground_truths is None:
            self.load_data()
        
        print("\n" + "=" * 70)
        print("计算评估指标")
        print("=" * 70)
        
        self.metrics = calculate_metrics(self.predictions, self.ground_truths)
        
        # 创建报告（传递原始预测数据以支持上下文分析）
        self.report = EvaluationReport(self.metrics, self.raw_predictions)
        
        print("✓ 评估指标计算完成")
    
    def print_report(self, detailed: bool = True):
        """
        打印评估报告
        
        Args:
            detailed: 是否打印详细报告
        """
        if self.report is None:
            self.evaluate()
        
        if detailed:
            self.report.print_detailed()
        else:
            self.report.print_summary()
    
    def _generate_report_name_from_prediction_file(self, output_dir: Path) -> str:
        """
        从预测文件名生成报告名称
        
        格式: {provider_abbr}_{sample_size}_{features}
        例如: ds_100_api_context, sf_50_api
        
        如果文件已存在，自动添加数字后缀 (2, 3, 4...)
        """
        from config import CURRENT_PROVIDER
        
        # 提供商缩写
        provider_map = {
            'deepseek': 'ds',
            'siliconflow': 'sf'
        }
        provider_abbr = provider_map.get(CURRENT_PROVIDER.lower(), 'eval')
        
        # 从预测文件名提取信息
        pred_name = self.prediction_file.stem  # 去掉扩展名
        pred_name = pred_name.replace('_external', '')  # 去掉 _external 后缀
        
        # 提取样本数量 (多种模式)
        # 1. XX_samples 或 XXsamples
        # 2. distillation_XX_api 中的 XX
        import re
        sample_match = re.search(r'(\d+)_?samples?', pred_name, re.IGNORECASE)
        if not sample_match:
            # 尝试查找 distillation_数字_ 模式
            sample_match = re.search(r'distillation_(\d+)', pred_name, re.IGNORECASE)
        sample_size = sample_match.group(1) if sample_match else None
        
        # 识别特征标记
        features = []
        feature_keywords = ['api', 'context', 'feature', 'fre', 'external']
        for keyword in feature_keywords:
            if keyword in pred_name.lower():
                features.append(keyword)
        
        # 构建基础名称
        if sample_size:
            base_name = f"{provider_abbr}_{sample_size}"
        else:
            base_name = f"{provider_abbr}"
        
        # 添加特征标记
        if features:
            base_name = f"{base_name}_{'_'.join(features)}"
        
        # 检查重复，添加数字后缀
        final_name = base_name
        counter = 2
        while (output_dir / f"{final_name}.json").exists() or \
              (output_dir / f"{final_name}.txt").exists():
            final_name = f"{base_name}{counter}"
            counter += 1
        
        return final_name
    
    def save_report(self, 
                   output_dir: Union[str, Path],
                   report_name: str = None,
                   add_timestamp: bool = False):
        """
        保存评估报告
        
        Args:
            output_dir: 输出目录
            report_name: 报告文件名（不含扩展名），如果为None则自动生成
            add_timestamp: 是否在文件名中添加时间戳（默认False）
        """
        if self.report is None:
            self.evaluate()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动生成报告名称
        if report_name is None:
            report_name = self._generate_report_name_from_prediction_file(output_dir)
        
        # 添加时间戳（通常不需要，因为已经有重复检测）
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name_with_timestamp = f"{report_name}_{timestamp}"
        else:
            report_name_with_timestamp = report_name
        
        # 保存JSON格式
        json_file = output_dir / f"{report_name_with_timestamp}.json"
        self.report.save_to_json(json_file)
        
        # 保存文本格式
        txt_file = output_dir / f"{report_name_with_timestamp}.txt"
        self.report.save_to_text(txt_file)
    
    def run(self, 
            output_dir: Optional[Union[str, Path]] = None,
            save_report: bool = True,
            detailed: bool = True):
        """
        运行完整的评估流程
        
        Args:
            output_dir: 输出目录
            save_report: 是否保存报告
            detailed: 是否打印详细报告
        """
        # 加载数据
        self.load_data()
        
        # 评估
        self.evaluate()
        
        # 打印报告
        self.print_report(detailed=detailed)
        
        # 保存报告
        if save_report and output_dir:
            self.save_report(output_dir)
        
        return self.metrics
