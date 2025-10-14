"""
DataExplainerAgent - 数据讲解Agent
用于分析数据集并生成详细的解读报告
"""
from pathlib import Path
from typing import Optional, Union, Dict
import pandas as pd

from agents.base_agent import BaseAgent
from utils import (
    load_csv,
    sample_data,
    print_data_info,
    get_data_statistics,
    load_prompt,
    format_prompt,
    save_json
)
from config import DATASET_PATH, OUTPUT_DIR


class DataExplainerAgent(BaseAgent):
    """
    数据讲解Agent
    随机抽取数据样本，生成对数据集的详细解读
    """
    
    def __init__(self,
                 dataset_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 sample_size: int = 20,
                 random_seed: Optional[int] = 42,
                 code_column: str = 'code',
                 label_column: str = 'label',
                 **kwargs):
        """
        初始化DataExplainerAgent
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            sample_size: 采样数量
            random_seed: 随机种子
            code_column: 代码列名
            label_column: 标签列名
            **kwargs: 传递给BaseAgent的其他参数
        """
        super().__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.code_column = code_column
        self.label_column = label_column
        
        # 加载prompt模板
        self.system_prompt = load_prompt('explainer_system')
        self.user_template = load_prompt('explainer_user')
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个专业的数据分析专家，擅长分析软件测试数据集并提供深入见解。"
    
    def format_samples(self, df_sample: pd.DataFrame) -> str:
        """
        格式化样本数据
        
        Args:
            df_sample: 采样的DataFrame
            
        Returns:
            格式化的样本字符串
        """
        samples_text = []
        
        for idx, row in df_sample.iterrows():
            code = row.get(self.code_column, row.get('full_code', ''))
            label = row.get(self.label_column, 'Unknown')
            
            sample_text = f"""
样本 {idx + 1}:
标签: {label}
代码:
```
{code}
```
"""
            samples_text.append(sample_text.strip())
        
        return "\n\n" + "-" * 60 + "\n\n".join(samples_text)
    
    def generate_user_prompt(self, df_sample: pd.DataFrame) -> str:
        """
        生成用户提示词
        
        Args:
            df_sample: 采样的DataFrame
            
        Returns:
            格式化的用户提示词
        """
        samples = self.format_samples(df_sample)
        return format_prompt(
            self.user_template,
            sample_size=len(df_sample),
            samples=samples
        )
    
    def run(self,
            dataset_path: Optional[Union[str, Path]] = None,
            output_name: str = 'dataset_explanation') -> Dict:
        """
        执行数据讲解任务
        
        Args:
            dataset_path: 数据集路径（可选，覆盖初始化参数）
            output_name: 输出文件名（不含扩展名）
            
        Returns:
            包含结果的字典
        """
        # 加载数据
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        print("\n" + "=" * 60)
        print("开始数据讲解任务")
        print("=" * 60)
        
        df = load_csv(self.dataset_path)
        
        # 打印数据集基本信息
        print_data_info(df)
        
        # 随机采样
        print(f"\n📊 随机采样 {self.sample_size} 条数据进行分析...")
        df_sample = sample_data(df, mode='random', n=self.sample_size, random_seed=self.random_seed)
        
        # 生成prompt
        print("\n🤔 生成分析prompt...")
        user_prompt = self.generate_user_prompt(df_sample)
        
        # 调用API获取解读
        print("\n🚀 调用AI进行数据分析...")
        self.reset_stats()
        explanation = self.call_api(user_prompt)
        
        if explanation is None:
            print("\n✗ 数据分析失败")
            return {
                "success": False,
                "error": "API调用失败"
            }
        
        # 保存结果
        result = {
            "dataset_path": str(self.dataset_path),
            "sample_size": self.sample_size,
            "total_records": len(df),
            "statistics": get_data_statistics(df),
            "explanation": explanation
        }
        
        # 保存JSON格式
        json_file = self.output_dir / f"{output_name}.json"
        save_json(result, json_file)
        
        # 保存纯文本格式
        txt_file = self.output_dir / f"{output_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("数据集分析报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"数据集路径: {self.dataset_path}\n")
            f.write(f"总记录数: {len(df)}\n")
            f.write(f"分析样本数: {self.sample_size}\n")
            f.write(f"随机种子: {self.random_seed}\n\n")
            f.write("=" * 80 + "\n")
            f.write("详细分析\n")
            f.write("=" * 80 + "\n\n")
            f.write(explanation)
            f.write("\n\n" + "=" * 80 + "\n")
        
        print(f"\n✓ 分析报告已保存到:")
        print(f"  - JSON格式: {json_file}")
        print(f"  - 文本格式: {txt_file}")
        
        # 打印API统计
        self.print_stats()
        
        # 打印解读结果（前500字符）
        print("\n" + "=" * 60)
        print("分析结果预览")
        print("=" * 60)
        preview = explanation[:500] + "..." if len(explanation) > 500 else explanation
        print(preview)
        print("=" * 60)
        
        return {
            "success": True,
            "explanation": explanation,
            "json_file": str(json_file),
            "txt_file": str(txt_file),
            "statistics": get_data_statistics(df),
            "api_stats": self.get_stats()
        }
