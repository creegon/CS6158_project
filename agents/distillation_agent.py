"""
DistillationAgent - 数据蒸馏Agent
用于生成包含推理过程的训练数据集
"""
import time
from pathlib import Path
from typing import Optional, Union, List, Dict
from tqdm import tqdm
import pandas as pd

from agents.base_agent import BaseAgent
from utils import (
    load_csv,
    sample_data,
    convert_to_alpaca_format,
    save_json,
    load_prompt,
    format_prompt
)
from config import (
    DATASET_PATH,
    OUTPUT_DIR,
    API_BATCH_SIZE,
    API_BATCH_DELAY,
    CHECKPOINT_INTERVAL
)


class DistillationAgent(BaseAgent):
    """
    数据蒸馏Agent
    读取原始测试数据，生成包含推理过程的Alpaca格式数据集
    """
    
    def __init__(self,
                 dataset_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 test_mode: str = 'all',
                 test_size: int = 10,
                 random_seed: Optional[int] = None,
                 code_column: str = 'code',
                 batch_size: Optional[int] = None,
                 batch_delay: Optional[float] = None,
                 checkpoint_interval: Optional[int] = None,
                 **kwargs):
        """
        初始化DistillationAgent
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            test_mode: 测试模式 ['all', 'first', 'last', 'random']
            test_size: 测试时使用的数据量
            random_seed: 随机种子
            code_column: 代码列名
            batch_size: 批次大小
            batch_delay: 批次延迟（秒）
            checkpoint_interval: 检查点保存间隔
            **kwargs: 传递给BaseAgent的其他参数
        """
        super().__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_mode = test_mode
        self.test_size = test_size
        self.random_seed = random_seed
        self.code_column = code_column
        
        self.batch_size = batch_size or API_BATCH_SIZE
        self.batch_delay = batch_delay if batch_delay is not None else API_BATCH_DELAY
        self.checkpoint_interval = checkpoint_interval or CHECKPOINT_INTERVAL
        
        # 加载prompt模板
        self.system_prompt = load_prompt('distillation_system')
        self.user_template = load_prompt('distillation_user')
        
        # 存储结果
        self.distilled_dataset: List[Dict] = []
        self.failed_indices: List[int] = []
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个专业的软件测试专家，擅长分析测试代码并识别Flaky Tests。"
    
    def generate_user_prompt(self, row: pd.Series) -> str:
        """
        生成用户提示词
        
        Args:
            row: 数据行
            
        Returns:
            格式化的用户提示词
        """
        test_code = row.get(self.code_column, row.get('full_code', ''))
        return format_prompt(self.user_template, test_code=test_code)
    
    def process_single_row(self, idx: int, row: pd.Series) -> Optional[Dict]:
        """
        处理单条数据
        
        Args:
            idx: 数据索引
            row: 数据行
            
        Returns:
            Alpaca格式的数据，失败返回None
        """
        # 生成prompt
        user_prompt = self.generate_user_prompt(row)
        
        # 调用API获取推理过程
        reasoning = self.call_api(user_prompt)
        
        if reasoning is None:
            print(f"\n⚠ 第 {idx} 条数据处理失败")
            self.failed_indices.append(idx)
            return None
        
        # 转换为Alpaca格式
        alpaca_item = convert_to_alpaca_format(row, reasoning, self.code_column)
        return alpaca_item
    
    def save_checkpoint(self, checkpoint_name: str = 'checkpoint') -> None:
        """
        保存检查点
        
        Args:
            checkpoint_name: 检查点文件名
        """
        checkpoint_file = self.output_dir / f"{checkpoint_name}.json"
        save_json(self.distilled_dataset, checkpoint_file)
    
    def run(self,
            dataset_path: Optional[Union[str, Path]] = None,
            output_name: str = 'distillation_dataset') -> Dict:
        """
        执行蒸馏任务
        
        Args:
            dataset_path: 数据集路径（可选，覆盖初始化参数）
            output_name: 输出文件名（不含扩展名）
            
        Returns:
            包含结果统计的字典
        """
        # 加载数据
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        print("\n" + "=" * 60)
        print("开始数据蒸馏任务")
        print("=" * 60)
        
        df = load_csv(self.dataset_path)
        
        # 根据测试模式采样数据
        if self.test_mode != 'all':
            print(f"\n📊 测试模式: {self.test_mode}, 采样 {self.test_size} 条数据")
            df = sample_data(df, mode=self.test_mode, n=self.test_size, random_seed=self.random_seed)
        
        print(f"\n🚀 开始处理 {len(df)} 条数据...")
        print(f"   批次大小: {self.batch_size}")
        print(f"   批次延迟: {self.batch_delay}秒")
        print(f"   检查点间隔: {self.checkpoint_interval}条")
        print("=" * 60 + "\n")
        
        # 重置结果
        self.distilled_dataset = []
        self.failed_indices = []
        self.reset_stats()
        
        # 处理数据
        start_time = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
            alpaca_item = self.process_single_row(idx, row)
            
            if alpaca_item:
                self.distilled_dataset.append(alpaca_item)
            
            # 批次延迟
            if (idx + 1) % self.batch_size == 0:
                time.sleep(self.batch_delay)
            
            # 保存检查点
            if (idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint('temp_checkpoint')
                print(f"\n✓ 已处理 {idx + 1} 条，检查点已保存")
        
        elapsed_time = time.time() - start_time
        
        # 保存最终结果
        output_file = self.output_dir / f"{output_name}.json"
        save_json(self.distilled_dataset, output_file)
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("蒸馏任务完成")
        print("=" * 60)
        print(f"✓ 成功: {len(self.distilled_dataset)} 条")
        print(f"✗ 失败: {len(self.failed_indices)} 条")
        print(f"⏱ 耗时: {elapsed_time:.2f} 秒")
        print(f"📁 输出文件: {output_file}")
        print("=" * 60)
        
        # 打印API统计
        self.print_stats()
        
        # 显示示例
        if self.distilled_dataset:
            print("\n示例数据:")
            print("=" * 60)
            import json
            print(json.dumps(self.distilled_dataset[0], ensure_ascii=False, indent=2))
            print("=" * 60)
        
        return {
            "success_count": len(self.distilled_dataset),
            "failed_count": len(self.failed_indices),
            "failed_indices": self.failed_indices,
            "elapsed_time": elapsed_time,
            "output_file": str(output_file),
            "api_stats": self.get_stats()
        }
