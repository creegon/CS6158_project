"""
DistillationAgent - 数据蒸馏Agent
用于生成包含推理过程的训练数据集
"""
import time
from pathlib import Path
from typing import Optional, Union, List, Dict
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
                 parallel_workers: int = 1,
                 api_matcher=None,
                 top_k_shots: int = 3,
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
            parallel_workers: 并行推理的工作线程数，1表示串行处理
            api_matcher: API签名匹配器，用于检索few-shot examples
            top_k_shots: 使用的few-shot样本数量
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
        self.parallel_workers = max(1, parallel_workers)  # 确保至少为1
        
        # API匹配相关
        self.api_matcher = api_matcher
        self.top_k_shots = top_k_shots if api_matcher else 0
        
        # 加载prompt模板
        self.system_prompt = load_prompt('distillation_system')
        self.user_template = load_prompt('distillation_user')
        
        # 存储结果
        self.distilled_dataset: List[Dict] = []
        self.failed_indices: List[int] = []
        
        # 线程安全锁
        self._lock = threading.Lock()
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个专业的软件测试专家，擅长分析测试代码并识别Flaky Tests。"
    
    def generate_user_prompt(self, row: pd.Series) -> str:
        """
        生成用户提示词（支持API匹配的few-shot examples）
        
        Args:
            row: 数据行
            
        Returns:
            格式化的用户提示词
        """
        prompt, _ = self.generate_user_prompt_with_examples(row)
        return prompt
    
    def generate_user_prompt_with_examples(self, row: pd.Series) -> tuple:
        """
        生成用户提示词并返回few-shot examples（支持API匹配）
        
        Args:
            row: 数据行
            
        Returns:
            (格式化的用户提示词, few-shot examples列表)
        """
        full_code = row.get(self.code_column, row.get('full_code', ''))
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        
        # 基础prompt
        base_prompt = format_prompt(
            self.user_template, 
            project=project,
            test_name=test_name,
            full_code=full_code
        )
        
        few_shot_examples = None
        
        # 如果启用API匹配，添加few-shot examples
        if self.api_matcher and self.top_k_shots > 0:
            try:
                # 检索最相似的案例
                similar_cases = self.api_matcher.retrieve_top_k(
                    full_code, 
                    top_k=self.top_k_shots,
                    min_similarity=0.1  # 最小相似度阈值
                )
                
                if similar_cases:
                    # 构建few-shot examples记录（用于debug）
                    few_shot_examples = []
                    for i, (idx, similarity, case_row) in enumerate(similar_cases, 1):
                        example_info = {
                            'index': int(idx),
                            'similarity': float(similarity),
                            'project': str(case_row.get('project', 'Unknown')),
                            'test_name': str(case_row.get('test_name', 'Unknown')),
                            'category': int(case_row.get('category', -1)),
                            'code_preview': str(case_row.get(self.code_column, case_row.get('full_code', ''))[:200])
                        }
                        if 'id' in case_row:
                            example_info['id'] = int(case_row['id'])
                        few_shot_examples.append(example_info)
                    
                    # 构建few-shot examples文本（插入prompt）
                    examples_text = "\n\n参考案例（根据API签名相似度检索）：\n"
                    examples_text += "=" * 60 + "\n"
                    
                    for i, (idx, similarity, case_row) in enumerate(similar_cases, 1):
                        case_code = case_row.get(self.code_column, case_row.get('full_code', ''))
                        case_category = case_row.get('category', 'Unknown')
                        case_project = case_row.get('project', 'Unknown')
                        
                        examples_text += f"\n【案例 {i}】(相似度: {similarity:.2f})\n"
                        examples_text += f"项目: {case_project}\n"
                        examples_text += f"分类: {case_category}\n"
                        examples_text += f"代码:\n{case_code}\n"
                        examples_text += "-" * 60 + "\n"
                    
                    # 将few-shot examples插入到prompt中
                    # 在原始代码之前插入参考案例
                    base_prompt = base_prompt.replace(
                        full_code,
                        examples_text + "\n待分析的测试代码:\n" + full_code
                    )
            
            except Exception as e:
                print(f"⚠ API匹配失败: {e}")
                # 如果匹配失败，继续使用基础prompt
        
        return base_prompt, few_shot_examples
    
    def process_single_row(self, idx: int, row: pd.Series, include_id: bool = False) -> Optional[Dict]:
        """
        处理单条数据
        
        Args:
            idx: 数据索引
            row: 数据行
            include_id: 是否包含ID字段
            
        Returns:
            Alpaca格式的数据，失败返回None
        """
        # 生成prompt（同时获取few-shot examples）
        user_prompt, few_shot_examples = self.generate_user_prompt_with_examples(row)
        
        # 调用API获取推理过程
        reasoning = self.call_api(user_prompt)
        
        if reasoning is None:
            with self._lock:
                print(f"\n⚠ 第 {idx} 条数据处理失败")
                self.failed_indices.append(idx)
            return None
        
        # 转换为Alpaca格式，传入 system_prompt、user_template 和 few_shot_examples
        alpaca_item = convert_to_alpaca_format(
            row, 
            reasoning, 
            self.code_column, 
            include_id=include_id,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
            few_shot_examples=few_shot_examples
        )
        return alpaca_item
    
    def process_single_row_with_index(self, task: tuple) -> tuple:
        """
        处理单条数据（带索引，用于并行处理）
        
        Args:
            task: (idx, row, include_id) 元组
            
        Returns:
            (idx, alpaca_item) 元组
        """
        idx, row, include_id = task
        alpaca_item = self.process_single_row(idx, row, include_id=include_id)
        return (idx, alpaca_item)
    
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
        print(f"   并行线程数: {self.parallel_workers}")
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
        
        if self.parallel_workers == 1:
            # 串行处理
            self._run_serial(df)
        else:
            # 并行处理
            self._run_parallel(df)
        
        elapsed_time = time.time() - start_time
        
        # 保存结果
        # 如果使用了API匹配，数据已包含id和few_shot_examples，保存为external版本
        # 否则只保存标准版本
        if self.api_matcher is not None:
            # 使用API匹配时，保存为external版本（包含id和few-shot examples）
            output_file_external = self.output_dir / f"{output_name}_external.json"
            save_json(self.distilled_dataset, output_file_external)
            
            # 同时保存一个不带额外信息的标准版本（用于训练）
            print("\n🔄 生成标准训练数据集（移除额外信息）...")
            dataset_standard = []
            for item in self.distilled_dataset:
                standard_item = {
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'output': item['output']
                }
                dataset_standard.append(standard_item)
            output_file = self.output_dir / f"{output_name}.json"
            save_json(dataset_standard, output_file)
        else:
            # 未使用API匹配时，只保存标准版本
            output_file = self.output_dir / f"{output_name}.json"
            save_json(self.distilled_dataset, output_file)
            output_file_external = None
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("蒸馏任务完成")
        print("=" * 60)
        print(f"✓ 成功: {len(self.distilled_dataset)} 条")
        print(f"✗ 失败: {len(self.failed_indices)} 条")
        print(f"⏱ 耗时: {elapsed_time:.2f} 秒")
        print(f"⚡ 平均速度: {len(df) / elapsed_time:.2f} 条/秒")
        if self.api_matcher is not None:
            print(f"📁 标准输出: {output_file}")
            print(f"📁 额外信息输出: {output_file_external}")
        else:
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
            "output_file_external": str(output_file_external) if output_file_external else None,
            "api_stats": self.get_stats()
        }
    
    def _run_serial(self, df: pd.DataFrame) -> None:
        """
        串行处理数据
        
        Args:
            df: 要处理的数据框
        """
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
            # 如果启用API匹配，直接生成包含额外信息的版本
            include_id = self.api_matcher is not None
            alpaca_item = self.process_single_row(idx, row, include_id=include_id)
            
            if alpaca_item:
                self.distilled_dataset.append(alpaca_item)
            
            # 批次延迟
            if (idx + 1) % self.batch_size == 0:
                time.sleep(self.batch_delay)
            
            # 保存检查点
            if (idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint('temp_checkpoint')
                print(f"\n✓ 已处理 {idx + 1} 条，检查点已保存")
    
    def _run_parallel(self, df: pd.DataFrame) -> None:
        """
        并行处理数据
        
        Args:
            df: 要处理的数据框
        """
        # 如果启用API匹配，直接生成包含额外信息的版本
        include_id = self.api_matcher is not None
        
        # 准备任务列表
        tasks = [(idx, row, include_id) for idx, row in df.iterrows()]
        results = {}  # 存储结果，保持原始顺序
        processed_count = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.process_single_row_with_index, task): task[0] 
                for task in tasks
            }
            
            # 使用进度条
            with tqdm(total=len(tasks), desc="处理进度") as pbar:
                for future in as_completed(future_to_task):
                    idx, alpaca_item = future.result()
                    results[idx] = alpaca_item
                    processed_count += 1
                    pbar.update(1)
                    
                    # 批次延迟（每处理一批后暂停）
                    if processed_count % self.batch_size == 0:
                        time.sleep(self.batch_delay)
                    
                    # 保存检查点
                    if processed_count % self.checkpoint_interval == 0:
                        # 按索引顺序整理当前结果
                        sorted_results = [
                            results[i] for i in sorted(results.keys()) 
                            if results[i] is not None
                        ]
                        with self._lock:
                            self.distilled_dataset = sorted_results
                            self.save_checkpoint('temp_checkpoint')
                        print(f"\n✓ 已处理 {processed_count} 条，检查点已保存")
        
        # 按原始索引顺序整理最终结果
        self.distilled_dataset = [
            results[idx] for idx in sorted(results.keys()) 
            if results[idx] is not None
        ]
