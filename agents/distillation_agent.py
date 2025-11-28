"""
DistillationAgent - 数据蒸馏Agent
用于生成包含推理过程的训练数据集
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from agents.base_agent import BaseAgent
from agents.reasoning_agent import ReasoningAgent
from agents.inferring_agent import InferringAgent
from agents.challenger_agent import ChallengerAgent
from utils import (
    load_csv,
    sample_data,
    convert_to_alpaca_format,
    save_json,
    load_prompt,
    format_prompt,
    ProjectContextFetcher
)
from config import (
    DATASET_PATH,
    OUTPUT_DIR,
    API_BATCH_SIZE,
    API_BATCH_DELAY,
    CHECKPOINT_INTERVAL,
    PROJECT_ROOT,
    FEATURE_HINT_MODE,
    FEATURE_HINT_MAX_PER_LEVEL
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
                 random_seed: Optional[int] = 42,
                 code_column: str = 'code',
                 batch_size: Optional[int] = None,
                 batch_delay: Optional[float] = None,
                 checkpoint_interval: Optional[int] = None,
                 parallel_workers: int = 1,
                 api_matcher=None,
                 top_k_shots: int = 3,
                 use_context: bool = False,
                 use_feature_hint: bool = True,
                 use_reasoning_guide: bool = True,
                 use_debate: bool = True,
                 target_ids: Optional[List[int]] = None,
                 **kwargs):
        """
        初始化DistillationAgent
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            test_mode: 测试模式 ('all', 'random', 'head')
            test_size: 测试样本数
            random_seed: 随机种子
            code_column: 代码列名
            use_reasoning_guide: 是否使用推理指引
            use_debate: 是否启用辩论模式 (Reasoning vs Challenger)
            parallel_workers: 并行工作线程数
            batch_size: 批次大小
            batch_delay: 批次延迟
            api_matcher: API匹配器
            top_k_shots: few-shot数量
            use_context: 是否使用上下文
            use_feature_hint: 是否使用特征提示
            target_ids: 指定要处理的目标ID列表
            **kwargs: 传递给BaseAgent的参数
        """
        super().__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) if dataset_path else Path(DATASET_PATH)
        self.output_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR)
        self.test_mode = test_mode
        self.test_size = test_size
        self.random_seed = random_seed
        self.code_column = code_column
        self.use_reasoning_guide = use_reasoning_guide
        self.use_debate = use_debate
        self.parallel_workers = parallel_workers
        self.batch_size = batch_size if batch_size is not None else 5
        self.batch_delay = batch_delay if batch_delay is not None else 1.0
        self.checkpoint_interval = checkpoint_interval if checkpoint_interval is not None else 10
        self.target_ids = target_ids
        
        # 初始化子Agent
        self.reasoning_agent = ReasoningAgent(**kwargs)
        self.inferring_agent = InferringAgent(
            code_column=code_column,
            api_matcher=api_matcher,
            top_k_shots=top_k_shots,
            use_context=use_context,
            use_feature_hint=use_feature_hint,
            **kwargs
        )
        self.challenger_agent = ChallengerAgent(**kwargs) if use_debate else None
        
        self.distilled_dataset = []
        self.failed_indices = []
        
        # 线程安全锁
        self._lock = threading.Lock()
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个专业的软件测试专家，擅长分析测试代码并识别Flaky Tests。"
    
    def generate_reasoning_guide(self, row: pd.Series) -> Optional[str]:
        """
        生成推理指引(委托给ReasoningAgent)
        
        Args:
            row: 数据行
            
        Returns:
            推理指引文本，失败返回None
        """
        if not self.use_reasoning_guide or not self.reasoning_agent:
            return None
        
        return self.reasoning_agent.generate_from_row(row, self.code_column)
    
    def generate_user_prompt(self, row: pd.Series) -> str:
        """
        生成用户提示词（委托给InferringAgent）
        
        Args:
            row: 数据行
            
        Returns:
            格式化的用户提示词
        """
        prompt, _, _ = self.generate_user_prompt_with_examples(row)
        return prompt
    
    def generate_user_prompt_with_examples(self, row: pd.Series) -> tuple:
        """
        生成用户提示词并返回元数据（委托给ReasoningAgent和InferringAgent）
        
        Args:
            row: 数据行
            
        Returns:
            (格式化的用户提示词, 元数据字典, 推理指引文本)
        """
        # 生成推理指引(如果启用)
        reasoning_guide_text = self.generate_reasoning_guide(row)
        
        # 生成用户提示词和元数据
        user_prompt, metadata = self.inferring_agent.generate_from_row(row, reasoning_guide_text)
        
        # 提取few_shot_examples（保持向后兼容）
        few_shot_examples = metadata.get('few_shot_examples', None)
        
        return user_prompt, few_shot_examples, reasoning_guide_text
    
    def process_single_row(self, idx: int, row: pd.Series, include_id: bool = False) -> Optional[Dict]:
        """
        处理单条数据
        
        Args:
            idx: 数据索引
            row: 数据行
            include_id: 是否包含ID字段（已废弃，总是包含额外信息）
            
        Returns:
            Alpaca格式的数据（包含所有额外信息），失败返回None
        """
        # 1. 生成推理指引 (Reasoning Guide)
        reasoning_guide = None
        debate_history = None
        debate_context = {}
        
        if self.use_reasoning_guide:
            reasoning_guide = self.reasoning_agent.generate_from_row(row, self.code_column)
            if reasoning_guide:
                debate_context['original_analysis'] = reasoning_guide
                
            if not reasoning_guide:
                print(f"⚠ 无法生成推理指引 (ID: {idx})")
                return None
                
            # 2. 辩论环节 (Debate Loop - 2 Rounds)
            if self.use_debate and self.challenger_agent:
                project = row.get('project', 'Unknown')
                test_name = row.get('test_name', 'Unknown')
                full_code = row.get(self.code_column, '')
                
                # --- Round 1 ---
                critique_1 = self.challenger_agent.challenge(project, test_name, full_code, reasoning_guide)
                
                if critique_1:
                    debate_context['critique_1'] = critique_1
                    
                    # Round 1 Defense
                    defense_1 = self.reasoning_agent.defend_analysis(project, test_name, full_code, reasoning_guide, critique_1)
                    
                    if defense_1:
                        debate_context['defense_1'] = defense_1
                        
                        # 构建第一轮历史
                        history_r1 = f"""
【原始分析 (Analyst)】
{reasoning_guide}

【审查员质疑 Round 1 (Challenger)】
{critique_1}

【分析师辩护 Round 1 (Defender)】
{defense_1}
"""
                        # --- Round 2 ---
                        # Challenger 基于第一轮历史进行第二轮质疑
                        critique_2 = self.challenger_agent.challenge(project, test_name, full_code, history_r1)
                        
                        if critique_2:
                            debate_context['critique_2'] = critique_2
                            
                            # 辩论在第二轮质疑后结束，不再进行辩护，确保双方发言机会均等 (2 vs 2)
                            debate_history = f"""{history_r1}

【审查员质疑 Round 2 (Challenger)】
{critique_2}
"""
                        else:
                            # Critique 2 failed or no further critique
                            debate_history = history_r1
                            
                        # 将完整的辩论历史作为最终的 reasoning_guide
                        reasoning_guide = debate_history
                    else:
                        # Defense 1 failed
                        reasoning_guide = f"""
【原始分析 (Analyst)】
{reasoning_guide}

【审查员质疑 (Challenger)】
{critique_1}

(Defender failed to respond)
"""

        # 3. 生成最终推理 (Final Inference)
        # 注意：此时的 reasoning_guide 可能包含了辩论历史
        result_tuple = self.inferring_agent.run(
            row.get('project', 'Unknown'),
            row.get('test_name', 'Unknown'),
            row.get(self.code_column, ''),
            reasoning_guide=reasoning_guide
        )
        
        if result_tuple is None:
            with self._lock:
                print(f"\n⚠ 第 {idx} 条数据处理失败")
                self.failed_indices.append(idx)
            return None
            
        reasoning, metadata = result_tuple
        
        # 从metadata中提取few_shot_examples（保持向后兼容）
        few_shot_examples = metadata.get('few_shot_examples', None)
        
        # 重新生成user_prompt（用于Alpaca格式）
        user_prompt, _ = self.inferring_agent.generate_from_row(row, reasoning_guide)
        
        # 转换为Alpaca格式
        alpaca_item = convert_to_alpaca_format(
            row, 
            reasoning, 
            self.code_column, 
            include_id=True,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=few_shot_examples
        )
        
        # 添加所有元数据到external版本
        if metadata.get('external_context'):
            alpaca_item['external_context'] = metadata['external_context']
        
        if metadata.get('feature_hints'):
            alpaca_item['feature_hints'] = metadata['feature_hints']
        
        if debate_context:
            alpaca_item['debate_context'] = debate_context
        
        return alpaca_item
    
    def process_single_row_with_index(self, task: tuple) -> tuple:
        """
        处理单条数据（带索引，用于并行处理）
        
        Args:
            task: (idx, row) 元组
            
        Returns:
            (idx, alpaca_item) 元组
        """
        idx, row = task
        alpaca_item = self.process_single_row(idx, row)
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
        
        # 如果指定了target_ids，优先使用ID过滤
        if self.target_ids:
            print(f"\n🎯 目标ID模式: 仅处理 {len(self.target_ids)} 个指定样本")
            # 确保ID列存在
            if 'id' not in df.columns:
                print("✗ 数据集中未找到 'id' 列，无法进行ID过滤")
                return {}
                
            df = df[df['id'].isin(self.target_ids)]
            if len(df) != len(self.target_ids):
                print(f"   ⚠️  警告: 仅找到 {len(df)} 个匹配样本 (目标 {len(self.target_ids)} 个)")
                
        # 根据测试模式采样数据
        elif self.test_mode != 'all':
            print(f"\n📊 测试模式: {self.test_mode}, 采样 {self.test_size} 条数据")
            if self.test_mode == 'random':
                if self.random_seed is not None:
                    print(f"   🎲 随机种子: {self.random_seed} (可复现)")
                else:
                    print(f"   ⚠️  未设置随机种子，结果不可复现")
            df = sample_data(df, mode=self.test_mode, n=self.test_size, random_seed=self.random_seed)
        
        print(f"\n🚀 开始处理 {len(df)} 条数据...")
        print(f"   并行线程数: {self.parallel_workers}")
        print(f"   批次大小: {self.batch_size}")
        print(f"   批次延迟: {self.batch_delay}秒")
        print(f"   检查点间隔: {self.checkpoint_interval}条")
        
        # 打印优化项配置
        optimizations = []
        if self.use_reasoning_guide:
            optimizations.append("推理指引(双Agent)")
        if self.inferring_agent.api_matcher:
            optimizations.append(f"Few-shot样本(top-{self.inferring_agent.top_k_shots})")
        if self.inferring_agent.use_context:
            optimizations.append("外部上下文")
        if self.inferring_agent.use_feature_hint:
            mode_desc = "全局最高级别" if FEATURE_HINT_MODE == "global-highest" else f"按类别分组(每级别最多{FEATURE_HINT_MAX_PER_LEVEL if FEATURE_HINT_MAX_PER_LEVEL > 0 else '不限'}个)"
            optimizations.append(f"特征词频({mode_desc})")
        
        if optimizations:
            print(f"   优化项: {', '.join(optimizations)}")
        
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
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name_with_timestamp = f"{output_name}_{timestamp}"
        
        # 总是生成两个文件:
        # 1. external 版本: 包含 id 和所有额外信息 (few_shot_examples, external_context, feature_hints)
        # 2. standard 版本: 仅包含 instruction, input, output (用于训练)
        
        print("\n💾 保存结果...")
        
        # 保存 external 版本 (包含所有额外信息)
        output_file_external = self.output_dir / f"{output_name_with_timestamp}_external.json"
        save_json(self.distilled_dataset, output_file_external)
        print(f"✓ External版本已保存: {output_file_external.name}")
        
        # 生成 standard 版本 (仅保留训练所需的基本字段)
        dataset_standard = []
        for item in self.distilled_dataset:
            standard_item = {
                'instruction': item['instruction'],
                'input': item['input'],
                'output': item['output']
            }
            dataset_standard.append(standard_item)
        
        output_file = self.output_dir / f"{output_name_with_timestamp}.json"
        save_json(dataset_standard, output_file)
        print(f"✓ Standard版本已保存: {output_file.name}")
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("蒸馏任务完成")
        print("=" * 60)
        print(f"✓ 成功: {len(self.distilled_dataset)} 条")
        print(f"✗ 失败: {len(self.failed_indices)} 条")
        print(f"⏱ 耗时: {elapsed_time:.2f} 秒")
        print(f"⚡ 平均速度: {len(df) / elapsed_time:.2f} 条/秒")
        print(f"\n📁 输出文件:")
        print(f"  Standard版本: {output_file}")
        print(f"  External版本: {output_file_external}")
        
        # 额外信息统计
        extra_info = []
        if self.use_reasoning_guide:
            extra_info.append("推理指引")
        if self.inferring_agent.api_matcher is not None:
            extra_info.append("Few-shot样本")
        if self.inferring_agent.use_context:
            extra_info.append("外部上下文")
        if self.inferring_agent.use_feature_hint:
            extra_info.append("特征词频")
        
        if extra_info:
            print(f"  External包含: {', '.join(extra_info)}")
        
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
    
    def _run_parallel(self, df: pd.DataFrame) -> None:
        """
        并行处理数据
        
        Args:
            df: 要处理的数据框
        """
        # 准备任务列表
        tasks = [(idx, row) for idx, row in df.iterrows()]
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
