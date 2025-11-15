"""
InferringAgent - 判断Agent
用于基于推理指引进行Flaky Test分类判断
"""
from typing import Optional, List, Dict
import pandas as pd

from agents.base_agent import BaseAgent
from utils import load_prompt, format_prompt
from utils.feature_matcher import FeatureMatcher
from utils.context_extractor import ProjectContextFetcher
from config import (
    PROJECT_ROOT,
    FEATURE_HINT_MODE,
    FEATURE_HINT_MAX_PER_LEVEL
)


class InferringAgent(BaseAgent):
    """
    判断Agent
    
    作为双Agent推理链的第二步，负责：
    1. 接收推理指引（可选）
    2. 结合few-shot examples（可选）
    3. 结合上下文信息（可选）
    4. 结合特征词频（可选）
    5. 进行最终的Flaky Test分类判断
    
    目标: 基于推理指引进行因果推理，避免简单的模式匹配错误
    """
    
    def __init__(self,
                 code_column: str = 'code',
                 api_matcher=None,
                 top_k_shots: int = 3,
                 use_context: bool = False,
                 use_feature_hint: bool = True,
                 **kwargs):
        """
        初始化InferringAgent
        
        Args:
            code_column: 代码列名
            api_matcher: API签名匹配器，用于检索few-shot examples
            top_k_shots: 使用的few-shot样本数量
            use_context: 是否启用上下文提取(从external_projects中提取)
            use_feature_hint: 是否启用特征词频提示(基于归一化频率分析)
            **kwargs: 传递给BaseAgent的其他参数
        """
        super().__init__(**kwargs)
        
        self.code_column = code_column
        
        # API匹配相关
        self.api_matcher = api_matcher
        self.top_k_shots = top_k_shots if api_matcher else 0
        
        # Context提取器 (仅在需要时初始化)
        self.use_context = use_context
        self.context_fetcher = ProjectContextFetcher() if use_context else None
        
        # 特征匹配器 (仅在需要时初始化)
        self.use_feature_hint = use_feature_hint
        self.feature_matcher = None
        if use_feature_hint:
            try:
                lookup_path = PROJECT_ROOT / 'output' / 'facet_analysis' / 'feature_lookup_table.json'
                if lookup_path.exists():
                    self.feature_matcher = FeatureMatcher(str(lookup_path))
                    print(f"✓ 特征匹配器已加载")
                else:
                    print(f"⚠️ 特征查找表不存在: {lookup_path}")
                    print("   提示: 运行 python analyze_normalized_features.py 生成特征表")
                    self.use_feature_hint = False
            except Exception as e:
                print(f"⚠️ 加载特征匹配器失败: {e}")
                self.use_feature_hint = False
        
        # 加载判断专用的prompt模板
        self.system_prompt = load_prompt('distillation_system')
        self.user_template = load_prompt('distillation_user')
        
        print("✓ InferringAgent已初始化")
    
    def get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个专业的软件测试专家，擅长分析测试代码并识别Flaky Tests。"
    
    def _filter_features_by_mode(self, matches: dict) -> list:
        """
        根据配置模式过滤特征
        
        Args:
            matches: feature_matcher返回的匹配结果
            
        Returns:
            过滤后的特征列表 [(category, level, features), ...]
        """
        if FEATURE_HINT_MODE == "global-highest":
            # 全局最高级别模式: 找到所有类别中的最高级别,只返回该级别的特征
            global_highest_level = None
            level_priority = ['unique', 'very_strong', 'strong', 'moderate']
            
            # 找到全局最高级别
            for level in level_priority:
                for category, levels in matches.items():
                    if levels[level]:
                        global_highest_level = level
                        break
                if global_highest_level:
                    break
            
            if not global_highest_level:
                return []
            
            # 收集所有类别中该级别的特征
            filtered = []
            for category, levels in matches.items():
                if levels[global_highest_level]:
                    features = sorted(levels[global_highest_level],
                                    key=lambda x: (x['discrimination'], x['flaky_density']),
                                    reverse=True)
                    filtered.append((category, global_highest_level, features))
            
            return filtered
            
        else:  # category-wise 模式
            # 每个类别独立选择最高级别
            filtered = []
            level_priority = ['unique', 'very_strong', 'strong', 'moderate']
            
            for category, levels in matches.items():
                for level in level_priority:
                    if levels[level]:
                        features = sorted(levels[level],
                                        key=lambda x: (x['discrimination'], x['flaky_density']),
                                        reverse=True)
                        # 如果设置了max_per_level限制
                        if FEATURE_HINT_MAX_PER_LEVEL > 0:
                            features = features[:FEATURE_HINT_MAX_PER_LEVEL]
                        filtered.append((category, level, features))
                        break  # 找到该类别的最高级别后跳出
            
            return filtered
    
    def _get_few_shot_examples(self, full_code: str) -> tuple:
        """
        获取few-shot examples
        
        Args:
            full_code: 完整测试代码
            
        Returns:
            (few_shot_examples列表, few_shots_text文本)
        """
        few_shot_examples = None
        few_shots_text = ""
        
        if self.api_matcher and self.top_k_shots > 0:
            try:
                # 检索最相似的案例
                similar_cases = self.api_matcher.retrieve_top_k(
                    full_code, 
                    top_k=self.top_k_shots,
                    min_similarity=0.1
                )
                
                if similar_cases:
                    # 构建few-shot examples记录（用于debug）
                    few_shot_examples = []
                    for i, (idx, similarity, case_row) in enumerate(similar_cases, 1):
                        example_info = {
                            'similarity': float(similarity),
                            'project': str(case_row.get('project', 'Unknown')),
                            'test_name': str(case_row.get('test_name', 'Unknown')),
                            'label': str(case_row.get('label', 'Unknown')),
                            'code_preview': str(case_row.get(self.code_column, case_row.get('full_code', ''))[:200])
                        }
                        if 'id' in case_row:
                            example_info['id'] = int(case_row['id'])
                        few_shot_examples.append(example_info)
                    
                    # 构建few-shot examples文本（用于模板替换）
                    examples_parts = []
                    for i, (idx, similarity, case_row) in enumerate(similar_cases, 1):
                        case_code = case_row.get(self.code_column, case_row.get('full_code', ''))
                        case_label = case_row.get('label', 'Unknown')
                        case_project = case_row.get('project', 'Unknown')
                        case_test_name = case_row.get('test_name', 'Unknown')
                        
                        example = f"""【案例 {i}】(相似度: {similarity:.2f})
项目: {case_project}
测试名称: {case_test_name}
标签: {case_label}
代码:
{case_code}
{'-' * 60}"""
                        examples_parts.append(example)
                    
                    few_shots_text = "\n".join(examples_parts)
            
            except Exception as e:
                print(f"⚠ API匹配失败: {e}")
        
        return few_shot_examples, few_shots_text
    
    def _get_context_info(self, project: str, test_name: str) -> tuple:
        """
        获取上下文信息
        
        Args:
            project: 项目名称
            test_name: 测试名称
            
        Returns:
            (context_windows_text, calling_functions_text, context_info字典)
        """
        context_windows_text = ""
        calling_functions_text = ""
        context_info = None
        
        if self.use_context and self.context_fetcher:
            try:
                context_info = self.context_fetcher.get_test_context(
                    project=project,
                    test_name=test_name,
                    context_lines=20,
                    invocation_limit=10
                )
                
                # 格式化surrounding_window
                if context_info.get('surrounding_window'):
                    context_windows_text = f"""文件路径: {context_info['file_path']}
类名: {context_info['class_name']}
方法名: {context_info['method_name']}

上下文代码:
{context_info['surrounding_window']}"""
                
                # 格式化invocations
                if context_info.get('invocations'):
                    invocations_list = []
                    for i, inv in enumerate(context_info['invocations'], 1):
                        invocations_list.append(
                            f"[{i}] {inv['file_path']}:{inv['line_number']}\n    {inv['line_preview']}"
                        )
                    calling_functions_text = "\n".join(invocations_list)
                    
            except Exception as e:
                print(f"⚠ 提取上下文信息失败 ({project}/{test_name}): {e}")
                context_info = {'error': str(e)}
        
        return context_windows_text, calling_functions_text, context_info
    
    def _get_feature_hints(self, full_code: str) -> tuple:
        """
        获取特征词频提示
        
        Args:
            full_code: 完整测试代码
            
        Returns:
            (feature_hint_text, feature_list)
        """
        feature_hint_text = ""
        feature_list = []
        
        if self.use_feature_hint and self.feature_matcher:
            try:
                matches = self.feature_matcher.match_features(full_code)
                
                if matches:
                    # 使用配置的过滤模式
                    filtered_features = self._filter_features_by_mode(matches)
                    
                    if filtered_features:
                        hint_parts = []
                        hint_parts.append("下面是一些给你提供的flaky种类的词频提示,包括它的类别以及它相对于non flaky的倍率。倍率越大就越有可能是对应的种类。可能会同时有很多种种类,这时,你需要结合你原有的判断去完成。并且,它们只是参考,你最终还是要依据题目本身做出逻辑分析和回答。\n")
                        
                        # 按过滤结果组织提示
                        for category, level, features in filtered_features:
                            for feat in features:
                                disc_str = '∞' if feat['discrimination'] == float('inf') else f"{feat['discrimination']:.1f}x"
                                hint_parts.append(
                                    f"【{feat['feature']}】: 它的类别是【{category}】，它的词频倍率是【{disc_str}】"
                                )
                                
                                # 添加到feature_list
                                feature_list.append({
                                    'feature': feat['feature'],
                                    'category': category,
                                    'level': level,
                                    'discrimination': feat['discrimination'],
                                    'flaky_density': feat['flaky_density']
                                })
                        
                        if len(hint_parts) > 1:  # 除了开头说明,还有实际特征
                            feature_hint_text = "\n".join(hint_parts)
                    
            except Exception as e:
                print(f"⚠ 生成特征提示失败: {e}")
                feature_list = [{'error': str(e)}]
        
        return feature_hint_text, feature_list
    
    def generate_user_prompt(self,
                           project: str,
                           test_name: str,
                           full_code: str,
                           reasoning_guide: Optional[str] = None) -> tuple:
        """
        生成用户提示词
        
        Args:
            project: 项目名称
            test_name: 测试名称
            full_code: 完整测试代码
            reasoning_guide: 推理指引文本（可选）
            
        Returns:
            (格式化的用户提示词, 元数据字典)
            元数据包含: few_shot_examples, context_info, feature_hints
        """
        # 动态构建prompt内容(只包含启用的部分)
        prompt_parts = []
        
        # 基本信息(必需)
        prompt_parts.append(f"该测试代码所属project的名称为：\n{project}")
        prompt_parts.append(f"它的测试名称为：\n{test_name}")
        prompt_parts.append(f"完整代码为：\n{full_code}")
        
        # 推理指引 (放在代码之后，其他信息之前)
        if reasoning_guide:
            prompt_parts.append(f"推理指引：\n{reasoning_guide}\n\n**请按照上述推理指引的建议路径进行分析，避免提到的常见陷阱。**")
        
        # 获取few-shot examples
        few_shot_examples, few_shots_text = self._get_few_shot_examples(full_code)
        if few_shots_text:
            prompt_parts.append(f"检索到的相近案例有：\n{few_shots_text}")
        
        # 获取上下文信息
        context_windows_text, calling_functions_text, context_info = self._get_context_info(project, test_name)
        if context_windows_text:
            prompt_parts.append(f"该测试案例在原项目中的上下文为：\n{context_windows_text}")
        if calling_functions_text:
            prompt_parts.append(f"在原项目中，涉及到调用该测试案例的原文为：\n{calling_functions_text}")
        
        # 获取特征提示
        feature_hint_text, feature_list = self._get_feature_hints(full_code)
        if feature_hint_text:
            prompt_parts.append(f"词频分析提示：\n{feature_hint_text}")
        
        # 用两个换行符连接各部分
        user_prompt = "\n\n".join(prompt_parts)
        
        # 构建元数据
        metadata = {}
        if few_shot_examples:
            metadata['few_shot_examples'] = few_shot_examples
        if context_info:
            metadata['external_context'] = context_info
        if feature_list:
            metadata['feature_hints'] = feature_list
        
        return user_prompt, metadata
    
    def generate_from_row(self,
                         row: pd.Series,
                         reasoning_guide: Optional[str] = None) -> tuple:
        """
        从DataFrame行生成用户提示词（便捷方法）
        
        Args:
            row: 数据行
            reasoning_guide: 推理指引文本（可选）
            
        Returns:
            (格式化的用户提示词, 元数据字典)
        """
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        full_code = row.get(self.code_column, row.get('full_code', ''))
        
        return self.generate_user_prompt(project, test_name, full_code, reasoning_guide)
    
    def infer(self,
             project: str,
             test_name: str,
             full_code: str,
             reasoning_guide: Optional[str] = None) -> tuple:
        """
        进行推断（调用API获取判断结果）
        
        Args:
            project: 项目名称
            test_name: 测试名称
            full_code: 完整测试代码
            reasoning_guide: 推理指引文本（可选）
            
        Returns:
            (推断结果文本, 元数据字典)
        """
        user_prompt, metadata = self.generate_user_prompt(
            project, test_name, full_code, reasoning_guide
        )
        
        # 调用API获取推理过程
        reasoning = self.call_api(user_prompt, system_prompt=self.system_prompt)
        
        return reasoning, metadata
    
    def infer_from_row(self,
                      row: pd.Series,
                      reasoning_guide: Optional[str] = None) -> tuple:
        """
        从DataFrame行进行推断（便捷方法）
        
        Args:
            row: 数据行
            reasoning_guide: 推理指引文本（可选）
            
        Returns:
            (推断结果文本, 元数据字典)
        """
        project = row.get('project', 'Unknown')
        test_name = row.get('test_name', 'Unknown')
        full_code = row.get(self.code_column, row.get('full_code', ''))
        
        return self.infer(project, test_name, full_code, reasoning_guide)
    
    def run(self, 
            project: str, 
            test_name: str, 
            full_code: str, 
            reasoning_guide: Optional[str] = None,
            **kwargs) -> tuple:
        """
        执行推断任务（实现BaseAgent的抽象方法）
        
        Args:
            project: 项目名称
            test_name: 测试名称
            full_code: 完整测试代码
            reasoning_guide: 推理指引文本（可选）
            **kwargs: 其他参数（未使用）
            
        Returns:
            (推断结果文本, 元数据字典)
        """
        return self.infer(project, test_name, full_code, reasoning_guide)
