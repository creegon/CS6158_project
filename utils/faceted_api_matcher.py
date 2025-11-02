"""
Faceted API Matcher - 多维标签过滤的Few-shot检索
基于RubikSQL的Faceted Search思想扩展

核心思想：
1. 为每个训练样本提取多维度标签（facets）
2. 检索时先按facets过滤，再按相似度排序
3. 提高检索的精准度和上下文相关性

设计模式：
- 继承自 APISignatureMatcher，复用 API 提取和相似度计算逻辑
- 扩展 Facet 提取和混合检索功能
"""
import re
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from dataclasses import dataclass

# 导入父类
from .api_matcher import APISignatureMatcher


@dataclass
class CodeFacets:
    """代码的多维度标签"""
    # 并发相关
    has_concurrency: bool = False
    concurrency_types: Set[str] = None  # {'Thread', 'ExecutorService', 'Lock', ...}
    
    # 断言相关
    assert_types: Set[str] = None  # {'assertEquals', 'assertNull', 'verify', ...}
    
    # Mock相关
    has_mock: bool = False
    mock_frameworks: Set[str] = None  # {'Mockito', 'PowerMock', ...}
    
    # 时间相关
    has_timing: bool = False
    timing_apis: Set[str] = None  # {'sleep', 'TimeUnit', 'currentTimeMillis', ...}
    
    # I/O相关
    has_io: bool = False
    io_types: Set[str] = None  # {'File', 'InputStream', 'Reader', ...}
    
    # 数据库相关
    has_database: bool = False
    db_types: Set[str] = None  # {'Connection', 'Statement', 'Transaction', ...}
    
    # 集合相关
    collection_types: Set[str] = None  # {'List', 'Map', 'Set', ...}
    
    # 异常相关
    has_exception: bool = False
    exception_types: Set[str] = None  # {'NullPointerException', 'IOException', ...}
    
    # 注解相关
    test_annotations: Set[str] = None  # {'@Test', '@Before', '@RunWith', ...}
    
    def __post_init__(self):
        """初始化空集合"""
        if self.concurrency_types is None:
            self.concurrency_types = set()
        if self.assert_types is None:
            self.assert_types = set()
        if self.mock_frameworks is None:
            self.mock_frameworks = set()
        if self.timing_apis is None:
            self.timing_apis = set()
        if self.io_types is None:
            self.io_types = set()
        if self.db_types is None:
            self.db_types = set()
        if self.collection_types is None:
            self.collection_types = set()
        if self.exception_types is None:
            self.exception_types = set()
        if self.test_annotations is None:
            self.test_annotations = set()
    
    def compute_match_score(self, other: 'CodeFacets') -> float:
        """
        计算两个facets的匹配度
        
        Args:
            other: 另一个CodeFacets对象
            
        Returns:
            匹配分数 (0-1之间)
        """
        scores = []
        
        # 1. 布尔facets匹配（权重高）
        bool_matches = 0
        bool_total = 0
        
        for attr in ['has_concurrency', 'has_mock', 'has_timing', 
                     'has_io', 'has_database', 'has_exception']:
            if getattr(self, attr) or getattr(other, attr):
                bool_total += 1
                if getattr(self, attr) == getattr(other, attr):
                    bool_matches += 1
        
        if bool_total > 0:
            scores.append(bool_matches / bool_total)
        
        # 2. 集合facets的Jaccard相似度
        set_attrs = ['concurrency_types', 'assert_types', 'mock_frameworks',
                     'timing_apis', 'io_types', 'db_types', 'collection_types',
                     'exception_types', 'test_annotations']
        
        for attr in set_attrs:
            set1 = getattr(self, attr)
            set2 = getattr(other, attr)
            if set1 or set2:
                union = set1 | set2
                if len(union) > 0:
                    jaccard = len(set1 & set2) / len(union)
                    scores.append(jaccard)
        
        # 综合得分
        return np.mean(scores) if scores else 0.0


class FacetedAPISignatureMatcher(APISignatureMatcher):
    """
    基于Faceted Search的API匹配器（继承自APISignatureMatcher）
    
    扩展功能：
    - 提取多维度标签（Facets）
    - 混合相似度计算（Facet + API）
    - 支持硬过滤和多样性检索
    """
    
    def __init__(self, train_data: pd.DataFrame, code_column: str = 'full_code'):
        """
        初始化Faceted API匹配器
        
        Args:
            train_data: 训练集DataFrame
            code_column: 代码列名
        """
        # 调用父类初始化（提取 API 签名）
        super().__init__(train_data, code_column)
        
        # 额外提取 Facets
        print("正在提取 Facets...")
        self.train_facets = []
        
        for idx, row in train_data.iterrows():
            code = row[code_column]
            facets = self.extract_facets(code)
            self.train_facets.append(facets)
        
        print(f"✓ Facets 提取完成！")
        self._print_facet_statistics()
    
    # 注意：extract_apis() 和 compute_similarity() 继承自父类 APISignatureMatcher
    # 无需在此重复定义，直接使用 self.extract_apis() 和 self.compute_similarity()
    
    @staticmethod
    def extract_facets(code: str) -> CodeFacets:
        """
        从测试代码中提取多维度标签
        
        Args:
            code: 测试代码
            
        Returns:
            CodeFacets对象
        """
        facets = CodeFacets()
        
        # 1. 并发相关
        concurrency_keywords = {
            'Thread': 'Thread',
            'Runnable': 'Runnable',
            'ExecutorService': 'ExecutorService',
            'Future': 'Future',
            'CompletableFuture': 'CompletableFuture',
            'synchronized': 'synchronized',
            'volatile': 'volatile',
            'CountDownLatch': 'CountDownLatch',
            'CyclicBarrier': 'CyclicBarrier',
            'Semaphore': 'Semaphore',
            'Lock': 'Lock',
            'ReentrantLock': 'ReentrantLock',
            'AtomicInteger': 'Atomic',
            'AtomicBoolean': 'Atomic',
        }
        for keyword, tag in concurrency_keywords.items():
            if keyword in code:
                facets.has_concurrency = True
                facets.concurrency_types.add(tag)
        
        # 2. 断言相关
        assert_patterns = {
            r'\bassertEquals\b': 'assertEquals',
            r'\bassertTrue\b': 'assertTrue',
            r'\bassertFalse\b': 'assertFalse',
            r'\bassertNull\b': 'assertNull',
            r'\bassertNotNull\b': 'assertNotNull',
            r'\bassertSame\b': 'assertSame',
            r'\bverify\s*\(': 'verify',
            r'\bfail\s*\(': 'fail',
        }
        for pattern, tag in assert_patterns.items():
            if re.search(pattern, code):
                facets.assert_types.add(tag)
        
        # 3. Mock相关
        if 'Mockito' in code or 'mock(' in code or '@Mock' in code:
            facets.has_mock = True
            facets.mock_frameworks.add('Mockito')
        if 'PowerMock' in code:
            facets.has_mock = True
            facets.mock_frameworks.add('PowerMock')
        if 'EasyMock' in code:
            facets.has_mock = True
            facets.mock_frameworks.add('EasyMock')
        
        # 4. 时间相关
        timing_keywords = {
            'Thread.sleep': 'sleep',
            'TimeUnit': 'TimeUnit',
            'System.currentTimeMillis': 'currentTimeMillis',
            'System.nanoTime': 'nanoTime',
            'Calendar': 'Calendar',
            'Date': 'Date',
        }
        for keyword, tag in timing_keywords.items():
            if keyword in code:
                facets.has_timing = True
                facets.timing_apis.add(tag)
        
        # 5. I/O相关
        io_keywords = {
            'InputStream': 'InputStream',
            'OutputStream': 'OutputStream',
            'Reader': 'Reader',
            'Writer': 'Writer',
            'File': 'File',
            'BufferedReader': 'BufferedReader',
            'BufferedWriter': 'BufferedWriter',
        }
        for keyword, tag in io_keywords.items():
            if keyword in code:
                facets.has_io = True
                facets.io_types.add(tag)
        
        # 6. 数据库相关
        db_keywords = {
            'Connection': 'Connection',
            'Statement': 'Statement',
            'PreparedStatement': 'PreparedStatement',
            'ResultSet': 'ResultSet',
            'Transaction': 'Transaction',
            'EntityManager': 'EntityManager',
        }
        for keyword, tag in db_keywords.items():
            if keyword in code:
                facets.has_database = True
                facets.db_types.add(tag)
        
        # 7. 集合相关
        collection_patterns = {
            r'\bList\b': 'List',
            r'\bSet\b': 'Set',
            r'\bMap\b': 'Map',
            r'\bArrayList\b': 'ArrayList',
            r'\bHashMap\b': 'HashMap',
            r'\bHashSet\b': 'HashSet',
        }
        for pattern, tag in collection_patterns.items():
            if re.search(pattern, code):
                facets.collection_types.add(tag)
        
        # 8. 异常相关
        exception_patterns = {
            r'\bthrows\s+(\w+Exception)': 'declared',
            r'\bcatch\s*\(\s*(\w+Exception)': 'caught',
            r'new\s+(\w+Exception)': 'thrown',
        }
        for pattern, tag in exception_patterns.items():
            matches = re.findall(pattern, code)
            if matches:
                facets.has_exception = True
                for match in matches:
                    if isinstance(match, str):
                        facets.exception_types.add(match)
        
        # 9. 注解相关
        test_annotations = ['@Test', '@Before', '@After', '@BeforeClass', 
                           '@AfterClass', '@RunWith', '@Mock', '@InjectMocks']
        for annotation in test_annotations:
            if annotation in code:
                facets.test_annotations.add(annotation)
        
        return facets
    
    # compute_similarity() 继承自父类 APISignatureMatcher，无需重复定义
    
    def retrieve_top_k(self, 
                      test_code: str, 
                      top_k: int = 3,
                      facet_weight: float = 0.3,
                      api_weight: float = 0.7,
                      min_similarity: float = 0.0,
                      require_facet_match: bool = False) -> List[Tuple[int, float, pd.Series]]:
        """
        检索最相似的K个训练案例（Faceted Search增强版）
        
        Args:
            test_code: 测试代码
            top_k: 返回最相似的K个案例
            facet_weight: Facet匹配的权重
            api_weight: API匹配的权重
            min_similarity: 最小相似度阈值
            require_facet_match: 是否要求Facet必须匹配（硬过滤）
            
        Returns:
            [(索引, 相似度, 数据行), ...] 列表
        """
        # 提取测试代码的API和Facets
        test_apis = self.extract_apis(test_code)
        test_facets = self.extract_facets(test_code)
        
        # 计算与所有训练样本的综合相似度
        scores = []
        for idx, (train_apis, train_facets) in enumerate(zip(self.train_apis, self.train_facets)):
            # Facet匹配分数
            facet_score = test_facets.compute_match_score(train_facets)
            
            # 如果要求硬过滤且facet不匹配，跳过
            if require_facet_match and facet_score < 0.3:
                continue
            
            # API相似度分数
            api_score = self.compute_similarity(test_apis, train_apis)
            
            # 综合得分
            final_score = facet_weight * facet_score + api_weight * api_score
            
            if final_score >= min_similarity:
                scores.append((idx, final_score, facet_score, api_score))
        
        # 按综合得分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回Top-K（包含完整数据行）
        results = []
        for idx, final_score, facet_score, api_score in scores[:top_k]:
            data_row = self.train_data.iloc[idx]
            results.append((idx, final_score, data_row))
        
        return results
    
    def retrieve_with_diversity(self,
                               test_code: str,
                               top_k: int = 3,
                               diversity_threshold: float = 0.3,
                               facet_weight: float = 0.3) -> List[Tuple[int, float, pd.Series]]:
        """
        检索最相似且多样化的K个训练案例（Faceted增强版）
        
        Args:
            test_code: 测试代码
            top_k: 返回最相似的K个案例
            diversity_threshold: 多样性阈值
            facet_weight: Facet权重
            
        Returns:
            [(索引, 相似度, 数据行), ...] 列表
        """
        # 提取特征
        test_apis = self.extract_apis(test_code)
        test_facets = self.extract_facets(test_code)
        
        # 计算候选案例
        candidates = []
        for idx, (train_apis, train_facets) in enumerate(zip(self.train_apis, self.train_facets)):
            facet_score = test_facets.compute_match_score(train_facets)
            api_score = self.compute_similarity(test_apis, train_apis)
            final_score = facet_weight * facet_score + (1 - facet_weight) * api_score
            
            candidates.append((idx, final_score, train_apis, train_facets))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 贪心选择多样化案例
        selected = []
        for idx, similarity, train_apis, train_facets in candidates:
            if len(selected) >= top_k:
                break
            
            # 检查多样性
            is_diverse = True
            for selected_idx, _, selected_apis, selected_facets in selected:
                # API多样性
                api_sim = self.compute_similarity(train_apis, selected_apis)
                # Facet多样性
                facet_sim = train_facets.compute_match_score(selected_facets)
                
                if api_sim > diversity_threshold or facet_sim > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append((idx, similarity, train_apis, train_facets))
        
        # 如果不足K个，补充高相似度案例
        if len(selected) < top_k:
            for idx, similarity, train_apis, train_facets in candidates:
                if len(selected) >= top_k:
                    break
                if idx not in [s[0] for s in selected]:
                    selected.append((idx, similarity, train_apis, train_facets))
        
        # 返回结果
        results = []
        for idx, similarity, _, _ in selected:
            data_row = self.train_data.iloc[idx]
            results.append((idx, similarity, data_row))
        
        return results
    
    def _print_facet_statistics(self):
        """打印Facet统计信息"""
        stats = {
            'has_concurrency': sum(f.has_concurrency for f in self.train_facets),
            'has_mock': sum(f.has_mock for f in self.train_facets),
            'has_timing': sum(f.has_timing for f in self.train_facets),
            'has_io': sum(f.has_io for f in self.train_facets),
            'has_database': sum(f.has_database for f in self.train_facets),
            'has_exception': sum(f.has_exception for f in self.train_facets),
        }
        
        print("\n✓ Facet统计:")
        for key, count in stats.items():
            percentage = count / len(self.train_facets) * 100
            print(f"  - {key}: {count} ({percentage:.1f}%)")
    
    def get_statistics(self) -> Dict:
        """
        获取详细统计信息（重写父类方法，添加Facet统计）
        """
        # 调用父类方法获取基础统计
        base_stats = super().get_statistics()
        
        # 添加 Facet 统计
        facet_stats = {
            'has_concurrency': sum(f.has_concurrency for f in self.train_facets),
            'has_mock': sum(f.has_mock for f in self.train_facets),
            'has_timing': sum(f.has_timing for f in self.train_facets),
            'has_io': sum(f.has_io for f in self.train_facets),
            'has_database': sum(f.has_database for f in self.train_facets),
        }
        
        # 合并统计信息
        base_stats['facet_distribution'] = facet_stats
        return base_stats
