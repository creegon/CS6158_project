"""
API签名匹配索引模块
用于从训练集中检索最相似的测试案例作为few-shot examples
"""
import re
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional
import numpy as np


class APISignatureMatcher:
    """API签名匹配器，用于检索最相似的代码案例"""
    
    def __init__(self, train_data: pd.DataFrame, code_column: str = 'full_code'):
        """
        初始化API签名匹配器
        
        Args:
            train_data: 训练集DataFrame
            code_column: 代码列名
        """
        self.train_data = train_data
        self.code_column = code_column
        
        # 预先提取所有训练集的API
        print("正在预处理训练集，提取API签名...")
        self.train_apis = []
        for idx, row in train_data.iterrows():
            code = row[code_column]
            apis = self.extract_apis(code)
            self.train_apis.append(apis)
        print(f"✓ 完成！共处理 {len(self.train_apis)} 个训练样本")
    
    @staticmethod
    def extract_apis(code: str) -> List[str]:
        """
        从测试代码中提取关键API调用
        
        Args:
            code: 测试代码
            
        Returns:
            API列表
        """
        apis = []
        
        # 规则1: Java方法调用 (object.method())
        # 匹配形如 obj.method()、Class.staticMethod() 等
        apis.extend(re.findall(r'\b(\w+)\.\w+\s*\(', code))
        
        # 规则2: 测试框架注解
        test_annotations = ['@Test', '@Before', '@After', '@BeforeClass', '@AfterClass',
                           '@RunWith', '@Mock', '@InjectMocks', '@Spy']
        for annotation in test_annotations:
            if annotation in code:
                apis.append(annotation)
        
        # 规则3: 断言API (JUnit/TestNG)
        assertion_patterns = [
            r'\bassert(True|False|Equals|NotEquals|Null|NotNull|Same|NotSame|ArrayEquals|Throws)',
            r'\bfail\s*\(',
            r'\bverify\s*\(',
            r'\bwhen\s*\(',
            r'\bthenReturn\s*\(',
            r'\bmock\s*\(',
        ]
        for pattern in assertion_patterns:
            matches = re.findall(pattern, code)
            apis.extend([f'assert_{m}' if isinstance(m, str) and m else 'assert' for m in matches])
        
        # 规则4: 异步/并发关键字
        async_keywords = ['Thread', 'Runnable', 'ExecutorService', 'Future', 'CompletableFuture',
                         'synchronized', 'volatile', 'CountDownLatch', 'CyclicBarrier',
                         'Semaphore', 'Lock', 'ReentrantLock']
        for keyword in async_keywords:
            if keyword in code:
                apis.append(f'async_{keyword}')
        
        # 规则5: 时间相关API
        time_apis = ['Thread.sleep', 'TimeUnit', 'System.currentTimeMillis', 'System.nanoTime',
                    'Calendar', 'Date', 'LocalDateTime', 'Instant']
        for time_api in time_apis:
            if time_api in code:
                apis.append(f'time_{time_api.replace(".", "_")}')
        
        # 规则6: 集合操作
        collection_patterns = [
            r'\bList\b', r'\bSet\b', r'\bMap\b', r'\bQueue\b',
            r'\bArrayList\b', r'\bHashSet\b', r'\bHashMap\b',
            r'\bLinkedList\b', r'\bTreeSet\b', r'\bTreeMap\b',
        ]
        for pattern in collection_patterns:
            if re.search(pattern, code):
                apis.append(pattern.replace(r'\b', '').replace(r'\\b', ''))
        
        # 规则7: I/O操作
        io_keywords = ['InputStream', 'OutputStream', 'Reader', 'Writer', 'File',
                      'FileInputStream', 'FileOutputStream', 'BufferedReader', 'BufferedWriter']
        for keyword in io_keywords:
            if keyword in code:
                apis.append(f'io_{keyword}')
        
        # 规则8: Mock框架
        if 'Mockito' in code or 'mock(' in code or '@Mock' in code:
            apis.append('mock_framework')
        if 'PowerMock' in code:
            apis.append('powermock')
        
        # 规则9: 数据库相关
        db_keywords = ['Connection', 'Statement', 'PreparedStatement', 'ResultSet',
                      'Transaction', 'EntityManager', 'Session']
        for keyword in db_keywords:
            if keyword in code:
                apis.append(f'db_{keyword}')
        
        return apis
    
    def compute_similarity(self, test_apis: List[str], train_apis: List[str]) -> float:
        """
        计算两个API列表的相似度
        
        Args:
            test_apis: 测试代码的API列表
            train_apis: 训练代码的API列表
            
        Returns:
            相似度分数 (0-1之间)
        """
        test_set = set(test_apis)
        train_set = set(train_apis)
        
        # Jaccard相似度
        if len(test_set | train_set) == 0:
            return 0.0
        
        jaccard = len(test_set & train_set) / len(test_set | train_set)
        
        # 考虑API频率的加权相似度
        test_counter = Counter(test_apis)
        train_counter = Counter(train_apis)
        
        common_apis = test_set & train_set
        if len(common_apis) > 0:
            # 计算频率加权分数
            freq_score = sum(min(test_counter[api], train_counter[api]) for api in common_apis)
            total_freq = sum(test_counter.values()) + sum(train_counter.values())
            freq_weight = freq_score / total_freq if total_freq > 0 else 0
            
            # 综合Jaccard和频率权重
            final_score = 0.6 * jaccard + 0.4 * freq_weight
        else:
            final_score = jaccard
        
        return final_score
    
    def retrieve_top_k(self, test_code: str, top_k: int = 3, 
                      min_similarity: float = 0.0) -> List[Tuple[int, float, pd.Series]]:
        """
        检索最相似的K个训练案例
        
        Args:
            test_code: 测试代码
            top_k: 返回最相似的K个案例
            min_similarity: 最小相似度阈值
            
        Returns:
            [(索引, 相似度, 数据行), ...] 列表
        """
        # 提取测试代码的API
        test_apis = self.extract_apis(test_code)
        
        # 计算与所有训练样本的相似度
        scores = []
        for idx, train_apis in enumerate(self.train_apis):
            similarity = self.compute_similarity(test_apis, train_apis)
            if similarity >= min_similarity:
                scores.append((idx, similarity))
        
        # 按相似度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回Top-K（包含完整数据行）
        results = []
        for idx, similarity in scores[:top_k]:
            data_row = self.train_data.iloc[idx]
            results.append((idx, similarity, data_row))
        
        return results
    
    def retrieve_with_diversity(self, test_code: str, top_k: int = 3,
                                diversity_threshold: float = 0.3) -> List[Tuple[int, float, pd.Series]]:
        """
        检索最相似且多样化的K个训练案例
        通过避免选择彼此过于相似的案例来增加多样性
        
        Args:
            test_code: 测试代码
            top_k: 返回最相似的K个案例
            diversity_threshold: 多样性阈值，候选案例之间的相似度应低于此值
            
        Returns:
            [(索引, 相似度, 数据行), ...] 列表
        """
        # 提取测试代码的API
        test_apis = self.extract_apis(test_code)
        
        # 计算与所有训练样本的相似度
        candidates = []
        for idx, train_apis in enumerate(self.train_apis):
            similarity = self.compute_similarity(test_apis, train_apis)
            candidates.append((idx, similarity, train_apis))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 贪心选择多样化的Top-K
        selected = []
        for idx, similarity, train_apis in candidates:
            if len(selected) >= top_k:
                break
            
            # 检查与已选择案例的多样性
            is_diverse = True
            for selected_idx, _, selected_apis in selected:
                inter_similarity = self.compute_similarity(train_apis, selected_apis)
                if inter_similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append((idx, similarity, train_apis))
        
        # 如果多样性选择不足K个，补充相似度最高的
        if len(selected) < top_k:
            for idx, similarity, train_apis in candidates:
                if len(selected) >= top_k:
                    break
                if idx not in [s[0] for s in selected]:
                    selected.append((idx, similarity, train_apis))
        
        # 返回结果（包含完整数据行）
        results = []
        for idx, similarity, _ in selected:
            data_row = self.train_data.iloc[idx]
            results.append((idx, similarity, data_row))
        
        return results
    
    def batch_retrieve(self, test_codes: List[str], top_k: int = 3,
                      use_diversity: bool = False) -> List[List[Tuple[int, float, pd.Series]]]:
        """
        批量检索
        
        Args:
            test_codes: 测试代码列表
            top_k: 每个测试返回的案例数
            use_diversity: 是否使用多样性策略
            
        Returns:
            每个测试对应的Top-K结果列表
        """
        results = []
        for test_code in test_codes:
            if use_diversity:
                top_k_results = self.retrieve_with_diversity(test_code, top_k)
            else:
                top_k_results = self.retrieve_top_k(test_code, top_k)
            results.append(top_k_results)
        return results
    
    def get_statistics(self) -> Dict:
        """
        获取匹配器的统计信息
        
        Returns:
            统计信息字典
        """
        all_apis = []
        for apis in self.train_apis:
            all_apis.extend(apis)
        
        api_counter = Counter(all_apis)
        
        return {
            'total_train_samples': len(self.train_apis),
            'total_unique_apis': len(api_counter),
            'most_common_apis': api_counter.most_common(10),
            'avg_apis_per_sample': np.mean([len(apis) for apis in self.train_apis]),
        }
