"""
特征匹配和提示生成工具

根据测试代码中的关键词,匹配对应的特征等级,
并生成强调提示用于 prompt
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class FeatureMatcher:
    """特征匹配器"""
    
    def __init__(self, lookup_table_path: str):
        """
        初始化匹配器
        
        Args:
            lookup_table_path: 特征查找表路径
        """
        with open(lookup_table_path, 'r', encoding='utf-8') as f:
            self.lookup_table = json.load(f)
        
        # 将 'inf' 字符串转回 float('inf')
        for feature, occurrences in self.lookup_table.items():
            for occ in occurrences:
                if occ['discrimination'] == 'inf':
                    occ['discrimination'] = float('inf')
        
        print(f"✓ 加载了 {len(self.lookup_table)} 个特征")
    
    def extract_features(self, code: str) -> Set[str]:
        """
        从代码中提取所有可能的特征
        
        Args:
            code: 测试代码
            
        Returns:
            特征集合
        """
        # 提取标识符
        identifiers = re.findall(r'\b[A-Za-z][a-zA-Z0-9]*\b', code)
        
        # 过滤Java关键字
        java_keywords = {
            'public', 'private', 'protected', 'static', 'final', 'void', 'return',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
            'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super',
            'class', 'interface', 'extends', 'implements', 'import', 'package',
            'int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short',
            'true', 'false', 'null', 'var', 'const', 'goto', 'assert', 'enum'
        }
        
        features = {id for id in identifiers if id not in java_keywords and len(id) >= 3}
        return features
    
    def match_features(self, code: str) -> Dict[str, List[Dict]]:
        """
        匹配代码中的特征
        
        Args:
            code: 测试代码
            
        Returns:
            {类别: [{特征信息}, ...]}
        """
        features_in_code = self.extract_features(code)
        
        # 按类别和等级组织匹配结果
        matches = defaultdict(lambda: {
            'unique': [],
            'very_strong': [],
            'strong': [],
            'moderate': []
        })
        
        for feature in features_in_code:
            if feature in self.lookup_table:
                for occurrence in self.lookup_table[feature]:
                    level = occurrence['level']
                    if level in ['unique', 'very_strong', 'strong', 'moderate']:
                        matches[occurrence['category']][level].append({
                            'feature': feature,
                            **occurrence
                        })
        
        return dict(matches)
    
    def generate_prompt_hint(self, code: str, max_features_per_level: int = 3) -> str:
        """
        生成 prompt 提示
        
        Args:
            code: 测试代码
            max_features_per_level: 每个等级最多显示的特征数
            
        Returns:
            prompt 提示文本
        """
        matches = self.match_features(code)
        
        if not matches:
            return ""
        
        hint_parts = []
        hint_parts.append("\n【关键特征分析】")
        
        for category, levels in matches.items():
            category_hints = []
            
            # 优先级: unique > very_strong > strong > moderate
            # 如果找到高优先级的,就不再处理低优先级的
            
            if levels['unique']:
                # 独有特征 - 最强信号
                features = sorted(levels['unique'], 
                                key=lambda x: x['flaky_density'], 
                                reverse=True)[:max_features_per_level]
                
                feature_names = [f['feature'] for f in features]
                category_hints.append(
                    f"发现 **独有特征**: {', '.join(feature_names)} - "
                    f"这些关键词几乎是 '{category}' 的独有特征,在 non-flaky 测试中完全不存在"
                )
            
            elif levels['very_strong']:
                # 极强区分特征
                features = sorted(levels['very_strong'],
                                key=lambda x: (x['discrimination'], x['flaky_density']),
                                reverse=True)[:max_features_per_level]
                
                feature_info = [
                    f"{f['feature']}({f['discrimination']:.1f}x)" 
                    if f['discrimination'] != float('inf') 
                    else f"{f['feature']}(∞)" 
                    for f in features
                ]
                
                category_hints.append(
                    f"发现 **极强区分特征** (20x+): {', '.join(feature_info)} - "
                    f"这些关键词在 '{category}' 中的出现频率远高于 non-flaky 测试"
                )
            
            elif levels['strong']:
                # 强区分特征
                features = sorted(levels['strong'],
                                key=lambda x: (x['discrimination'], x['flaky_density']),
                                reverse=True)[:max_features_per_level]
                
                feature_info = [f"{f['feature']}({f['discrimination']:.1f}x)" for f in features]
                
                category_hints.append(
                    f"发现 **强区分特征** (10-20x): {', '.join(feature_info)} - "
                    f"这些关键词强烈指向 '{category}'"
                )
            
            elif levels['moderate']:
                # 中等区分特征
                features = sorted(levels['moderate'],
                                key=lambda x: (x['discrimination'], x['flaky_density']),
                                reverse=True)[:max_features_per_level]
                
                feature_info = [f"{f['feature']}({f['discrimination']:.1f}x)" for f in features]
                
                category_hints.append(
                    f"发现 **中等区分特征** (5-10x): {', '.join(feature_info)} - "
                    f"这些关键词可能指向 '{category}'"
                )
            
            if category_hints:
                hint_parts.append(f"\n[{category}]")
                hint_parts.extend([f"  • {hint}" for hint in category_hints])
        
        return "\n".join(hint_parts)
    
    def generate_structured_hints(self, code: str) -> Dict[str, Dict[str, List[str]]]:
        """
        生成结构化的提示信息
        
        Args:
            code: 测试代码
            
        Returns:
            {类别: {等级: [特征列表]}}
        """
        matches = self.match_features(code)
        
        structured = {}
        for category, levels in matches.items():
            structured[category] = {}
            
            for level in ['unique', 'very_strong', 'strong', 'moderate']:
                if levels[level]:
                    features = sorted(levels[level],
                                    key=lambda x: (x['discrimination'], x['flaky_density']),
                                    reverse=True)
                    structured[category][level] = [f['feature'] for f in features]
        
        return structured


def demo_usage():
    """演示用法"""
    import sys
    
    # 查找特征查找表
    lookup_path = Path(__file__).parent / 'output' / 'facet_analysis' / 'feature_lookup_table.json'
    
    if not lookup_path.exists():
        print(f"错误: 找不到特征查找表: {lookup_path}")
        print("请先运行 analyze_normalized_features.py 生成特征查找表")
        return
    
    # 初始化匹配器
    matcher = FeatureMatcher(str(lookup_path))
    
    # 测试代码示例
    test_codes = [
        # async wait 相关
        """
        @Test
        public void testAsyncExecution() {
            CountDownLatch latch = new CountDownLatch(1);
            Thread.sleep(1000);
            await().atMost(5, TimeUnit.SECONDS)
                   .until(() -> service.isReady());
            latch.await(10, TimeUnit.SECONDS);
        }
        """,
        
        # test order dependency 相关
        """
        @Test
        public void testNamingContext() {
            CompositeName name = new CompositeName("test");
            context.lookup(name);
            context.bind("key", value);
        }
        """,
        
        # time 相关
        """
        @Test
        public void testTimeSensitive() {
            long now = System.currentTimeMillis();
            Path file = fileSys.getPath("/test");
            long modTime = file.getModificationTime();
            assertEquals(expected, now);
        }
        """
    ]
    
    print("\n" + "=" * 80)
    print("特征匹配示例")
    print("=" * 80)
    
    for i, code in enumerate(test_codes, 1):
        print(f"\n{'=' * 80}")
        print(f"测试代码 {i}:")
        print(f"{'=' * 80}")
        print(code.strip())
        
        # 生成提示
        hint = matcher.generate_prompt_hint(code, max_features_per_level=5)
        print(hint)
        
        # 生成结构化信息
        structured = matcher.generate_structured_hints(code)
        print(f"\n结构化结果: {json.dumps(structured, indent=2, ensure_ascii=False)}")


if __name__ == '__main__':
    demo_usage()
