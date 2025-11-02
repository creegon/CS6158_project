"""
Faceted Search 使用示例

演示如何使用多维标签过滤来增强Few-shot检索
对比原版API Matcher和Faceted版本的效果差异
"""
from pathlib import Path
import pandas as pd
from utils import load_csv
from utils.api_matcher import APISignatureMatcher
from utils.faceted_api_matcher import FacetedAPISignatureMatcher, CodeFacets


def demo_facet_extraction():
    """演示Facet提取功能"""
    print("=" * 80)
    print("演示1: Facet提取功能")
    print("=" * 80)
    
    # 测试代码1: 并发测试
    code1 = """
    @Test
    public void testConcurrency() {
        ExecutorService executor = Executors.newFixedThreadPool(10);
        CountDownLatch latch = new CountDownLatch(10);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                try {
                    Thread.sleep(100);
                    assertEquals(1, service.getValue());
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await(5, TimeUnit.SECONDS);
        verify(mockService).process();
    }
    """
    
    print("\n【代码1】并发测试")
    print(code1[:200] + "...")
    
    facets1 = FacetedAPISignatureMatcher.extract_facets(code1)
    print("\n提取的Facets:")
    print(f"  has_concurrency: {facets1.has_concurrency}")
    print(f"  concurrency_types: {facets1.concurrency_types}")
    print(f"  has_timing: {facets1.has_timing}")
    print(f"  timing_apis: {facets1.timing_apis}")
    print(f"  has_mock: {facets1.has_mock}")
    print(f"  assert_types: {facets1.assert_types}")
    
    # 测试代码2: I/O测试
    code2 = """
    @Test
    public void testFileRead() throws IOException {
        File file = new File("test.txt");
        BufferedReader reader = new BufferedReader(new FileReader(file));
        
        String line = reader.readLine();
        assertNotNull(line);
        assertEquals("Hello", line);
        
        reader.close();
    }
    """
    
    print("\n\n【代码2】I/O测试")
    print(code2[:200] + "...")
    
    facets2 = FacetedAPISignatureMatcher.extract_facets(code2)
    print("\n提取的Facets:")
    print(f"  has_io: {facets2.has_io}")
    print(f"  io_types: {facets2.io_types}")
    print(f"  has_exception: {facets2.has_exception}")
    print(f"  assert_types: {facets2.assert_types}")
    
    # 计算两个代码的Facet相似度
    similarity = facets1.compute_match_score(facets2)
    print(f"\n代码1和代码2的Facet相似度: {similarity:.3f}")
    print("(预期很低，因为一个是并发测试，一个是I/O测试)")


def demo_comparison():
    """演示原版API Matcher vs Faceted版本的对比"""
    print("\n\n" + "=" * 80)
    print("演示2: 原版 vs Faceted版本对比")
    print("=" * 80)
    
    # 加载数据集
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'FlakyLens_dataset_with_nonflaky_indented.csv'
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    print("\n加载数据集...")
    data = load_csv(dataset_path)
    print(f"✓ 数据集大小: {len(data)}")
    
    # 划分训练集和测试集
    train_data = data.head(500)  # 前500条作为训练集
    test_data = data.iloc[500:510]  # 10条测试
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 测试集: {len(test_data)} 条")
    
    # 构建两种匹配器
    print("\n\n【构建原版API Matcher】")
    original_matcher = APISignatureMatcher(train_data, code_column='full_code')
    
    print("\n【构建Faceted API Matcher】")
    faceted_matcher = FacetedAPISignatureMatcher(train_data, code_column='full_code')
    
    # 选一个测试样本
    test_sample = test_data.iloc[0]
    test_code = test_sample['full_code']
    test_label = test_sample.get('category', test_sample.get('label', 'Unknown'))
    
    print("\n\n" + "-" * 80)
    print("【测试样本】")
    print(f"项目: {test_sample.get('project', 'Unknown')}")
    print(f"真实标签: {test_label}")
    print(f"代码长度: {len(test_code)} 字符")
    print(f"代码预览:\n{test_code[:300]}...")
    
    # 提取测试样本的Facets
    test_facets = FacetedAPISignatureMatcher.extract_facets(test_code)
    print("\n测试样本的Facets:")
    print(f"  并发: {test_facets.has_concurrency} {test_facets.concurrency_types}")
    print(f"  Mock: {test_facets.has_mock} {test_facets.mock_frameworks}")
    print(f"  时间: {test_facets.has_timing} {test_facets.timing_apis}")
    print(f"  I/O: {test_facets.has_io} {test_facets.io_types}")
    print(f"  数据库: {test_facets.has_database} {test_facets.db_types}")
    
    # 原版检索
    print("\n\n" + "=" * 80)
    print("【原版API Matcher检索结果】(纯API相似度)")
    print("=" * 80)
    
    original_results = original_matcher.retrieve_top_k(test_code, top_k=5)
    
    for i, (idx, similarity, row) in enumerate(original_results, 1):
        print(f"\n案例 {i}:")
        print(f"  相似度: {similarity:.3f}")
        print(f"  项目: {row.get('project', 'Unknown')}")
        print(f"  标签: {row.get('category', row.get('label', 'Unknown'))}")
        
        # 显示该案例的Facets
        case_facets = FacetedAPISignatureMatcher.extract_facets(row['full_code'])
        print(f"  Facets: 并发={case_facets.has_concurrency}, "
              f"Mock={case_facets.has_mock}, "
              f"时间={case_facets.has_timing}, "
              f"I/O={case_facets.has_io}")
        print(f"  代码预览: {row['full_code'][:150]}...")
    
    # Faceted版本检索
    print("\n\n" + "=" * 80)
    print("【Faceted API Matcher检索结果】(Facet 30% + API 70%)")
    print("=" * 80)
    
    faceted_results = faceted_matcher.retrieve_top_k(
        test_code, 
        top_k=5,
        facet_weight=0.3,
        api_weight=0.7
    )
    
    for i, (idx, similarity, row) in enumerate(faceted_results, 1):
        print(f"\n案例 {i}:")
        print(f"  综合相似度: {similarity:.3f}")
        print(f"  项目: {row.get('project', 'Unknown')}")
        print(f"  标签: {row.get('category', row.get('label', 'Unknown'))}")
        
        # 显示该案例的Facets
        case_facets = FacetedAPISignatureMatcher.extract_facets(row['full_code'])
        facet_match = test_facets.compute_match_score(case_facets)
        print(f"  Facet匹配度: {facet_match:.3f}")
        print(f"  Facets: 并发={case_facets.has_concurrency}, "
              f"Mock={case_facets.has_mock}, "
              f"时间={case_facets.has_timing}, "
              f"I/O={case_facets.has_io}")
        print(f"  代码预览: {row['full_code'][:150]}...")
    
    # 对比分析
    print("\n\n" + "=" * 80)
    print("【对比分析】")
    print("=" * 80)
    
    print("\n观察:")
    print("1. 原版可能检索到API重叠但场景不同的案例")
    print("2. Faceted版本会优先考虑场景相似性（并发/Mock/I/O等）")
    print("3. 如果测试样本有明显的Facet特征，Faceted版本的案例更有针对性")


def demo_hard_filter():
    """演示硬过滤模式（require_facet_match）"""
    print("\n\n" + "=" * 80)
    print("演示3: 硬过滤模式 - 只检索Facet匹配的案例")
    print("=" * 80)
    
    # 加载数据
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'FlakyLens_dataset_with_nonflaky_indented.csv'
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    data = load_csv(dataset_path)
    train_data = data.head(500)
    
    print("构建Faceted Matcher...")
    matcher = FacetedAPISignatureMatcher(train_data, code_column='full_code')
    
    # 构造一个明确的并发测试
    concurrent_test = """
    @Test
    public void testThreadSafety() {
        ExecutorService executor = Executors.newFixedThreadPool(10);
        CountDownLatch latch = new CountDownLatch(10);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                Thread.sleep(100);
                counter.increment();
                latch.countDown();
            });
        }
        
        latch.await();
        assertEquals(10, counter.getValue());
    }
    """
    
    print("\n【测试代码】明确的并发测试")
    print(concurrent_test)
    
    # 普通检索
    print("\n【普通检索】(可能包含非并发案例)")
    normal_results = matcher.retrieve_top_k(
        concurrent_test,
        top_k=5,
        require_facet_match=False
    )
    
    for i, (idx, similarity, row) in enumerate(normal_results, 1):
        case_facets = FacetedAPISignatureMatcher.extract_facets(row['full_code'])
        print(f"  案例{i}: 相似度={similarity:.3f}, "
              f"并发={case_facets.has_concurrency}, "
              f"项目={row.get('project', 'Unknown')[:30]}")
    
    # 硬过滤检索
    print("\n【硬过滤检索】(只要并发相关案例)")
    filtered_results = matcher.retrieve_top_k(
        concurrent_test,
        top_k=5,
        require_facet_match=True  # 要求Facet匹配度 >= 0.3
    )
    
    for i, (idx, similarity, row) in enumerate(filtered_results, 1):
        case_facets = FacetedAPISignatureMatcher.extract_facets(row['full_code'])
        print(f"  案例{i}: 相似度={similarity:.3f}, "
              f"并发={case_facets.has_concurrency}, "
              f"项目={row.get('project', 'Unknown')[:30]}")
    
    print("\n✓ 硬过滤模式确保检索到的案例与查询在Facet维度上匹配")


def demo_diversity():
    """演示多样性检索"""
    print("\n\n" + "=" * 80)
    print("演示4: 多样性检索 - 避免检索相似的重复案例")
    print("=" * 80)
    
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'FlakyLens_dataset_with_nonflaky_indented.csv'
    
    if not dataset_path.exists():
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    data = load_csv(dataset_path)
    train_data = data.head(500)
    
    print("构建Faceted Matcher...")
    matcher = FacetedAPISignatureMatcher(train_data, code_column='full_code')
    
    test_code = train_data.iloc[0]['full_code']
    
    print("\n【标准检索】(可能选到相似案例)")
    standard_results = matcher.retrieve_top_k(test_code, top_k=5)
    
    print("\n案例相似度矩阵:")
    for i, (idx_i, _, row_i) in enumerate(standard_results, 1):
        similarities = []
        for j, (idx_j, _, row_j) in enumerate(standard_results, 1):
            if i == j:
                similarities.append("  -  ")
            else:
                apis_i = matcher.extract_apis(row_i['full_code'])
                apis_j = matcher.extract_apis(row_j['full_code'])
                sim = matcher.compute_similarity(apis_i, apis_j)
                similarities.append(f"{sim:.2f}")
        print(f"  案例{i}: {' '.join(similarities)}")
    
    print("\n【多样性检索】(强制案例间差异)")
    diverse_results = matcher.retrieve_with_diversity(
        test_code, 
        top_k=5,
        diversity_threshold=0.3  # 案例间相似度要 < 0.3
    )
    
    print("\n案例相似度矩阵:")
    for i, (idx_i, _, row_i) in enumerate(diverse_results, 1):
        similarities = []
        for j, (idx_j, _, row_j) in enumerate(diverse_results, 1):
            if i == j:
                similarities.append("  -  ")
            else:
                apis_i = matcher.extract_apis(row_i['full_code'])
                apis_j = matcher.extract_apis(row_j['full_code'])
                sim = matcher.compute_similarity(apis_i, apis_j)
                similarities.append(f"{sim:.2f}")
        print(f"  案例{i}: {' '.join(similarities)}")
    
    print("\n✓ 多样性检索确保选出的案例彼此不同，提供更丰富的参考信息")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Faceted Search Few-shot 检索演示")
    print("=" * 80)
    
    # 运行所有演示
    demo_facet_extraction()
    
    try:
        demo_comparison()
    except Exception as e:
        print(f"\n⚠ demo_comparison 执行失败: {e}")
    
    try:
        demo_hard_filter()
    except Exception as e:
        print(f"\n⚠ demo_hard_filter 执行失败: {e}")
    
    try:
        demo_diversity()
    except Exception as e:
        print(f"\n⚠ demo_diversity 执行失败: {e}")
    
    print("\n\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n下一步:")
    print("1. 在 distillation_agent.py 中集成 FacetedAPISignatureMatcher")
    print("2. 运行完整的数据蒸馏实验，对比效果")
    print("3. 分析不同 facet_weight 参数的影响")
    print("4. (可选) 进一步扩展到 Multi-vector 混合检索")
