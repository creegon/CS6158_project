#!/usr/bin/env python3
"""
Flaky Test Analyzer - Command Line Interface
=============================================
Main entry point for analyzing flaky tests from command line or batch processing.

Usage:
    python main.py --code "test code string"
    python main.py --file test.java
    python main.py --dataset data.xlsx --output results.json
    python main.py --demo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analyzer import FlakyTestAnalyzer, create_llm_input, analyze_test
from src.llm_formatter import OutputFormat
from config.flaky_patterns import FlakyCategory, CATEGORY_LABELS


def analyze_single(code: str, output_format: str = "json") -> str:
    """Analyze a single test and return formatted result"""
    analyzer = FlakyTestAnalyzer()
    
    if output_format == "llm_prompt":
        return analyzer.format_for_llm(code, OutputFormat.LLM_PROMPT)
    elif output_format == "markdown":
        return analyzer.format_for_llm(code, OutputFormat.MARKDOWN)
    elif output_format == "structured_text":
        return analyzer.format_for_llm(code, OutputFormat.STRUCTURED_TEXT)
    else:
        result = analyzer.get_structured_info(code)
        return json.dumps(result, indent=2, ensure_ascii=False)


def analyze_file(filepath: str, output_format: str = "json") -> str:
    """Analyze test code from a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return analyze_single(code, output_format)


def analyze_dataset(dataset_path: str, output_path: Optional[str] = None) -> dict:
    """
    Analyze all tests in a dataset (Excel or CSV file).
    
    Expected columns: id, project, test_name, full_code, label, category
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for dataset analysis. Install with: pip install pandas openpyxl")
        sys.exit(1)
    
    # Load dataset
    if dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    print(f"Loaded {len(df)} tests from {dataset_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Analyze each test
    analyzer = FlakyTestAnalyzer()
    results = []
    
    for idx, row in df.iterrows():
        code = row.get('full_code', row.get('code', ''))
        if not code:
            continue
        
        result = analyzer.analyze(
            code=code,
            test_id=str(row.get('id', idx)),
            test_name=row.get('test_name', ''),
            project=row.get('project', ''),
            actual_category=row.get('category'),
            actual_label=row.get('label')
        )
        results.append(result)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} tests...")
    
    # Evaluate if ground truth is available
    evaluation = analyzer.evaluate(results)
    
    # Prepare output
    output = {
        "summary": {
            "total_tests": len(results),
            "evaluation": evaluation
        },
        "results": [r.to_dict() for r in results]
    }
    
    # Save output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return output


def run_demo():
    """Run a demonstration with sample test code"""
    demo_tests = [
        {
            "name": "Async Wait Example",
            "code": """@Test
public void testAsyncOperation() throws InterruptedException {
    MyService service = new MyService();
    service.startAsync();
    Thread.sleep(1000);
    assertTrue("Service should be running", service.isRunning());
}"""
        },
        {
            "name": "Concurrency Example",
            "code": """@Test
public void testConcurrentCounter() {
    AtomicInteger counter = new AtomicInteger(0);
    ExecutorService executor = Executors.newFixedThreadPool(4);
    for (int i = 0; i < 100; i++) {
        executor.submit(() -> counter.incrementAndGet());
    }
    executor.shutdown();
    executor.awaitTermination(5, TimeUnit.SECONDS);
    assertEquals(100, counter.get());
}"""
        },
        {
            "name": "Time-dependent Example",
            "code": """@Test
public void testTimestamp() {
    long before = System.currentTimeMillis();
    MyObject obj = new MyObject();
    long after = System.currentTimeMillis();
    assertTrue(obj.getCreatedAt() >= before);
    assertTrue(obj.getCreatedAt() <= after);
}"""
        },
        {
            "name": "Unordered Collection Example",
            "code": """@Test
public void testJsonSerialization() {
    Map<String, String> data = new HashMap<>();
    data.put("key1", "value1");
    data.put("key2", "value2");
    String json = JsonUtil.toJson(data);
    assertEquals("{\"key1\":\"value1\",\"key2\":\"value2\"}", json);
}"""
        },
        {
            "name": "Non-Flaky Example",
            "code": """@Test
public void testAddition() {
    Calculator calc = new Calculator();
    int result = calc.add(2, 3);
    assertEquals(5, result);
}"""
        }
    ]
    
    analyzer = FlakyTestAnalyzer()
    
    print("=" * 70)
    print("FLAKY TEST ANALYZER - DEMONSTRATION")
    print("=" * 70)
    
    for demo in demo_tests:
        print(f"\n{'='*70}")
        print(f"Test: {demo['name']}")
        print("=" * 70)
        print("\nCode:")
        print(demo['code'])
        print("\n" + "-" * 40)
        print("Analysis Result:")
        
        info = analyzer.get_structured_info(demo['code'])
        pred = info['prediction']
        
        print(f"  Predicted Category: {pred['category']} ({pred['label']})")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Non-deterministic Operations: {info['nondeterministic_operations']['count']}")
        
        ops_by_cat = info['nondeterministic_operations']['by_category']
        if ops_by_cat:
            print("  Detected Patterns:")
            for cat, ops in ops_by_cat.items():
                for op in ops[:2]:
                    print(f"    - [{cat}] Line {op['line']}: {op['pattern']} ({op['type']})")
        
        print(f"  High-risk Assertions: {info['assertion_analysis']['high_risk_count']}")
        print(f"  Tainted Variables: {list(info['variable_analysis']['tainted_variables'].keys())}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Flaky Test Analyzer - Generate structured information for LLM-based classification"
    )
    
    parser.add_argument("--code", "-c", type=str, help="Test code string to analyze")
    parser.add_argument("--file", "-f", type=str, help="Path to Java test file")
    parser.add_argument("--dataset", "-d", type=str, help="Path to dataset file (xlsx/csv)")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--format", type=str, default="json",
                       choices=["json", "llm_prompt", "markdown", "structured_text"],
                       help="Output format (default: json)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.code:
        result = analyze_single(args.code, args.format)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Output saved to {args.output}")
        else:
            print(result)
    elif args.file:
        result = analyze_file(args.file, args.format)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Output saved to {args.output}")
        else:
            print(result)
    elif args.dataset:
        analyze_dataset(args.dataset, args.output)
    else:
        parser.print_help()
        print("\nRun --demo to see examples")


if __name__ == "__main__":
    main()
