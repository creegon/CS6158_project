#!/usr/bin/env python3
"""
Usage Examples
==============
Demonstrates how to use the Flaky Test Analyzer.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analyzer import FlakyTestAnalyzer, create_llm_input, analyze_test
from src.llm_formatter import OutputFormat


# Example test codes for each category
EXAMPLE_TESTS = {
    "async_wait": {
        "code": """@Test
public void testAsyncCallback() throws Exception {
    CountDownLatch latch = new CountDownLatch(1);
    AtomicBoolean success = new AtomicBoolean(false);
    
    service.asyncOperation(result -> {
        success.set(result.isSuccess());
        latch.countDown();
    });
    
    latch.await(5, TimeUnit.SECONDS);
    assertTrue("Operation should succeed", success.get());
}""",
        "expected_category": 0
    },
    
    "concurrency": {
        "code": """@Test
public void testConcurrentModification() {
    AtomicInteger counter = new AtomicInteger(0);
    List<Thread> threads = new ArrayList<>();
    
    for (int i = 0; i < 10; i++) {
        Thread t = new Thread(() -> {
            for (int j = 0; j < 100; j++) {
                counter.incrementAndGet();
            }
        });
        threads.add(t);
        t.start();
    }
    
    for (Thread t : threads) {
        t.join();
    }
    
    assertEquals(1000, counter.get());
}""",
        "expected_category": 1
    },
    
    "time": {
        "code": """@Test
public void testTimestampOrdering() {
    long before = System.currentTimeMillis();
    Event event = new Event("test");
    long after = System.currentTimeMillis();
    
    assertTrue(event.getTimestamp() >= before);
    assertTrue(event.getTimestamp() <= after);
}""",
        "expected_category": 2
    },
    
    "unordered_collection": {
        "code": """@Test
public void testMapSerialization() {
    Map<String, Integer> data = new HashMap<>();
    data.put("a", 1);
    data.put("b", 2);
    data.put("c", 3);
    
    String json = serializer.toJson(data);
    assertEquals("{\"a\":1,\"b\":2,\"c\":3}", json);
}""",
        "expected_category": 3
    },
    
    "test_order_dependency": {
        "code": """@Test
public void testFileOutput() throws Exception {
    File outputFile = new File(testDir, "output.txt");
    writer.write(outputFile, "test content");
    
    assertTrue(outputFile.exists());
    assertEquals("test content", Files.readString(outputFile.toPath()));
}""",
        "expected_category": 4
    },
    
    "non_flaky": {
        "code": """@Test
public void testCalculation() {
    Calculator calc = new Calculator();
    
    assertEquals(5, calc.add(2, 3));
    assertEquals(10, calc.multiply(2, 5));
    assertEquals(2, calc.divide(10, 5));
}""",
        "expected_category": 5
    }
}


def example_basic_analysis():
    """Basic analysis example"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Analysis")
    print("=" * 60)
    
    code = EXAMPLE_TESTS["async_wait"]["code"]
    
    # Method 1: Using convenience function
    result = analyze_test(code)
    
    print("\nAnalysis Result:")
    print(f"  Predicted: {result['prediction']['label']} (category {result['prediction']['category']})")
    print(f"  Confidence: {result['prediction']['confidence']:.1%}")
    print(f"  Non-deterministic operations: {result['nondeterministic_operations']['count']}")
    print(f"  Tainted variables: {list(result['variable_analysis']['tainted_variables'].keys())}")


def example_llm_prompt_generation():
    """Generate LLM-ready prompt"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: LLM Prompt Generation")
    print("=" * 60)
    
    code = EXAMPLE_TESTS["concurrency"]["code"]
    
    # Generate LLM prompt
    prompt = create_llm_input(code, "llm_prompt")
    
    print("\nGenerated LLM Prompt (truncated):")
    print("-" * 40)
    # Show first 1500 characters
    print(prompt[:1500])
    print("...")
    print("-" * 40)


def example_detailed_analysis():
    """Detailed analysis with all information"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Detailed Analysis")
    print("=" * 60)
    
    analyzer = FlakyTestAnalyzer()
    code = EXAMPLE_TESTS["time"]["code"]
    
    # Full analysis
    result = analyzer.analyze(
        code=code,
        test_id="demo_001",
        test_name="testTimestampOrdering",
        project="demo_project"
    )
    
    print("\nFull Analysis Result:")
    print(f"  Test ID: {result.test_id}")
    print(f"  Test Name: {result.test_name}")
    print(f"  Predicted: {result.predicted_label}")
    print(f"  Confidence: {result.confidence:.3f}")
    
    print("\n  Non-deterministic Operations by Category:")
    ops = result.nondeterminism_analysis.get('nondeterministic_operations', {})
    by_cat = ops.get('by_category', {})
    for cat, cat_ops in by_cat.items():
        print(f"    [{cat}]")
        for op in cat_ops[:3]:
            print(f"      - Line {op['location']['line']}: {op['pattern_matched']}")
    
    print("\n  Data Flow Analysis:")
    data_flow = result.nondeterminism_analysis.get('data_flow', {})
    tainted = data_flow.get('tainted_variables', {})
    for var, cats in tainted.items():
        print(f"    {var} <- {cats}")
    
    print("\n  High-Risk Assertions:")
    assertions = result.nondeterminism_analysis.get('assertion_analysis', {})
    high_risk = assertions.get('high_risk_assertions', [])
    for a in high_risk:
        print(f"    - Line {a['location']['line']}: {a['assertion_type']} "
              f"(exposure: {a['nondeterminism_exposure']:.2f})")


def example_batch_analysis():
    """Batch analysis example"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Analysis")
    print("=" * 60)
    
    analyzer = FlakyTestAnalyzer()
    
    # Prepare test data
    tests = [
        {
            "id": i,
            "code": test_data["code"],
            "category": test_data["expected_category"],
            "label": name
        }
        for i, (name, test_data) in enumerate(EXAMPLE_TESTS.items())
    ]
    
    # Run batch analysis
    results = analyzer.analyze_batch(tests)
    
    # Evaluate
    evaluation = analyzer.evaluate(results)
    
    print(f"\nBatch Analysis Results:")
    print(f"  Total tests: {evaluation['total']}")
    print(f"  Correct: {evaluation['correct']}")
    print(f"  Accuracy: {evaluation['overall_accuracy']:.1%}")
    
    print("\n  Per-test Results:")
    for result in results:
        status = "✓" if result.predicted_category == result.actual_category else "✗"
        print(f"    {status} {result.actual_label}: predicted={result.predicted_label}, "
              f"actual_cat={result.actual_category}")


def example_different_formats():
    """Show different output formats"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different Output Formats")
    print("=" * 60)
    
    analyzer = FlakyTestAnalyzer()
    code = EXAMPLE_TESTS["unordered_collection"]["code"]
    
    formats = [
        ("JSON Compact", OutputFormat.JSON_COMPACT),
        ("Markdown", OutputFormat.MARKDOWN),
        ("Structured Text", OutputFormat.STRUCTURED_TEXT)
    ]
    
    for name, fmt in formats:
        output = analyzer.format_for_llm(code, fmt)
        print(f"\n--- {name} ---")
        # Truncate long outputs
        if len(output) > 800:
            print(output[:800] + "\n...")
        else:
            print(output)


def example_structured_info():
    """Get structured info for custom processing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Structured Information for Custom Use")
    print("=" * 60)
    
    analyzer = FlakyTestAnalyzer()
    code = EXAMPLE_TESTS["async_wait"]["code"]
    
    # Get structured info
    info = analyzer.get_structured_info(code)
    
    print("\nStructured Information (JSON):")
    print(json.dumps(info, indent=2))


def main():
    """Run all examples"""
    print("=" * 60)
    print("FLAKY TEST ANALYZER - USAGE EXAMPLES")
    print("=" * 60)
    
    example_basic_analysis()
    example_llm_prompt_generation()
    example_detailed_analysis()
    example_batch_analysis()
    example_different_formats()
    example_structured_info()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
