#!/usr/bin/env python3
"""
Batch Processor and Evaluator
=============================
Process datasets in batch and evaluate classification accuracy.
Generates comprehensive reports and LLM-ready outputs.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.analyzer import FlakyTestAnalyzer, AnalysisResult
from src.llm_formatter import OutputFormat
from config.flaky_patterns import FlakyCategory, CATEGORY_LABELS


class BatchProcessor:
    """
    Processes multiple tests and generates comprehensive analysis reports.
    """
    
    def __init__(self):
        self.analyzer = FlakyTestAnalyzer()
        self.results: List[AnalysisResult] = []
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset from Excel or CSV file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas openpyxl")
        
        if filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        return df.to_dict('records')
    
    def process(self, tests: List[Dict], verbose: bool = True) -> List[AnalysisResult]:
        """
        Process all tests in the dataset.
        
        Args:
            tests: List of test dictionaries
            verbose: Print progress
        
        Returns:
            List of AnalysisResult
        """
        self.results = []
        total = len(tests)
        
        for i, test in enumerate(tests):
            code = test.get('full_code', test.get('code', ''))
            if not code:
                continue
            
            result = self.analyzer.analyze(
                code=code,
                test_id=str(test.get('id', i)),
                test_name=test.get('test_name', ''),
                project=test.get('project', ''),
                actual_category=test.get('category'),
                actual_label=test.get('label')
            )
            self.results.append(result)
            
            if verbose and (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{total} tests...")
        
        if verbose:
            print(f"Processing complete: {len(self.results)} tests analyzed")
        
        return self.results
    
    def evaluate(self) -> Dict:
        """Evaluate prediction accuracy"""
        return self.analyzer.evaluate(self.results)
    
    def generate_llm_prompts(self, output_dir: str, 
                            output_format: OutputFormat = OutputFormat.LLM_PROMPT):
        """
        Generate LLM-ready prompts for each test.
        
        Args:
            output_dir: Directory to save prompts
            output_format: Format for output
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in self.results:
            prompt = self.analyzer.format_for_llm(result.code, output_format)
            
            filename = f"test_{result.test_id}.txt"
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        print(f"Generated {len(self.results)} prompts in {output_dir}")
    
    def generate_structured_dataset(self, output_path: str):
        """
        Generate a structured dataset with analysis for each test.
        This is the primary output format for training/fine-tuning LLMs.
        """
        structured_data = []
        
        for result in self.results:
            info = self.analyzer.get_structured_info(result.code)
            
            entry = {
                "id": result.test_id,
                "project": result.project,
                "test_name": result.test_name,
                "code": result.code,
                "ground_truth": {
                    "category": result.actual_category,
                    "label": result.actual_label
                },
                "analysis": info,
                "prediction": {
                    "category": result.predicted_category,
                    "label": result.predicted_label,
                    "confidence": result.confidence,
                    "is_correct": result.predicted_category == result.actual_category 
                                 if result.actual_category is not None else None
                }
            }
            structured_data.append(entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        print(f"Structured dataset saved to {output_path}")
        return structured_data
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        evaluation = self.evaluate()
        
        lines = []
        lines.append("=" * 70)
        lines.append("FLAKY TEST ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # Overall metrics
        lines.append("\n## OVERALL METRICS")
        lines.append(f"Total Tests: {evaluation.get('total', 0)}")
        lines.append(f"Correct Predictions: {evaluation.get('correct', 0)}")
        lines.append(f"Accuracy: {evaluation.get('overall_accuracy', 0):.1%}")
        
        # Per-category metrics
        lines.append("\n## PER-CATEGORY METRICS")
        cat_metrics = evaluation.get('category_metrics', {})
        for cat, metrics in sorted(cat_metrics.items()):
            lines.append(f"\n  {cat}:")
            lines.append(f"    Total: {metrics.get('total', 0)}")
            lines.append(f"    Correct: {metrics.get('correct', 0)}")
            lines.append(f"    Accuracy: {metrics.get('accuracy', 0):.1%}")
        
        # Confusion matrix
        lines.append("\n## CONFUSION MATRIX")
        confusion = evaluation.get('confusion_matrix', {})
        if confusion:
            # Get all labels
            all_labels = sorted(set(confusion.keys()) | 
                              set(l for row in confusion.values() for l in row.keys()))
            
            # Header
            header = "Actual \\ Predicted | " + " | ".join(f"{l[:12]:^12}" for l in all_labels)
            lines.append(header)
            lines.append("-" * len(header))
            
            # Rows
            for actual in all_labels:
                row_data = confusion.get(actual, {})
                row = f"{actual[:18]:18} | "
                row += " | ".join(f"{row_data.get(pred, 0):^12}" for pred in all_labels)
                lines.append(row)
        
        # Sample errors
        lines.append("\n## SAMPLE MISCLASSIFICATIONS")
        errors = [r for r in self.results 
                 if r.actual_category is not None and r.predicted_category != r.actual_category]
        
        for error in errors[:5]:
            lines.append(f"\n  Test ID: {error.test_id}")
            lines.append(f"    Actual: {error.actual_label} ({error.actual_category})")
            lines.append(f"    Predicted: {error.predicted_label} ({error.predicted_category})")
            lines.append(f"    Confidence: {error.confidence:.2f}")
            
            # Show key indicators
            evidence = error.nondeterminism_analysis.get('classification_evidence', {})
            indicators = evidence.get('key_indicators', [])[:2]
            if indicators:
                lines.append(f"    Key Indicators:")
                for ind in indicators:
                    lines.append(f"      - {ind.get('pattern')} (line {ind.get('line')})")
        
        lines.append("\n" + "=" * 70)
        
        return '\n'.join(lines)


class FlakyPatternStatistics:
    """
    Generate statistics about flaky patterns across a dataset.
    """
    
    def __init__(self, results: List[AnalysisResult]):
        self.results = results
    
    def pattern_frequency(self) -> Dict[str, Dict[str, int]]:
        """Get frequency of each pattern by category"""
        frequency = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            ops = result.nondeterminism_analysis.get('nondeterministic_operations', {})
            all_ops = ops.get('all_operations', [])
            
            for op in all_ops:
                pattern = op.get('pattern_matched', '')
                op_type = op.get('operation_type', '')
                category = op.get('category_name', '')
                
                frequency[category][f"{op_type}: {pattern}"] += 1
        
        return dict(frequency)
    
    def variable_taint_analysis(self) -> Dict[str, int]:
        """Analyze which types of taint are most common"""
        taint_counts = defaultdict(int)
        
        for result in self.results:
            data_flow = result.nondeterminism_analysis.get('data_flow', {})
            tainted = data_flow.get('tainted_variables', {})
            
            for var, categories in tainted.items():
                for cat in categories:
                    taint_counts[cat] += 1
        
        return dict(taint_counts)
    
    def assertion_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of assertion risk levels"""
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for result in self.results:
            assertions = result.nondeterminism_analysis.get('assertion_analysis', {})
            all_assertions = assertions.get('all_assertions', [])
            
            for a in all_assertions:
                risk = a.get('risk_level', 'LOW')
                risk_counts[risk] += 1
        
        return risk_counts
    
    def generate_statistics_report(self) -> str:
        """Generate a statistics report"""
        lines = []
        lines.append("=" * 60)
        lines.append("FLAKY PATTERN STATISTICS")
        lines.append("=" * 60)
        
        # Pattern frequency
        lines.append("\n## PATTERN FREQUENCY BY CATEGORY")
        freq = self.pattern_frequency()
        for cat, patterns in sorted(freq.items()):
            lines.append(f"\n  [{cat}]")
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:10]:
                lines.append(f"    {count:4d}x {pattern}")
        
        # Taint analysis
        lines.append("\n## VARIABLE TAINT DISTRIBUTION")
        taint = self.variable_taint_analysis()
        for cat, count in sorted(taint.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {cat}: {count}")
        
        # Assertion risk
        lines.append("\n## ASSERTION RISK DISTRIBUTION")
        risk = self.assertion_risk_distribution()
        total = sum(risk.values())
        for level, count in risk.items():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {level}: {count} ({pct:.1f}%)")
        
        lines.append("\n" + "=" * 60)
        
        return '\n'.join(lines)


def process_dataset(dataset_path: str, output_dir: str = "./output"):
    """
    Complete pipeline for processing a dataset.
    
    Generates:
    - structured_analysis.json: Full structured analysis for each test
    - evaluation_report.txt: Classification accuracy report
    - statistics_report.txt: Pattern frequency statistics
    - llm_prompts/: Individual LLM prompts for each test
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    processor = BatchProcessor()
    tests = processor.load_dataset(dataset_path)
    processor.process(tests)
    
    # Generate structured dataset
    processor.generate_structured_dataset(output_path / "structured_analysis.json")
    
    # Generate evaluation report
    report = processor.generate_report()
    with open(output_path / "evaluation_report.txt", 'w') as f:
        f.write(report)
    print(f"Evaluation report saved to {output_path / 'evaluation_report.txt'}")
    
    # Generate statistics
    stats = FlakyPatternStatistics(processor.results)
    stats_report = stats.generate_statistics_report()
    with open(output_path / "statistics_report.txt", 'w') as f:
        f.write(stats_report)
    print(f"Statistics report saved to {output_path / 'statistics_report.txt'}")
    
    # Generate LLM prompts
    processor.generate_llm_prompts(str(output_path / "llm_prompts"))
    
    print(f"\nAll outputs saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    evaluation = processor.evaluate()
    print(f"Total Tests: {evaluation.get('total', 0)}")
    print(f"Overall Accuracy: {evaluation.get('overall_accuracy', 0):.1%}")
    
    return processor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process flaky test dataset")
    parser.add_argument("dataset", help="Path to dataset file (xlsx/csv)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.output)
