"""
Flaky Test Analyzer - Main Module
=================================
Integrates all components for comprehensive flaky test analysis.
Provides a unified interface for analyzing Java test code.
"""

import json
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ast_analyzer import JavaASTAnalyzer, analyze_java_code
from src.nondeterminism_detector import NonDeterminismDetector, detect_nondeterminism
from src.llm_formatter import LLMOutputFormatter, OutputFormat, format_for_llm
from config.flaky_patterns import FlakyCategory, CATEGORY_LABELS, PatternConfig


@dataclass
class AnalysisResult:
    """Complete analysis result for a test"""
    test_id: Optional[str] = None
    test_name: Optional[str] = None
    project: Optional[str] = None
    code: str = ""
    
    # Analysis results
    ast_analysis: Dict = field(default_factory=dict)
    nondeterminism_analysis: Dict = field(default_factory=dict)
    
    # Prediction
    predicted_category: int = -1
    predicted_label: str = ""
    confidence: float = 0.0
    
    # Ground truth (if available)
    actual_category: Optional[int] = None
    actual_label: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "project": self.project,
            "code": self.code,
            "predicted_category": self.predicted_category,
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 3),
            "actual_category": self.actual_category,
            "actual_label": self.actual_label,
            "is_correct": self.predicted_category == self.actual_category if self.actual_category is not None else None,
            "ast_analysis": self.ast_analysis,
            "nondeterminism_analysis": self.nondeterminism_analysis
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FlakyTestAnalyzer:
    """
    Main analyzer class that orchestrates the complete analysis pipeline.
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        self.config = config or PatternConfig()
        self.ast_analyzer = JavaASTAnalyzer()
        self.nd_detector = NonDeterminismDetector(self.config)
        self.formatter = LLMOutputFormatter()
    
    def analyze(self, code: str, 
                test_id: Optional[str] = None,
                test_name: Optional[str] = None,
                project: Optional[str] = None,
                actual_category: Optional[int] = None,
                actual_label: Optional[str] = None) -> AnalysisResult:
        """
        Perform complete analysis on a test code snippet.
        
        Args:
            code: Java test code
            test_id: Optional identifier
            test_name: Optional test name
            project: Optional project name
            actual_category: Ground truth category (for evaluation)
            actual_label: Ground truth label (for evaluation)
        
        Returns:
            AnalysisResult with complete analysis
        """
        # Step 1: AST Analysis
        ast_result = self.ast_analyzer.analyze(code)
        
        # Step 2: Non-determinism Detection
        nd_result = self.nd_detector.detect(code)
        
        # Step 3: Extract prediction
        summary = nd_result.get("analysis_summary", {})
        evidence = nd_result.get("classification_evidence", {})
        
        predicted_category = summary.get("predicted_category", -1)
        predicted_label = summary.get("predicted_label", "unknown")
        confidence = evidence.get("confidence", 0.0)
        
        # Create result
        result = AnalysisResult(
            test_id=test_id,
            test_name=test_name,
            project=project,
            code=code,
            ast_analysis=ast_result,
            nondeterminism_analysis=nd_result,
            predicted_category=predicted_category,
            predicted_label=predicted_label,
            confidence=confidence,
            actual_category=actual_category,
            actual_label=actual_label
        )
        
        return result
    
    def analyze_batch(self, tests: List[Dict]) -> List[AnalysisResult]:
        """
        Analyze multiple tests in batch.
        
        Args:
            tests: List of dicts with 'code' and optional metadata
        
        Returns:
            List of AnalysisResult objects
        """
        results = []
        for test in tests:
            result = self.analyze(
                code=test.get("code", test.get("full_code", "")),
                test_id=str(test.get("id", "")),
                test_name=test.get("test_name", ""),
                project=test.get("project", ""),
                actual_category=test.get("category"),
                actual_label=test.get("label")
            )
            results.append(result)
        return results
    
    def format_for_llm(self, code: str, 
                       output_format: OutputFormat = OutputFormat.LLM_PROMPT) -> str:
        """
        Analyze code and format result for LLM consumption.
        
        Args:
            code: Java test code
            output_format: Desired output format
        
        Returns:
            Formatted string for LLM
        """
        nd_result = self.nd_detector.detect(code)
        return self.formatter.format(nd_result, code, output_format)
    
    def get_structured_info(self, code: str) -> Dict:
        """
        Get structured analysis information.
        
        This is the primary method for getting information to feed to an LLM.
        Returns a clean, structured dictionary with all relevant analysis.
        """
        nd_result = self.nd_detector.detect(code)
        ast_result = self.ast_analyzer.analyze(code)
        
        summary = nd_result.get("analysis_summary", {})
        evidence = nd_result.get("classification_evidence", {})
        ops = nd_result.get("nondeterministic_operations", {})
        data_flow = nd_result.get("data_flow", {})
        assertions = nd_result.get("assertion_analysis", {})
        
        return {
            "prediction": {
                "category": summary.get("predicted_category"),
                "label": summary.get("predicted_label"),
                "confidence": evidence.get("confidence", 0.0),
                "all_scores": summary.get("category_scores", {})
            },
            
            "nondeterministic_operations": {
                "count": summary.get("total_nondeterministic_operations", 0),
                "by_category": self._simplify_operations(ops.get("by_category", {}))
            },
            
            "variable_analysis": {
                "total_variables": ast_result.get("summary", {}).get("total_variables", 0),
                "tainted_variables": data_flow.get("tainted_variables", {}),
                "data_flow_edges": len(data_flow.get("flow_edges", []))
            },
            
            "assertion_analysis": {
                "total_assertions": summary.get("total_assertions", 0),
                "high_risk_count": summary.get("high_risk_assertions", 0),
                "high_risk_details": self._simplify_assertions(
                    assertions.get("high_risk_assertions", [])
                )
            },
            
            "key_evidence": {
                "indicators": evidence.get("key_indicators", []),
                "risk_factors": evidence.get("risk_factors", [])
            },
            
            "code_structure": {
                "method_calls": ast_result.get("summary", {}).get("total_method_calls", 0),
                "loops": ast_result.get("summary", {}).get("total_loops", 0)
            }
        }
    
    def _simplify_operations(self, ops_by_category: Dict) -> Dict:
        """Simplify operations for cleaner output"""
        simplified = {}
        for cat, ops in ops_by_category.items():
            simplified[cat] = [
                {
                    "pattern": op.get("pattern_matched"),
                    "type": op.get("operation_type"),
                    "line": op.get("location", {}).get("line"),
                    "confidence": op.get("confidence"),
                    "affects": op.get("affected_variables", [])
                }
                for op in ops[:5]  # Limit to 5 per category
            ]
        return simplified
    
    def _simplify_assertions(self, assertions: List[Dict]) -> List[Dict]:
        """Simplify assertion details"""
        return [
            {
                "type": a.get("assertion_type"),
                "line": a.get("location", {}).get("line"),
                "exposure": a.get("nondeterminism_exposure"),
                "risk": a.get("risk_level"),
                "variables": a.get("involved_variables", [])[:3]
            }
            for a in assertions[:5]
        ]
    
    def evaluate(self, results: List[AnalysisResult]) -> Dict:
        """
        Evaluate prediction accuracy against ground truth.
        
        Args:
            results: List of AnalysisResult with actual_category set
        
        Returns:
            Evaluation metrics
        """
        # Filter results with ground truth
        valid_results = [r for r in results if r.actual_category is not None]
        
        if not valid_results:
            return {"error": "No results with ground truth"}
        
        # Calculate metrics
        correct = sum(1 for r in valid_results if r.predicted_category == r.actual_category)
        total = len(valid_results)
        accuracy = correct / total
        
        # Per-category metrics
        category_metrics = {}
        for cat in FlakyCategory:
            cat_results = [r for r in valid_results if r.actual_category == cat.value]
            if cat_results:
                cat_correct = sum(1 for r in cat_results 
                                if r.predicted_category == r.actual_category)
                category_metrics[CATEGORY_LABELS[cat]] = {
                    "total": len(cat_results),
                    "correct": cat_correct,
                    "accuracy": cat_correct / len(cat_results)
                }
        
        # Confusion matrix
        confusion = {}
        for r in valid_results:
            actual = r.actual_label or str(r.actual_category)
            predicted = r.predicted_label or str(r.predicted_category)
            if actual not in confusion:
                confusion[actual] = {}
            confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
        
        return {
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "category_metrics": category_metrics,
            "confusion_matrix": confusion
        }


def create_llm_input(code: str, output_format: str = "llm_prompt") -> str:
    """
    Convenience function to create LLM-ready input from test code.
    
    Args:
        code: Java test code
        output_format: One of "llm_prompt", "json", "json_compact", "markdown", "structured_text"
    
    Returns:
        Formatted string for LLM consumption
    """
    analyzer = FlakyTestAnalyzer()
    fmt = OutputFormat(output_format) if output_format in [f.value for f in OutputFormat] else OutputFormat.LLM_PROMPT
    return analyzer.format_for_llm(code, fmt)


def analyze_test(code: str) -> Dict:
    """
    Convenience function to analyze a single test.
    
    Args:
        code: Java test code
    
    Returns:
        Structured analysis dictionary
    """
    analyzer = FlakyTestAnalyzer()
    return analyzer.get_structured_info(code)
