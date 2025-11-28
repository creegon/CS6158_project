"""
Flaky Test Analyzer Package
============================
A comprehensive tool for analyzing Java test code and generating
structured information for LLM-based flaky test classification.
"""

from .analyzer import (
    FlakyTestAnalyzer,
    AnalysisResult,
    create_llm_input,
    analyze_test
)

from .nondeterminism_detector import (
    NonDeterminismDetector,
    NonDeterministicOperation,
    detect_nondeterminism
)

from .ast_analyzer import (
    JavaASTAnalyzer,
    analyze_java_code
)

from .llm_formatter import (
    LLMOutputFormatter,
    OutputFormat,
    format_for_llm
)

__version__ = "1.0.0"
__all__ = [
    "FlakyTestAnalyzer",
    "AnalysisResult", 
    "create_llm_input",
    "analyze_test",
    "NonDeterminismDetector",
    "NonDeterministicOperation",
    "detect_nondeterminism",
    "JavaASTAnalyzer",
    "analyze_java_code",
    "LLMOutputFormatter",
    "OutputFormat",
    "format_for_llm"
]
