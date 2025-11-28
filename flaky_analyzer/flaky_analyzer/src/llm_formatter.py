"""
LLM Output Formatter
====================
Formats analysis results into clear, structured formats optimized for LLM comprehension.
Supports multiple output formats: JSON, Markdown, and Compact text.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.flaky_patterns import FlakyCategory, CATEGORY_LABELS


class OutputFormat(Enum):
    JSON = "json"
    JSON_COMPACT = "json_compact"
    MARKDOWN = "markdown"
    STRUCTURED_TEXT = "structured_text"
    LLM_PROMPT = "llm_prompt"


@dataclass
class FormatterConfig:
    """Configuration for output formatting"""
    include_code_context: bool = True
    include_data_flow: bool = True
    include_all_operations: bool = False  # If False, only include top operations
    max_operations_per_category: int = 5
    max_context_lines: int = 3
    confidence_threshold: float = 0.3
    include_raw_evidence: bool = True


class LLMOutputFormatter:
    """
    Formats flaky test analysis results for LLM consumption.
    Optimized for clarity, structure, and informativeness.
    """
    
    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()
    
    def format(self, analysis: Dict, code: str, 
               output_format: OutputFormat = OutputFormat.JSON) -> str:
        """
        Format analysis results in the specified format.
        """
        if output_format == OutputFormat.JSON:
            return self._format_json(analysis, code)
        elif output_format == OutputFormat.JSON_COMPACT:
            return self._format_json_compact(analysis, code)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(analysis, code)
        elif output_format == OutputFormat.STRUCTURED_TEXT:
            return self._format_structured_text(analysis, code)
        elif output_format == OutputFormat.LLM_PROMPT:
            return self._format_llm_prompt(analysis, code)
        else:
            return self._format_json(analysis, code)
    
    def _format_json(self, analysis: Dict, code: str) -> str:
        """Format as detailed JSON"""
        output = {
            "test_code": code if self.config.include_code_context else None,
            "analysis": analysis,
            "formatted_summary": self._create_summary(analysis)
        }
        
        if not self.config.include_code_context:
            del output["test_code"]
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def _format_json_compact(self, analysis: Dict, code: str) -> str:
        """Format as compact JSON with essential information only"""
        summary = analysis.get("analysis_summary", {})
        evidence = analysis.get("classification_evidence", {})
        
        compact = {
            "prediction": {
                "category": summary.get("predicted_category"),
                "label": summary.get("predicted_label"),
                "confidence": summary.get("category_scores", {}).get(
                    summary.get("predicted_label"), 0
                )
            },
            "scores": summary.get("category_scores", {}),
            "key_patterns": [
                {
                    "pattern": ind.get("pattern"),
                    "line": ind.get("line"),
                    "confidence": ind.get("confidence")
                }
                for ind in evidence.get("key_indicators", [])[:5]
            ],
            "risk_assertions": len(analysis.get("assertion_analysis", {}).get(
                "high_risk_assertions", []
            )),
            "tainted_vars": list(analysis.get("data_flow", {}).get(
                "tainted_variables", {}
            ).keys())
        }
        
        return json.dumps(compact, indent=2, ensure_ascii=False)
    
    def _format_markdown(self, analysis: Dict, code: str) -> str:
        """Format as Markdown for human readability"""
        lines = []
        summary = analysis.get("analysis_summary", {})
        evidence = analysis.get("classification_evidence", {})
        
        # Header
        lines.append("# Flaky Test Analysis Report\n")
        
        # Prediction
        lines.append("## Prediction")
        lines.append(f"- **Category**: {summary.get('predicted_label', 'Unknown')}")
        lines.append(f"- **Category ID**: {summary.get('predicted_category', -1)}")
        lines.append(f"- **Confidence**: {evidence.get('confidence', 0):.1%}\n")
        
        # Category Scores
        lines.append("## Category Scores")
        scores = summary.get("category_scores", {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_scores:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"- {label}: {bar} {score:.1%}")
        lines.append("")
        
        # Key Indicators
        lines.append("## Key Indicators")
        indicators = evidence.get("key_indicators", [])
        if indicators:
            for ind in indicators[:5]:
                lines.append(f"- Line {ind.get('line')}: `{ind.get('pattern')}` "
                           f"({ind.get('type')}, conf: {ind.get('confidence', 0):.2f})")
        else:
            lines.append("- No strong indicators detected")
        lines.append("")
        
        # Risk Factors
        lines.append("## Risk Factors")
        risk_factors = evidence.get("risk_factors", [])
        if risk_factors:
            for rf in risk_factors[:5]:
                lines.append(f"- **{rf.get('assertion')}** at line {rf.get('line')}: "
                           f"exposure {rf.get('exposure', 0):.1%}")
                for src in rf.get("sources", []):
                    lines.append(f"  - {src}")
        else:
            lines.append("- No high-risk assertions detected")
        lines.append("")
        
        # Data Flow
        if self.config.include_data_flow:
            lines.append("## Tainted Variables")
            tainted = analysis.get("data_flow", {}).get("tainted_variables", {})
            if tainted:
                for var, categories in tainted.items():
                    lines.append(f"- `{var}`: {', '.join(categories)}")
            else:
                lines.append("- No tainted variables detected")
        
        return '\n'.join(lines)
    
    def _format_structured_text(self, analysis: Dict, code: str) -> str:
        """Format as structured plain text"""
        lines = []
        summary = analysis.get("analysis_summary", {})
        evidence = analysis.get("classification_evidence", {})
        
        lines.append("=" * 60)
        lines.append("FLAKY TEST ANALYSIS")
        lines.append("=" * 60)
        
        # Prediction
        lines.append(f"\nPREDICTION: {summary.get('predicted_label', 'Unknown')}")
        lines.append(f"CATEGORY ID: {summary.get('predicted_category', -1)}")
        lines.append(f"CONFIDENCE: {evidence.get('confidence', 0):.3f}")
        
        # Scores
        lines.append("\nCATEGORY SCORES:")
        scores = summary.get("category_scores", {})
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {label}: {score:.3f}")
        
        # Operations
        lines.append("\nNON-DETERMINISTIC OPERATIONS:")
        ops_by_cat = analysis.get("nondeterministic_operations", {}).get("by_category", {})
        for cat, ops in ops_by_cat.items():
            lines.append(f"\n  [{cat.upper()}]")
            for op in ops[:self.config.max_operations_per_category]:
                lines.append(f"    - Line {op.get('location', {}).get('line')}: "
                           f"{op.get('pattern_matched')} ({op.get('operation_type')})")
        
        # High-risk assertions
        lines.append("\nHIGH-RISK ASSERTIONS:")
        high_risk = analysis.get("assertion_analysis", {}).get("high_risk_assertions", [])
        if high_risk:
            for a in high_risk[:5]:
                lines.append(f"  - Line {a.get('location', {}).get('line')}: "
                           f"{a.get('assertion_type')} (exposure: {a.get('nondeterminism_exposure', 0):.2f})")
        else:
            lines.append("  None detected")
        
        lines.append("\n" + "=" * 60)
        
        return '\n'.join(lines)
    
    def _format_llm_prompt(self, analysis: Dict, code: str) -> str:
        """
        Format as a complete prompt for LLM classification.
        This is the primary format for feeding to another LLM for classification.
        """
        summary = analysis.get("analysis_summary", {})
        evidence = analysis.get("classification_evidence", {})
        ops = analysis.get("nondeterministic_operations", {})
        data_flow = analysis.get("data_flow", {})
        assertions = analysis.get("assertion_analysis", {})
        
        prompt_parts = []
        
        # System context
        prompt_parts.append("""<task>
Classify the following Java test code into one of 6 flaky test categories based on the provided analysis.
</task>

<categories>
0: async wait - Tests with timing dependencies (Thread.sleep, await, Future.get)
1: concurrency - Tests with thread safety issues (AtomicInteger, synchronized, Executor)
2: time - Tests depending on current time (Date, System.currentTimeMillis, nanoTime)
3: unordered collections - Tests with iteration order assumptions (HashMap, HashSet, keySet)
4: test order dependency - Tests depending on external state (static fields, files, config)
5: non-flaky - Deterministic tests with no flakiness sources
</categories>
""")
        
        # Test code
        prompt_parts.append("<test_code>")
        prompt_parts.append(code)
        prompt_parts.append("</test_code>")
        
        # Analysis summary
        prompt_parts.append("\n<analysis>")
        
        # Category scores
        prompt_parts.append("\n<category_scores>")
        scores = summary.get("category_scores", {})
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            prompt_parts.append(f"{label}: {score:.3f}")
        prompt_parts.append("</category_scores>")
        
        # Detected operations
        prompt_parts.append("\n<detected_operations>")
        ops_by_cat = ops.get("by_category", {})
        if ops_by_cat:
            for cat, cat_ops in ops_by_cat.items():
                prompt_parts.append(f"\n[{cat}]")
                for op in cat_ops[:self.config.max_operations_per_category]:
                    prompt_parts.append(
                        f"  - Line {op.get('location', {}).get('line')}: "
                        f"{op.get('pattern_matched')} "
                        f"(type: {op.get('operation_type')}, "
                        f"confidence: {op.get('confidence', 0):.2f})"
                    )
                    if op.get('affected_variables'):
                        prompt_parts.append(
                            f"    affects: {', '.join(op.get('affected_variables'))}"
                        )
        else:
            prompt_parts.append("No non-deterministic operations detected")
        prompt_parts.append("</detected_operations>")
        
        # Data flow / tainted variables
        prompt_parts.append("\n<tainted_variables>")
        tainted = data_flow.get("tainted_variables", {})
        if tainted:
            for var, categories in tainted.items():
                prompt_parts.append(f"  {var} <- [{', '.join(categories)}]")
        else:
            prompt_parts.append("No tainted variables")
        prompt_parts.append("</tainted_variables>")
        
        # Assertion analysis
        prompt_parts.append("\n<assertion_risk>")
        high_risk = assertions.get("high_risk_assertions", [])
        if high_risk:
            for a in high_risk[:5]:
                prompt_parts.append(
                    f"  - {a.get('assertion_type')} at line {a.get('location', {}).get('line')}: "
                    f"exposure={a.get('nondeterminism_exposure', 0):.2f}, "
                    f"risk={a.get('risk_level')}"
                )
                if a.get('exposure_sources'):
                    prompt_parts.append(f"    sources: {'; '.join(a.get('exposure_sources')[:3])}")
        else:
            prompt_parts.append("No high-risk assertions")
        prompt_parts.append("</assertion_risk>")
        
        # Key evidence
        prompt_parts.append("\n<key_evidence>")
        indicators = evidence.get("key_indicators", [])
        if indicators:
            for ind in indicators[:5]:
                prompt_parts.append(
                    f"  - {ind.get('pattern')} (line {ind.get('line')}, "
                    f"type: {ind.get('type')}, conf: {ind.get('confidence', 0):.2f})"
                )
        else:
            prompt_parts.append("No strong indicators")
        prompt_parts.append("</key_evidence>")
        
        prompt_parts.append("\n</analysis>")
        
        # Request
        prompt_parts.append("""
<instruction>
Based on the test code and analysis above, determine the most likely flaky category.
Consider:
1. The detected non-deterministic operations and their types
2. Which variables are tainted by non-determinism
3. Whether assertions check tainted variables
4. The confidence scores for each category

Output your classification as a JSON object:
{
  "category": <0-5>,
  "label": "<category_label>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}
</instruction>
""")
        
        return '\n'.join(prompt_parts)
    
    def _create_summary(self, analysis: Dict) -> Dict:
        """Create a human-readable summary"""
        summary = analysis.get("analysis_summary", {})
        evidence = analysis.get("classification_evidence", {})
        
        return {
            "predicted_category": summary.get("predicted_label", "Unknown"),
            "confidence": evidence.get("confidence", 0),
            "num_operations": summary.get("total_nondeterministic_operations", 0),
            "num_risky_assertions": summary.get("high_risk_assertions", 0),
            "primary_indicators": [
                ind.get("pattern") for ind in evidence.get("key_indicators", [])[:3]
            ]
        }


def format_for_llm(analysis: Dict, code: str, 
                   output_format: str = "llm_prompt") -> str:
    """Convenience function to format analysis for LLM"""
    formatter = LLMOutputFormatter()
    fmt = OutputFormat(output_format) if output_format in [f.value for f in OutputFormat] else OutputFormat.LLM_PROMPT
    return formatter.format(analysis, code, fmt)
