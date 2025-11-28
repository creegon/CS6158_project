"""
Non-Determinism Detector
========================
Detects and classifies non-deterministic operations in Java test code.
Maps operations to flaky test categories.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.flaky_patterns import (
    FlakyCategory, PatternConfig, PATTERN_WEIGHTS, 
    CATEGORY_LABELS, NONDETERMINISM_SOURCES
)


@dataclass
class NonDeterministicOperation:
    """Represents a detected non-deterministic operation"""
    operation_type: str  # e.g., "ASYNC", "TIME", "CONCURRENCY"
    category: FlakyCategory
    pattern_matched: str
    location: Dict  # {"line": int, "column": int}
    context: str  # surrounding code context
    confidence: float  # 0.0 to 1.0
    affected_variables: List[str] = field(default_factory=list)
    risk_description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "operation_type": self.operation_type,
            "category": self.category.value,
            "category_name": CATEGORY_LABELS[self.category],
            "pattern_matched": self.pattern_matched,
            "location": self.location,
            "context": self.context,
            "confidence": round(self.confidence, 3),
            "affected_variables": self.affected_variables,
            "risk_description": self.risk_description
        }


@dataclass
class DataFlowEdge:
    """Represents a data flow relationship"""
    source: str  # source variable or operation
    target: str  # target variable
    flow_type: str  # "assignment", "method_return", "parameter"
    location: Dict
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "flow_type": self.flow_type,
            "location": self.location
        }


@dataclass  
class AssertionAnalysis:
    """Analysis of an assertion statement"""
    assertion_type: str
    location: Dict
    expression: str
    involved_variables: List[str]
    nondeterminism_exposure: float  # 0.0 to 1.0
    exposure_sources: List[str] = field(default_factory=list)
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    
    def to_dict(self) -> Dict:
        return {
            "assertion_type": self.assertion_type,
            "location": self.location,
            "expression": self.expression,
            "involved_variables": self.involved_variables,
            "nondeterminism_exposure": round(self.nondeterminism_exposure, 3),
            "exposure_sources": self.exposure_sources,
            "risk_level": self.risk_level
        }


class NonDeterminismDetector:
    """
    Detects non-deterministic operations and builds a comprehensive
    analysis for LLM-based flaky test classification.
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        self.config = config or PatternConfig()
        self.operations: List[NonDeterministicOperation] = []
        self.data_flows: List[DataFlowEdge] = []
        self.assertions: List[AssertionAnalysis] = []
        self.variable_taint: Dict[str, Set[FlakyCategory]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.compiled_patterns = {
            FlakyCategory.ASYNC_WAIT: [
                (re.compile(p, re.IGNORECASE), p) 
                for p in self.config.async_wait_patterns
            ],
            FlakyCategory.CONCURRENCY: [
                (re.compile(p, re.IGNORECASE), p)
                for p in self.config.concurrency_patterns  
            ],
            FlakyCategory.TIME: [
                (re.compile(p, re.IGNORECASE), p)
                for p in self.config.time_patterns
            ],
            FlakyCategory.UNORDERED_COLLECTIONS: [
                (re.compile(p, re.IGNORECASE), p)
                for p in self.config.unordered_collection_patterns
            ],
            FlakyCategory.TEST_ORDER_DEPENDENCY: [
                (re.compile(p, re.IGNORECASE), p)
                for p in self.config.order_dependency_patterns
            ]
        }
        
        self.assertion_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.assertion_patterns
        ]
    
    def detect(self, code: str) -> Dict:
        """
        Main detection method - analyzes code and returns structured analysis.
        """
        self._reset()
        lines = code.split('\n')
        
        # Phase 1: Detect all non-deterministic operations
        for line_num, line in enumerate(lines, 1):
            self._detect_operations_in_line(line, line_num, lines)
        
        # Phase 2: Extract variable declarations and build initial taint
        self._extract_variables(code)
        
        # Phase 3: Propagate taint through data flow
        self._propagate_taint(code)
        
        # Phase 4: Analyze assertions
        for line_num, line in enumerate(lines, 1):
            self._analyze_assertion(line, line_num)
        
        # Phase 5: Calculate category scores
        category_scores = self._calculate_category_scores()
        
        return self._build_structured_output(code, category_scores)
    
    def _reset(self):
        """Reset state for new analysis"""
        self.operations = []
        self.data_flows = []
        self.assertions = []
        self.variable_taint = {}
    
    def _detect_operations_in_line(self, line: str, line_num: int, all_lines: List[str]):
        """Detect non-deterministic operations in a single line"""
        for category, patterns in self.compiled_patterns.items():
            for pattern, pattern_str in patterns:
                match = pattern.search(line)
                if match:
                    # Get context (surrounding lines)
                    context_start = max(0, line_num - 2)
                    context_end = min(len(all_lines), line_num + 2)
                    context = '\n'.join(all_lines[context_start:context_end])
                    
                    # Calculate confidence based on pattern weights
                    confidence = self._calculate_confidence(match.group(), category)
                    
                    # Extract affected variables
                    affected_vars = self._extract_affected_variables(line, match)
                    
                    # Determine operation type
                    op_type = self._determine_operation_type(pattern_str, category)
                    
                    operation = NonDeterministicOperation(
                        operation_type=op_type,
                        category=category,
                        pattern_matched=match.group(),
                        location={"line": line_num, "column": match.start()},
                        context=context,
                        confidence=confidence,
                        affected_variables=affected_vars,
                        risk_description=self._get_risk_description(category, pattern_str)
                    )
                    
                    # Avoid duplicates on same line
                    if not any(op.location["line"] == line_num and 
                              op.pattern_matched == match.group() 
                              for op in self.operations):
                        self.operations.append(operation)
    
    def _calculate_confidence(self, matched_text: str, category: FlakyCategory) -> float:
        """Calculate confidence score for a pattern match"""
        weights = PATTERN_WEIGHTS.get(category, {})
        
        # Check each weight pattern
        for pattern_key, weight in weights.items():
            if pattern_key.lower() in matched_text.lower():
                return weight
        
        # Default confidence based on category
        defaults = {
            FlakyCategory.ASYNC_WAIT: 0.7,
            FlakyCategory.CONCURRENCY: 0.65,
            FlakyCategory.TIME: 0.7,
            FlakyCategory.UNORDERED_COLLECTIONS: 0.5,
            FlakyCategory.TEST_ORDER_DEPENDENCY: 0.55
        }
        return defaults.get(category, 0.5)
    
    def _extract_affected_variables(self, line: str, match) -> List[str]:
        """Extract variables affected by the matched operation"""
        affected = []
        
        # Check for assignment pattern: var = ...match...
        assignment_pattern = re.compile(r'(\w+)\s*=\s*.*' + re.escape(match.group()))
        assign_match = assignment_pattern.search(line)
        if assign_match:
            affected.append(assign_match.group(1))
        
        # Check for variable as method target: var.method()
        target_pattern = re.compile(r'(\w+)\s*\.' + re.escape(match.group().split('.')[0] if '.' in match.group() else match.group()))
        target_match = target_pattern.search(line)
        if target_match:
            affected.append(target_match.group(1))
        
        # Extract variables from method arguments
        args_pattern = re.compile(r'\(([^)]+)\)')
        args_match = args_pattern.search(line[match.start():])
        if args_match:
            args = args_match.group(1).split(',')
            for arg in args:
                # Extract variable names from argument
                var_names = re.findall(r'\b([a-z_]\w*)\b', arg)
                affected.extend([v for v in var_names if v not in 
                               ['new', 'null', 'true', 'false', 'this']])
        
        return list(set(affected))
    
    def _determine_operation_type(self, pattern_str: str, category: FlakyCategory) -> str:
        """Determine the type of operation based on pattern and category"""
        pattern_lower = pattern_str.lower()
        
        if category == FlakyCategory.ASYNC_WAIT:
            if 'sleep' in pattern_lower:
                return "SLEEP_WAIT"
            elif 'await' in pattern_lower:
                return "ASYNC_AWAIT"
            elif 'future' in pattern_lower or 'get' in pattern_lower:
                return "FUTURE_BLOCKING"
            elif 'countdownlatch' in pattern_lower:
                return "LATCH_SYNC"
            return "ASYNC_OPERATION"
        
        elif category == FlakyCategory.CONCURRENCY:
            if 'atomic' in pattern_lower:
                return "ATOMIC_OPERATION"
            elif 'thread' in pattern_lower:
                return "THREAD_OPERATION"
            elif 'executor' in pattern_lower or 'schedule' in pattern_lower:
                return "EXECUTOR_TASK"
            elif 'synchronized' in pattern_lower:
                return "SYNCHRONIZED_BLOCK"
            return "CONCURRENT_OPERATION"
        
        elif category == FlakyCategory.TIME:
            if 'currenttimemillis' in pattern_lower or 'nanotime' in pattern_lower:
                return "SYSTEM_TIME"
            elif 'date' in pattern_lower:
                return "DATE_OPERATION"
            elif 'format' in pattern_lower or 'parse' in pattern_lower:
                return "TIME_PARSING"
            return "TIME_DEPENDENT"
        
        elif category == FlakyCategory.UNORDERED_COLLECTIONS:
            if 'hashmap' in pattern_lower or 'hashset' in pattern_lower:
                return "HASH_COLLECTION"
            elif 'keyset' in pattern_lower or 'entryset' in pattern_lower:
                return "ITERATION_ORDER"
            elif 'json' in pattern_lower:
                return "JSON_COMPARISON"
            return "UNORDERED_ITERATION"
        
        elif category == FlakyCategory.TEST_ORDER_DEPENDENCY:
            if 'static' in pattern_lower:
                return "STATIC_STATE"
            elif 'file' in pattern_lower or 'path' in pattern_lower:
                return "FILE_SYSTEM"
            elif 'config' in pattern_lower:
                return "CONFIGURATION"
            return "EXTERNAL_DEPENDENCY"
        
        return "UNKNOWN"
    
    def _get_risk_description(self, category: FlakyCategory, pattern_str: str) -> str:
        """Get human-readable risk description"""
        descriptions = {
            FlakyCategory.ASYNC_WAIT: {
                "sleep": "Thread.sleep creates timing dependency - test may fail under high system load",
                "await": "Async await may not complete in expected time window",
                "future": "Future.get blocking may timeout unpredictably",
                "countdownlatch": "Latch may not reach zero if thread execution varies",
                "default": "Async operation timing may vary between runs"
            },
            FlakyCategory.CONCURRENCY: {
                "atomic": "Atomic operations may have race conditions with other threads",
                "thread": "Thread execution order is non-deterministic",
                "executor": "Task scheduling order varies between runs",
                "synchronized": "Lock contention may cause timing variations",
                "default": "Concurrent operation results depend on thread scheduling"
            },
            FlakyCategory.TIME: {
                "currenttimemillis": "System time changes between test runs",
                "nanotime": "Nano time varies with system load",
                "date": "Date comparisons sensitive to execution time",
                "format": "Date parsing affected by locale and timezone",
                "default": "Time-dependent operation varies between runs"
            },
            FlakyCategory.UNORDERED_COLLECTIONS: {
                "hashmap": "HashMap iteration order not guaranteed",
                "hashset": "HashSet iteration order not guaranteed",
                "keyset": "Key iteration order varies between JVM runs",
                "json": "JSON field order may vary",
                "default": "Collection iteration order is non-deterministic"
            },
            FlakyCategory.TEST_ORDER_DEPENDENCY: {
                "static": "Static state persists across tests - order dependent",
                "file": "File system operations depend on test execution order",
                "config": "Configuration state may persist from previous tests",
                "default": "External state dependency - test order sensitive"
            }
        }
        
        cat_descs = descriptions.get(category, {"default": "Non-deterministic operation detected"})
        pattern_lower = pattern_str.lower()
        
        for key, desc in cat_descs.items():
            if key in pattern_lower:
                return desc
        
        return cat_descs["default"]
    
    def _extract_variables(self, code: str):
        """Extract variable declarations and initialize taint tracking"""
        var_pattern = re.compile(
            r'(?:final\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*=',
            re.MULTILINE
        )
        
        for match in var_pattern.finditer(code):
            var_name = match.group(2)
            self.variable_taint[var_name] = set()
    
    def _propagate_taint(self, code: str):
        """Propagate taint from non-deterministic operations to variables"""
        for op in self.operations:
            # Taint directly affected variables
            for var in op.affected_variables:
                if var in self.variable_taint:
                    self.variable_taint[var].add(op.category)
                else:
                    self.variable_taint[var] = {op.category}
                
                # Create data flow edge
                self.data_flows.append(DataFlowEdge(
                    source=op.pattern_matched,
                    target=var,
                    flow_type="direct_effect",
                    location=op.location
                ))
        
        # Simple propagation: check assignments
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Pattern: var1 = var2.method() or var1 = operation(var2)
            assign_match = re.search(r'(\w+)\s*=\s*(.+);', line)
            if assign_match:
                target_var = assign_match.group(1)
                rhs = assign_match.group(2)
                
                # Check if RHS contains any tainted variables
                for tainted_var, categories in list(self.variable_taint.items()):
                    if tainted_var in rhs and categories:
                        if target_var not in self.variable_taint:
                            self.variable_taint[target_var] = set()
                        self.variable_taint[target_var].update(categories)
                        
                        self.data_flows.append(DataFlowEdge(
                            source=tainted_var,
                            target=target_var,
                            flow_type="assignment",
                            location={"line": line_num}
                        ))
    
    def _analyze_assertion(self, line: str, line_num: int):
        """Analyze an assertion statement for nondeterminism exposure"""
        for pattern in self.assertion_patterns:
            match = pattern.search(line)
            if match:
                assertion_type = match.group()
                
                # Extract variables used in assertion
                involved_vars = self._extract_assertion_variables(line)
                
                # Calculate exposure based on tainted variables
                exposure = 0.0
                exposure_sources = []
                
                for var in involved_vars:
                    if var in self.variable_taint and self.variable_taint[var]:
                        for cat in self.variable_taint[var]:
                            exposure = max(exposure, self._get_category_exposure(cat))
                            exposure_sources.append(f"{var} <- {CATEGORY_LABELS[cat]}")
                
                # Determine risk level
                if exposure >= 0.7:
                    risk_level = "HIGH"
                elif exposure >= 0.4:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                self.assertions.append(AssertionAnalysis(
                    assertion_type=assertion_type.split('(')[0],
                    location={"line": line_num, "column": match.start()},
                    expression=line.strip(),
                    involved_variables=involved_vars,
                    nondeterminism_exposure=exposure,
                    exposure_sources=exposure_sources,
                    risk_level=risk_level
                ))
                break
    
    def _extract_assertion_variables(self, line: str) -> List[str]:
        """Extract variables used in an assertion"""
        # Remove string literals
        clean_line = re.sub(r'"[^"]*"', '', line)
        clean_line = re.sub(r"'[^']*'", '', clean_line)
        
        # Find potential variables
        var_pattern = re.compile(r'\b([a-z_]\w*)\b')
        candidates = var_pattern.findall(clean_line)
        
        # Filter to known variables
        keywords = {'assert', 'assertEquals', 'assertTrue', 'assertFalse', 
                   'assertNull', 'assertNotNull', 'assertThat', 'verify',
                   'new', 'null', 'true', 'false', 'this', 'super'}
        
        return [v for v in candidates if v not in keywords and v in self.variable_taint]
    
    def _get_category_exposure(self, category: FlakyCategory) -> float:
        """Get default exposure level for a category"""
        exposures = {
            FlakyCategory.ASYNC_WAIT: 0.85,
            FlakyCategory.CONCURRENCY: 0.8,
            FlakyCategory.TIME: 0.75,
            FlakyCategory.UNORDERED_COLLECTIONS: 0.6,
            FlakyCategory.TEST_ORDER_DEPENDENCY: 0.65
        }
        return exposures.get(category, 0.5)
    
    def _calculate_category_scores(self) -> Dict[FlakyCategory, float]:
        """Calculate confidence scores for each category"""
        scores = {cat: 0.0 for cat in FlakyCategory}
        
        # Aggregate operation confidences by category
        category_confidences = {cat: [] for cat in FlakyCategory}
        for op in self.operations:
            category_confidences[op.category].append(op.confidence)
        
        # Calculate weighted scores
        for cat, confidences in category_confidences.items():
            if confidences:
                # Use max confidence with count bonus
                max_conf = max(confidences)
                count_bonus = min(0.15, len(confidences) * 0.03)
                scores[cat] = min(1.0, max_conf + count_bonus)
        
        # Factor in assertion exposure
        for assertion in self.assertions:
            if assertion.risk_level == "HIGH":
                for var in assertion.involved_variables:
                    if var in self.variable_taint:
                        for cat in self.variable_taint[var]:
                            scores[cat] = min(1.0, scores[cat] + 0.1)
        
        # If no flaky patterns detected, boost non-flaky score
        total_flaky_score = sum(scores[cat] for cat in FlakyCategory if cat != FlakyCategory.NON_FLAKY)
        if total_flaky_score < 0.2:
            scores[FlakyCategory.NON_FLAKY] = 0.9
        elif total_flaky_score < 0.5:
            scores[FlakyCategory.NON_FLAKY] = 0.3
        else:
            scores[FlakyCategory.NON_FLAKY] = 0.1
        
        return scores
    
    def _build_structured_output(self, code: str, category_scores: Dict) -> Dict:
        """Build the final structured output for LLM consumption"""
        
        # Get predicted category (highest score among flaky categories)
        predicted_category = max(
            [cat for cat in FlakyCategory if cat != FlakyCategory.NON_FLAKY],
            key=lambda c: category_scores[c]
        )
        
        # If non-flaky has highest score, use that
        if category_scores[FlakyCategory.NON_FLAKY] > category_scores[predicted_category]:
            predicted_category = FlakyCategory.NON_FLAKY
        
        # Group operations by category
        ops_by_category = {cat: [] for cat in FlakyCategory}
        for op in self.operations:
            ops_by_category[op.category].append(op.to_dict())
        
        # Build taint summary
        taint_summary = {}
        for var, categories in self.variable_taint.items():
            if categories:
                taint_summary[var] = [CATEGORY_LABELS[cat] for cat in categories]
        
        # Identify high-risk assertions
        high_risk_assertions = [
            a.to_dict() for a in self.assertions 
            if a.risk_level in ["HIGH", "MEDIUM"]
        ]
        
        return {
            "analysis_summary": {
                "predicted_category": predicted_category.value,
                "predicted_label": CATEGORY_LABELS[predicted_category],
                "category_scores": {
                    CATEGORY_LABELS[cat]: round(score, 3) 
                    for cat, score in category_scores.items()
                },
                "total_nondeterministic_operations": len(self.operations),
                "total_assertions": len(self.assertions),
                "high_risk_assertions": len(high_risk_assertions)
            },
            
            "nondeterministic_operations": {
                "by_category": {
                    CATEGORY_LABELS[cat]: ops 
                    for cat, ops in ops_by_category.items() if ops
                },
                "all_operations": [op.to_dict() for op in self.operations]
            },
            
            "data_flow": {
                "tainted_variables": taint_summary,
                "flow_edges": [edge.to_dict() for edge in self.data_flows]
            },
            
            "assertion_analysis": {
                "all_assertions": [a.to_dict() for a in self.assertions],
                "high_risk_assertions": high_risk_assertions
            },
            
            "classification_evidence": self._build_classification_evidence(
                predicted_category, category_scores
            )
        }
    
    def _build_classification_evidence(self, predicted: FlakyCategory, 
                                       scores: Dict) -> Dict:
        """Build evidence summary for the classification decision"""
        evidence = {
            "primary_category": CATEGORY_LABELS[predicted],
            "confidence": round(scores[predicted], 3),
            "key_indicators": [],
            "risk_factors": []
        }
        
        # Add key indicators based on detected operations
        for op in self.operations:
            if op.category == predicted and op.confidence >= 0.7:
                indicator = {
                    "pattern": op.pattern_matched,
                    "type": op.operation_type,
                    "line": op.location["line"],
                    "confidence": round(op.confidence, 3)
                }
                if indicator not in evidence["key_indicators"]:
                    evidence["key_indicators"].append(indicator)
        
        # Add risk factors from assertions
        for assertion in self.assertions:
            if assertion.risk_level in ["HIGH", "MEDIUM"]:
                risk = {
                    "assertion": assertion.assertion_type,
                    "line": assertion.location["line"],
                    "exposure": round(assertion.nondeterminism_exposure, 3),
                    "sources": assertion.exposure_sources[:3]  # Limit for brevity
                }
                evidence["risk_factors"].append(risk)
        
        return evidence


def detect_nondeterminism(code: str) -> Dict:
    """Convenience function to detect nondeterminism in code"""
    detector = NonDeterminismDetector()
    return detector.detect(code)
