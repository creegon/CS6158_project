"""
Java AST Analyzer for Flaky Test Detection
===========================================
Parses Java test code and extracts structural information for flaky test analysis.
"""

import re
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    import javalang
    from javalang.tree import MethodInvocation, VariableDeclarator, LocalVariableDeclaration
    from javalang.tree import MethodDeclaration, ClassDeclaration, FieldDeclaration
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False


class NodeType(Enum):
    METHOD_CALL = "method_call"
    VARIABLE_DECL = "variable_declaration"
    FIELD_DECL = "field_declaration"
    ASSERTION = "assertion"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    TRY_CATCH = "try_catch"
    THREAD_OP = "thread_operation"
    SYNC_OP = "synchronization"


@dataclass
class CodeLocation:
    """Represents a location in source code"""
    line: int
    column: int = 0
    end_line: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {"line": self.line, "column": self.column, "end_line": self.end_line}


@dataclass  
class MethodCall:
    """Represents a method invocation"""
    name: str
    qualifier: Optional[str]  # e.g., Thread for Thread.sleep
    arguments: List[str]
    location: CodeLocation
    is_nondeterministic: bool = False
    nondeterminism_type: Optional[str] = None
    affected_variables: List[str] = field(default_factory=list)
    
    def full_name(self) -> str:
        if self.qualifier:
            return f"{self.qualifier}.{self.name}"
        return self.name
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "qualifier": self.qualifier,
            "full_name": self.full_name(),
            "arguments": self.arguments,
            "location": self.location.to_dict(),
            "is_nondeterministic": self.is_nondeterministic,
            "nondeterminism_type": self.nondeterminism_type,
            "affected_variables": self.affected_variables
        }


@dataclass
class VariableInfo:
    """Information about a variable"""
    name: str
    var_type: str
    location: CodeLocation
    is_affected_by_nondeterminism: bool = False
    nondeterminism_sources: List[str] = field(default_factory=list)
    used_in_assertions: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.var_type,
            "location": self.location.to_dict(),
            "is_affected_by_nondeterminism": self.is_affected_by_nondeterminism,
            "nondeterminism_sources": self.nondeterminism_sources,
            "used_in_assertions": self.used_in_assertions
        }


@dataclass
class AssertionInfo:
    """Information about an assertion"""
    assertion_type: str  # assertEquals, assertTrue, etc.
    location: CodeLocation
    variables_checked: List[str]
    is_checking_nondeterministic: bool = False
    nondeterminism_path: List[str] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "assertion_type": self.assertion_type,
            "location": self.location.to_dict(),
            "variables_checked": self.variables_checked,
            "is_checking_nondeterministic": self.is_checking_nondeterministic,
            "nondeterminism_path": self.nondeterminism_path,
            "raw_text": self.raw_text
        }


@dataclass
class LoopInfo:
    """Information about a loop construct"""
    loop_type: str  # for, while, do-while
    location: CodeLocation
    contains_wait: bool = False
    contains_assertion: bool = False
    iteration_variable: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "loop_type": self.loop_type,
            "location": self.location.to_dict(),
            "contains_wait": self.contains_wait,
            "contains_assertion": self.contains_assertion,
            "iteration_variable": self.iteration_variable
        }


class JavaASTAnalyzer:
    """Analyzes Java code using AST parsing and pattern matching"""
    
    def __init__(self):
        self.method_calls: List[MethodCall] = []
        self.variables: Dict[str, VariableInfo] = {}
        self.assertions: List[AssertionInfo] = []
        self.loops: List[LoopInfo] = []
        self.raw_code: str = ""
        
        # Patterns for fallback regex-based analysis
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for code analysis"""
        # Method call patterns
        self.method_call_pattern = re.compile(
            r'(\w+(?:\.\w+)*)\s*\.\s*(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        
        # Static method call patterns
        self.static_method_pattern = re.compile(
            r'([A-Z]\w+)\s*\.\s*(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        
        # Variable declaration patterns
        self.var_decl_pattern = re.compile(
            r'(?:final\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*=\s*([^;]+);',
            re.MULTILINE
        )
        
        # Loop patterns
        self.for_loop_pattern = re.compile(
            r'for\s*\(([^)]+)\)\s*\{',
            re.MULTILINE
        )
        self.while_loop_pattern = re.compile(
            r'while\s*\(([^)]+)\)\s*\{',
            re.MULTILINE
        )
        
        # Assertion patterns  
        self.assertion_pattern = re.compile(
            r'(assert\w*|Assert\.\w+|verify\w*|fail)\s*\(([^;]+)\);',
            re.MULTILINE | re.DOTALL
        )
        
        # Thread/async patterns
        self.thread_sleep_pattern = re.compile(r'Thread\.sleep\s*\(([^)]+)\)')
        self.await_pattern = re.compile(r'\.await\s*\(([^)]*)\)')
        self.future_get_pattern = re.compile(r'\.get\s*\(([^)]*)\)')
        
    def analyze(self, code: str) -> Dict:
        """
        Main analysis method - returns structured information about the test
        """
        self.raw_code = code
        self._reset()
        
        # Try AST parsing first, fall back to regex
        if HAS_JAVALANG:
            try:
                self._analyze_with_ast(code)
            except Exception:
                self._analyze_with_regex(code)
        else:
            self._analyze_with_regex(code)
        
        # Build data flow graph
        self._build_data_flow()
        
        # Mark assertions checking nondeterministic values
        self._mark_nondeterministic_assertions()
        
        return self._build_result()
    
    def _reset(self):
        """Reset state for new analysis"""
        self.method_calls = []
        self.variables = {}
        self.assertions = []
        self.loops = []
    
    def _analyze_with_ast(self, code: str):
        """Analyze code using javalang AST parser"""
        # Wrap code in class if needed for parsing
        wrapped_code = self._wrap_code_for_parsing(code)
        
        try:
            tree = javalang.parse.parse(wrapped_code)
            self._extract_from_ast(tree)
        except javalang.parser.JavaSyntaxError:
            # Fall back to regex
            self._analyze_with_regex(code)
    
    def _wrap_code_for_parsing(self, code: str) -> str:
        """Wrap code snippet in a class for parsing"""
        if "class " in code and "public class" not in code:
            code = "public " + code
        if "class " not in code:
            code = f"public class TestWrapper {{\n{code}\n}}"
        return code
    
    def _extract_from_ast(self, tree):
        """Extract information from parsed AST"""
        for path, node in tree:
            if isinstance(node, MethodInvocation):
                self._process_method_invocation(node, path)
            elif isinstance(node, LocalVariableDeclaration):
                self._process_variable_declaration(node, path)
    
    def _process_method_invocation(self, node, path):
        """Process a method invocation node"""
        qualifier = None
        if hasattr(node, 'qualifier') and node.qualifier:
            qualifier = str(node.qualifier)
        
        args = []
        if hasattr(node, 'arguments') and node.arguments:
            args = [str(arg) for arg in node.arguments]
        
        location = CodeLocation(
            line=node.position.line if hasattr(node, 'position') and node.position else 0
        )
        
        method_call = MethodCall(
            name=node.member,
            qualifier=qualifier,
            arguments=args,
            location=location
        )
        
        self.method_calls.append(method_call)
    
    def _process_variable_declaration(self, node, path):
        """Process a variable declaration node"""
        var_type = str(node.type.name) if hasattr(node.type, 'name') else str(node.type)
        
        for declarator in node.declarators:
            location = CodeLocation(
                line=declarator.position.line if hasattr(declarator, 'position') and declarator.position else 0
            )
            
            var_info = VariableInfo(
                name=declarator.name,
                var_type=var_type,
                location=location
            )
            self.variables[declarator.name] = var_info
    
    def _analyze_with_regex(self, code: str):
        """Analyze code using regex patterns (fallback method)"""
        lines = code.split('\n')
        
        # Extract method calls
        for line_num, line in enumerate(lines, 1):
            self._extract_method_calls_from_line(line, line_num)
            self._extract_variables_from_line(line, line_num)
            self._extract_assertions_from_line(line, line_num)
            self._extract_loops_from_line(line, line_num)
    
    def _extract_method_calls_from_line(self, line: str, line_num: int):
        """Extract method calls from a single line"""
        # Instance method calls
        for match in self.method_call_pattern.finditer(line):
            qualifier = match.group(1)
            method_name = match.group(2)
            args = [a.strip() for a in match.group(3).split(',') if a.strip()]
            
            method_call = MethodCall(
                name=method_name,
                qualifier=qualifier,
                arguments=args,
                location=CodeLocation(line=line_num, column=match.start())
            )
            self.method_calls.append(method_call)
        
        # Static method calls
        for match in self.static_method_pattern.finditer(line):
            class_name = match.group(1)
            method_name = match.group(2)
            args = [a.strip() for a in match.group(3).split(',') if a.strip()]
            
            # Avoid duplicates
            full_name = f"{class_name}.{method_name}"
            if not any(mc.full_name() == full_name and mc.location.line == line_num 
                      for mc in self.method_calls):
                method_call = MethodCall(
                    name=method_name,
                    qualifier=class_name,
                    arguments=args,
                    location=CodeLocation(line=line_num, column=match.start())
                )
                self.method_calls.append(method_call)
    
    def _extract_variables_from_line(self, line: str, line_num: int):
        """Extract variable declarations from a single line"""
        for match in self.var_decl_pattern.finditer(line):
            var_type = match.group(1)
            var_name = match.group(2)
            
            if var_name not in self.variables:
                self.variables[var_name] = VariableInfo(
                    name=var_name,
                    var_type=var_type,
                    location=CodeLocation(line=line_num, column=match.start())
                )
    
    def _extract_assertions_from_line(self, line: str, line_num: int):
        """Extract assertions from a single line"""
        for match in self.assertion_pattern.finditer(line):
            assertion_type = match.group(1)
            assertion_body = match.group(2)
            
            # Extract variables used in assertion
            variables_checked = self._extract_variables_from_expression(assertion_body)
            
            assertion = AssertionInfo(
                assertion_type=assertion_type,
                location=CodeLocation(line=line_num, column=match.start()),
                variables_checked=variables_checked,
                raw_text=line.strip()
            )
            self.assertions.append(assertion)
    
    def _extract_loops_from_line(self, line: str, line_num: int):
        """Extract loop constructs"""
        for_match = self.for_loop_pattern.search(line)
        if for_match:
            loop_info = LoopInfo(
                loop_type="for",
                location=CodeLocation(line=line_num)
            )
            self.loops.append(loop_info)
        
        while_match = self.while_loop_pattern.search(line)
        if while_match:
            loop_info = LoopInfo(
                loop_type="while",
                location=CodeLocation(line=line_num)
            )
            self.loops.append(loop_info)
    
    def _extract_variables_from_expression(self, expr: str) -> List[str]:
        """Extract variable names from an expression"""
        # Remove string literals
        expr = re.sub(r'"[^"]*"', '', expr)
        expr = re.sub(r"'[^']*'", '', expr)
        
        # Find identifiers (variable names)
        var_pattern = re.compile(r'\b([a-z_]\w*)\b', re.IGNORECASE)
        potential_vars = var_pattern.findall(expr)
        
        # Filter out keywords and common method names
        keywords = {'new', 'null', 'true', 'false', 'this', 'super', 'if', 'else',
                   'for', 'while', 'return', 'void', 'int', 'long', 'double', 
                   'float', 'boolean', 'String', 'Object', 'assertEquals', 
                   'assertTrue', 'assertFalse', 'assertNull', 'assertNotNull',
                   'Assert', 'Matchers', 'greaterThan', 'lessThan', 'equalTo'}
        
        return [v for v in potential_vars if v not in keywords and v in self.variables]
    
    def _build_data_flow(self):
        """Build data flow relationships between variables and method calls"""
        # Mark variables affected by nondeterministic operations
        for var_name, var_info in self.variables.items():
            sources = self._find_nondeterminism_sources_for_variable(var_name)
            if sources:
                var_info.is_affected_by_nondeterminism = True
                var_info.nondeterminism_sources = sources
            
            # Check if variable is used in any assertion
            for assertion in self.assertions:
                if var_name in assertion.variables_checked:
                    var_info.used_in_assertions = True
                    break
    
    def _find_nondeterminism_sources_for_variable(self, var_name: str) -> List[str]:
        """Find nondeterminism sources that affect a variable"""
        sources = []
        var_info = self.variables.get(var_name)
        if not var_info:
            return sources
        
        # Check the line where variable is declared
        var_line_num = var_info.location.line
        
        # Find method calls on or before this line that might affect it
        for mc in self.method_calls:
            if mc.is_nondeterministic:
                # Check if this method call is in the variable's definition
                if mc.location.line == var_line_num:
                    sources.append(f"{mc.full_name()} (line {mc.location.line})")
                # Or if variable is passed as argument
                elif var_name in mc.arguments:
                    sources.append(f"{mc.full_name()} (line {mc.location.line})")
        
        return sources
    
    def _mark_nondeterministic_assertions(self):
        """Mark assertions that check nondeterministic values"""
        for assertion in self.assertions:
            for var_name in assertion.variables_checked:
                var_info = self.variables.get(var_name)
                if var_info and var_info.is_affected_by_nondeterminism:
                    assertion.is_checking_nondeterministic = True
                    assertion.nondeterminism_path = var_info.nondeterminism_sources.copy()
                    break
    
    def _build_result(self) -> Dict:
        """Build the final analysis result"""
        return {
            "method_calls": [mc.to_dict() for mc in self.method_calls],
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "assertions": [a.to_dict() for a in self.assertions],
            "loops": [l.to_dict() for l in self.loops],
            "summary": {
                "total_method_calls": len(self.method_calls),
                "total_variables": len(self.variables),
                "total_assertions": len(self.assertions),
                "total_loops": len(self.loops),
                "nondeterministic_method_calls": sum(1 for mc in self.method_calls if mc.is_nondeterministic),
                "affected_variables": sum(1 for v in self.variables.values() if v.is_affected_by_nondeterminism),
                "risky_assertions": sum(1 for a in self.assertions if a.is_checking_nondeterministic)
            }
        }


def analyze_java_code(code: str) -> Dict:
    """Convenience function to analyze Java code"""
    analyzer = JavaASTAnalyzer()
    return analyzer.analyze(code)
