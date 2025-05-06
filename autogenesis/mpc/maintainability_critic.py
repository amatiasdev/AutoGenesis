"""
Maintainability Critic for AutoGenesis MPC.

This module evaluates the maintainability aspects of generated agent code
as part of the Multi-Perspective Criticism (MPC) process.
"""

import logging
import os
import re
import ast
import subprocess
import glob
from typing import Dict, Any, List, Set, Tuple, Optional

class MaintainabilityCritic:
    """
    Evaluates agent code for maintainability issues.
    
    Analyzes Python code for adherence to best practices, code complexity,
    documentation, and other factors affecting long-term maintainability.
    """
    
    def __init__(self):
        """Initialize the maintainability critic."""
        self.logger = logging.getLogger(__name__)
        
        # Define maintainability checks
        self.maintainability_checks = {
            "docstring_missing": {
                "pattern": r'(?<=def\s+[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)(?:\s*->.*?)?\s*:)(?!\s*[\'"])',
                "description": "Function missing docstring",
                "severity": "medium"
            },
            "too_long_function": {
                "pattern": None,  # Will use AST for better detection
                "description": "Function is too long (>50 lines)",
                "severity": "medium"
            },
            "too_many_parameters": {
                "pattern": None,  # Will use AST for better detection
                "description": "Function has too many parameters (>5)",
                "severity": "medium"
            },
            "too_deep_nesting": {
                "pattern": None,  # Will use AST for better detection
                "description": "Code has deep nesting (>3 levels)",
                "severity": "medium"
            },
            "inconsistent_naming": {
                "pattern": r'(?:class\s+[a-z])|(?:def\s+[A-Z])',
                "description": "Inconsistent naming conventions",
                "severity": "low"
            },
            "magic_numbers": {
                "pattern": r'(?<!\w)[-+]?[0-9]+(?:\.[0-9]+)?(?!\w)',
                "description": "Use of magic numbers that should be constants",
                "severity": "low"
            },
            "commented_code": {
                "pattern": r'^\s*#\s*(?:def|class|import|from|if|for|while)',
                "description": "Commented out code",
                "severity": "low"
            },
            "long_line": {
                "pattern": r'^.{100,}$',
                "description": "Line is too long (>100 characters)",
                "severity": "low"
            },
            "redundant_code": {
                "pattern": None,  # Will use AST for better detection
                "description": "Redundant or duplicate code",
                "severity": "medium"
            },
            "complex_condition": {
                "pattern": r'if\s+(?:[^:]*\band\b[^:]*\band\b)|(?:[^:]*\bor\b[^:]*\bor\b)',
                "description": "Complex conditional expression",
                "severity": "medium"
            }
        }
        
        # Define weights for severity levels
        self.severity_weights = {
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        # Define patterns for ignored files
        self.ignored_file_patterns = [
            r'__pycache__',
            r'\.git',
            r'venv',
            r'\.env',
            r'\.pytest_cache',
            r'\.cache'
        ]
    
    def evaluate(self, code_path: str) -> Dict[str, Any]:
        """
        Evaluate agent code for maintainability issues.
        
        Args:
            code_path: Path to the code to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        self.logger.info(f"Evaluating maintainability of code at {code_path}")
        
        # Find all Python files
        python_files = self._find_python_files(code_path)
        self.logger.info(f"Found {len(python_files)} Python files to evaluate")
        
        # Analyze each file
        findings = []
        for file_path in python_files:
            pattern_findings = self._analyze_file_patterns(file_path)
            findings.extend(pattern_findings)
            
            ast_findings = self._analyze_file_ast(file_path)
            findings.extend(ast_findings)
        
        # Run linting tools if available
        lint_findings = self._run_linting_tools(code_path)
        findings.extend(lint_findings)
        
        # Check for duplicate code
        duplication_findings = self._check_for_duplication(python_files)
        findings.extend(duplication_findings)
        
        # Calculate maintainability score
        max_score = 100
        deductions = self._calculate_deductions(findings)
        score = max(0, max_score - deductions)
        
        # Determine pass/fail threshold
        passed = score >= 70  # 70% is the passing threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings)
        
        return {
            "passed": passed,
            "score": score,
            "findings": findings,
            "recommendations": recommendations,
            "files_analyzed": len(python_files)
        }
    
    def _find_python_files(self, code_path: str) -> List[str]:
        """
        Find all Python files in the code path.
        
        Args:
            code_path: Path to search for Python files
            
        Returns:
            List of Python file paths
        """
        python_files = []
        
        # Walk through directory
        for root, dirs, files in os.walk(code_path):
            # Skip ignored directories
            dirs_to_remove = []
            for i, directory in enumerate(dirs):
                if any(re.match(pattern, directory) for pattern in self.ignored_file_patterns):
                    dirs_to_remove.append(directory)
            
            for directory in dirs_to_remove:
                dirs.remove(directory)
            
            # Add Python files
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_file_patterns(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for maintainability issues using regex patterns.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of maintainability findings
        """
        findings = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for patterns that work on the full content
            for issue_type, pattern_info in self.maintainability_checks.items():
                pattern = pattern_info.get("pattern")
                if pattern:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        # Extract line number
                        line_number = content[:match.start()].count('\n') + 1
                        
                        # Extract the matched line
                        line = content.split('\n')[line_number - 1]
                        
                        # Add finding
                        findings.append({
                            "type": issue_type,
                            "description": pattern_info["description"],
                            "severity": pattern_info["severity"],
                            "file_path": file_path,
                            "line_number": line_number,
                            "line": line.strip()
                        })
            
            # Check for long lines
            for i, line in enumerate(lines):
                if len(line) > 100 and not line.strip().startswith('#'):
                    findings.append({
                        "type": "long_line",
                        "description": "Line is too long (>100 characters)",
                        "severity": "low",
                        "file_path": file_path,
                        "line_number": i + 1,
                        "line": line.strip()
                    })
        
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return findings
    
    def _analyze_file_ast(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for maintainability issues using AST.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of maintainability findings
        """
        findings = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Analyze functions and classes
            findings.extend(self._check_functions(tree, file_path))
            findings.extend(self._check_classes(tree, file_path))
            findings.extend(self._check_complexity(tree, file_path))
            
        except SyntaxError:
            self.logger.warning(f"Syntax error in file {file_path}, skipping AST analysis")
        except Exception as e:
            self.logger.error(f"Error analyzing AST of file {file_path}: {e}")
        
        return findings
    
    def _check_functions(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """
        Check for maintainability issues in functions.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            
        Returns:
            List of findings
        """
        findings = []
        
        # Visitor class to find function issues
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_FunctionDef(self, node):
                # Check for missing docstring
                if (not ast.get_docstring(node) and
                    not node.name.startswith('_') and
                    node.name != '__init__'):
                    self.issues.append({
                        "type": "docstring_missing",
                        "description": f"Function '{node.name}' missing docstring",
                        "severity": "medium",
                        "line_number": node.lineno
                    })
                
                # Check function length (lines of code)
                if hasattr(node, 'end_lineno'):
                    func_length = node.end_lineno - node.lineno
                    if func_length > 50:
                        self.issues.append({
                            "type": "too_long_function",
                            "description": f"Function '{node.name}' is too long ({func_length} lines)",
                            "severity": "medium",
                            "line_number": node.lineno
                        })
                
                # Check number of parameters
                params = [arg for arg in node.args.args 
                          if arg.arg != 'self' and arg.arg != 'cls']
                if len(params) > 5:
                    self.issues.append({
                        "type": "too_many_parameters",
                        "description": f"Function '{node.name}' has too many parameters ({len(params)})",
                        "severity": "medium",
                        "line_number": node.lineno
                    })
                
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                # Same checks for async functions
                self.visit_FunctionDef(node)
        
        visitor = FunctionVisitor()
        visitor.visit(tree)
        
        # Add file path to issues
        for issue in visitor.issues:
            issue["file_path"] = file_path
        
        findings.extend(visitor.issues)
        return findings
    
    def _check_classes(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """
        Check for maintainability issues in classes.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            
        Returns:
            List of findings
        """
        findings = []
        
        # Visitor class to find class issues
        class ClassVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_ClassDef(self, node):
                # Check for missing docstring
                if not ast.get_docstring(node):
                    self.issues.append({
                        "type": "docstring_missing",
                        "description": f"Class '{node.name}' missing docstring",
                        "severity": "medium",
                        "line_number": node.lineno
                    })
                
                # Check for too many methods
                methods = [child for child in node.body 
                           if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if len(methods) > 20:
                    self.issues.append({
                        "type": "too_many_methods",
                        "description": f"Class '{node.name}' has too many methods ({len(methods)})",
                        "severity": "medium",
                        "line_number": node.lineno
                    })
                
                self.generic_visit(node)
        
        visitor = ClassVisitor()
        visitor.visit(tree)
        
        # Add file path to issues
        for issue in visitor.issues:
            issue["file_path"] = file_path
        
        findings.extend(visitor.issues)
        return findings
    
    def _check_complexity(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """
        Check for complexity issues in the code.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            
        Returns:
            List of findings
        """
        findings = []
        
        # Visitor class to find complexity issues
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.current_nesting = 0
                self.max_nesting = 0
                self.nesting_lines = set()
            
            def _increment_nesting(self):
                self.current_nesting += 1
                self.max_nesting = max(self.max_nesting, self.current_nesting)
            
            def _decrement_nesting(self):
                self.current_nesting -= 1
            
            def visit_If(self, node):
                self._increment_nesting()
                if self.current_nesting > 3:
                    self.nesting_lines.add(node.lineno)
                self.generic_visit(node)
                self._decrement_nesting()
            
            def visit_For(self, node):
                self._increment_nesting()
                if self.current_nesting > 3:
                    self.nesting_lines.add(node.lineno)
                self.generic_visit(node)
                self._decrement_nesting()
            
            def visit_While(self, node):
                self._increment_nesting()
                if self.current_nesting > 3:
                    self.nesting_lines.add(node.lineno)
                self.generic_visit(node)
                self._decrement_nesting()
            
            def visit_Try(self, node):
                self._increment_nesting()
                if self.current_nesting > 3:
                    self.nesting_lines.add(node.lineno)
                self.generic_visit(node)
                self._decrement_nesting()
            
            def visit_With(self, node):
                self._increment_nesting()
                if self.current_nesting > 3:
                    self.nesting_lines.add(node.lineno)
                self.generic_visit(node)
                self._decrement_nesting()
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Report deep nesting
        if visitor.max_nesting > 3:
            for line in visitor.nesting_lines:
                findings.append({
                    "type": "too_deep_nesting",
                    "description": f"Code has deep nesting (>{visitor.max_nesting} levels)",
                    "severity": "medium",
                    "file_path": file_path,
                    "line_number": line
                })
        
        return findings
    
    def _run_linting_tools(self, code_path: str) -> List[Dict[str, Any]]:
        """
        Run linting tools on the code.
        
        Args:
            code_path: Path to the code to analyze
            
        Returns:
            List of maintainability findings from linting tools
        """
        findings = []
        
        # Try to run flake8 if available
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=100", "--ignore=E203,E501,W503", code_path],
                capture_output=True,
                text=True
            )
            
            # Parse results
            for line in result.stdout.split('\n'):
                if line.strip():
                    # Parse flake8 output (format: file:line:col: code message)
                    match = re.match(r'(.+):(\d+):(\d+): ([A-Z]\d+) (.+)', line)
                    if match:
                        file_path, line_num, col, code, message = match.groups()
                        
                        # Determine severity based on code
                        severity = "low"
                        if code.startswith(('E', 'F')):
                            severity = "medium"
                        elif code.startswith('W'):
                            severity = "low"
                        
                        findings.append({
                            "type": f"flake8_{code}",
                            "description": message,
                            "severity": severity,
                            "file_path": file_path,
                            "line_number": int(line_num),
                            "column": int(col),
                            "tool": "flake8"
                        })
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.info("Flake8 linter not available")
        
        # Try to run pylint if available
        try:
            # Run pylint for each Python file (to avoid directory issues)
            python_files = self._find_python_files(code_path)
            for file_path in python_files:
                result = subprocess.run(
                    ["pylint", "--output-format=json", file_path],
                    capture_output=True,
                    text=True
                )
                
                # Parse JSON output
                try:
                    import json
                    pylint_results = json.loads(result.stdout)
                    
                    for issue in pylint_results:
                        # Map pylint score to severity
                        severity = "low"
                        if issue.get("type") in ["error", "fatal"]:
                            severity = "high"
                        elif issue.get("type") == "warning":
                            severity = "medium"
                        
                        findings.append({
                            "type": f"pylint_{issue.get('symbol', 'unknown')}",
                            "description": issue.get('message', 'Unknown issue'),
                            "severity": severity,
                            "file_path": issue.get('path', file_path),
                            "line_number": issue.get('line', 0),
                            "column": issue.get('column', 0),
                            "tool": "pylint"
                        })
                
                except (json.JSONDecodeError, ImportError):
                    self.logger.warning(f"Failed to parse Pylint JSON output for {file_path}")
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.info("Pylint linter not available")
        
        return findings
    
    def _check_for_duplication(self, python_files: List[str]) -> List[Dict[str, Any]]:
        """
        Check for code duplication across files.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            List of duplication findings
        """
        findings = []
        
        # Simple duplication detection (line-based)
        # In a real implementation, you would use a more sophisticated tool like CPD
        try:
            # Extract function and method definitions
            functions = {}
            
            for file_path in python_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content, filename=file_path)
                    
                    # Extract functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                            # Skip very small functions (less than 5 lines)
                            if hasattr(node, 'end_lineno'):
                                func_length = node.end_lineno - node.lineno
                                if func_length < 5:
                                    continue
                            
                            # Get function body as string
                            func_body = ast.unparse(node)
                            
                            # Skip very small functions
                            if len(func_body) < 100:
                                continue
                            
                            # Normalize by removing comments and whitespace
                            normalized_body = self._normalize_code(func_body)
                            
                            # Store function
                            functions[(file_path, node.lineno, node.name)] = normalized_body
                
                except (SyntaxError, Exception) as e:
                    self.logger.warning(f"Error parsing {file_path} for duplication check: {e}")
            
            # Check for duplicates
            seen_bodies = {}
            for (file_path, line_num, func_name), body in functions.items():
                if body in seen_bodies:
                    dup_file, dup_line, dup_name = seen_bodies[body]
                    
                    findings.append({
                        "type": "redundant_code",
                        "description": f"Function '{func_name}' is similar to '{dup_name}'",
                        "severity": "medium",
                        "file_path": file_path,
                        "line_number": line_num,
                        "details": f"Similar to function in {dup_file} at line {dup_line}"
                    })
                else:
                    seen_bodies[body] = (file_path, line_num, func_name)
        
        except Exception as e:
            self.logger.error(f"Error checking for code duplication: {e}")
        
        return findings
    
    def _normalize_code(self, code: str) -> str:
        """
        Normalize code for duplication detection.
        
        Args:
            code: Code string
            
        Returns:
            Normalized code string
        """
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Remove blank lines
        code = re.sub(r'\n\s*\n', '\n', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Normalize variable names (replace with placeholders)
        # This is simplified; a real implementation would use a proper parser
        code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', code)
        
        return code
    
    def _calculate_deductions(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate score deductions based on findings.
        
        Args:
            findings: List of maintainability findings
            
        Returns:
            Total deduction points
        """
        deductions = 0
        
        # Deduct points based on severity
        for finding in findings:
            severity = finding.get("severity", "low")
            deductions += self.severity_weights.get(severity, 1)
        
        return deductions
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """
        Generate maintainability recommendations based on findings.
        
        Args:
            findings: List of maintainability findings
            
        Returns:
            List of maintainability recommendations
        """
        recommendations = []
        
        # Group findings by type
        findings_by_type = {}
        for finding in findings:
            finding_type = finding.get("type")
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(finding)
        
        # Generate recommendations for each type
        for finding_type, type_findings in findings_by_type.items():
            # Get first finding for reference
            first_finding = type_findings[0]
            severity = first_finding.get("severity", "low")
            
            # Generate recommendation based on finding type
            if finding_type == "docstring_missing":
                recommendations.append(
                    "Add docstrings to all public functions and classes to improve code documentation."
                )
            elif finding_type == "too_long_function":
                recommendations.append(
                    "Refactor long functions into smaller, more focused functions to improve readability and maintainability."
                )
            elif finding_type == "too_many_parameters":
                recommendations.append(
                    "Reduce the number of function parameters, consider using data structures or classes to group related parameters."
                )
            elif finding_type == "too_deep_nesting":
                recommendations.append(
                    "Reduce code nesting depth by extracting logic into separate functions or using early returns."
                )
            elif finding_type == "inconsistent_naming":
                recommendations.append(
                    "Follow consistent naming conventions: CamelCase for classes and snake_case for functions and variables."
                )
            elif finding_type == "magic_numbers":
                recommendations.append(
                    "Replace magic numbers with named constants to improve code readability and maintainability."
                )
            elif finding_type == "commented_code":
                recommendations.append(
                    "Remove commented-out code. Use version control instead of keeping old code in comments."
                )
            elif finding_type == "long_line":
                recommendations.append(
                    "Break up long lines (>100 characters) to improve code readability."
                )
            elif finding_type == "redundant_code":
                recommendations.append(
                    "Remove or refactor duplicate code into shared functions or classes to avoid maintenance issues."
                )
            elif finding_type == "complex_condition":
                recommendations.append(
                    "Simplify complex conditional expressions by breaking them up or extracting them into named functions."
                )
            elif finding_type == "too_many_methods":
                recommendations.append(
                    "Consider splitting large classes with many methods into smaller, more focused classes."
                )
            elif finding_type.startswith("flake8_"):
                recommendations.append(
                    "Address Flake8 code style and quality issues to improve code consistency."
                )
            elif finding_type.startswith("pylint_"):
                recommendations.append(
                    "Address Pylint code quality issues to improve code maintainability and prevent bugs."
                )
            else:
                # Generic recommendation for other findings
                recommendations.append(
                    f"Address {severity} maintainability issue: {first_finding.get('description', 'Unknown issue')}"
                )
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations