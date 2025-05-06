"""
Scalability Critic for AutoGenesis MPC.

This module evaluates the scalability aspects of generated agent code
as part of the Multi-Perspective Criticism (MPC) process.
"""

import logging
import os
import re
import ast
import json
from typing import Dict, Any, List, Set, Tuple, Optional, Union

class ScalabilityCritic:
    """
    Evaluates agent code for scalability issues.
    
    Analyzes Python code for patterns that may impact performance at scale,
    providing a scalability score and recommendations for improvement.
    """
    
    def __init__(self):
        """Initialize the scalability critic."""
        self.logger = logging.getLogger(__name__)
        
        # Define scalability patterns to check for
        self.scalability_patterns = {
            "inefficient_loops": {
                "pattern": r'for\s+.*\s+in\s+range\(.+\):\s*\n\s+for\s+.*\s+in\s+range\(.+\):', 
                "description": "Nested loops may cause quadratic time complexity",
                "severity": "high"
            },
            "large_data_in_memory": {
                "pattern": r'\.read\(\)|\.readlines\(\)', 
                "description": "Reading entire file into memory may cause issues with large files",
                "severity": "medium"
            },
            "inefficient_list_operations": {
                "pattern": r'for\s+.*\s+in\s+.*:\s*\n\s+if\s+.*\s+in\s+', 
                "description": "Using list for membership testing is inefficient, consider using a set or dictionary",
                "severity": "medium"
            },
            "blocking_io": {
                "pattern": r'requests\.|urllib\.|ftplib\.|telnetlib\.', 
                "description": "Synchronous I/O operations may block execution",
                "severity": "medium"
            },
            "unbounded_memory": {
                "pattern": r'\.append\(|\.extend\(', 
                "description": "Continuously growing data structures without limits may cause memory issues",
                "severity": "low"
            },
            "inefficient_string_concat": {
                "pattern": r'\+=\s*[\'"]{1}', 
                "description": "Repeated string concatenation in loops is inefficient, use list and join instead",
                "severity": "low"
            },
            "global_variables": {
                "pattern": r'^\s*global\s+', 
                "description": "Global variables can cause thread safety issues",
                "severity": "medium"
            },
            "sleep_in_loop": {
                "pattern": r'for\s+.*:.*\n\s+.*sleep\(', 
                "description": "Using sleep in loops can cause performance issues",
                "severity": "medium"
            },
            "recursion_without_limit": {
                "pattern": r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\(.*\).*\n[^#]*\1\(', 
                "description": "Recursion without depth limit may cause stack overflow",
                "severity": "high"
            }
        }
        
        # Define weights for severity levels
        self.severity_weights = {
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        # Define AST patterns for more advanced analysis
        self.ast_checks = [
            self._check_for_large_loops,
            self._check_for_memory_issues,
            self._check_for_cpu_intensive_operations,
            self._check_for_database_issues,
            self._check_for_parallelization
        ]
    
    def evaluate(self, 
                code_path: str,
                resource_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate agent code for scalability issues.
        
        Args:
            code_path: Path to the code to evaluate
            resource_requirements: Resource requirements for the agent
            
        Returns:
            Dict containing evaluation results
        """
        self.logger.info(f"Evaluating scalability of code at {code_path}")
        
        # Find all Python files
        python_files = self._find_python_files(code_path)
        self.logger.info(f"Found {len(python_files)} Python files to evaluate")
        
        # Analyze each file
        findings = []
        for file_path in python_files:
            pattern_findings = self._analyze_file_patterns(file_path)
            findings.extend(pattern_findings)
            
            ast_findings = self._analyze_file_ast(file_path, resource_requirements)
            findings.extend(ast_findings)
        
        # Calculate scalability score
        max_score = 100
        deductions = self._calculate_deductions(findings)
        score = max(0, max_score - deductions)
        
        # Determine pass/fail threshold
        passed = score >= 70  # 70% is the passing threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, resource_requirements)
        
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
        for root, _, files in os.walk(code_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_file_patterns(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for scalability issues using regex patterns.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of scalability findings
        """
        findings = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for each scalability pattern
            for issue_type, pattern_info in self.scalability_patterns.items():
                matches = re.finditer(pattern_info["pattern"], content, re.MULTILINE)
                
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
        
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return findings
    
    def _analyze_file_ast(self, 
                        file_path: str,
                        resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for scalability issues using AST.
        
        Args:
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of scalability findings
        """
        findings = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Run each AST check
            for check_func in self.ast_checks:
                check_findings = check_func(tree, file_path, resource_requirements)
                findings.extend(check_findings)
        
        except SyntaxError:
            self.logger.warning(f"Syntax error in file {file_path}, skipping AST analysis")
        except Exception as e:
            self.logger.error(f"Error analyzing AST of file {file_path}: {e}")
        
        return findings
    
    def _check_for_large_loops(self, 
                              tree: ast.AST, 
                              file_path: str,
                              resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for nested loops and loops with large iteration counts.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of findings
        """
        findings = []
        
        # Visitor class to find nested loops
        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.max_loop_depth = 0
                self.loops = []
            
            def visit_For(self, node):
                self.loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
                self.loops.append((node, self.loop_depth))
                self.generic_visit(node)
                self.loop_depth -= 1
            
            def visit_While(self, node):
                self.loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
                self.loops.append((node, self.loop_depth))
                self.generic_visit(node)
                self.loop_depth -= 1
        
        visitor = LoopVisitor()
        visitor.visit(tree)
        
        # Check for deeply nested loops
        if visitor.max_loop_depth >= 3:
            findings.append({
                "type": "deeply_nested_loops",
                "description": f"Found loops nested {visitor.max_loop_depth} levels deep, which can lead to exponential time complexity",
                "severity": "high",
                "file_path": file_path,
                "line_number": 0,  # We don't have a specific line number
            })
        
        # Check for large iteration counts in loops
        for node, depth in visitor.loops:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range' and len(node.iter.args) > 0:
                    # Try to determine the range size
                    range_size = None
                    if isinstance(node.iter.args[0], ast.Num):
                        if len(node.iter.args) == 1:
                            range_size = node.iter.args[0].n
                        elif len(node.iter.args) >= 2 and isinstance(node.iter.args[1], ast.Num):
                            range_size = node.iter.args[1].n - node.iter.args[0].n
                    
                    if range_size and range_size > 10000:
                        findings.append({
                            "type": "large_iteration_count",
                            "description": f"Loop with potentially large iteration count ({range_size}), which may cause performance issues",
                            "severity": "medium",
                            "file_path": file_path,
                            "line_number": node.lineno,
                            "details": f"Loop at line {node.lineno} has range size of {range_size}"
                        })
        
        return findings
    
    def _check_for_memory_issues(self, 
                               tree: ast.AST, 
                               file_path: str,
                               resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for memory-related issues.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of findings
        """
        findings = []
        
        # Get memory limit if specified
        memory_limit_mb = resource_requirements.get('memory_mb', 512)
        
        # Visitor class to find memory issues
        class MemoryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.in_loop = False
                self.loop_stack = []
            
            def visit_For(self, node):
                self.loop_stack.append(node)
                self.in_loop = True
                self.generic_visit(node)
                self.loop_stack.pop()
                self.in_loop = len(self.loop_stack) > 0
            
            def visit_While(self, node):
                self.loop_stack.append(node)
                self.in_loop = True
                self.generic_visit(node)
                self.loop_stack.pop()
                self.in_loop = len(self.loop_stack) > 0
            
            def visit_Call(self, node):
                # Check for memory-intensive operations inside loops
                if self.in_loop and isinstance(node.func, ast.Attribute):
                    # Check for list/dict/set growing operations inside loops
                    if node.func.attr in ['append', 'extend', 'update']:
                        self.issues.append({
                            "type": "unbounded_growth",
                            "description": "Collection growing inside loop without clear bounds",
                            "severity": "medium",
                            "line_number": node.lineno,
                            "details": f"Collection growth via {node.func.attr} inside loop at line {node.lineno}"
                        })
                
                # Check for reading entire files into memory
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['read', 'readlines', 'load', 'loads']):
                    self.issues.append({
                        "type": "full_data_load",
                        "description": "Loading entire data set into memory at once",
                        "severity": "medium",
                        "line_number": node.lineno,
                        "details": f"Full data load via {node.func.attr} at line {node.lineno}"
                    })
                
                self.generic_visit(node)
            
            def visit_List(self, node):
                # Check for large list literals
                if len(node.elts) > 1000:
                    self.issues.append({
                        "type": "large_literal",
                        "description": f"Large list literal with {len(node.elts)} elements",
                        "severity": "low",
                        "line_number": node.lineno,
                        "details": f"Large list literal at line {node.lineno}"
                    })
                self.generic_visit(node)
            
            def visit_Dict(self, node):
                # Check for large dict literals
                if len(node.keys) > 1000:
                    self.issues.append({
                        "type": "large_literal",
                        "description": f"Large dictionary literal with {len(node.keys)} elements",
                        "severity": "low",
                        "line_number": node.lineno,
                        "details": f"Large dictionary literal at line {node.lineno}"
                    })
                self.generic_visit(node)
        
        visitor = MemoryVisitor()
        visitor.visit(tree)
        
        # Add issues to findings
        for issue in visitor.issues:
            issue["file_path"] = file_path
            findings.append(issue)
        
        # Check for memory limit considerations
        if memory_limit_mb < 256:
            # If memory is very constrained, add a general warning
            findings.append({
                "type": "limited_memory",
                "description": f"Available memory is limited to {memory_limit_mb}MB, which may affect large data processing capabilities",
                "severity": "medium",
                "file_path": file_path,
                "line_number": 0,
                "details": f"Memory limit: {memory_limit_mb}MB"
            })
        
        return findings
    
    def _check_for_cpu_intensive_operations(self, 
                                          tree: ast.AST, 
                                          file_path: str,
                                          resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for CPU-intensive operations.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of findings
        """
        findings = []
        
        # Get CPU limit if specified
        cpu_limit = resource_requirements.get('cpu_cores', 0.5)
        
        # Visitor class to find CPU-intensive operations
        class CPUVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.cpu_intensive_imports = {
                    'numpy': ['linalg', 'fft', 'convolve', 'correlate', 'einsum'],
                    'pandas': ['apply', 'groupby', 'merge', 'join', 'pivot', 'resample'],
                    'scipy': ['optimize', 'signal', 'ndimage'],
                    'sklearn': ['fit', 'predict', 'transform'],
                    'tensorflow': ['fit', 'predict', 'compile'],
                    'torch': ['backward', 'cuda', 'train']
                }
                self.imports = {}
            
            def visit_Import(self, node):
                for name in node.names:
                    self.imports[name.asname or name.name] = name.name
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module = node.module
                for name in node.names:
                    self.imports[name.asname or name.name] = f"{module}.{name.name}"
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for CPU-intensive operations
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        # Check if the module is in our CPU-intensive list
                        module_name = node.func.value.id
                        if module_name in self.imports:
                            real_module = self.imports[module_name]
                            # Check for the base module (e.g., 'numpy' from 'numpy.fft')
                            base_module = real_module.split('.')[0]
                            
                            if base_module in self.cpu_intensive_imports:
                                intensive_functions = self.cpu_intensive_imports[base_module]
                                if node.func.attr in intensive_functions:
                                    self.issues.append({
                                        "type": "cpu_intensive_operation",
                                        "description": f"CPU-intensive operation: {module_name}.{node.func.attr}",
                                        "severity": "medium",
                                        "line_number": node.lineno,
                                        "details": f"CPU-intensive operation at line {node.lineno}"
                                    })
                
                self.generic_visit(node)
        
        visitor = CPUVisitor()
        visitor.visit(tree)
        
        # Add issues to findings
        for issue in visitor.issues:
            issue["file_path"] = file_path
            findings.append(issue)
        
        # Check for CPU limit considerations
        if cpu_limit < 0.5:
            # If CPU is very constrained, add a general warning
            findings.append({
                "type": "limited_cpu",
                "description": f"Available CPU is limited to {cpu_limit} cores, which may affect computation-heavy operations",
                "severity": "medium",
                "file_path": file_path,
                "line_number": 0,
                "details": f"CPU limit: {cpu_limit} cores"
            })
        
        return findings
    
    def _check_for_database_issues(self, 
                                 tree: ast.AST, 
                                 file_path: str,
                                 resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for database-related scalability issues.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of findings
        """
        findings = []
        
        # Visitor class to find database issues
        class DBVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.in_loop = False
                self.loop_stack = []
                self.db_modules = ['sqlite3', 'psycopg2', 'mysql', 'pymongo', 'sqlalchemy', 'django.db']
                self.db_operations = ['execute', 'query', 'find', 'find_one', 'insert', 'update', 'delete', 'remove']
                self.imports = {}
            
            def visit_Import(self, node):
                for name in node.names:
                    module_name = name.name.split('.')[0]
                    if module_name in self.db_modules:
                        self.imports[name.asname or name.name] = name.name
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module.split('.')[0] in self.db_modules:
                    for name in node.names:
                        self.imports[name.asname or name.name] = f"{node.module}.{name.name}"
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.loop_stack.append(node)
                self.in_loop = True
                self.generic_visit(node)
                self.loop_stack.pop()
                self.in_loop = len(self.loop_stack) > 0
            
            def visit_While(self, node):
                self.loop_stack.append(node)
                self.in_loop = True
                self.generic_visit(node)
                self.loop_stack.pop()
                self.in_loop = len(self.loop_stack) > 0
            
            def visit_Call(self, node):
                # Check for database operations in loops
                if self.in_loop and isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.db_operations:
                        # Check if it's a database operation
                        is_db_op = False
                        if isinstance(node.func.value, ast.Name):
                            # Direct module attribute access
                            module_name = node.func.value.id
                            if module_name in self.imports:
                                is_db_op = True
                        elif isinstance(node.func.value, ast.Attribute):
                            # Nested attribute access (e.g., connection.cursor().execute)
                            is_db_op = True
                        
                        if is_db_op:
                            self.issues.append({
                                "type": "db_operation_in_loop",
                                "description": f"Database operation '{node.func.attr}' inside loop",
                                "severity": "high",
                                "line_number": node.lineno,
                                "details": f"Database operation in loop at line {node.lineno}"
                            })
                
                self.generic_visit(node)
        
        visitor = DBVisitor()
        visitor.visit(tree)
        
        # Add issues to findings
        for issue in visitor.issues:
            issue["file_path"] = file_path
            findings.append(issue)
        
        return findings
    
    def _check_for_parallelization(self, 
                                 tree: ast.AST, 
                                 file_path: str,
                                 resource_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for opportunities to use parallelization.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of findings
        """
        findings = []
        
        # Get CPU limit to determine if parallelization is beneficial
        cpu_limit = resource_requirements.get('cpu_cores', 0.5)
        
        # Visitor class to find parallelization opportunities
        class ParallelVisitor(ast.NodeVisitor):
            def __init__(self, cpu_cores):
                self.issues = []
                self.cpu_cores = cpu_cores
                self.has_parallelization = False
                self.parallel_modules = ['multiprocessing', 'threading', 'concurrent.futures', 'asyncio']
                self.imports = set()
            
            def visit_Import(self, node):
                for name in node.names:
                    module_name = name.name.split('.')[0]
                    if module_name in self.parallel_modules:
                        self.has_parallelization = True
                    self.imports.add(module_name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.parallel_modules:
                        self.has_parallelization = True
                    self.imports.add(module_name)
                self.generic_visit(node)
            
            def visit_For(self, node):
                # If we find a significant loop that could benefit from parallelization,
                # and the code isn't already using parallel processing, suggest it
                # Look for loops that iterate over a potentially large collection
                if self.cpu_cores >= 2 and not self.has_parallelization:
                    # Check if the loop is iterating over a potentially large collection
                    is_large_iteration = False
                    
                    # Check for iteration over range with large number
                    if (isinstance(node.iter, ast.Call) and 
                        isinstance(node.iter.func, ast.Name) and 
                        node.iter.func.id == 'range' and
                        len(node.iter.args) > 0 and
                        isinstance(node.iter.args[0], ast.Num) and
                        node.iter.args[0].n > 1000):
                        is_large_iteration = True
                    
                    # Check for iteration over a variable
                    if isinstance(node.iter, ast.Name):
                        # This is a heuristic, we can't be sure if it's large
                        is_large_iteration = True
                    
                    if is_large_iteration:
                        self.issues.append({
                            "type": "parallelization_opportunity",
                            "description": "Loop at line {node.lineno} could potentially benefit from parallelization",
                            "severity": "low",
                            "line_number": node.lineno,
                            "details": f"Consider parallel processing for loop at line {node.lineno} on a {self.cpu_cores} core system"
                        })
                
                self.generic_visit(node)
        
        visitor = ParallelVisitor(cpu_limit)
        visitor.visit(tree)
        
        # Add issues to findings
        for issue in visitor.issues:
            issue["file_path"] = file_path
            findings.append(issue)
        
        # If no parallelization is used but we have multiple cores, suggest it as a general recommendation
        if cpu_limit >= 2 and not visitor.has_parallelization:
            # Check if the file has any significant processing that might benefit from parallelization
            has_significant_processing = False
            for module in ['pandas', 'numpy', 'scipy', 'sklearn', 'requests']:
                if module in visitor.imports:
                    has_significant_processing = True
                    break
            
            if has_significant_processing:
                findings.append({
                    "type": "missing_parallelization",
                    "description": f"Code could benefit from parallelization on a {cpu_limit} core system",
                    "severity": "medium",
                    "file_path": file_path,
                    "line_number": 0,
                    "details": "Consider using multiprocessing, threading, or async for better performance"
                })
        
        return findings
    
    def _calculate_deductions(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate score deductions based on findings.
        
        Args:
            findings: List of scalability findings
            
        Returns:
            Total deduction points
        """
        deductions = 0
        
        # Deduct points based on severity
        for finding in findings:
            severity = finding.get("severity", "low")
            deductions += self.severity_weights.get(severity, 1)
        
        return deductions
    
    def _generate_recommendations(self, 
                                findings: List[Dict[str, Any]],
                                resource_requirements: Dict[str, Any]) -> List[str]:
        """
        Generate scalability recommendations based on findings.
        
        Args:
            findings: List of scalability findings
            resource_requirements: Resource requirements for the agent
            
        Returns:
            List of scalability recommendations
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
            if finding_type == "inefficient_loops":
                recommendations.append(
                    "Optimize nested loops to reduce time complexity. Consider using better algorithms or data structures."
                )
            elif finding_type == "large_data_in_memory":
                recommendations.append(
                    "Use streaming or chunking approaches when reading large files instead of loading entire files into memory."
                )
            elif finding_type == "inefficient_list_operations":
                recommendations.append(
                    "Use sets or dictionaries instead of lists for membership testing to improve performance."
                )
            elif finding_type == "blocking_io":
                recommendations.append(
                    "Consider using asynchronous I/O operations to avoid blocking execution, especially for network operations."
                )
            elif finding_type == "unbounded_memory":
                recommendations.append(
                    "Implement limits for growing data structures to prevent memory issues, or process data in chunks."
                )
            elif finding_type == "inefficient_string_concat":
                recommendations.append(
                    "Replace repeated string concatenation with list append and join operations for better performance."
                )
            elif finding_type == "global_variables":
                recommendations.append(
                    "Avoid using global variables to improve thread safety and reduce potential concurrency issues."
                )
            elif finding_type == "sleep_in_loop":
                recommendations.append(
                    "Replace sleep calls in loops with event-based approaches or async patterns for better responsiveness."
                )
            elif finding_type == "recursion_without_limit":
                recommendations.append(
                    "Implement a depth limit for recursive functions to prevent stack overflow errors."
                )
            elif finding_type == "deeply_nested_loops":
                recommendations.append(
                    "Refactor deeply nested loops to reduce time complexity. Consider using vectorized operations or more efficient algorithms."
                )
            elif finding_type == "large_iteration_count":
                recommendations.append(
                    "For loops with large iteration counts, consider chunking the work or implementing pagination."
                )
            elif finding_type == "unbounded_growth":
                recommendations.append(
                    "Implement size limits for collections that grow inside loops to prevent memory issues."
                )
            elif finding_type == "full_data_load":
                recommendations.append(
                    "Use iterators, generators, or streaming approaches instead of loading entire datasets into memory."
                )
            elif finding_type == "limited_memory":
                recommendations.append(
                    "Optimize memory usage due to limited available memory. Use streaming processing and avoid keeping large datasets in memory."
                )
            elif finding_type == "cpu_intensive_operation":
                recommendations.append(
                    "Optimize CPU-intensive operations. Consider caching results, optimizing algorithms, or distributing workloads."
                )
            elif finding_type == "limited_cpu":
                recommendations.append(
                    "Optimize CPU usage due to limited available CPU cores. Avoid compute-intensive operations or implement efficient algorithms."
                )
            elif finding_type == "db_operation_in_loop":
                recommendations.append(
                    "Avoid database operations inside loops. Use batch operations, bulk inserts, or proper query conditions instead."
                )
            elif finding_type == "parallelization_opportunity" or finding_type == "missing_parallelization":
                recommendations.append(
                    "Implement parallelization using multiprocessing, threading, or async IO to better utilize available CPU cores."
                )
            else:
                # Generic recommendation for other findings
                recommendations.append(
                    f"Address {severity} scalability issue: {first_finding.get('description', 'Unknown issue')}"
                )
        
        # Add general recommendations based on resource requirements
        memory_mb = resource_requirements.get('memory_mb', 512)
        cpu_cores = resource_requirements.get('cpu_cores', 0.5)
        
        if memory_mb < 256:
            recommendations.append(
                "Memory is very limited. Implement memory-efficient data structures and processing approaches."
            )
        
        if cpu_cores < 0.5:
            recommendations.append(
                "CPU is very limited. Focus on efficient algorithms and avoid computation-heavy operations."
            )
        elif cpu_cores >= 2 and "parallelization_opportunity" not in findings_by_type:
            recommendations.append(
                f"Multiple CPU cores ({cpu_cores}) are available. Consider parallelism for better performance."
            )
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations