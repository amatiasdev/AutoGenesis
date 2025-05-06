"""
Security Critic for AutoGenesis MPC.

This module evaluates the security aspects of generated agent code
as part of the Multi-Perspective Criticism (MPC) process.
"""

import logging
import os
import re
import subprocess
import json
import glob
from typing import Dict, Any, List, Set, Tuple

class SecurityCritic:
    """
    Evaluates agent code for security vulnerabilities.
    
    Analyzes Python code for common security issues and vulnerabilities,
    providing a security score and recommendations for improvement.
    """
    
    def __init__(self):
        """Initialize the security critic."""
        self.logger = logging.getLogger(__name__)
        
        # Define security patterns to check for
        self.security_patterns = {
            "hardcoded_secrets": {
                "pattern": r'(password|secret|key|token|auth).*=.*[\'"][A-Za-z0-9+/=_\-\.]{16,}[\'"]',
                "description": "Hardcoded secret or credential",
                "severity": "high"
            },
            "sql_injection": {
                "pattern": r'.*execute\([\'"](.*%s.*)[\'"].*\)',
                "description": "Potential SQL injection vulnerability",
                "severity": "high"
            },
            "os_command_injection": {
                "pattern": r'os\.system\(|subprocess\.call\(|subprocess\.Popen\(',
                "description": "Potential OS command injection",
                "severity": "medium"
            },
            "insecure_pickle": {
                "pattern": r'pickle\.loads\(|pickle\.load\(',
                "description": "Use of insecure pickle deserialization",
                "severity": "high"
            },
            "eval_use": {
                "pattern": r'eval\(',
                "description": "Use of eval() which can be dangerous",
                "severity": "high"
            },
            "temp_file_insecure": {
                "pattern": r'open\([\'"]\/tmp\/|tempfile\.mktemp\(',
                "description": "Insecure temporary file creation",
                "severity": "medium"
            },
            "http_without_tls": {
                "pattern": r'http:\/\/(?!localhost)',
                "description": "HTTP connection without TLS",
                "severity": "medium"
            },
            "weak_crypto": {
                "pattern": r'import md5|hashlib\.md5\(|import sha|hashlib\.sha1\(',
                "description": "Use of weak cryptographic functions",
                "severity": "medium"
            },
            "debug_enabled": {
                "pattern": r'DEBUG.*=.*True|debug.*=.*True',
                "description": "Debug mode enabled in production",
                "severity": "low"
            },
            "logging_sensitive": {
                "pattern": r'log.*\.(info|debug|warning|error|critical)\(.*password.*\)|print\(.*password.*\)',
                "description": "Potentially logging sensitive information",
                "severity": "medium"
            }
        }
        
        # Define weights for severity levels
        self.severity_weights = {
            "high": 10,
            "medium": 5,
            "low": 2
        }
    
    def evaluate(self, code_path: str) -> Dict[str, Any]:
        """
        Evaluate agent code for security issues.
        
        Args:
            code_path: Path to the code to evaluate
            
        Returns:
            Dict containing evaluation results
        """
        self.logger.info(f"Evaluating security of code at {code_path}")
        
        # Find all Python files
        python_files = self._find_python_files(code_path)
        self.logger.info(f"Found {len(python_files)} Python files to evaluate")
        
        # Analyze each file
        findings = []
        for file_path in python_files:
            file_findings = self._analyze_file(file_path)
            findings.extend(file_findings)
        
        # Run static analysis tools if available
        tool_findings = self._run_static_analysis(code_path)
        findings.extend(tool_findings)
        
        # Calculate security score
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
        for root, _, files in os.walk(code_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for security issues.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of security findings
        """
        findings = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for each security pattern
            for issue_type, pattern_info in self.security_patterns.items():
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
    
    def _run_static_analysis(self, code_path: str) -> List[Dict[str, Any]]:
        """
        Run static analysis tools on the code.
        
        Args:
            code_path: Path to the code to analyze
            
        Returns:
            List of security findings from static analysis tools
        """
        findings = []
        
        # Try to run Bandit security scanner if available
        try:
            # Run bandit with JSON output
            result = subprocess.run(
                ["bandit", "-r", code_path, "-f", "json"],
                capture_output=True,
                text=True
            )
            
            # Parse results
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_results = json.loads(result.stdout)
                    
                    # Extract findings
                    for result in bandit_results.get("results", []):
                        severity = result.get("issue_severity", "").lower()
                        if severity not in self.severity_weights:
                            severity = "medium"  # Default to medium if unknown
                        
                        findings.append({
                            "type": f"bandit_{result.get('test_id', 'unknown')}",
                            "description": result.get("issue_text", "Unknown issue"),
                            "severity": severity,
                            "file_path": result.get("filename", "unknown"),
                            "line_number": result.get("line_number", 0),
                            "line": result.get("code", "").strip(),
                            "tool": "bandit"
                        })
                
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse Bandit JSON output")
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.info("Bandit security scanner not available")
        
        # Try to run Safety for dependency checking if available
        try:
            # Look for requirements.txt
            req_files = glob.glob(os.path.join(code_path, "**/requirements.txt"), recursive=True)
            
            if req_files:
                for req_file in req_files:
                    result = subprocess.run(
                        ["safety", "check", "-r", req_file, "--json"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Parse results
                    try:
                        safety_results = json.loads(result.stdout)
                        
                        # Extract findings
                        for vuln in safety_results.get("vulnerabilities", []):
                            findings.append({
                                "type": "dependency_vulnerability",
                                "description": f"Vulnerable dependency: {vuln.get('package_name', 'unknown')} {vuln.get('vulnerable_spec', '')}",
                                "severity": "high",  # Assume high severity for vulnerable dependencies
                                "file_path": req_file,
                                "details": vuln.get("advisory", ""),
                                "tool": "safety"
                            })
                    
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse Safety JSON output")
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.info("Safety dependency checker not available")
        
        return findings
    
    def _calculate_deductions(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate score deductions based on findings.
        
        Args:
            findings: List of security findings
            
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
        Generate security recommendations based on findings.
        
        Args:
            findings: List of security findings
            
        Returns:
            List of security recommendations
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
            description = first_finding.get("description", "Unknown issue")
            
            # Generate recommendation based on finding type
            if finding_type == "hardcoded_secrets":
                recommendations.append(
                    "Replace hardcoded secrets with environment variables or a proper secret management solution."
                )
            elif finding_type == "sql_injection":
                recommendations.append(
                    "Use parameterized queries or an ORM instead of string formatting for SQL queries."
                )
            elif finding_type == "os_command_injection":
                recommendations.append(
                    "Avoid using os.system or subprocess with shell=True. Use subprocess.run with a list of arguments instead."
                )
            elif finding_type == "insecure_pickle":
                recommendations.append(
                    "Avoid using pickle for deserializing untrusted data. Consider using JSON or a more secure alternative."
                )
            elif finding_type == "eval_use":
                recommendations.append(
                    "Avoid using eval() as it can execute arbitrary code. Use safer alternatives specific to your use case."
                )
            elif finding_type == "temp_file_insecure":
                recommendations.append(
                    "Use tempfile.mkstemp() or tempfile.TemporaryFile() for secure temporary file creation."
                )
            elif finding_type == "http_without_tls":
                recommendations.append(
                    "Use HTTPS instead of HTTP for all external connections to ensure data encryption."
                )
            elif finding_type == "weak_crypto":
                recommendations.append(
                    "Replace MD5 and SHA-1 with more secure hashing algorithms like SHA-256 or higher."
                )
            elif finding_type == "debug_enabled":
                recommendations.append(
                    "Ensure DEBUG mode is disabled in production environments."
                )
            elif finding_type == "logging_sensitive":
                recommendations.append(
                    "Avoid logging sensitive information like passwords or tokens. Mask or remove sensitive data before logging."
                )
            elif finding_type == "dependency_vulnerability":
                recommendations.append(
                    "Update vulnerable dependencies to secure versions as specified in the security advisories."
                )
            else:
                # Generic recommendation for other findings
                recommendations.append(
                    f"Address {severity} security issue: {description}"
                )
        
        return recommendations