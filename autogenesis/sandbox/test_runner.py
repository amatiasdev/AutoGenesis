"""
Test Runner for AutoGenesis.

This module handles the execution of tests for the target agent
within the sandbox environment.
"""

import logging
import os
import json
import subprocess
import re
from typing import Dict, Any, List, Optional, Tuple
import xml.etree.ElementTree as ET

from sandbox.docker_manager import DockerManager

class TestRunner:
    def _run_unit_tests(self, sandbox_id: str, test_paths: List[str]) -> Dict[str, Any]:
        """
        Run unit tests with pytest.
        
        Args:
            sandbox_id: ID of the sandbox environment
            test_paths: List of paths to test files
            
        Returns:
            Dict containing unit test results
        """
        self.logger.info(f"Running unit tests in sandbox {sandbox_id}")
        
        # Run pytest with JUnit XML output
        exit_code, stdout, stderr = self.docker_manager.run_command(
            sandbox_id,
            ["pytest", "-xvs", "--junitxml=test-results.xml"]
        )
        
        # Get the XML results
        _, xml_stdout, _ = self.docker_manager.run_command(
            sandbox_id,
            ["cat", "test-results.xml"]
        )
        
        # Parse test results from XML
        test_results = self._parse_junit_xml(xml_stdout)
        
        # Add console output for debugging
        test_results["stdout"] = stdout
        test_results["stderr"] = stderr
        
        return test_results
    
    def _run_integration_tests(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Run integration tests.
        
        Args:
            sandbox_id: ID of the sandbox environment
            
        Returns:
            Dict containing integration test results
        """
        self.logger.info(f"Running integration tests in sandbox {sandbox_id}")
        
        # For now, integration tests are assumed to be in the tests directory
        # with names starting with 'test_integration'
        exit_code, stdout, stderr = self.docker_manager.run_command(
            sandbox_id,
            ["bash", "-c", "find tests -name 'test_integration*.py' | xargs pytest -xvs --junitxml=integration-results.xml || true"]
        )
        
        # Check if any integration tests were found
        if "no tests ran" in stdout or "no tests ran" in stderr:
            return {
                "passed": True,
                "total": 0,
                "passed_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "error_count": 0,
                "tests": []
            }
        
        # Get the XML results
        _, xml_stdout, _ = self.docker_manager.run_command(
            sandbox_id,
            ["cat", "integration-results.xml"]
        )
        
        # Parse test results from XML
        test_results = self._parse_junit_xml(xml_stdout)
        
        # Add console output for debugging
        test_results["stdout"] = stdout
        test_results["stderr"] = stderr
        
        return test_results
    
    def _parse_junit_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse JUnit XML test results.
        
        Args:
            xml_content: JUnit XML content
            
        Returns:
            Dict containing parsed test results
        """
        # Default result in case parsing fails
        default_result = {
            "passed": False,
            "total": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "tests": []
        }
        
        if not xml_content or xml_content.strip() == "":
            return default_result
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Get test counts
            total = int(root.attrib.get('tests', 0))
            failures = int(root.attrib.get('failures', 0))
            errors = int(root.attrib.get('errors', 0))
            skipped = int(root.attrib.get('skipped', 0))
            passed_count = total - failures - errors - skipped
            
            # Extract individual test case results
            tests = []
            for testcase in root.findall('.//testcase'):
                test_name = testcase.attrib.get('name', '')
                class_name = testcase.attrib.get('classname', '')
                time = float(testcase.attrib.get('time', 0))
                
                # Check if test failed
                failure = testcase.find('failure')
                error = testcase.find('error')
                skipped_node = testcase.find('skipped')
                
                status = "passed"
                message = ""
                
                if failure is not None:
                    status = "failed"
                    message = failure.attrib.get('message', '')
                elif error is not None:
                    status = "error"
                    message = error.attrib.get('message', '')
                elif skipped_node is not None:
                    status = "skipped"
                    message = skipped_node.attrib.get('message', '')
                
                tests.append({
                    "name": test_name,
                    "class_name": class_name,
                    "time": time,
                    "status": status,
                    "message": message
                })
            
            return {
                "passed": failures == 0 and errors == 0,
                "total": total,
                "passed_count": passed_count,
                "failed_count": failures,
                "skipped_count": skipped,
                "error_count": errors,
                "tests": tests
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing JUnit XML: {e}")
            return default_result
    """
    Runs tests for the generated agent in a sandbox environment.
    
    Executes different types of tests (syntax checks, unit tests, etc.)
    and processes the results.
    """
    
    def __init__(self):
        """Initialize the test runner."""
        self.logger = logging.getLogger(__name__)
        self.docker_manager = DockerManager()
    
    def run_tests(self, sandbox_id: str, test_paths: List[str]) -> Dict[str, Any]:
        """
        Run all tests for the target agent.
        
        Args:
            sandbox_id: ID of the sandbox environment
            test_paths: List of paths to test files
            
        Returns:
            Dict containing test results
        """
        self.logger.info(f"Running tests in sandbox {sandbox_id}")
        
        results = {
            "syntax_check": self._run_syntax_check(sandbox_id),
            "lint_check": self._run_lint_check(sandbox_id),
            "unit_tests": self._run_unit_tests(sandbox_id, test_paths),
            "integration_tests": self._run_integration_tests(sandbox_id),
            "passed": False,
            "summary": {}
        }
        
        # Compute overall pass/fail status
        results["passed"] = (
            results["syntax_check"]["passed"] and
            results["lint_check"]["passed"] and
            results["unit_tests"]["passed"] and
            results["integration_tests"]["passed"]
        )
        
        # Generate summary
        total_tests = (
            results["unit_tests"].get("total", 0) +
            results["integration_tests"].get("total", 0)
        )
        passed_tests = (
            results["unit_tests"].get("passed_count", 0) +
            results["integration_tests"].get("passed_count", 0)
        )
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "passed_percentage": round(passed_tests / max(total_tests, 1) * 100, 2),
            "syntax_errors": results["syntax_check"].get("error_count", 0),
            "lint_errors": results["lint_check"].get("error_count", 0),
            "overall_status": "Passed" if results["passed"] else "Failed"
        }
        
        self.logger.info(f"Test results: {results['summary']}")
        return results
    
    def _run_syntax_check(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Run syntax checks on the Python code.
        
        Args:
            sandbox_id: ID of the sandbox environment
            
        Returns:
            Dict containing syntax check results
        """
        self.logger.info(f"Running syntax check in sandbox {sandbox_id}")
        
        # Run pyflakes for syntax checking
        exit_code, stdout, stderr = self.docker_manager.run_command(
            sandbox_id,
            ["bash", "-c", "find . -name '*.py' | xargs pyflakes"]
        )
        
        # Process results
        error_lines = stdout.strip().split('\n')
        error_count = len([line for line in error_lines if line.strip()])
        
        return {
            "passed": exit_code == 0,
            "error_count": error_count,
            "errors": error_lines if error_count > 0 else []
        }
    
    def _run_lint_check(self, sandbox_id: str) -> Dict[str, Any]:
        """
        Run lint checks on the Python code.
        
        Args:
            sandbox_id: ID of the sandbox environment
            
        Returns:
            Dict containing lint check results
        """
        self.logger.info(f"Running lint check in sandbox {sandbox_id}")
        
        # Run flake8 for linting
        exit_code, stdout, stderr = self.docker_manager.run_command(
            sandbox_id,
            ["flake8", "--max-line-length=100", "--ignore=E203,E501,W503", "."]
        )
        
        # Process results
        error_lines = stdout.strip().split('\n')
        error_count = len([line for line in error_lines if line.strip()])
        
        return {
            "passed": exit_code == 0,
            "error_count": error_count,
            "errors": error_lines if error_count > 0 else []
        }