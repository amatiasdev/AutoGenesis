"""
Tests for Multi-Perspective Criticism (MPC) components of AutoGenesis.

This module contains tests for the security, scalability, maintainability,
and cost critics, as well as the DynamoDB Logger.
"""

import unittest
import os
import tempfile
import shutil
import uuid
from unittest.mock import patch, MagicMock
from pathlib import Path
from autogenesis.mpc.security_critic import SecurityCritic
from autogenesis.mpc.scalability_critic import ScalabilityCritic
from autogenesis.mpc.maintainability_critic import MaintainabilityCritic
from autogenesis.mpc.cost_estimator import CostEstimator
from autogenesis.mpc.dynamodb_logger import DynamoDBLogger

class TestSecurityCritic(unittest.TestCase):
    """Test cases for the Security Critic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.security_critic = SecurityCritic()
        
        # Create temporary directory with test code
        self.temp_dir = tempfile.mkdtemp()
        
        # Write test file with security issues
        self.test_file_path = os.path.join(self.temp_dir, 'insecure.py')
        with open(self.test_file_path, 'w') as f:
            f.write("""
# Test file with security issues
import os

def execute_command(cmd):
    # Insecure: uses os.system directly with user input
    os.system(cmd)

def get_password():
    # Insecure: hardcoded credentials
    return "hardcoded_password_123"

def connect_db(connection_string="mysql://root:password@localhost/db"):
    # Insecure: hardcoded credentials in default parameter
    pass
""")
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_analyze_security(self):
        """Test security analysis of generated code."""
        # Analyze security
        results = self.security_critic.analyze(self.temp_dir)
        
        # Assert results
        self.assertIsNotNone(results)
        self.assertIn('passed', results)
        self.assertIn('score', results)
        self.assertIn('findings', results)
        
        # Check findings
        findings = results['findings']
        self.assertTrue(len(findings) > 0)
        
        # Verify detection of os.system issue
        os_system_finding = next((f for f in findings if 'os.system' in f['description']), None)
        self.assertIsNotNone(os_system_finding)
        
        # Verify detection of hardcoded credentials
        hardcoded_finding = next((f for f in findings if 'hardcoded' in f['description']), None)
        self.assertIsNotNone(hardcoded_finding)


class TestScalabilityCritic(unittest.TestCase):
    """Test cases for the Scalability Critic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scalability_critic = ScalabilityCritic()
        
        # Create temporary directory with test code
        self.temp_dir = tempfile.mkdtemp()
        
        # Write test file with scalability issues
        self.test_file_path = os.path.join(self.temp_dir, 'unscalable.py')
        with open(self.test_file_path, 'w') as f:
            f.write("""
# Test file with scalability issues
def process_large_data(data):
    # Inefficient: loads all data into memory
    all_data = list(data)
    
    # Inefficient: nested loops
    results = []
    for item in all_data:
        for other_item in all_data:
            results.append(item * other_item)
    
    return results

def fetch_urls(urls):
    # Inefficient: sequential processing
    results = []
    for url in urls:
        results.append(fetch_single_url(url))
    return results

def fetch_single_url(url):
    # Placeholder
    return f"Content from {url}"
""")
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_analyze_scalability(self):
        """Test scalability analysis of generated code."""
        # Resource constraints
        constraints = {
            'memory_mb': 512,
            'cpu_cores': 1
        }
        
        # Analyze scalability
        results = self.scalability_critic.analyze(
            self.temp_dir,
            resource_constraints=constraints
        )
        
        # Assert results
        self.assertIsNotNone(results)
        self.assertIn('passed', results)
        self.assertIn('score', results)
        self.assertIn('findings', results)
        
        # Check findings
        findings = results['findings']
        self.assertTrue(len(findings) > 0)
        
        # Verify detection of memory inefficiency
        memory_finding = next((f for f in findings if 'memory' in f['description']), None)
        self.assertIsNotNone(memory_finding)
        
        # Verify detection of nested loops
        loop_finding = next((f for f in findings if 'loop' in f['description']), None)
        self.assertIsNotNone(loop_finding)


class TestMaintainabilityCritic(unittest.TestCase):
    """Test cases for the Maintainability Critic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.maintainability_critic = MaintainabilityCritic()
        
        # Create temporary directory with test code
        self.temp_dir = tempfile.mkdtemp()
        
        # Write test file with maintainability issues
        self.test_file_path = os.path.join(self.temp_dir, 'unmaintainable.py')
        with open(self.test_file_path, 'w') as f:
            f.write("""
# Test file with maintainability issues
def a(x, y, z):
    # Unclear function name, no docstring
    v = 0
    for i in range(x):
        # Magic number
        if i % 2 == 0:
            v += i * 1.5
        else:
            v += i * 2.5
    return v

class Helper:
    # No docstring
    def __init__(self):
        self.data = {}
    
    def process(self, x):
        # Too many branches, high cyclomatic complexity
        if x < 0:
            return -1
        elif x == 0:
            return 0
        elif x < 10:
            return x * 2
        elif x < 20:
            return x * 3
        elif x < 30:
            return x * 4
        elif x < 40:
            return x * 5
        elif x < 50:
            return x * 6
        else:
            return x * 7
""")
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_analyze_maintainability(self):
        """Test maintainability analysis of generated code."""
        # Analyze maintainability
        results = self.maintainability_critic.analyze(self.temp_dir)
        
        # Assert results
        self.assertIsNotNone(results)
        self.assertIn('passed', results)
        self.assertIn('score', results)
        self.assertIn('findings', results)
        
        # Check findings
        findings = results['findings']
        self.assertTrue(len(findings) > 0)
        
        # Verify detection of unclear naming
        naming_finding = next((f for f in findings if 'naming' in f['description'] or 'name' in f['description']), None)
        self.assertIsNotNone(naming_finding)
        
        # Verify detection of missing docstrings
        docstring_finding = next((f for f in findings if 'docstring' in f['description']), None)
        self.assertIsNotNone(docstring_finding)
        
        # Verify detection of complexity
        complexity_finding = next((f for f in findings if 'complex' in f['description']), None)
        self.assertIsNotNone(complexity_finding)


class TestCostEstimator(unittest.TestCase):
    """Test cases for the Cost Estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cost_estimator = CostEstimator()
        
        # Sample blueprint with AWS resources
        self.blueprint = {
            'name': 'DataProcessingAgent',
            'tools': {
                'boto3': {'version': '1.24.0'}
            },
            'aws_resources': {
                'lambda': {
                    'memory_mb': 512,
                    'estimated_invocations_per_month': 10000,
                    'average_duration_ms': 500
                },
                's3': {
                    'estimated_storage_gb': 10,
                    'estimated_get_requests': 5000,
                    'estimated_put_requests': 1000
                }
            }
        }
        
    def test_estimate_cost(self):
        """Test cost estimation for a blueprint."""
        # Estimate cost
        results = self.cost_estimator.estimate(self.blueprint)
        
        # Assert results
        self.assertIsNotNone(results)
        self.assertIn('passed', results)
        self.assertIn('total_monthly_cost', results)
        self.assertIn('breakdown', results)
        
        # Check breakdown
        breakdown = results['breakdown']
        self.assertIn('lambda', breakdown)
        self.assertIn('s3', breakdown)
        
        # Verify costs are positive
        self.assertTrue(results['total_monthly_cost'] > 0)
        self.assertTrue(breakdown['lambda'] > 0)
        self.assertTrue(breakdown['s3'] > 0)


class TestDynamoDBLogger(unittest.TestCase):
    """Test cases for the DynamoDB Logger."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock DynamoDB to avoid actual AWS calls
        self.patcher = patch('boto3.resource')
        self.mock_boto3_resource = self.patcher.start()
        
        # Mock table
        self.mock_table = MagicMock()
        self.mock_dynamodb = MagicMock()
        self.mock_dynamodb.Table.return_value = self.mock_table
        self.mock_boto3_resource.return_value = self.mock_dynamodb
        
        # Configure mock response
        mock_response = {'ResponseMetadata': {'HTTPStatusCode': 200}}
        self.mock_table.put_item.return_value = mock_response
        
        # Create logger with test table
        self.logger = DynamoDBLogger(table_name='test-table')
        
        # Create temporary directory with test code
        self.temp_dir = tempfile.mkdtemp()
        with open(os.path.join(self.temp_dir, 'test.py'), 'w') as f:
            f.write("print('Hello, world!')")
        
        # Sample data
        self.agent_id = str(uuid.uuid4())
        self.blueprint = {
            'name': 'TestAgent',
            'task_type': 'data_processing',
            'tools': {'pandas': {'version': '1.5.3'}}
        }
        self.mpc_results = {
            'satisfied': True,
            'security': {
                'passed': True,
                'score': 0.9,
                'findings': [],
                'recommendations': ['Use environment variables for credentials']
            },
            'scalability': {
                'passed': True,
                'score': 0.85,
                'findings': [],
                'recommendations': ['Consider chunking large datasets']
            },
            'maintainability': {
                'passed': True,
                'score': 0.95,
                'findings': [],
                'recommendations': ['Add more docstrings']
            },
            'cost': {
                'passed': True,
                'total_monthly_cost': 5.75,
                'recommendations': ['Consider Lambda instead of EC2 for this workload']
            }
        }
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patch
        self.patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_log_mpc_results(self):
        """Test logging MPC results to DynamoDB."""
        # Log results
        success = self.logger.log_mpc_results(
            agent_id=self.agent_id,
            code_path=self.temp_dir,
            blueprint=self.blueprint,
            mpc_results=self.mpc_results
        )
        
        # Assert results
        self.assertTrue(success)
        
        # Verify DynamoDB interaction
        self.mock_table.put_item.assert_called_once()
        
        # Check item format
        call_args = self.mock_table.put_item.call_args
        item = call_args[1]['Item']
        
        self.assertIn('critique_id', item)
        self.assertEqual(item['agent_id'], self.agent_id)
        self.assertIn('timestamp', item)
        self.assertIn('code_hash', item)
        self.assertIn('blueprint_summary', item)
        self.assertIn('evaluation_summary', item)
        self.assertIn('scores', item)


if __name__ == '__main__':
    unittest.main()