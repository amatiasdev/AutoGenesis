"""
Tests for input processing components of AutoGenesis.

This module contains tests for the API and Natural Language processors
to ensure they correctly interpret user requirements.
"""

import unittest
import os
import json
from pathlib import Path
from autogenesis.input_processing.api_processor import APIProcessor
from autogenesis.input_processing.nl_processor import NLProcessor
from autogenesis.input_processing.task_decomposer import TaskDecomposer

class TestAPIProcessor(unittest.TestCase):
    """Test cases for the API input processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_processor = APIProcessor()
        
        # Create path to example request
        self.project_root = Path(__file__).parent.parent.parent
        self.example_json_path = self.project_root / "example_request.json"
        
    def test_parse_valid_request(self):
        """Test parsing a valid JSON request."""
        # Read test data
        with open(self.example_json_path, 'r') as f:
            request_data = json.load(f)
        
        # Process the request
        result = self.api_processor.process(request_data)
        
        # Assert results
        self.assertIsNotNone(result)
        self.assertIn('task_type', result)
        self.assertIn('requirements', result)
        
    def test_validate_schema(self):
        """Test validation of request schema."""
        # Test with invalid schema (missing required fields)
        invalid_data = {"incomplete": "data"}
        
        with self.assertRaises(Exception):
            self.api_processor.process(invalid_data)


class TestNLProcessor(unittest.TestCase):
    """Test cases for the Natural Language processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nl_processor = NLProcessor()
        
        # Create path to example NL request
        self.project_root = Path(__file__).parent.parent.parent
        self.nl_request_path = self.project_root / "nl_request.txt"
        
    def test_process_nl_request(self):
        """Test processing a natural language request."""
        # Read test data
        with open(self.nl_request_path, 'r') as f:
            nl_text = f.read()
        
        # Process the request
        result = self.nl_processor.process(nl_text)
        
        # Assert results
        self.assertIsNotNone(result)
        self.assertIn('task_type', result)
        self.assertIn('requirements', result)
        
    def test_detect_ambiguity(self):
        """Test detection of ambiguous requests."""
        ambiguous_text = "Create an agent that processes data."
        
        # Process the ambiguous request
        result = self.nl_processor.process(ambiguous_text)
        
        # Check if ambiguity was detected
        self.assertIn('ambiguities', result)
        self.assertTrue(len(result['ambiguities']) > 0)


class TestTaskDecomposer(unittest.TestCase):
    """Test cases for the Task Decomposer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.decomposer = TaskDecomposer()
        
        # Sample task data
        self.sample_task = {
            'task_type': 'web_scraping',
            'requirements': {
                'source': {
                    'type': 'url',
                    'value': 'example.com/data'
                },
                'processing_steps': [
                    {'action': 'extract_data', 'fields': ['title', 'price']}
                ],
                'output': {
                    'format': 'csv',
                    'destination_type': 'local_file',
                    'destination_value': 'output.csv'
                }
            }
        }
        
    def test_decompose_task(self):
        """Test task decomposition into atomic actions."""
        # Decompose the task
        actions = self.decomposer.decompose(self.sample_task)
        
        # Assert results
        self.assertIsNotNone(actions)
        self.assertTrue(len(actions) > 0)
        
        # Check that actions include expected steps
        action_types = [action['type'] for action in actions]
        self.assertIn('http_request', action_types)
        self.assertIn('parse_html', action_types)
        self.assertIn('extract_data', action_types)
        self.assertIn('write_file', action_types)


if __name__ == '__main__':
    unittest.main()