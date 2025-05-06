"""
Tests for blueprint generation components of AutoGenesis.

This module contains tests for the Blueprint Designer, Tool Selector,
and Resource Analyzer to ensure they correctly design agent architecture.
"""

import unittest
import os
import json
from pathlib import Path
from autogenesis.blueprint.designer import BlueprintDesigner
from autogenesis.blueprint.tool_selector import ToolSelector
from autogenesis.blueprint.resource_analyzer import ResourceAnalyzer

class TestBlueprintDesigner(unittest.TestCase):
    """Test cases for the Blueprint Designer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.designer = BlueprintDesigner()
        
        # Sample decomposed task
        self.decomposed_task = {
            'task_type': 'web_scraping',
            'actions': [
                {'type': 'http_request', 'url': 'example.com/data'},
                {'type': 'parse_html', 'selectors': ['.product-title', '.product-price']},
                {'type': 'extract_data', 'fields': ['title', 'price']},
                {'type': 'write_file', 'format': 'csv', 'path': 'output.csv'}
            ]
        }
        
    def test_design_blueprint(self):
        """Test blueprint generation for a decomposed task."""
        # Design a blueprint
        blueprint = self.designer.design(self.decomposed_task)
        
        # Assert results
        self.assertIsNotNone(blueprint)
        self.assertIn('name', blueprint)
        self.assertIn('architecture_type', blueprint)
        self.assertIn('tools', blueprint)
        
        # Check that blueprint includes expected components
        self.assertTrue(len(blueprint['tools']) > 0)
        self.assertIn('requests', blueprint['tools'])
        self.assertIn('beautifulsoup4', blueprint['tools'])


class TestToolSelector(unittest.TestCase):
    """Test cases for the Tool Selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.selector = ToolSelector()
        
    def test_select_tools_for_actions(self):
        """Test tool selection for specific actions."""
        # Sample actions
        actions = [
            {'type': 'http_request', 'url': 'example.com/data'},
            {'type': 'parse_html', 'selectors': ['.product-title']}
        ]
        
        # Select tools
        tools = self.selector.select_tools(actions)
        
        # Assert results
        self.assertIsNotNone(tools)
        self.assertTrue(len(tools) > 0)
        self.assertIn('requests', tools)
        self.assertIn('beautifulsoup4', tools)
        
    def test_select_tools_with_constraints(self):
        """Test tool selection with memory/CPU constraints."""
        # Sample actions
        actions = [
            {'type': 'http_request', 'url': 'example.com/data'}
        ]
        
        # Resource constraints
        constraints = {
            'memory_mb': 256,
            'cpu_cores': 0.5
        }
        
        # Select tools with constraints
        tools = self.selector.select_tools(actions, constraints)
        
        # Assert results - should prefer lightweight alternatives when constrained
        self.assertIsNotNone(tools)
        self.assertIn('requests', tools)  # Should use lightweight HTTP client
        self.assertNotIn('selenium', tools)  # Should avoid heavy tools


class TestResourceAnalyzer(unittest.TestCase):
    """Test cases for the Resource Analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResourceAnalyzer()
        
    def test_detect_host_resources(self):
        """Test detection of host EC2 instance resources."""
        # Get host resources (note: this may not be an EC2 instance in test environment)
        resources = self.analyzer.detect_host_resources()
        
        # Assert results
        self.assertIsNotNone(resources)
        self.assertIn('cpu_cores', resources)
        self.assertIn('memory_mb', resources)
        
    def test_analyze_blueprint_resources(self):
        """Test analysis of blueprint resource requirements."""
        # Sample blueprint
        blueprint = {
            'name': 'Test Agent',
            'architecture_type': 'modular',
            'tools': {
                'requests': {'version': '2.28.1'},
                'beautifulsoup4': {'version': '4.11.1'},
                'pandas': {'version': '1.5.3'}
            }
        }
        
        # Analyze resource requirements
        requirements = self.analyzer.analyze_blueprint_resources(blueprint)
        
        # Assert results
        self.assertIsNotNone(requirements)
        self.assertIn('estimated_memory_mb', requirements)
        self.assertIn('estimated_cpu_cores', requirements)
        self.assertTrue(requirements['estimated_memory_mb'] > 0)


if __name__ == '__main__':
    unittest.main()