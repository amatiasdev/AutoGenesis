"""
Tests for code generation components of AutoGenesis.

This module contains tests for the Module Generator, Scaffold Generator,
and Test Generator to ensure they correctly generate code for target agents.
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from autogenesis.code_generator.module_generator import ModuleGenerator
from autogenesis.code_generator.scaffold_generator import ScaffoldGenerator
from autogenesis.code_generator.test_generator import TestGenerator

class TestModuleGenerator(unittest.TestCase):
    """Test cases for the Module Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.module_generator = ModuleGenerator()
        
        # Create temporary directory for generated code
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample blueprint
        self.blueprint = {
            'name': 'WebScraperAgent',
            'architecture_type': 'modular',
            'tools': {
                'requests': {'version': '2.28.1'},
                'beautifulsoup4': {'version': '4.11.1'}
            },
            'modules': [
                {
                    'name': 'scraper',
                    'functions': [
                        {
                            'name': 'fetch_url',
                            'parameters': [{'name': 'url', 'type': 'str'}],
                            'return_type': 'str',
                            'description': 'Fetches content from a URL'
                        }
                    ]
                }
            ]
        }
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_generate_module(self):
        """Test generation of a Python module."""
        # Generate module
        module_path = self.module_generator.generate(
            self.blueprint['modules'][0],
            output_dir=self.temp_dir
        )
        
        # Assert results
        self.assertTrue(os.path.exists(module_path))
        
        # Check file content
        with open(module_path, 'r') as f:
            content = f.read()
            
        # Verify essential components
        self.assertIn('import requests', content)
        self.assertIn('def fetch_url', content)
        self.assertIn('"""Fetches content from a URL"""', content)


class TestScaffoldGenerator(unittest.TestCase):
    """Test cases for the Scaffold Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scaffold_generator = ScaffoldGenerator()
        
        # Create temporary directory for generated code
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample blueprint
        self.blueprint = {
            'name': 'WebScraperAgent',
            'architecture_type': 'modular',
            'tools': {
                'requests': {'version': '2.28.1'},
                'beautifulsoup4': {'version': '4.11.1'}
            },
            'modules': [
                {'name': 'scraper'},
                {'name': 'parser'},
                {'name': 'writer'}
            ]
        }
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_generate_project_scaffold(self):
        """Test generation of a project scaffold."""
        # Generate scaffold
        project_dir = self.scaffold_generator.generate(
            self.blueprint,
            output_dir=self.temp_dir
        )
        
        # Assert results
        self.assertTrue(os.path.exists(project_dir))
        
        # Check directory structure
        self.assertTrue(os.path.exists(os.path.join(project_dir, 'main.py')))
        self.assertTrue(os.path.exists(os.path.join(project_dir, 'config.yaml')))
        
        # Check module files
        for module in self.blueprint['modules']:
            module_path = os.path.join(project_dir, f"{module['name']}.py")
            self.assertTrue(os.path.exists(module_path))


class TestTestGenerator(unittest.TestCase):
    """Test cases for the Test Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_generator = TestGenerator()
        
        # Create temporary directory for generated code
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample blueprint
        self.blueprint = {
            'name': 'WebScraperAgent',
            'modules': [
                {
                    'name': 'scraper',
                    'functions': [
                        {
                            'name': 'fetch_url',
                            'parameters': [{'name': 'url', 'type': 'str'}],
                            'return_type': 'str'
                        }
                    ]
                }
            ]
        }
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_generate_tests(self):
        """Test generation of unit tests."""
        # Generate tests
        test_dir = self.test_generator.generate(
            self.blueprint,
            output_dir=self.temp_dir
        )
        
        # Assert results
        self.assertTrue(os.path.exists(test_dir))
        
        # Check test files
        test_file = os.path.join(test_dir, 'test_scraper.py')
        self.assertTrue(os.path.exists(test_file))
        
        # Check file content
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Verify essential components
        self.assertIn('import unittest', content)
        self.assertIn('class TestScraper', content)
        self.assertIn('test_fetch_url', content)


if __name__ == '__main__':
    unittest.main()