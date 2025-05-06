"""
Module Generator for AutoGenesis Agent Factory.

This module is the orchestrator responsible for coordinating the generation 
of Python code modules for Target Agents based on blueprint specifications.
"""

import os
import logging
import black
from typing import Dict, List, Optional, Any

# Import the specific module generators
from generators.main_module_generator import MainModuleGenerator
from generators.utils_module_generator import UtilsModuleGenerator
from generators.config_module_generator import ConfigModuleGenerator
from generators.api_client_module_generator import ApiClientModuleGenerator
from generators.data_processor_module_generator import DataProcessorModuleGenerator

# Set up logging
logger = logging.getLogger(__name__)

class ModuleGenerator:
    """
    Orchestrator for generating Python code modules for Target Agents.
    
    This class coordinates the generation of Python modules with proper structure,
    documentation, logging, and error handling based on blueprint specifications.
    It generates code that adheres to PEP 8 standards and includes Google-style docstrings.
    """
    
    def __init__(self, output_dir: str= None):
        """
        Initialize the ModuleGenerator.
        
        Args:
            output_dir: Directory where generated modules will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize specific module generators
        self.main_generator = MainModuleGenerator()
        self.utils_generator = UtilsModuleGenerator()
        self.config_generator = ConfigModuleGenerator()
        self.api_client_generator = ApiClientModuleGenerator()
        self.data_processor_generator = DataProcessorModuleGenerator()
        
        logger.debug(f"Initialized ModuleGenerator with output directory: {output_dir}")
    
    def generate_module(self, 
                      module_name: str, 
                      blueprint: Dict[str, Any],
                      module_type: str = "main") -> str:
        """
        Generate a Python module based on blueprint specifications.
        
        Args:
            module_name: Name of the module to generate
            blueprint: Blueprint specifications for the Target Agent
            module_type: Type of module to generate (main, utils, config, etc.)
            
        Returns:
            Path to the generated module file
        
        Raises:
            ValueError: If module_type is invalid or blueprint is missing required keys
        """
        if not blueprint:
            raise ValueError("Blueprint cannot be empty")
            
        # Select the appropriate generator method based on module_type
        generator_methods = {
            "main": self.main_generator.generate,
            "utils": self.utils_generator.generate,
            "config": self.config_generator.generate,
            "api_client": self.api_client_generator.generate,
            "data_processor": self.data_processor_generator.generate,
        }
        
        if module_type not in generator_methods:
            raise ValueError(f"Invalid module_type: {module_type}. Valid types are: {list(generator_methods.keys())}")
        
        # Generate the module code
        code = generator_methods[module_type](module_name, blueprint)
        
        # Format the code with black
        formatted_code = self._format_code(code)
        
        # Save the module to a file
        file_path = os.path.join(self.output_dir, f"{module_name}.py")
        with open(file_path, 'w') as f:
            f.write(formatted_code)
        
        logger.info(f"Generated {module_type} module: {file_path}")
        return file_path
    
    def _format_code(self, code: str) -> str:
        """
        Format Python code using Black formatter.
        
        Args:
            code: Python code to format
            
        Returns:
            Formatted Python code
            
        Raises:
            Exception: If Black formatting fails
        """
        try:
            return black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.warning(f"Black formatting failed: {e}. Returning unformatted code.")
            return code


if __name__ == "__main__":
    # Example usage
    generator = ModuleGenerator("output")
    blueprint = {
        "task_type": "web_scraping",
        "requirements": {
            "source": {
                "type": "url",
                "value": "https://example.com"
            },
            "processing_steps": [
                {"action": "scrape_data", "fields": ["name", "price", "description"]},
                {"action": "filter_data", "criteria": "price > 100"},
                {"action": "save_data", "format": "csv", "destination_type": "local_file", "destination_value": "output.csv"}
            ],
            "output": {
                "format": "csv",
                "destination_type": "local_file",
                "destination_value": "data/output.csv"
            }
        },
        "selected_tools": ["requests", "beautifulsoup", "pandas"]
    }
    
    generator.generate_module("main", blueprint, "main")
    generator.generate_module("utils", blueprint, "utils")
    generator.generate_module("config", blueprint, "config")