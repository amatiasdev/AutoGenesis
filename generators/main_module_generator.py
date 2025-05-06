"""
Main Module Generator for AutoGenesis Agent Factory.

This module is responsible for generating the main Python module for Target Agents.
"""

import logging
from typing import Dict, List, Any

from generators.base_module_generator import BaseModuleGenerator
from generators.step_function_generator import StepFunctionGenerator

# Set up logging
logger = logging.getLogger(__name__)

class MainModuleGenerator(BaseModuleGenerator):
    """
    Generates the main module for Target Agents.
    """
    
    def __init__(self):
        """Initialize the MainModuleGenerator."""
        self.step_function_generator = StepFunctionGenerator()
    
    def generate(self, module_name: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate the main module for the Target Agent.
        
        Args:
            module_name: Name of the module
            blueprint: Blueprint specifications
            
        Returns:
            Complete Python code for the main module
        """
        # Extract relevant information from the blueprint
        task_type = blueprint.get('task_type', 'custom')
        processing_steps = blueprint.get('requirements', {}).get('processing_steps', [])
        source = blueprint.get('requirements', {}).get('source', {})
        output = blueprint.get('requirements', {}).get('output', {})
        tools = blueprint.get('selected_tools', [])
        
        # Generate module header
        code = self.generate_module_header(
            module_name, 
            f"Main module for {task_type} agent"
        )
        
        # Add imports based on selected tools
        for tool in tools:
            if tool == 'pandas':
                code += 'import pandas as pd\n'
            elif tool == 'requests':
                code += 'import requests\n'
            elif tool == 'beautifulsoup':
                code += 'from bs4 import BeautifulSoup\n'
            elif tool == 'selenium':
                code += 'from selenium import webdriver\n'
                code += 'from selenium.webdriver.common.by import By\n'
                code += 'from selenium.webdriver.support.ui import WebDriverWait\n'
                code += 'from selenium.webdriver.support import expected_conditions as EC\n'
        
        # Add standard imports
        code += '''
import json
import time
import sys
from datetime import datetime
import traceback

'''
        
        # Add configuration loading
        code += '''
def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment or config file.
    
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file cannot be found
        json.JSONDecodeError: If config file is not valid JSON
    """
    try:
        # Try to load from environment variable
        if os.environ.get('AGENT_CONFIG'):
            logger.info("Loading configuration from environment variable")
            return json.loads(os.environ.get('AGENT_CONFIG'))
        
        # Fall back to config file
        config_path = os.environ.get('CONFIG_PATH', 'config.json')
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
            
    except FileNotFoundError:
        logger.error(f"Config file not found")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

'''
        
        # Generate step functions based on processing steps
        step_functions = []
        for i, step in enumerate(processing_steps):
            step_name = step.get('action', f'step_{i+1}')
            step_function = self.step_function_generator.generate_step_function(step_name, step, tools)
            step_functions.append(step_function)
            code += step_function + '\n\n'
        
        # Generate main function
        code += '''
def main() -> int:
    """
    Main entry point for the agent.
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        logger.info("Starting agent execution")
        
        # Load configuration
        config = load_config()
        logger.debug(f"Loaded configuration: {json.dumps(config, indent=2)}")
        
'''

        # Add calls to step functions
        for i, step in enumerate(processing_steps):
            step_name = step.get('action', f'step_{i+1}')
            result_var = f"result_{i+1}"
            
            # Add the function call with error handling
            code += f'''
        # Execute {step_name}
        try:
            logger.info(f"Executing step: {step_name}")
            {result_var} = {step_name}(config)
            logger.info(f"Completed step: {step_name}")
        except Exception as e:
            logger.error(f"Error in {step_name}: {{str(e)}}")
            logger.error(traceback.format_exc())
            return 1
            
'''
        
        # Add final success
        code += '''
        logger.info("Agent execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

'''
        
        # Add the script entry point
        code += '''
if __name__ == "__main__":
    # Configure logging
    logging_level = os.environ.get("LOGGING_LEVEL", "INFO")
    numeric_level = getattr(logging, logging_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.environ.get("LOG_FILE", "/var/log/agent_factory/agent.log"))
        ]
    )
    
    # Execute main function
    exit_code = main()
    sys.exit(exit_code)
'''
        
        return code