"""
Configuration module for AutoGenesis.

This module defines default configuration and constants.
"""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "file": "logs/autogenesis.log",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "templates": {
        "path": "autogenesis/data/templates",
    },
    "output": {
        "path": "output"
    },
    "sandbox": {
        "enabled": True,
        "docker_image": "python:3.10-slim"
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment variable overrides.
    
    Returns:
        Dict containing configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    if os.environ.get('LOG_LEVEL'):
        config['logging']['level'] = os.environ.get('LOG_LEVEL')
    
    if os.environ.get('LOG_FILE'):
        config['logging']['file'] = os.environ.get('LOG_FILE')
    
    if os.environ.get('OUTPUT_PATH'):
        config['output']['path'] = os.environ.get('OUTPUT_PATH')
    
    if os.environ.get('TEMPLATE_PATH'):
        config['templates']['path'] = os.environ.get('TEMPLATE_PATH')
    
    return config