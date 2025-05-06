"""
Config Module Generator for AutoGenesis Agent Factory.

This module is responsible for generating configuration modules for Target Agents.
"""

import logging
from typing import Dict, Any

from generators.base_module_generator import BaseModuleGenerator

# Set up logging
logger = logging.getLogger(__name__)

class ConfigModuleGenerator(BaseModuleGenerator):
    """
    Generates configuration modules for Target Agents.
    """
    
    def generate(self, module_name: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate a configuration module.
        
        Args:
            module_name: Name of the module
            blueprint: Blueprint specifications
            
        Returns:
            Complete Python code for the config module
        """
        # Get configuration from blueprint
        constraints = blueprint.get('requirements', {}).get('constraints', {})
        rate_limit = constraints.get('rate_limit_per_minute', 10)
        max_runtime = constraints.get('max_runtime_seconds', 3600)
        run_schedule = constraints.get('run_schedule', 'daily@3am')
        
        # Start with the module header
        code = self.generate_module_header(
            module_name, 
            "Configuration handling for the agent"
        )
        
        # Add configuration handling code
        code += f'''
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {{
    "rate_limit": {{
        "requests_per_minute": {rate_limit}
    }},
    "runtime": {{
        "max_seconds": {max_runtime}
    }},
    "schedule": "{run_schedule}",
    "logging": {{
        "level": "INFO",
        "file": "/var/log/agent_factory/agent.log",
        "format": "json"
    }}
}}


class ConfigManager:
    """
    Manages configuration loading, validation, and access.
    
    This class handles loading configuration from various sources
    (environment variables, config files) and provides access to
    configuration values with validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self._load_config()
        
    def _load_config(self) -> None:
        """
        Load configuration from environment and files.
        
        Loads in order of precedence:
        1. Default configuration
        2. Configuration file (if specified)
        3. Environment variables
        
        Raises:
            FileNotFoundError: If specified config file cannot be found
        """
        # Try to load from config file if specified
        if self.config_path:
            self._load_from_file(self.config_path)
        
        # Load from environment variables (overrides file config)
        self._load_from_env()
        
        logger.info(f"Loaded configuration: {{json.dumps(self.config, indent=2)}}")
        
    def _load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Config file not found: {{file_path}}")
            raise FileNotFoundError(f"Config file not found: {{file_path}}")
        
        try:
            # Determine file type from extension
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    file_config = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {{path.suffix}}")
                return
                
            # Update configuration with file values
            self._update_config(file_config)
            logger.info(f"Loaded configuration from {{file_path}}")
            
        except Exception as e:
            logger.error(f"Error loading config from file: {{str(e)}}")
            raise ValueError(f"Invalid configuration file: {{str(e)}}")
    
    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Looks for environment variables with prefix AGENT_ and updates
        the configuration accordingly.
        """
        env_config = {{}}
        
        # Look for environment variables with AGENT_ prefix
        for key, value in os.environ.items():
            if key.startswith('AGENT_'):
                # Remove prefix and convert to lowercase
                config_key = key[6:].lower()
                
                # Try to parse as JSON, fall back to string if not valid JSON
                try:
                    env_config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    env_config[config_key] = value
        
        # Update configuration with environment values
        self._update_config(env_config)
        
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: New configuration values to apply
        """
        def update_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_dict(target[key], value)
                else:
                    target[key] = value
        
        update_dict(self.config, new_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested values)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default if not found
            
        Example:
            >>> config_manager.get('logging.level')
            'INFO'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        
        return value
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check required fields
        required_fields = [
            'rate_limit.requests_per_minute',
            'runtime.max_seconds',
            'logging.level'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                logger.error(f"Missing required configuration field: {{field}}")
                return False
        
        # Validate specific fields
        rate_limit = self.get('rate_limit.requests_per_minute')
        if not isinstance(rate_limit, (int, float)) or rate_limit <= 0:
            logger.error(f"Invalid rate limit: {{rate_limit}}")
            return False
        
        max_runtime = self.get('runtime.max_seconds')
        if not isinstance(max_runtime, (int, float)) or max_runtime <= 0:
            logger.error(f"Invalid max runtime: {{max_runtime}}")
            return False
        
        return True


# Create a global instance
config = ConfigManager()
'''
        
        return code