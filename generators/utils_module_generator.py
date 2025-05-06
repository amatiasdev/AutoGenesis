"""
Utils Module Generator for AutoGenesis Agent Factory.

This module is responsible for generating utility modules for Target Agents.
"""

import logging
from typing import Dict, Any

from generators.base_module_generator import BaseModuleGenerator

# Set up logging
logger = logging.getLogger(__name__)

class UtilsModuleGenerator(BaseModuleGenerator):
    """
    Generates utility modules for Target Agents.
    """
    
    def generate(self, module_name: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate a utility module with helper functions.
        
        Args:
            module_name: Name of the module
            blueprint: Blueprint specifications
            
        Returns:
            Complete Python code for the utils module
        """
        # Start with the module header
        code = self.generate_module_header(
            module_name, 
            "Utility functions for the agent"
        )
        
        # Add common utility functions
        code += '''
def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, 
                    max_delay: float = 60.0, backoff_factor: float = 2.0):
    """
    Decorator that retries the wrapped function with exponential backoff.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        
    Returns:
        Wrapped function with retry logic
        
    Example:
        @retry_with_backoff
        def fetch_data(url):
            return requests.get(url)
    """
    def wrapper(*args, **kwargs):
        """Wrapper function that implements the retry logic."""
        retry_count = 0
        delay = base_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                    raise
                
                logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                logger.warning(f"Waiting {delay:.2f} seconds before next attempt")
                
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
    
    return wrapper


def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """
    Parse a date string using multiple possible formats.
    
    Args:
        date_str: Date string to parse
        formats: List of formats to try (if None, uses common formats)
        
    Returns:
        Parsed datetime object or None if parsing fails
        
    Example:
        >>> parse_date("2023-01-15")
        datetime.datetime(2023, 1, 15, 0, 0)
    """
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ"
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_str}")
    return None


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace and normalizing line endings.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
        
    Example:
        >>> clean_text("Hello   world\\r\\n  ")
        "Hello world"
    """
    if not text:
        return ""
        
    # Replace multiple whitespace with single space
    import re
    text = re.sub(r'\\s+', ' ', text)
    
    # Replace various line endings with standard line ending
    text = text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
    
    # Strip leading/trailing whitespace
    return text.strip()


def safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely access nested dictionary values without raising KeyError.
    
    Args:
        data: Dictionary to access
        keys: List of keys representing the path to the value
        default: Default value if key doesn't exist
        
    Returns:
        Value from the dictionary or default if not found
        
    Example:
        >>> data = {"user": {"profile": {"name": "John"}}}
        >>> safe_get(data, ["user", "profile", "name"])
        "John"
        >>> safe_get(data, ["user", "settings", "theme"], "default")
        "default"
    """
    current = data
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current

'''
        
        return code