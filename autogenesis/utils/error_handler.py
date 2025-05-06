"""
Error handling utilities for AutoGenesis.

This module provides error handling helpers and decorators for the AutoGenesis system.
"""

import logging
import functools
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union

class AutoGenesisError(Exception):
    """Base exception class for AutoGenesis errors."""
    
    def __init__(self, message: str, details: Any = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(message)


class ValidationError(AutoGenesisError):
    """Exception raised for validation errors."""
    pass


class RuntimeError(AutoGenesisError):
    """Exception raised for runtime errors."""
    pass


class ConfigurationError(AutoGenesisError):
    """Exception raised for configuration errors."""
    pass


class ResourceError(AutoGenesisError):
    """Exception raised for resource-related errors."""
    pass


def handle_exceptions(
    logger: Optional[logging.Logger] = None,
    default_message: str = "An error occurred",
    reraise: bool = True,
    expected_exceptions: Optional[tuple] = None
) -> Callable:
    """
    Decorator to handle exceptions in a consistent way.
    
    Args:
        logger: Logger to use (defaults to module logger)
        default_message: Default error message
        reraise: Whether to re-raise the exception
        expected_exceptions: Tuple of expected exception types
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if exception is expected
                if expected_exceptions and isinstance(e, expected_exceptions):
                    # Log at info level for expected exceptions
                    logger.info(
                        f"Expected exception in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                else:
                    # Log at error level for unexpected exceptions
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                # Re-raise if requested
                if reraise:
                    raise
                
                # Otherwise, return default error response
                return {
                    "success": False,
                    "error": {
                        "message": str(e) or default_message,
                        "type": e.__class__.__name__,
                        "traceback": traceback.format_exc()
                    }
                }
        
        return wrapper
    
    return decorator


def format_error_response(
    exception: Exception, 
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Format an exception as a standardized error response.
    
    Args:
        exception: Exception to format
        include_traceback: Whether to include traceback
        
    Returns:
        Dict containing error response
    """
    response = {
        "success": False,
        "error": {
            "message": str(exception),
            "type": exception.__class__.__name__
        }
    }
    
    # Add details if available
    if hasattr(exception, 'details') and exception.details:
        response["error"]["details"] = exception.details
    
    # Add traceback if requested
    if include_traceback:
        response["error"]["traceback"] = traceback.format_exc()
    
    return response


def safe_execute(
    func: Callable,
    *args,
    error_message: str = "Execution failed",
    logger: Optional[logging.Logger] = None,
    default_return: Any = None,
    **kwargs
) -> Any:
    """
    Safely execute a function and handle any exceptions.
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_message: Error message to log
        logger: Logger to use (defaults to root logger)
        default_return: Default return value if execution fails
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default return value
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}", exc_info=True)
        return default_return