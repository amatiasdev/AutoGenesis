"""
Logging utilities for AutoGenesis.

This module provides logging setup and configuration for the AutoGenesis system.
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logging(log_level: str = None, log_file: str = None) -> None:
    """
    Set up logging for the AutoGenesis system.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    # Determine log level
    level_str = log_level or os.environ.get('LOG_LEVEL', 'INFO')
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    
    # Determine log file
    log_path = log_file or os.environ.get('LOG_FILE', None)
    
    # Create logs directory if not exists
    if log_path and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    elif not log_path:
        # Default to logs directory in current path
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'autogenesis.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # File handler (if log file specified)
    if log_path:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    # Create formatter
    formatter = JsonFormatter()
    
    # Add formatter to handlers and handlers to logger
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Log initial message
    logging.info(f"Logging initialized at level {level_str}")


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add filename, function name and line number
        log_data["file"] = record.filename
        log_data["function"] = record.funcName
        log_data["line"] = record.lineno
        
        # Add extra attributes from record
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text',
                           'filename', 'funcName', 'id', 'levelname', 'levelno',
                           'lineno', 'module', 'msecs', 'message', 'msg',
                           'name', 'pathname', 'process', 'processName',
                           'relativeCreated', 'stack_info', 'thread', 'threadName']:
                try:
                    # Try to serialize value to JSON
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, ValueError):
                    # If value is not JSON serializable, convert to string
                    log_data[key] = str(value)
        
        return json.dumps(log_data)