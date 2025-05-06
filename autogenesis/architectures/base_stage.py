"""
Base Stage for Pipeline Architecture.

This module provides the base class for all pipeline stages.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List

class BaseStage:
    """
    Base class for all pipeline stages.
    
    Provides common functionality and interface for all pipeline stages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.config = config
        
        # Extract stage-specific configuration if available
        stage_name = self._get_stage_name()
        self.stage_config = config.get('actions', {}).get(stage_name, {})
        
        # Initialize any resources needed
        self._initialize_resources()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through this stage.
        
        Args:
            data: Input data for processing
            
        Returns:
            Dict containing processing results and output data
            
        Note:
            This method should be overridden by subclasses.
        """
        try:
            self.logger.info(f"Processing data through {self._get_stage_name()} stage")
            
            # Validate input data
            self._validate_input(data)
            
            # Process the data (to be implemented by subclasses)
            result = self._process_data(data)
            
            # Validate output
            self._validate_output(result)
            
            return {
                "status": "success",
                "stage": self._get_stage_name(),
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "stage": self._get_stage_name(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data.
        
        Args:
            data: Input data
            
        Returns:
            Dict containing processed data
            
        Note:
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_data method")
        
    def _initialize_resources(self) -> None:
        """
        Initialize any resources needed by this stage.
        
        Subclasses should override this method if they need to initialize
        resources such as database connections, file handles, etc.
        """
        pass
        
    def _validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            raise ValueError("Input data cannot be None")
            
    def _validate_output(self, data: Dict[str, Any]) -> None:
        """
        Validate output data.
        
        Args:
            data: Output data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            raise ValueError("Output data cannot be None")
            
    def _get_stage_name(self) -> str:
        """
        Get the name of this stage.
        
        Returns:
            String name of the stage (derived from class name)
        """
        return self.__class__.__name__.lower().replace('stage', '')
        
    def cleanup(self) -> None:
        """
        Clean up resources used by this stage.
        
        This method should be called when the stage is no longer needed,
        to free up any resources it has allocated.
        """
        pass