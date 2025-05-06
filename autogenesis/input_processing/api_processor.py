"""
API Processor for AutoGenesis.

This module handles the processing and validation of API requests
for agent generation.
"""

import logging
import json
import jsonschema
from typing import Dict, Any, List, Optional

class APIProcessor:
    """
    Processes and validates API requests for agent generation.
    
    Ensures that incoming API requests meet the expected schema and 
    contain all necessary information for agent creation.
    """
    
    def __init__(self):
        """Initialize the API processor with validation schema."""
        self.logger = logging.getLogger(__name__)
        
        # Define JSON schema for validating API requests
        self.request_schema = {
            "type": "object",
            "required": ["task_type", "requirements"],
            "properties": {
                "task_type": {"type": "string"},
                "user_description_nl": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "required": ["source", "output"],
                    "properties": {
                        "source": {
                            "type": "object",
                            "required": ["type", "value"],
                            "properties": {
                                "type": {"type": "string"},
                                "value": {"type": "string"},
                                "format": {"type": "string"}
                            }
                        },
                        "processing_steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["action"],
                                "properties": {
                                    "action": {"type": "string"}
                                },
                                "additionalProperties": True
                            }
                        },
                        "output": {
                            "type": "object",
                            "required": ["format", "destination_type", "destination_value"],
                            "properties": {
                                "format": {"type": "string"},
                                "destination_type": {"type": "string"},
                                "destination_value": {"type": "string"}
                            }
                        },
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "rate_limit_per_minute": {"type": "integer"},
                                "run_schedule": {"type": "string"},
                                "max_runtime_seconds": {"type": "integer"}
                            },
                            "additionalProperties": True
                        },
                        "preferred_tools": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "deployment_format": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["docker", "lambda", "script"]}
                        }
                    }
                }
            }
        }
    
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate an API request payload.
        
        Args:
            payload: The API request payload
            
        Returns:
            Dict containing validated requirements
            
        Raises:
            ValueError: If the payload is invalid
        """
        self.logger.info("Processing API request payload")
        
        try:
            # Validate against schema
            jsonschema.validate(instance=payload, schema=self.request_schema)
            self.logger.info("Payload validation successful")
            
            # Additional validation and normalization
            requirements = self._normalize_requirements(payload)
            
            return requirements
            
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"Payload validation failed: {str(e)}")
            raise ValueError(f"Invalid API request: {str(e)}")
    
    def _normalize_requirements(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and enhance requirements with default values as needed.
        
        Args:
            payload: The validated API payload
            
        Returns:
            Dict containing normalized requirements
        """
        # Create a copy to avoid modifying the original
        normalized = json.loads(json.dumps(payload))
        
        # Ensure deployment_format has at least one value
        if "deployment_format" not in normalized["requirements"]:
            normalized["requirements"]["deployment_format"] = ["script"]
        
        # Ensure processing_steps exists (empty list if not provided)
        if "processing_steps" not in normalized["requirements"]:
            normalized["requirements"]["processing_steps"] = []
        
        # Ensure constraints exists
        if "constraints" not in normalized["requirements"]:
            normalized["requirements"]["constraints"] = {}
        
        # Ensure preferred_tools exists
        if "preferred_tools" not in normalized["requirements"]:
            normalized["requirements"]["preferred_tools"] = []
        
        # Add additional metadata
        normalized["metadata"] = {
            "source": "api",
            "timestamp": self._get_timestamp()
        }
        
        self.logger.debug(f"Normalized requirements: {json.dumps(normalized)}")
        return normalized
    
    def _get_timestamp(self) -> str:
        """Get current ISO-formatted timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()