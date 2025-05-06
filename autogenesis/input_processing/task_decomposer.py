"""
Task Decomposer for AutoGenesis.

This module handles the decomposition of high-level tasks into granular,
executable atomic actions with dependencies.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple

class TaskDecomposer:
    """
    Decomposes high-level tasks into atomic actions with dependencies.
    
    Takes structured requirements and breaks them down into a sequence
    of atomic actions that the generated agent will need to perform.
    """
    
    def __init__(self):
        """Initialize the task decomposer."""
        self.logger = logging.getLogger(__name__)
        
        # Define task type templates for common tasks
        self.task_templates = {
            "web_scraping": self._web_scraping_template,
            "data_processing": self._data_processing_template,
            "api_integration": self._api_integration_template,
            "file_processing": self._file_processing_template,
            "ui_automation": self._ui_automation_template,
            "email_processing": self._email_processing_template
        }
    
    def decompose(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose a task into atomic actions.
        
        Args:
            requirements: Structured requirements for agent generation
            
        Returns:
            List of atomic action dictionaries with dependencies
        """
        self.logger.info("Decomposing task into atomic actions")
        
        task_type = requirements.get("task_type", "unknown")
        
        # Check if user already provided processing steps
        user_steps = requirements.get("requirements", {}).get("processing_steps", [])
        
        # If user provided steps, use them as a starting point
        if user_steps:
            self.logger.info(f"User provided {len(user_steps)} processing steps")
            actions = self._convert_user_steps_to_actions(user_steps, requirements)
        else:
            # Otherwise, use template based on task type
            self.logger.info(f"Using template for task type: {task_type}")
            template_func = self.task_templates.get(task_type, self._default_template)
            actions = template_func(requirements)
        
        # Add standard actions (setup, error handling, cleanup)
        complete_actions = self._add_standard_actions(actions, requirements)
        
        # Resolve dependencies
        actions_with_deps = self._resolve_dependencies(complete_actions)
        
        self.logger.info(f"Task decomposed into {len(actions_with_deps)} atomic actions")
        return actions_with_deps
    
    def _convert_user_steps_to_actions(self, 
                                      user_steps: List[Dict[str, Any]], 
                                      requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert user-provided processing steps to atomic actions.
        
        Args:
            user_steps: Processing steps provided by user
            requirements: Overall requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Process each user step
        for i, step in enumerate(user_steps):
            # Basic validation
            if not step.get("action"):
                self.logger.warning(f"Skipping user step {i} with no action")
                continue
                
            # Create core action
            action = {
                "id": f"user_action_{i}",
                "name": step["action"],
                "description": step.get("description", f"Perform {step['action']}"),
                "parameters": {},
                "requires": []
            }
            
            # Copy all non-standard fields as parameters
            for key, value in step.items():
                if key not in ["action", "description"]:
                    action["parameters"][key] = value
            
            # Set dependencies for sequential steps
            if i > 0:
                action["requires"].append(f"user_action_{i-1}")
            
            actions.append(action)
        
        return actions
    
    def _add_standard_actions(self, 
                             actions: List[Dict[str, Any]], 
                             requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Add standard actions for setup, validation, error handling, and cleanup.
        
        Args:
            actions: Core actions for the task
            requirements: Overall requirements
            
        Returns:
            List of actions including standard ones
        """
        result = []
        
        # Add setup action
        setup_action = {
            "id": "setup",
            "name": "setup",
            "description": "Initialize required resources and configurations",
            "parameters": {
                "logging_level": "DEBUG",
                "config_path": "/etc/agent/config.yaml"
            },
            "requires": []
        }
        result.append(setup_action)
        
        # Add validate inputs action
        validate_action = {
            "id": "validate_inputs",
            "name": "validate_inputs",
            "description": "Validate that all required inputs are available and correct",
            "parameters": {},
            "requires": ["setup"]
        }
        result.append(validate_action)
        
        # Add core actions, with dependency on validation
        for action in actions:
            # Ensure requires list exists
            if "requires" not in action:
                action["requires"] = []
                
            # If action has no dependencies, add dependency on validation
            if not action["requires"]:
                action["requires"].append("validate_inputs")
                
            result.append(action)
        
        # Add cleanup action that depends on all core actions
        cleanup_action = {
            "id": "cleanup",
            "name": "cleanup",
            "description": "Clean up resources and temporary files",
            "parameters": {},
            "requires": [action["id"] for action in actions]
        }
        result.append(cleanup_action)
        
        return result
    
    def _resolve_dependencies(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve and validate action dependencies.
        
        Args:
            actions: List of actions with potential dependencies
            
        Returns:
            List of actions with validated dependencies
        """
        # Build a set of valid action IDs
        valid_ids = {action["id"] for action in actions}
        
        # Validate dependencies
        for action in actions:
            valid_requires = []
            for req in action.get("requires", []):
                if req in valid_ids:
                    valid_requires.append(req)
                else:
                    self.logger.warning(f"Removing invalid dependency '{req}' from action '{action['id']}'")
            
            # Update with validated dependencies
            action["requires"] = valid_requires
        
        return actions
    
    def _web_scraping_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for a web scraping task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        url = source.get("value", "")
        
        # Check if authentication might be needed
        auth_needed = any(domain in url.lower() for domain in 
                         ["linkedin", "facebook", "github", "twitter", "instagram"])
        
        # Initialize browser/session
        actions.append({
            "id": "initialize_browser",
            "name": "initialize_browser",
            "description": "Initialize web browser or HTTP session",
            "parameters": {
                "headless": True,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            "requires": []
        })
        
        # Add authentication if needed
        if auth_needed:
            actions.append({
                "id": "authenticate",
                "name": "authenticate",
                "description": "Authenticate with the website",
                "parameters": {
                    "credentials_source": "environment",
                    "credentials_secret_name": f"{url.split('.')[1]}-credentials"
                },
                "requires": ["initialize_browser"]
            })
            last_action = "authenticate"
        else:
            last_action = "initialize_browser"
        
        # Navigate to URL
        actions.append({
            "id": "navigate_to_url",
            "name": "navigate_to_url",
            "description": f"Navigate to the target URL: {url}",
            "parameters": {
                "url": url,
                "timeout": 30
            },
            "requires": [last_action]
        })
        last_action = "navigate_to_url"
        
        # For paginated content, add pagination handling
        if "list" in url.lower() or "search" in url.lower():
            actions.append({
                "id": "setup_pagination",
                "name": "setup_pagination",
                "description": "Set up pagination to iterate through results",
                "parameters": {
                    "max_pages": 10,
                    "pagination_selector": ".pagination a.next"
                },
                "requires": [last_action]
            })
            last_action = "setup_pagination"
        
        # Extract data
        actions.append({
            "id": "extract_data",
            "name": "extract_data",
            "description": "Extract target data from the webpage",
            "parameters": {
                "css_selectors": {
                    "items": ".item-container",
                    "title": ".item-title",
                    "description": ".item-description",
                    "price": ".item-price"
                }
            },
            "requires": [last_action]
        })
        last_action = "extract_data"
        
        # Transform data
        actions.append({
            "id": "transform_data",
            "name": "transform_data",
            "description": "Transform and clean extracted data",
            "parameters": {
                "remove_html": True,
                "trim_whitespace": True
            },
            "requires": [last_action]
        })
        last_action = "transform_data"
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "csv")
        
        actions.append({
            "id": "save_data",
            "name": "save_data",
            "description": f"Save data to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", "output.csv")
            },
            "requires": [last_action]
        })
        
        return actions
    
    def _data_processing_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for a data processing task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        source_type = source.get("type", "file")
        source_value = source.get("value", "")
        
        # Load data
        actions.append({
            "id": "load_data",
            "name": "load_data",
            "description": f"Load data from {source_type}",
            "parameters": {
                "source_type": source_type,
                "source_value": source_value,
                "encoding": "utf-8"
            },
            "requires": []
        })
        last_action = "load_data"
        
        # Clean data
        actions.append({
            "id": "clean_data",
            "name": "clean_data",
            "description": "Clean and preprocess data",
            "parameters": {
                "remove_nulls": True,
                "convert_types": True,
                "normalize_text": True
            },
            "requires": [last_action]
        })
        last_action = "clean_data"
        
        # Transform data
        actions.append({
            "id": "transform_data",
            "name": "transform_data",
            "description": "Apply transformations to the data",
            "parameters": {
                "aggregation": "none",
                "filters": []
            },
            "requires": [last_action]
        })
        last_action = "transform_data"
        
        # Analyze (if appropriate)
        if "analyze" in json.dumps(requirements).lower() or "analytics" in json.dumps(requirements).lower():
            actions.append({
                "id": "analyze_data",
                "name": "analyze_data",
                "description": "Perform analysis on the data",
                "parameters": {
                    "statistics": ["mean", "median", "min", "max"],
                    "groupby": []
                },
                "requires": [last_action]
            })
            last_action = "analyze_data"
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "csv")
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", "output.csv"),
                "include_metadata": True
            },
            "requires": [last_action]
        })
        
        return actions
    
    def _api_integration_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for an API integration task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        api_url = source.get("value", "")
        
        # Initialize API client
        actions.append({
            "id": "initialize_client",
            "name": "initialize_client",
            "description": "Initialize API client and configuration",
            "parameters": {
                "base_url": api_url,
                "timeout": 30,
                "retry_attempts": 3
            },
            "requires": []
        })
        
        # Authenticate
        actions.append({
            "id": "authenticate",
            "name": "authenticate",
            "description": "Authenticate with the API",
            "parameters": {
                "auth_type": "oauth2",  # Default, will be overridden if specified
                "credentials_source": "environment",
                "credentials_secret_name": "api-credentials"
            },
            "requires": ["initialize_client"]
        })
        
        # Fetch data
        actions.append({
            "id": "fetch_data",
            "name": "fetch_data",
            "description": "Fetch data from the API",
            "parameters": {
                "endpoint": "/data",  # Placeholder
                "method": "GET",
                "query_params": {},
                "pagination": True
            },
            "requires": ["authenticate"]
        })
        
        # Process data
        actions.append({
            "id": "process_data",
            "name": "process_data",
            "description": "Process the API response data",
            "parameters": {
                "extract_fields": [],
                "transform": True
            },
            "requires": ["fetch_data"]
        })
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "json")
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", "output.json")
            },
            "requires": ["process_data"]
        })
        
        return actions
    
    def _file_processing_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for a file processing task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        source_type = source.get("type", "file")
        source_value = source.get("value", "")
        file_format = source.get("format", "")
        
        if not file_format and "." in source_value:
            file_format = source_value.split(".")[-1]
        
        # Determine file type and appropriate reader
        file_type_readers = {
            "csv": "csv_reader",
            "json": "json_reader",
            "xlsx": "excel_reader",
            "xls": "excel_reader",
            "xml": "xml_reader",
            "pdf": "pdf_reader",
            "txt": "text_reader"
        }
        
        reader = file_type_readers.get(file_format.lower(), "text_reader")
        
        # Read file
        actions.append({
            "id": "read_file",
            "name": reader,
            "description": f"Read {file_format} file from {source_type}",
            "parameters": {
                "source_type": source_type,
                "source_value": source_value,
                "encoding": "utf-8"
            },
            "requires": []
        })
        
        # Parse file content
        actions.append({
            "id": "parse_content",
            "name": "parse_content",
            "description": f"Parse {file_format} content into structured data",
            "parameters": {
                "schema_validation": True,
                "handle_errors": True
            },
            "requires": ["read_file"]
        })
        
        # Process data
        actions.append({
            "id": "process_data",
            "name": "process_data",
            "description": "Process and transform the parsed data",
            "parameters": {
                "operations": ["filter", "transform", "aggregate"],
                "custom_logic": False
            },
            "requires": ["parse_content"]
        })
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "csv")
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", f"output.{output_format}")
            },
            "requires": ["process_data"]
        })
        
        return actions
    
    def _ui_automation_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for a UI automation task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        url = source.get("value", "")
        
        # Initialize browser
        actions.append({
            "id": "initialize_browser",
            "name": "initialize_browser",
            "description": "Initialize web browser for UI automation",
            "parameters": {
                "browser": "chrome",
                "headless": False,  # UI automation often needs visible browser
                "window_size": "1920x1080"
            },
            "requires": []
        })
        
        # Navigate to URL
        actions.append({
            "id": "navigate_to_url",
            "name": "navigate_to_url",
            "description": f"Navigate to the target URL: {url}",
            "parameters": {
                "url": url,
                "timeout": 30,
                "wait_for": "page_load"
            },
            "requires": ["initialize_browser"]
        })
        
        # Authenticate if needed
        actions.append({
            "id": "authenticate",
            "name": "authenticate",
            "description": "Log in to the application",
            "parameters": {
                "username_selector": "#username",
                "password_selector": "#password",
                "submit_selector": "button[type='submit']",
                "credentials_source": "environment"
            },
            "requires": ["navigate_to_url"]
        })
        
        # Perform UI interaction
        actions.append({
            "id": "interact_with_ui",
            "name": "interact_with_ui",
            "description": "Perform UI interactions according to scenario",
            "parameters": {
                "max_steps": 10,
                "screenshot_on_error": True,
                "scenario_path": "/etc/agent/scenario.yaml"
            },
            "requires": ["authenticate"]
        })
        
        # Extract results
        actions.append({
            "id": "extract_results",
            "name": "extract_results",
            "description": "Extract results from UI interaction",
            "parameters": {
                "result_selector": ".results-container",
                "item_selector": ".result-item"
            },
            "requires": ["interact_with_ui"]
        })
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "json")
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", f"output.{output_format}")
            },
            "requires": ["extract_results"]
        })
        
        return actions
    
    def _email_processing_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate atomic actions for an email processing task.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        email_source = source.get("value", "")
        
        # Connect to email server
        actions.append({
            "id": "connect_to_email",
            "name": "connect_to_email",
            "description": "Connect to email server",
            "parameters": {
                "protocol": "imap",
                "server": email_source or "imap.gmail.com",
                "port": 993,
                "use_ssl": True
            },
            "requires": []
        })
        
        # Authenticate
        actions.append({
            "id": "authenticate",
            "name": "authenticate",
            "description": "Authenticate with email server",
            "parameters": {
                "credentials_source": "environment",
                "credentials_secret_name": "email-credentials"
            },
            "requires": ["connect_to_email"]
        })
        
        # Search for emails
        actions.append({
            "id": "search_emails",
            "name": "search_emails",
            "description": "Search for relevant emails",
            "parameters": {
                "folder": "INBOX",
                "search_criteria": {
                    "from": "",
                    "subject": "",
                    "since": "1d",
                    "unseen": True
                }
            },
            "requires": ["authenticate"]
        })
        
        # Process emails
        actions.append({
            "id": "process_emails",
            "name": "process_emails",
            "description": "Process matched emails",
            "parameters": {
                "extract_attachments": True,
                "extract_body": True,
                "mark_as_read": True
            },
            "requires": ["search_emails"]
        })
        
        # Process content
        actions.append({
            "id": "process_content",
            "name": "process_content",
            "description": "Process email content and attachments",
            "parameters": {
                "parse_attachments": True,
                "extract_information": True
            },
            "requires": ["process_emails"]
        })
        
        # Output destination
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "json")
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", f"output.{output_format}")
            },
            "requires": ["process_content"]
        })
        
        return actions
    
    def _default_template(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a generic template for unknown task types.
        
        Args:
            requirements: Task requirements
            
        Returns:
            List of atomic actions
        """
        actions = []
        
        # Get source details
        source = requirements.get("requirements", {}).get("source", {})
        source_type = source.get("type", "unknown")
        source_value = source.get("value", "")
        
        # Get output details
        output = requirements.get("requirements", {}).get("output", {})
        output_format = output.get("format", "json")
        
        # Default actions for any task
        actions.append({
            "id": "fetch_input",
            "name": "fetch_input",
            "description": f"Fetch input data from {source_type}",
            "parameters": {
                "source_type": source_type,
                "source_value": source_value
            },
            "requires": []
        })
        
        actions.append({
            "id": "process_data",
            "name": "process_data",
            "description": "Process the input data",
            "parameters": {
                "operations": ["transform"]
            },
            "requires": ["fetch_input"]
        })
        
        actions.append({
            "id": "save_results",
            "name": "save_results",
            "description": f"Save results to {output_format} format",
            "parameters": {
                "format": output_format,
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", f"output.{output_format}")
            },
            "requires": ["process_data"]
        })
        
        return actions