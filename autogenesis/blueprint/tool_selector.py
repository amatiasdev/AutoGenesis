"""
Tool Selector for AutoGenesis.

This module handles the selection of appropriate tools and libraries
for agent implementation based on task requirements and resource constraints.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set

class ToolSelector:
    """
    Selects appropriate tools and libraries for agent implementation.
    
    Uses task type, actions, and resource constraints to determine
    the optimal set of tools for the agent being generated.
    """
    
    def __init__(self):
        """Initialize the tool selector with tool definitions."""
        self.logger = logging.getLogger(__name__)
        
        # Define known tools with their capabilities and resource requirements
        self.tools = {
            # Web tools
            "requests": {
                "description": "HTTP library for API calls and simple web interactions",
                "capabilities": ["http_requests", "api_integration", "web_scraping_basic"],
                "task_types": ["api_integration", "web_scraping"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "2.31.0"
            },
            "beautifulsoup4": {
                "description": "Library for parsing HTML and XML documents",
                "capabilities": ["html_parsing", "data_extraction"],
                "task_types": ["web_scraping", "data_extraction"],
                "related_packages": ["html5lib", "lxml"],
                "resource_requirements": {
                    "memory_mb": 20,
                    "cpu_cores": 0.1,
                    "disk_mb": 5
                },
                "version": "4.12.2"
            },
            "selenium": {
                "description": "Browser automation tool for complex web interactions",
                "capabilities": ["browser_automation", "javascript_execution", "web_scraping_advanced"],
                "task_types": ["web_scraping", "ui_automation"],
                "related_packages": ["webdriver_manager"],
                "resource_requirements": {
                    "memory_mb": 200,
                    "cpu_cores": 0.5,
                    "disk_mb": 100
                },
                "version": "4.14.0"
            },
            "playwright": {
                "description": "Modern browser automation for reliable end-to-end testing",
                "capabilities": ["browser_automation", "javascript_execution", "web_scraping_advanced"],
                "task_types": ["web_scraping", "ui_automation"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 250,
                    "cpu_cores": 0.5,
                    "disk_mb": 150
                },
                "version": "1.39.0"
            },
            "scrapy": {
                "description": "Framework for large-scale web scraping projects",
                "capabilities": ["web_scraping_advanced", "data_extraction", "crawling"],
                "task_types": ["web_scraping"],
                "related_packages": ["itemadapter", "parsel"],
                "resource_requirements": {
                    "memory_mb": 100,
                    "cpu_cores": 0.3,
                    "disk_mb": 20
                },
                "version": "2.11.0"
            },
            
            # Data processing tools
            "pandas": {
                "description": "Data analysis and manipulation library",
                "capabilities": ["data_processing", "data_analysis", "tabular_data"],
                "task_types": ["data_processing", "data_extraction", "etl"],
                "related_packages": ["numpy"],
                "resource_requirements": {
                    "memory_mb": 150,
                    "cpu_cores": 0.3,
                    "disk_mb": 50
                },
                "version": "2.1.2"
            },
            "numpy": {
                "description": "Numerical computing library",
                "capabilities": ["numerical_computing", "array_operations"],
                "task_types": ["data_processing", "data_analysis"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 50,
                    "cpu_cores": 0.2,
                    "disk_mb": 20
                },
                "version": "1.26.1"
            },
            "openpyxl": {
                "description": "Library for Excel file operations",
                "capabilities": ["excel_read", "excel_write"],
                "task_types": ["file_processing", "data_extraction"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 30,
                    "cpu_cores": 0.1,
                    "disk_mb": 10
                },
                "version": "3.1.2"
            },
            "xlrd": {
                "description": "Library for reading Excel files",
                "capabilities": ["excel_read"],
                "task_types": ["file_processing", "data_extraction"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 20,
                    "cpu_cores": 0.1,
                    "disk_mb": 5
                },
                "version": "2.0.1"
            },
            "xlwt": {
                "description": "Library for writing Excel files",
                "capabilities": ["excel_write"],
                "task_types": ["file_processing", "data_extraction"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 20,
                    "cpu_cores": 0.1,
                    "disk_mb": 5
                },
                "version": "1.3.0"
            },
            "pymongo": {
                "description": "MongoDB driver for Python",
                "capabilities": ["mongodb_integration", "nosql_database"],
                "task_types": ["data_processing", "api_integration", "database_integration"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 30,
                    "cpu_cores": 0.1,
                    "disk_mb": 10
                },
                "version": "4.5.0"
            },
            "sqlalchemy": {
                "description": "SQL toolkit and Object-Relational Mapping library",
                "capabilities": ["sql_database", "orm", "database_integration"],
                "task_types": ["data_processing", "api_integration", "database_integration"],
                "related_packages": ["psycopg2-binary"],
                "resource_requirements": {
                    "memory_mb": 50,
                    "cpu_cores": 0.2,
                    "disk_mb": 15
                },
                "version": "2.0.22"
            },
            
            # PDF processing
            "pdfminer.six": {
                "description": "Tool for extracting information from PDF documents",
                "capabilities": ["pdf_extraction", "text_extraction"],
                "task_types": ["file_processing", "data_extraction"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 80,
                    "cpu_cores": 0.2,
                    "disk_mb": 15
                },
                "version": "20221105"
            },
            "PyPDF2": {
                "description": "PDF toolkit for splitting, merging, cropping, and transforming PDF files",
                "capabilities": ["pdf_manipulation", "pdf_extraction"],
                "task_types": ["file_processing"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 40,
                    "cpu_cores": 0.1,
                    "disk_mb": 10
                },
                "version": "3.0.1"
            },
            
            # Email processing
            "imaplib": {
                "description": "Standard library for IMAP protocol client",
                "capabilities": ["email_access", "imap_protocol"],
                "task_types": ["email_processing"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "standard_library"
            },
            "email": {
                "description": "Standard library for parsing email messages",
                "capabilities": ["email_parsing"],
                "task_types": ["email_processing"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "standard_library"
            },
            "smtplib": {
                "description": "Standard library for SMTP protocol client",
                "capabilities": ["email_sending", "smtp_protocol"],
                "task_types": ["email_processing", "notification"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "standard_library"
            },
            
            # API frameworks
            "flask": {
                "description": "Lightweight web application framework",
                "capabilities": ["web_api", "web_server"],
                "task_types": ["api_integration", "web_service"],
                "related_packages": ["werkzeug", "jinja2"],
                "resource_requirements": {
                    "memory_mb": 50,
                    "cpu_cores": 0.2,
                    "disk_mb": 15
                },
                "version": "2.3.3"
            },
            "fastapi": {
                "description": "Modern, fast web framework for building APIs",
                "capabilities": ["web_api", "web_server", "async_processing"],
                "task_types": ["api_integration", "web_service"],
                "related_packages": ["uvicorn", "pydantic"],
                "resource_requirements": {
                    "memory_mb": 70,
                    "cpu_cores": 0.3,
                    "disk_mb": 20
                },
                "version": "0.104.0"
            },
            
            # AWS tools
            "boto3": {
                "description": "AWS SDK for Python",
                "capabilities": ["aws_integration", "s3_storage", "lambda_integration", "dynamodb"],
                "task_types": ["api_integration", "data_processing", "automation"],
                "related_packages": ["botocore"],
                "resource_requirements": {
                    "memory_mb": 40,
                    "cpu_cores": 0.1,
                    "disk_mb": 15
                },
                "version": "1.29.0"
            },
            
            # NLP tools
            "nltk": {
                "description": "Natural Language Toolkit",
                "capabilities": ["text_processing", "nlp"],
                "task_types": ["data_processing", "text_analysis"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 100,
                    "cpu_cores": 0.2,
                    "disk_mb": 200
                },
                "version": "3.8.1"
            },
            "spacy": {
                "description": "Advanced Natural Language Processing library",
                "capabilities": ["text_processing", "nlp", "entity_extraction"],
                "task_types": ["data_processing", "text_analysis"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 200,
                    "cpu_cores": 0.3,
                    "disk_mb": 500
                },
                "version": "3.7.2"
            },
            
            # Misc tools
            "python-crontab": {
                "description": "Library for crontab operations",
                "capabilities": ["scheduling", "automation"],
                "task_types": ["automation"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "3.0.0"
            },
            "schedule": {
                "description": "Python job scheduling for humans",
                "capabilities": ["scheduling", "automation"],
                "task_types": ["automation"],
                "related_packages": [],
                "resource_requirements": {
                    "memory_mb": 10,
                    "cpu_cores": 0.1,
                    "disk_mb": 1
                },
                "version": "1.2.1"
            }
        }
    
    def select_tools(self,
                task_type: str = "default",
                actions: List[Dict[str, Any]] = None,
                preferred_tools: List[str] = None,
                resource_specs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Select the appropriate tools for agent implementation.
        
        Args:
            task_type: Type of task (default: "default")
            actions: List of atomic actions (default: empty list)
            preferred_tools: User-preferred tools (default: empty list)
            resource_specs: Host resource specifications (default: empty dict)
            
        Returns:
            Dict of selected tools with their details
        """
        if actions is None:
            actions = []
        
        if preferred_tools is None:
            preferred_tools = []
        
        if resource_specs is None:
            resource_specs = {}
            
        self.logger.info(f"Selecting tools for task type '{task_type}'")
        
        # Extract required capabilities from actions
        required_capabilities = self._extract_required_capabilities(task_type, actions)
        self.logger.debug(f"Required capabilities: {required_capabilities}")
        
        # Calculate tool scores based on capabilities, task type, and resource requirements
        tool_scores = self._calculate_tool_scores(
            required_capabilities, 
            task_type, 
            preferred_tools,
            resource_specs
        )
        self.logger.debug(f"Tool scores: {tool_scores}")
        
        # Select tools based on scores and compatibility
        selected_tools = self._select_best_tools(tool_scores, required_capabilities)
        self.logger.info(f"Selected tools: {list(selected_tools.keys())}")
        
        return selected_tools
    
    def _extract_required_capabilities(self, 
                                     task_type: str, 
                                     actions: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract required capabilities from task type and actions.
        
        Args:
            task_type: Type of task
            actions: List of atomic actions
            
        Returns:
            Set of required capability strings
        """
        capabilities = set()
        
        # Add capabilities based on task type
        task_type_capability_map = {
            "web_scraping": ["http_requests", "html_parsing", "data_extraction"],
            "data_processing": ["data_processing"],
            "api_integration": ["http_requests", "api_integration"],
            "file_processing": ["data_processing"],
            "ui_automation": ["browser_automation"],
            "email_processing": ["email_access", "email_parsing"],
            "etl": ["data_processing", "data_extraction"],
            "notification": ["email_sending"],
            "automation": ["scheduling"],
            "web_service": ["web_api"],
            "text_analysis": ["text_processing", "nlp"]
        }
        
        if task_type in task_type_capability_map:
            capabilities.update(task_type_capability_map[task_type])
        
        # Add capabilities based on actions
        for action in actions:
            action_name = action.get("name", "")
            action_params = action.get("parameters", {})
            
            # Check for action-specific capabilities
            if "authenticate" in action_name:
                capabilities.add("authentication")
            
            if "navigate" in action_name and "url" in action_params:
                capabilities.add("http_requests")
                
            if "browser" in action_name:
                capabilities.add("browser_automation")
                
            if "extract" in action_name:
                capabilities.add("data_extraction")
                
            if "parse" in action_name:
                if "html" in json.dumps(action).lower():
                    capabilities.add("html_parsing")
                if "pdf" in json.dumps(action).lower():
                    capabilities.add("pdf_extraction")
                if "excel" in json.dumps(action).lower() or "xlsx" in json.dumps(action).lower():
                    capabilities.add("excel_read")
            
            if "save" in action_name or "write" in action_name:
                if "excel" in json.dumps(action).lower() or "xlsx" in json.dumps(action).lower():
                    capabilities.add("excel_write")
                    
            if "api" in action_name or "api" in json.dumps(action_params).lower():
                capabilities.add("api_integration")
                
            if "transform" in action_name or "process" in action_name:
                capabilities.add("data_processing")
                
            if "email" in action_name:
                if "send" in action_name:
                    capabilities.add("email_sending")
                else:
                    capabilities.add("email_access")
                    capabilities.add("email_parsing")
            
            if "database" in json.dumps(action).lower() or "sql" in json.dumps(action).lower():
                capabilities.add("sql_database")
                
            if "mongo" in json.dumps(action).lower():
                capabilities.add("mongodb_integration")
                
            if "s3" in json.dumps(action).lower() or "aws" in json.dumps(action).lower():
                capabilities.add("aws_integration")
                
            if "schedule" in action_name or "cron" in action_name:
                capabilities.add("scheduling")
        
        return capabilities
    
    def _calculate_tool_scores(self,
                             required_capabilities: Set[str],
                             task_type: str,
                             preferred_tools: List[str],
                             resource_specs: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate scores for each tool based on capabilities, task type, and resources.
        
        Args:
            required_capabilities: Set of required capabilities
            task_type: Type of task
            preferred_tools: User-preferred tools (if any)
            resource_specs: Host resource specifications
            
        Returns:
            Dict mapping tool names to their scores
        """
        scores = {}
        
        for tool_name, tool_info in self.tools.items():
            # Start with base score of 0
            score = 0
            
            # Score based on capability match
            tool_capabilities = set(tool_info.get("capabilities", []))
            capability_match = tool_capabilities.intersection(required_capabilities)
            
            if capability_match:
                # Add points for each matched capability
                score += len(capability_match) * 2
                
                # Add extra points if all required capabilities are met by this tool
                if required_capabilities.issubset(tool_capabilities):
                    score += 5
            else:
                # If no capabilities match, skip this tool
                continue
            
            # Score based on task type match
            if task_type in tool_info.get("task_types", []):
                score += 3
            
            # Score based on user preference
            if tool_name in preferred_tools:
                score += 10
            
            # Score based on resource requirements (if available)
            if resource_specs:
                tool_resources = tool_info.get("resource_requirements", {})
                
                # Check if the tool's resource requirements are within host capabilities
                host_memory = resource_specs.get("memory_mb", 0)
                host_cpu = resource_specs.get("cpu_cores", 0)
                
                tool_memory = tool_resources.get("memory_mb", 0)
                tool_cpu = tool_resources.get("cpu_cores", 0)
                
                # Penalize tools that require too many resources
                if tool_memory > host_memory * 0.5:
                    score -= 2
                
                if tool_cpu > host_cpu * 0.5:
                    score -= 2
                
                # Favor lightweight tools when resources are limited
                if host_memory < 1024 and tool_memory < 50:
                    score += 1
                
                if host_cpu < 2 and tool_cpu < 0.2:
                    score += 1
            
            scores[tool_name] = score
        
        return scores
    
    def _select_best_tools(self,
                         tool_scores: Dict[str, float],
                         required_capabilities: Set[str]) -> Dict[str, Any]:
        """
        Select the best tools based on scores and ensure all capabilities are covered.
        
        Args:
            tool_scores: Dict mapping tool names to their scores
            required_capabilities: Set of required capabilities
            
        Returns:
            Dict of selected tools with their details
        """
        # Sort tools by score (descending)
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Start with an empty selection
        selected_tools = {}
        covered_capabilities = set()
        
        # Add tools until all capabilities are covered or no more tools available
        for tool_name, score in sorted_tools:
            # If all capabilities are covered, stop unless this is a high-scoring tool
            if required_capabilities.issubset(covered_capabilities) and score < 5:
                break
                
            # Get tool details
            tool_info = self.tools[tool_name]
            tool_capabilities = set(tool_info.get("capabilities", []))
            
            # Check if this tool adds new capabilities
            new_capabilities = tool_capabilities - covered_capabilities
            
            if new_capabilities or score >= 5:
                # Add tool to selection
                selected_tools[tool_name] = tool_info
                covered_capabilities.update(tool_capabilities)
        
        # Check if all required capabilities are covered
        if not required_capabilities.issubset(covered_capabilities):
            missing_capabilities = required_capabilities - covered_capabilities
            self.logger.warning(f"Not all required capabilities could be covered. Missing: {missing_capabilities}")
            
            # Try to add tools to cover missing capabilities
            for capability in missing_capabilities:
                for tool_name, tool_info in self.tools.items():
                    if capability in tool_info.get("capabilities", []) and tool_name not in selected_tools:
                        selected_tools[tool_name] = tool_info
                        covered_capabilities.add(capability)
                        self.logger.info(f"Added {tool_name} to cover missing capability: {capability}")
                        break
        
        return selected_tools