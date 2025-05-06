"""
Blueprint Designer for AutoGenesis.

This module handles the design of agent blueprints, including architecture
selection, tool selection, and resource analysis.
"""

import logging
import json
import os
import subprocess
import platform
from typing import Dict, Any, List, Optional, Set

from autogenesis.blueprint.tool_selector import ToolSelector
from autogenesis.blueprint.resource_analyzer import ResourceAnalyzer


class BlueprintDesigner:
    """
    Designs agent blueprints based on requirements and actions.
    
    Determines the architecture, tools, and implementation details
    for the agent being generated.
    """
    
    def __init__(self):
        """Initialize the blueprint designer with necessary components."""
        self.logger = logging.getLogger(__name__)
        self.tool_selector = ToolSelector()
        self.resource_analyzer = ResourceAnalyzer()
        
        # Define architecture patterns
        self.architecture_patterns = {
            "simple_script": {
                "description": "Single script implementation for simple tasks",
                "conditions": {
                    "max_actions": 5,
                    "complexity": "low",
                    "requires_state": False
                }
            },
            "modular": {
                "description": "Modular implementation with separate modules for different functions",
                "conditions": {
                    "max_actions": 15,
                    "complexity": "medium",
                    "requires_state": False
                }
            },
            "pipeline": {
                "description": "Data pipeline architecture for sequential processing steps",
                "conditions": {
                    "task_types": ["data_processing", "etl", "file_processing"],
                    "requires_state": True
                }
            },
            "service": {
                "description": "Service-based architecture with API endpoints",
                "conditions": {
                    "task_types": ["api_integration", "web_service"],
                    "complexity": "high",
                    "requires_api": True
                }
            },
            "event_driven": {
                "description": "Event-driven architecture for handling asynchronous operations",
                "conditions": {
                    "task_types": ["automation", "notification", "monitoring"],
                    "requires_scheduling": True
                }
            }
        }
    
    def design(self, 
            requirements: Dict[str, Any], 
            actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Design an agent blueprint based on requirements and actions.
        
        Args:
            requirements: Structured requirements for agent generation
            actions: List of atomic actions with dependencies
            
        Returns:
            Dict containing the agent blueprint
        """
        self.logger.info("Designing agent blueprint")
        
        # Ensure actions is not None to avoid iteration errors
        if actions is None:
            actions = []
        
        # Extract basic details
        task_type = requirements.get("task_type", "unknown")
        description = requirements.get("description", "")
        req_details = requirements.get("requirements", {})
        
        # Analyze resource constraints
        resource_specs = self.resource_analyzer.analyze_host_resources()
        self.logger.info(f"Host resource specs: {resource_specs}")
        
        # Select tools based on task type and actions
        selected_tools = self.tool_selector.select_tools(
            task_type, 
            actions, 
            req_details.get("preferred_tools", []),
            resource_specs
        )
        self.logger.info(f"Selected tools: {selected_tools}")
        
        # Determine architecture pattern
        architecture_type = self._determine_architecture(
            task_type, 
            actions, 
            req_details,
            resource_specs
        )
        self.logger.info(f"Selected architecture: {architecture_type}")
        
        # Define file structure based on architecture
        file_structure = self._define_file_structure(architecture_type, task_type, actions)
        
        # Define configuration points
        config_points = self._define_configuration_points(
            architecture_type, 
            task_type, 
            actions, 
            req_details
        )
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(
            architecture_type,
            selected_tools,
            actions,
            resource_specs
        )
        
        # Define expected usage patterns
        expected_usage = self._define_expected_usage(req_details)
        
        # Create blueprint
        blueprint = {
            "agent_id": requirements.get("agent_id", ""),
            "name": f"{task_type}_agent",
            "description": description,
            "task_type": task_type,
            "architecture_type": architecture_type,
            "tools": selected_tools,
            "actions": actions,
            "file_structure": file_structure,
            "config_points": config_points,
            "resource_requirements": resource_requirements,
            "expected_usage": expected_usage,
            "deployment_formats": req_details.get("deployment_format", ["script"]),
            "environment_variables": self._define_environment_variables(selected_tools, actions),
            "external_dependencies": self._define_external_dependencies(selected_tools)
        }
        
        self.logger.info("Blueprint design complete")
        return blueprint
    
    def _determine_architecture(self,
                               task_type: str,
                               actions: List[Dict[str, Any]],
                               req_details: Dict[str, Any],
                               resource_specs: Dict[str, Any]) -> str:
        """
        Determine the best architecture pattern for the agent.
        
        Args:
            task_type: Type of task
            actions: List of atomic actions
            req_details: Detailed requirements
            resource_specs: Host resource specifications
            
        Returns:
            String identifying the selected architecture type
        """
        # Calculate complexity
        num_actions = len(actions)
        complexity = "low"
        if num_actions > 10:
            complexity = "high"
        elif num_actions > 5:
            complexity = "medium"
        
        # Check if task requires state persistence
        requires_state = self._check_requires_state(actions)
        
        # Check if task requires API
        requires_api = "api" in task_type.lower() or any("api" in str(a).lower() for a in actions)
        
        # Check if task requires scheduling
        requires_scheduling = "schedule" in json.dumps(req_details).lower()
        
        # Calculate scores for each architecture
        scores = {}
        for arch_name, arch_details in self.architecture_patterns.items():
            score = 0
            conditions = arch_details["conditions"]
            
            # Check max actions condition
            if "max_actions" in conditions and num_actions <= conditions["max_actions"]:
                score += 1
            
            # Check complexity condition
            if "complexity" in conditions and complexity == conditions["complexity"]:
                score += 2
            
            # Check task type condition
            if "task_types" in conditions and task_type in conditions["task_types"]:
                score += 3
            
            # Check state requirement
            if "requires_state" in conditions and requires_state == conditions["requires_state"]:
                score += 1
            
            # Check API requirement
            if "requires_api" in conditions and requires_api == conditions["requires_api"]:
                score += 1
            
            # Check scheduling requirement
            if "requires_scheduling" in conditions and requires_scheduling == conditions["requires_scheduling"]:
                score += 1
            
            scores[arch_name] = score
        
        # Select architecture with highest score
        best_arch = max(scores.items(), key=lambda x: x[1])
        
        # If tie or very low score, default to modular
        if best_arch[1] == 0 or list(scores.values()).count(best_arch[1]) > 1:
            return "modular"
        
        return best_arch[0]
    
    def _check_requires_state(self, actions: List[Dict[str, Any]]) -> bool:
        """
        Check if the actions require state persistence.
        
        Args:
            actions: List of atomic actions
            
        Returns:
            Boolean indicating whether state persistence is required
        """
        # Look for actions that typically require state
        state_keywords = ["save_state", "load_state", "resume", "checkpoint", "persist"]
        
        for action in actions:
            action_str = json.dumps(action).lower()
            if any(keyword in action_str for keyword in state_keywords):
                return True
        
        # Check for complex dependencies that might indicate state needs
        dependency_graph = {}
        for action in actions:
            action_id = action.get("id", "")
            requires = action.get("requires", [])
            dependency_graph[action_id] = requires
        
        # If we have complex dependency chains, might need state
        max_chain = self._get_max_dependency_chain(dependency_graph)
        
        return max_chain > 5  # Arbitrary threshold
    
    def _get_max_dependency_chain(self, dependency_graph: Dict[str, List[str]]) -> int:
        """
        Calculate the maximum dependency chain length.
        
        Args:
            dependency_graph: Dictionary mapping action IDs to their dependencies
            
        Returns:
            Integer representing the longest dependency chain
        """
        memo = {}
        
        def get_chain_length(node):
            if node in memo:
                return memo[node]
            
            if node not in dependency_graph or not dependency_graph[node]:
                return 1
            
            max_length = 0
            for dep in dependency_graph[node]:
                max_length = max(max_length, get_chain_length(dep))
            
            memo[node] = 1 + max_length
            return memo[node]
        
        max_length = 0
        for node in dependency_graph:
            max_length = max(max_length, get_chain_length(node))
        
        return max_length
    
    def _define_file_structure(self,
                              architecture_type: str,
                              task_type: str,
                              actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Define the file structure for the agent.
        
        Args:
            architecture_type: Type of architecture
            task_type: Type of task
            actions: List of atomic actions
            
        Returns:
            Dict describing the file structure
        """
        # Base structure for all architecture types
        base_structure = {
            "root": {
                "files": ["README.md", "requirements.txt", "setup.py", "Dockerfile"],
                "directories": []
            }
        }
        
        # Architecture-specific structures
        if architecture_type == "simple_script":
            base_structure["root"]["files"].extend(["main.py", "config.yaml"])
        
        elif architecture_type == "modular":
            agent_name = f"{task_type}_agent"
            base_structure["root"]["directories"].extend([
                {
                    "name": agent_name,
                    "files": ["__init__.py", "main.py", "config.py"],
                    "directories": [
                        {
                            "name": "utils",
                            "files": ["__init__.py", "logging.py", "error_handler.py"]
                        },
                        {
                            "name": "core",
                            "files": ["__init__.py"]
                        }
                    ]
                },
                {
                    "name": "tests",
                    "files": ["__init__.py", "test_main.py"]
                },
                {
                    "name": "config",
                    "files": ["config.yaml", "logging.yaml"]
                }
            ])
            
            # Add action modules
            action_modules = set()
            for action in actions:
                module_name = action.get("name", "").split("_")[0]
                action_modules.add(module_name)
            
            core_dir = next(d for d in base_structure["root"]["directories"][0]["directories"] 
                          if d["name"] == "core")
            
            for module in action_modules:
                if module and module not in ["setup", "cleanup"]:
                    core_dir["files"].append(f"{module}.py")
        
        elif architecture_type == "pipeline":
            agent_name = f"{task_type}_agent"
            base_structure["root"]["directories"].extend([
                {
                    "name": agent_name,
                    "files": ["__init__.py", "main.py", "config.py", "pipeline.py"],
                    "directories": [
                        {
                            "name": "utils",
                            "files": ["__init__.py", "logging.py", "error_handler.py"]
                        },
                        {
                            "name": "stages",
                            "files": ["__init__.py"]
                        },
                        {
                            "name": "models",
                            "files": ["__init__.py", "data_model.py"]
                        }
                    ]
                },
                {
                    "name": "tests",
                    "files": ["__init__.py", "test_pipeline.py"]
                },
                {
                    "name": "config",
                    "files": ["config.yaml", "logging.yaml", "pipeline.yaml"]
                }
            ])
            
            # Add stage modules
            stages_dir = next(d for d in base_structure["root"]["directories"][0]["directories"] 
                           if d["name"] == "stages")
            
            for action in actions:
                if action.get("id") not in ["setup", "cleanup", "validate_inputs"]:
                    stages_dir["files"].append(f"{action.get('name')}.py")
        
        elif architecture_type == "service":
            agent_name = f"{task_type}_service"
            base_structure["root"]["directories"].extend([
                {
                    "name": agent_name,
                    "files": ["__init__.py", "main.py", "config.py", "api.py", "wsgi.py"],
                    "directories": [
                        {
                            "name": "utils",
                            "files": ["__init__.py", "logging.py", "error_handler.py"]
                        },
                        {
                            "name": "services",
                            "files": ["__init__.py"]
                        },
                        {
                            "name": "models",
                            "files": ["__init__.py", "data_model.py"]
                        },
                        {
                            "name": "routes",
                            "files": ["__init__.py", "api_routes.py"]
                        }
                    ]
                },
                {
                    "name": "tests",
                    "files": ["__init__.py", "test_api.py"]
                },
                {
                    "name": "config",
                    "files": ["config.yaml", "logging.yaml"]
                }
            ])
            
            # Add service modules
            services_dir = next(d for d in base_structure["root"]["directories"][0]["directories"] 
                             if d["name"] == "services")
            
            action_services = set()
            for action in actions:
                if action.get("id") not in ["setup", "cleanup", "validate_inputs"]:
                    service_name = action.get("name", "").split("_")[0]
                    action_services.add(service_name)
            
            for service in action_services:
                if service:
                    services_dir["files"].append(f"{service}_service.py")
        
        elif architecture_type == "event_driven":
            agent_name = f"{task_type}_agent"
            base_structure["root"]["directories"].extend([
                {
                    "name": agent_name,
                    "files": ["__init__.py", "main.py", "config.py", "event_handler.py"],
                    "directories": [
                        {
                            "name": "utils",
                            "files": ["__init__.py", "logging.py", "error_handler.py"]
                        },
                        {
                            "name": "handlers",
                            "files": ["__init__.py", "base_handler.py"]
                        },
                        {
                            "name": "models",
                            "files": ["__init__.py", "event.py", "state.py"]
                        }
                    ]
                },
                {
                    "name": "tests",
                    "files": ["__init__.py", "test_handlers.py"]
                },
                {
                    "name": "config",
                    "files": ["config.yaml", "logging.yaml", "events.yaml"]
                }
            ])
            
            # Add handler modules
            handlers_dir = next(d for d in base_structure["root"]["directories"][0]["directories"] 
                              if d["name"] == "handlers")
            
            for action in actions:
                if action.get("id") not in ["setup", "cleanup", "validate_inputs"]:
                    handlers_dir["files"].append(f"{action.get('name')}_handler.py")
        
        return base_structure
    
    def _define_configuration_points(self,
                                   architecture_type: str,
                                   task_type: str,
                                   actions: List[Dict[str, Any]],
                                   req_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define the configuration points for the agent.
        
        Args:
            architecture_type: Type of architecture
            task_type: Type of task
            actions: List of atomic actions
            req_details: Detailed requirements
            
        Returns:
            Dict describing the configuration points
        """
        # Common configuration points
        config = {
            "logging": {
                "level": "INFO",
                "format": "json",
                "file_path": "/var/log/agent_factory/",
                "configurable": True
            },
            "resource_limits": {
                "max_memory_mb": 512,
                "max_cpu_percent": 50,
                "configurable": True
            }
        }
        
        # Add source configuration
        source = req_details.get("source", {})
        if source:
            config["source"] = {
                "type": source.get("type", "unknown"),
                "value": source.get("value", ""),
                "configurable": True
            }
        
        # Add output configuration
        output = req_details.get("output", {})
        if output:
            config["output"] = {
                "format": output.get("format", "csv"),
                "destination_type": output.get("destination_type", "local_file"),
                "destination_value": output.get("destination_value", "output.csv"),
                "configurable": True
            }
        
        # Add constraints as configuration
        constraints = req_details.get("constraints", {})
        if constraints:
            config["constraints"] = {
                **constraints,
                "configurable": True
            }
        
        # Architecture-specific configuration
        if architecture_type == "pipeline":
            config["pipeline"] = {
                "max_retries": 3,
                "retry_delay_seconds": 5,
                "parallel_stages": False,
                "configurable": True
            }
        
        elif architecture_type == "service":
            config["api"] = {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "workers": 4,
                "timeout_seconds": 30,
                "configurable": True
            }
        
        elif architecture_type == "event_driven":
            config["events"] = {
                "max_queue_size": 1000,
                "worker_threads": 4,
                "configurable": True
            }
        
        # Add action-specific configuration
        action_config = {}
        for action in actions:
            if action.get("parameters"):
                action_id = action.get("id", "")
                action_config[action_id] = {
                    **action.get("parameters", {}),
                    "configurable": True
                }
        
        if action_config:
            config["actions"] = action_config
        
        return config
    
    def _calculate_resource_requirements(self,
                                       architecture_type: str,
                                       selected_tools: Dict[str, Any],
                                       actions: List[Dict[str, Any]],
                                       resource_specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate resource requirements for the agent.
        
        Args:
            architecture_type: Type of architecture
            selected_tools: Selected tools and libraries
            actions: List of atomic actions
            resource_specs: Host resource specifications
            
        Returns:
            Dict describing the resource requirements
        """
        # Base resource requirements
        resources = {
            "memory_mb": 256,
            "cpu_cores": 0.5,
            "disk_mb": 100
        }
        
        # Adjust based on architecture
        architecture_multipliers = {
            "simple_script": 1,
            "modular": 1.2,
            "pipeline": 1.5,
            "service": 2.0,
            "event_driven": 1.8
        }
        
        multiplier = architecture_multipliers.get(architecture_type, 1)
        
        resources["memory_mb"] *= multiplier
        resources["cpu_cores"] *= multiplier
        resources["disk_mb"] *= multiplier
        
        # Adjust based on tools
        for tool, details in selected_tools.items():
            resources["memory_mb"] += details.get("memory_mb", 0)
            resources["cpu_cores"] += details.get("cpu_cores", 0)
            resources["disk_mb"] += details.get("disk_mb", 0)
        
        # Adjust based on number of actions
        num_actions = len(actions)
        resources["memory_mb"] += num_actions * 5  # 5MB per action
        resources["cpu_cores"] += num_actions * 0.05  # 0.05 cores per action
        
        # Cap based on host resources (if available)
        if resource_specs:
            max_memory = resource_specs.get("memory_mb", 0) * 0.8  # Use at most 80% of available memory
            max_cpu = resource_specs.get("cpu_cores", 0) * 0.8     # Use at most 80% of available CPU
            
            resources["memory_mb"] = min(resources["memory_mb"], max_memory)
            resources["cpu_cores"] = min(resources["cpu_cores"], max_cpu)
        
        # Round to reasonable values
        resources["memory_mb"] = int(resources["memory_mb"])
        resources["cpu_cores"] = round(resources["cpu_cores"] * 2) / 2  # Round to nearest 0.5
        resources["disk_mb"] = int(resources["disk_mb"])
        
        return resources
    
    def _define_expected_usage(self, req_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define expected usage patterns for the agent.
        
        Args:
            req_details: Detailed requirements
            
        Returns:
            Dict describing the expected usage patterns
        """
        # Default usage patterns
        usage = {
            "frequency": "on_demand",
            "concurrent_users": 1,
            "expected_runtime_seconds": 60,
            "data_volume_mb": 10
        }
        
        # Update based on constraints
        constraints = req_details.get("constraints", {})
        
        if "run_schedule" in constraints:
            schedule = constraints["run_schedule"]
            
            if "daily" in schedule:
                usage["frequency"] = "daily"
            elif "hourly" in schedule:
                usage["frequency"] = "hourly"
            elif "weekly" in schedule:
                usage["frequency"] = "weekly"
            elif "monthly" in schedule:
                usage["frequency"] = "monthly"
        
        if "max_runtime_seconds" in constraints:
            usage["expected_runtime_seconds"] = constraints["max_runtime_seconds"]
        
        # Estimate data volume based on source and output
        source = req_details.get("source", {})
        output = req_details.get("output", {})
        
        source_type = source.get("type", "")
        if source_type == "url" and "file" in source_type:
            # For file sources, estimate based on common file sizes
            if "csv" in source.get("value", "").lower():
                usage["data_volume_mb"] = 50
            elif "json" in source.get("value", "").lower():
                usage["data_volume_mb"] = 30
            elif "xml" in source.get("value", "").lower():
                usage["data_volume_mb"] = 40
            elif "excel" in source.get("value", "").lower() or "xlsx" in source.get("value", "").lower():
                usage["data_volume_mb"] = 20
            elif "pdf" in source.get("value", "").lower():
                usage["data_volume_mb"] = 15
        elif source_type == "api":
            # For API sources, estimate based on typical API response sizes
            usage["data_volume_mb"] = 5
        elif source_type == "database":
            # For database sources, estimate based on typical query result sizes
            usage["data_volume_mb"] = 100
        
        return usage
    
    def _define_environment_variables(self, 
                                    selected_tools: Dict[str, Any],
                                    actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Define required environment variables for the agent.
        
        Args:
            selected_tools: Selected tools and libraries
            actions: List of atomic actions
            
        Returns:
            List of environment variable definitions
        """
        env_vars = []
        
        # Add logging environment variables
        env_vars.append({
            "name": "LOG_LEVEL",
            "description": "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            "default": "INFO",
            "required": False
        })
        
        env_vars.append({
            "name": "LOG_FILE",
            "description": "Path to log file",
            "default": "/var/log/agent_factory/agent.log",
            "required": False
        })
        
        # Check for authentication requirements in actions
        for action in actions:
            if action.get("name") == "authenticate":
                params = action.get("parameters", {})
                secret_name = params.get("credentials_secret_name", "")
                
                if secret_name:
                    # For each credential, add username and password/token
                    username_var = f"{secret_name.upper().replace('-', '_')}_USERNAME"
                    password_var = f"{secret_name.upper().replace('-', '_')}_PASSWORD"
                    
                    env_vars.append({
                        "name": username_var,
                        "description": f"Username for {secret_name}",
                        "default": "",
                        "required": True
                    })
                    
                    env_vars.append({
                        "name": password_var,
                        "description": f"Password for {secret_name}",
                        "default": "",
                        "required": True
                    })
        
        # Add tool-specific environment variables
        for tool, details in selected_tools.items():
            if tool == "selenium":
                env_vars.append({
                    "name": "SELENIUM_BROWSER",
                    "description": "Browser to use for Selenium (chrome, firefox)",
                    "default": "chrome",
                    "required": False
                })
            
            elif tool == "boto3":
                env_vars.extend([
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "description": "AWS access key ID",
                        "default": "",
                        "required": True
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "description": "AWS secret access key",
                        "default": "",
                        "required": True
                    },
                    {
                        "name": "AWS_REGION",
                        "description": "AWS region",
                        "default": "us-east-1",
                        "required": False
                    }
                ])
        
        return env_vars
    
    def _define_external_dependencies(self, selected_tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define external dependencies for the agent.
        
        Args:
            selected_tools: Selected tools and libraries
            
        Returns:
            Dict describing external dependencies
        """
        dependencies = {
            "python_packages": [],
            "system_packages": [],
            "services": []
        }
        
        # Add Python package dependencies
        for tool, details in selected_tools.items():
            # Add the main tool package
            dependencies["python_packages"].append({
                "name": tool,
                "version": details.get("version", "latest"),
                "purpose": details.get("description", "")
            })
            
            # Add related packages if any
            for related in details.get("related_packages", []):
                dependencies["python_packages"].append({
                    "name": related,
                    "version": "latest",
                    "purpose": f"Required by {tool}"
                })
        
        # Add system package dependencies
        for tool, details in selected_tools.items():
            if tool == "selenium":
                dependencies["system_packages"].extend([
                    {
                        "name": "chromium-browser",
                        "purpose": "Required for Selenium with Chrome"
                    },
                    {
                        "name": "chromedriver",
                        "purpose": "Driver for Chrome with Selenium"
                    }
                ])
            
            elif tool == "pandas":
                dependencies["system_packages"].append({
                    "name": "build-essential",
                    "purpose": "Required for compiling numerical libraries"
                })
            
            elif tool == "psycopg2":
                dependencies["system_packages"].append({
                    "name": "libpq-dev",
                    "purpose": "Required for PostgreSQL connection"
                })
        
        # Add service dependencies
        for tool, details in selected_tools.items():
            if tool == "boto3":
                dependencies["services"].append({
                    "name": "AWS",
                    "purpose": "Cloud services access"
                })
            
            elif tool == "sqlalchemy" or tool == "psycopg2":
                dependencies["services"].append({
                    "name": "Database",
                    "purpose": "Data storage and retrieval"
                })
        
        return dependencies