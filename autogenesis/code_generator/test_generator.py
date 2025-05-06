"""
Test Generator for AutoGenesis.

This module handles the generation of test files for the target agent
based on the blueprint and generated code.
"""

import logging
import os
import re
import inspect
from typing import Dict, Any, List, Optional, Set
import importlib.util
from jinja2 import Environment, FileSystemLoader, select_autoescape

class TestGenerator:
    """
    Generates test files for the target agent.
    
    Creates pytest test files for unit testing and integration testing
    of the generated agent code.
    """
    
    def __init__(self):
        """Initialize the test generator with templates and utilities."""
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment for templates
        self.template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates")
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters['to_snake_case'] = self._to_snake_case
        self.env.filters['to_camel_case'] = self._to_camel_case
        self.env.filters['to_class_name'] = self._to_class_name
    
    def generate(self, 
                blueprint: Dict[str, Any], 
                scaffold_path: str, 
                code_paths: List[str]) -> List[str]:
        """
        Generate test files for the target agent.
        
        Args:
            blueprint: Agent blueprint with architecture and tools
            scaffold_path: Path to the scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of paths to generated test files
        """
        self.logger.info(f"Generating tests for architecture: {blueprint.get('architecture_type')}")
        
        # Select generator function based on architecture type
        architecture_type = blueprint.get('architecture_type')
        generator_func = getattr(self, f"_generate_{architecture_type}_tests", self._generate_modular_tests)
        
        # Generate tests
        generated_files = generator_func(blueprint, scaffold_path, code_paths)
        
        self.logger.info(f"Generated {len(generated_files)} test files")
        return generated_files
    
    def _generate_simple_script_tests(self, 
                                    blueprint: Dict[str, Any], 
                                    scaffold_path: str, 
                                    code_paths: List[str]) -> List[str]:
        """
        Generate tests for a simple script architecture.
        
        Args:
            blueprint: Agent blueprint
            scaffold_path: Path to scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of generated test file paths
        """
        generated_files = []
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(scaffold_path, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate __init__.py in tests directory
        init_path = os.path.join(test_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write('"""Test package for the agent."""\n\n')
        generated_files.append(init_path)
        
        # Generate test_main.py
        test_main_path = os.path.join(test_dir, "test_main.py")
        self._generate_simple_script_main_test(blueprint, test_main_path, code_paths)
        generated_files.append(test_main_path)
        
        # Generate conftest.py
        conftest_path = os.path.join(test_dir, "conftest.py")
        self._generate_conftest(blueprint, conftest_path)
        generated_files.append(conftest_path)
        
        return generated_files
    
    def _generate_modular_tests(self, 
                              blueprint: Dict[str, Any], 
                              scaffold_path: str, 
                              code_paths: List[str]) -> List[str]:
        """
        Generate tests for a modular architecture.
        
        Args:
            blueprint: Agent blueprint
            scaffold_path: Path to scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of generated test file paths
        """
        generated_files = []
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(scaffold_path, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate __init__.py in tests directory
        init_path = os.path.join(test_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write('"""Test package for the agent."""\n\n')
        generated_files.append(init_path)
        
        # Generate conftest.py
        conftest_path = os.path.join(test_dir, "conftest.py")
        self._generate_conftest(blueprint, conftest_path)
        generated_files.append(conftest_path)
        
        # Generate test_main.py
        test_main_path = os.path.join(test_dir, "test_main.py")
        self._generate_modular_main_test(blueprint, test_main_path)
        generated_files.append(test_main_path)
        
        # Generate tests for core modules
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        core_dir = os.path.join(scaffold_path, agent_name, "core")
        
        # Find all Python files in core directory
        core_modules = [f for f in os.listdir(core_dir) if f.endswith('.py') and f != '__init__.py']
        
        for module_file in core_modules:
            module_name = module_file[:-3]  # Remove .py extension
            test_file_path = os.path.join(test_dir, f"test_{module_name}.py")
            self._generate_module_test(blueprint, test_file_path, module_name, os.path.join(core_dir, module_file))
            generated_files.append(test_file_path)
        
        return generated_files
    
    def _generate_pipeline_tests(self, 
                               blueprint: Dict[str, Any], 
                               scaffold_path: str, 
                               code_paths: List[str]) -> List[str]:
        """
        Generate tests for a pipeline architecture.
        
        Args:
            blueprint: Agent blueprint
            scaffold_path: Path to scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of generated test file paths
        """
        generated_files = []
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(scaffold_path, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate __init__.py in tests directory
        init_path = os.path.join(test_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write('"""Test package for the agent."""\n\n')
        generated_files.append(init_path)
        
        # Generate conftest.py
        conftest_path = os.path.join(test_dir, "conftest.py")
        self._generate_conftest(blueprint, conftest_path)
        generated_files.append(conftest_path)
        
        # Generate test_pipeline.py
        test_pipeline_path = os.path.join(test_dir, "test_pipeline.py")
        self._generate_pipeline_test(blueprint, test_pipeline_path)
        generated_files.append(test_pipeline_path)
        
        # Generate tests for each stage
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        stages_dir = os.path.join(scaffold_path, agent_name, "stages")
        
        # Find all Python files in stages directory that aren't base or __init__
        stage_modules = [
            f for f in os.listdir(stages_dir) 
            if f.endswith('.py') and f != '__init__.py' and f != 'base_stage.py'
        ]
        
        for stage_file in stage_modules:
            stage_name = stage_file[:-3]  # Remove .py extension
            test_file_path = os.path.join(test_dir, f"test_{stage_name}.py")
            self._generate_stage_test(blueprint, test_file_path, stage_name, os.path.join(stages_dir, stage_file))
            generated_files.append(test_file_path)
        
        return generated_files
    
    def _generate_service_tests(self, 
                              blueprint: Dict[str, Any], 
                              scaffold_path: str, 
                              code_paths: List[str]) -> List[str]:
        """
        Generate tests for a service architecture.
        
        Args:
            blueprint: Agent blueprint
            scaffold_path: Path to scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of generated test file paths
        """
        generated_files = []
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(scaffold_path, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate __init__.py in tests directory
        init_path = os.path.join(test_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write('"""Test package for the agent."""\n\n')
        generated_files.append(init_path)
        
        # Generate conftest.py
        conftest_path = os.path.join(test_dir, "conftest.py")
        self._generate_conftest(blueprint, conftest_path)
        generated_files.append(conftest_path)
        
        # Generate test_api.py
        test_api_path = os.path.join(test_dir, "test_api.py")
        self._generate_api_test(blueprint, test_api_path)
        generated_files.append(test_api_path)
        
        # Generate tests for each service
        agent_name = f"{blueprint.get('task_type', 'agent')}_service"
        services_dir = os.path.join(scaffold_path, agent_name, "services")
        
        # Find all Python files in services directory
        service_modules = [
            f for f in os.listdir(services_dir) 
            if f.endswith('.py') and f != '__init__.py'
        ]
        
        for service_file in service_modules:
            service_name = service_file[:-3]  # Remove .py extension
            test_file_path = os.path.join(test_dir, f"test_{service_name}.py")
            self._generate_service_test(blueprint, test_file_path, service_name, os.path.join(services_dir, service_file))
            generated_files.append(test_file_path)
        
        return generated_files
    
    def _generate_event_driven_tests(self, 
                                    blueprint: Dict[str, Any], 
                                    scaffold_path: str, 
                                    code_paths: List[str]) -> List[str]:
        """
        Generate tests for an event-driven architecture.
        
        Args:
            blueprint: Agent blueprint
            scaffold_path: Path to scaffolded project
            code_paths: List of paths to generated code files
            
        Returns:
            List of generated test file paths
        """
        generated_files = []
        
        # Create test directory if it doesn't exist
        test_dir = os.path.join(scaffold_path, "tests")
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate __init__.py in tests directory
        init_path = os.path.join(test_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write('"""Test package for the agent."""\n\n')
        generated_files.append(init_path)
        
        # Generate conftest.py
        conftest_path = os.path.join(test_dir, "conftest.py")
        self._generate_conftest(blueprint, conftest_path)
        generated_files.append(conftest_path)
        
        # Generate test_event_handler.py
        test_handler_path = os.path.join(test_dir, "test_event_handler.py")
        self._generate_event_handler_test(blueprint, test_handler_path)
        generated_files.append(test_handler_path)
        
        # Generate tests for each handler
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        handlers_dir = os.path.join(scaffold_path, agent_name, "handlers")
        
        # Find all Python files in handlers directory that aren't base or __init__
        handler_modules = [
            f for f in os.listdir(handlers_dir) 
            if f.endswith('.py') and f != '__init__.py' and f != 'base_handler.py'
        ]
        
        for handler_file in handler_modules:
            handler_name = handler_file[:-3]  # Remove .py extension
            test_file_path = os.path.join(test_dir, f"test_{handler_name}.py")
            self._generate_handler_test(blueprint, test_file_path, handler_name, os.path.join(handlers_dir, handler_file))
            generated_files.append(test_file_path)
        
        return generated_files
    
    def _generate_simple_script_main_test(self,
                                        blueprint: Dict[str, Any],
                                        file_path: str,
                                        code_paths: List[str]) -> None:
        """
        Generate test for the main.py in a simple script architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
            code_paths: List of paths to generated code files
        """
        self.logger.info(f"Generating simple script main test at {file_path}")
        
        # Load template
        template = self.env.get_template("test_simple_script_main.py.jinja")
        
        # Extract functions from main.py
        main_path = None
        for path in code_paths:
            if path.endswith('main.py'):
                main_path = path
                break
        
        functions = []
        if main_path:
            functions = self._extract_functions_from_file(main_path)
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            functions=functions
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_modular_main_test(self,
                                  blueprint: Dict[str, Any],
                                  file_path: str) -> None:
        """
        Generate test for the main.py in a modular architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
        """
        self.logger.info(f"Generating modular main test at {file_path}")
        
        # Load template
        template = self.env.get_template("test_modular_main.py.jinja")
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_pipeline_test(self,
                              blueprint: Dict[str, Any],
                              file_path: str) -> None:
        """
        Generate test for the pipeline in a pipeline architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
        """
        self.logger.info(f"Generating pipeline test at {file_path}")
        
        # Load template
        template = self.env.get_template("test_pipeline.py.jinja")
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_api_test(self,
                         blueprint: Dict[str, Any],
                         file_path: str) -> None:
        """
        Generate test for the API in a service architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
        """
        self.logger.info(f"Generating API test at {file_path}")
        
        # Determine which web framework to use
        tools = blueprint.get('tools', {})
        web_framework = "flask"  # Default
        if "fastapi" in tools:
            web_framework = "fastapi"
        
        # Load appropriate template
        template = self.env.get_template(f"test_{web_framework}_api.py.jinja")
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_service"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_event_handler_test(self,
                                   blueprint: Dict[str, Any],
                                   file_path: str) -> None:
        """
        Generate test for the event handler in an event-driven architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
        """
        self.logger.info(f"Generating event handler test at {file_path}")
        
        # Load template
        template = self.env.get_template("test_event_handler.py.jinja")
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_module_test(self,
                            blueprint: Dict[str, Any],
                            file_path: str,
                            module_name: str,
                            module_path: str) -> None:
        """
        Generate test for a core module in a modular architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
            module_name: Name of the module
            module_path: Path to the module file
        """
        self.logger.info(f"Generating test for module {module_name} at {file_path}")
        
        # Load template
        template = self.env.get_template("test_module.py.jinja")
        
        # Extract functions and classes from module
        functions = self._extract_functions_from_file(module_path)
        classes = self._extract_classes_from_file(module_path)
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            module_name=module_name,
            functions=functions,
            classes=classes
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_stage_test(self,
                           blueprint: Dict[str, Any],
                           file_path: str,
                           stage_name: str,
                           stage_path: str) -> None:
        """
        Generate test for a stage in a pipeline architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
            stage_name: Name of the stage
            stage_path: Path to the stage file
        """
        self.logger.info(f"Generating test for stage {stage_name} at {file_path}")
        
        # Load template
        template = self.env.get_template("test_stage.py.jinja")
        
        # Extract classes from stage file
        classes = self._extract_classes_from_file(stage_path)
        
        # Find stage class
        stage_class = None
        for cls in classes:
            if cls['name'].endswith('Stage'):
                stage_class = cls
                break
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            stage_name=stage_name,
            stage_class=stage_class
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_service_test(self,
                             blueprint: Dict[str, Any],
                             file_path: str,
                             service_name: str,
                             service_path: str) -> None:
        """
        Generate test for a service in a service architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
            service_name: Name of the service
            service_path: Path to the service file
        """
        self.logger.info(f"Generating test for service {service_name} at {file_path}")
        
        # Load template
        template = self.env.get_template("test_service.py.jinja")
        
        # Extract classes from service file
        classes = self._extract_classes_from_file(service_path)
        
        # Find service class
        service_class = None
        for cls in classes:
            if cls['name'].endswith('Service'):
                service_class = cls
                break
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_service"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            service_name=service_name,
            service_class=service_class
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_handler_test(self,
                             blueprint: Dict[str, Any],
                             file_path: str,
                             handler_name: str,
                             handler_path: str) -> None:
        """
        Generate test for a handler in an event-driven architecture.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the test file
            handler_name: Name of the handler
            handler_path: Path to the handler file
        """
        self.logger.info(f"Generating test for handler {handler_name} at {file_path}")
        
        # Load template
        template = self.env.get_template("test_handler.py.jinja")
        
        # Extract classes from handler file
        classes = self._extract_classes_from_file(handler_path)
        
        # Find handler class
        handler_class = None
        for cls in classes:
            if cls['name'].endswith('Handler'):
                handler_class = cls
                break
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            handler_name=handler_name,
            handler_class=handler_class
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _generate_conftest(self,
                         blueprint: Dict[str, Any],
                         file_path: str) -> None:
        """
        Generate conftest.py for pytest fixtures.
        
        Args:
            blueprint: Agent blueprint
            file_path: Path to write the file
        """
        self.logger.info(f"Generating conftest.py at {file_path}")
        
        # Load template
        template = self.env.get_template("conftest.py.jinja")
        
        # Define template variables
        agent_name = f"{blueprint.get('task_type', 'agent')}_agent"
        architecture_type = blueprint.get('architecture_type', 'modular')
        
        # Render template
        rendered = template.render(
            agent_name=agent_name,
            architecture_type=architecture_type
        )
        
        # Write file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(rendered)
    
    def _extract_functions_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of dictionaries with function information
        """
        functions = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Use regex to find function definitions
            pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
            matches = re.finditer(pattern, content)
            
            for match in matches:
                func_name = match.group(1)
                
                # Ignore private functions (starting with underscore)
                if func_name.startswith('_'):
                    continue
                
                params_str = match.group(2).strip()
                return_type = match.group(3).strip() if match.group(3) else None
                
                # Parse parameters
                params = []
                if params_str:
                    # Simple parameter parsing (doesn't handle complex annotations or defaults well)
                    param_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*([^=,]+))?(?:=([^,]+))?'
                    param_matches = re.finditer(param_pattern, params_str)
                    
                    for param_match in param_matches:
                        param_name = param_match.group(1)
                        
                        # Skip self parameter
                        if param_name == 'self':
                            continue
                            
                        param_type = param_match.group(2).strip() if param_match.group(2) else None
                        param_default = param_match.group(3).strip() if param_match.group(3) else None
                        
                        params.append({
                            'name': param_name,
                            'type': param_type,
                            'default': param_default
                        })
                
                # Extract docstring
                docstring = self._extract_docstring(content, match.end())
                
                functions.append({
                    'name': func_name,
                    'params': params,
                    'return_type': return_type,
                    'docstring': docstring
                })
        
        except Exception as e:
            self.logger.error(f"Error extracting functions from {file_path}: {e}")
        
        return functions
    
    def _extract_classes_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of dictionaries with class information
        """
        classes = []
        
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Use regex to find class definitions
            pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s*:'
            matches = re.finditer(pattern, content)
            
            for match in matches:
                class_name = match.group(1)
                
                # Get parent classes
                parents = []
                if match.group(2):
                    parents_str = match.group(2).strip()
                    parents = [p.strip() for p in parents_str.split(',')]
                
                # Extract docstring
                docstring = self._extract_docstring(content, match.end())
                
                # Extract methods
                methods = self._extract_methods(content, match.end())
                
                classes.append({
                    'name': class_name,
                    'parents': parents,
                    'docstring': docstring,
                    'methods': methods
                })
        
        except Exception as e:
            self.logger.error(f"Error extracting classes from {file_path}: {e}")
        
        return classes
    
    def _extract_docstring(self, content: str, start_pos: int) -> str:
        """
        Extract docstring following a function or class definition.
        
        Args:
            content: File content
            start_pos: Position after the function/class definition
            
        Returns:
            Docstring or empty string if none found
        """
        # Look for triple quotes after the definition
        triple_quote_pattern = r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(?:\s*$)?'
        remaining_content = content[start_pos:].strip()
        
        match = re.search(triple_quote_pattern, remaining_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_methods(self, content: str, class_start_pos: int) -> List[Dict[str, Any]]:
        """
        Extract method definitions from a class.
        
        Args:
            content: File content
            class_start_pos: Position after the class definition
            
        Returns:
            List of dictionaries with method information
        """
        methods = []
        
        # Find class block
        class_block = content[class_start_pos:]
        
        # Find next class or end of file
        next_class_match = re.search(r'^\s*class\s+', class_block, re.MULTILINE)
        if next_class_match:
            class_block = class_block[:next_class_match.start()]
        
        # Use regex to find method definitions
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
        matches = re.finditer(pattern, class_block)
        
        for match in matches:
            method_name = match.group(1)
            
            # Ignore private methods (starting with underscore)
            if method_name.startswith('_') and method_name != '__init__':
                continue
                
            params_str = match.group(2).strip()
            return_type = match.group(3).strip() if match.group(3) else None
            
            # Parse parameters
            params = []
            if params_str:
                # Simple parameter parsing
                param_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*([^=,]+))?(?:=([^,]+))?'
                param_matches = re.finditer(param_pattern, params_str)
                
                for param_match in param_matches:
                    param_name = param_match.group(1)
                    
                    # Skip self parameter
                    if param_name == 'self':
                        continue
                        
                    param_type = param_match.group(2).strip() if param_match.group(2) else None
                    param_default = param_match.group(3).strip() if param_match.group(3) else None
                    
                    params.append({
                        'name': param_name,
                        'type': param_type,
                        'default': param_default
                    })
            
            # Extract docstring
            docstring = self._extract_docstring(class_block, match.end())
            
            methods.append({
                'name': method_name,
                'params': params,
                'return_type': return_type,
                'docstring': docstring
            })
        
        return methods
    
    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return re.sub(r'\W+', '_', s2).lower()
    
    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        s = self._to_snake_case(text)
        components = s.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    
    def _to_class_name(self, text: str) -> str:
        """Convert text to CamelCase (for class names)."""
        s = self._to_snake_case(text)
        return ''.join(x.title() for x in s.split('_'))