"""
Lambda Packager for AutoGenesis.

This module handles the packaging of generated agent code into AWS Lambda functions.
"""

import logging
import os
import subprocess
import tempfile
import shutil
import zipfile
import json
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

class LambdaPackager:
    """
    Packages agent code into AWS Lambda deployment packages.
    
    Creates ZIP archives suitable for AWS Lambda deployment,
    with appropriate dependencies and handler configuration.
    """
    
    def __init__(self):
        """Initialize the Lambda packager."""
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment for Lambda handler template
        self.template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def package(self, 
               scaffold_path: str, 
               blueprint: Dict[str, Any], 
               output_path: Optional[str] = None) -> str:
        """
        Package agent code into an AWS Lambda deployment package.
        
        Args:
            scaffold_path: Path to the scaffolded project
            blueprint: Agent blueprint with tools and requirements
            output_path: Optional path for the output ZIP file
            
        Returns:
            String path to the created Lambda package
            
        Raises:
            RuntimeError: If Lambda packaging fails
        """
        self.logger.info("Packaging agent as AWS Lambda function")
        
        # Generate output path if not provided
        if not output_path:
            agent_name = blueprint.get('name', 'agent').lower()
            output_path = f"{agent_name}_lambda.zip"
        
        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            try:
                # Copy scaffold to build directory
                self._copy_scaffold(scaffold_path, build_dir)
                
                # Generate Lambda handler
                self._generate_lambda_handler(build_dir, blueprint)
                
                # Install dependencies
                self._install_dependencies(build_dir, blueprint)
                
                # Create ZIP archive
                self._create_zip_archive(build_dir, output_path)
                
                self.logger.info(f"Lambda package created successfully: {output_path}")
                return output_path
                
            except Exception as e:
                self.logger.error(f"Lambda packaging failed: {str(e)}")
                raise RuntimeError(f"Lambda packaging failed: {str(e)}")
    
    def _copy_scaffold(self, scaffold_path: str, build_dir: str) -> None:
        """
        Copy scaffold to build directory.
        
        Args:
            scaffold_path: Path to scaffold
            build_dir: Path to build directory
        """
        self.logger.info(f"Copying scaffold from {scaffold_path} to {build_dir}")
        
        # Copy all files except .git directory
        for item in os.listdir(scaffold_path):
            if item == ".git":
                continue
                
            source = os.path.join(scaffold_path, item)
            destination = os.path.join(build_dir, item)
            
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    
    def _generate_lambda_handler(self, build_dir: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate Lambda handler from blueprint.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
            
        Returns:
            Path to generated Lambda handler
        """
        self.logger.info("Generating Lambda handler")
        
        # Load Lambda handler template
        template = self.env.get_template("lambda_handler.py.jinja")
        
        # Extract required information from blueprint
        agent_name = blueprint.get('name', 'agent')
        architecture_type = blueprint.get('architecture_type', 'modular')
        task_type = blueprint.get('task_type', 'agent')
        
        # Create template context
        context = {
            "agent_name": agent_name,
            "architecture_type": architecture_type,
            "task_type": task_type,
            "main_module": self._determine_main_module(blueprint)
        }
        
        # Render template
        handler_content = template.render(**context)
        
        # Write Lambda handler
        handler_path = os.path.join(build_dir, "lambda_handler.py")
        with open(handler_path, 'w') as f:
            f.write(handler_content)
        
        return handler_path
    
    def _determine_main_module(self, blueprint: Dict[str, Any]) -> str:
        """
        Determine the main module import path based on the blueprint.
        
        Args:
            blueprint: Agent blueprint
            
        Returns:
            String representing the main module import path
        """
        architecture_type = blueprint.get('architecture_type', 'modular')
        task_type = blueprint.get('task_type', 'agent')
        
        if architecture_type == "simple_script":
            return "main"
        elif architecture_type == "service":
            return f"{task_type}_service.main"
        else:
            return f"{task_type}_agent.main"
    
    def _install_dependencies(self, build_dir: str, blueprint: Dict[str, Any]) -> None:
        """
        Install required dependencies in the build directory.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
            
        Raises:
            RuntimeError: If dependency installation fails
        """
        self.logger.info("Installing dependencies for Lambda package")
        
        # Check if requirements.txt exists
        requirements_path = os.path.join(build_dir, "requirements.txt")
        if not os.path.exists(requirements_path):
            # Generate requirements.txt from blueprint
            self._generate_requirements_file(build_dir, blueprint)
        
        # Create lib directory for dependencies
        lib_dir = os.path.join(build_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        
        try:
            # Install dependencies into lib directory
            subprocess.run(
                ["pip", "install", "-r", requirements_path, "-t", lib_dir],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dependency installation failed: {e.stderr}")
            raise RuntimeError(f"Dependency installation failed: {e.stderr}")
    
    def _generate_requirements_file(self, build_dir: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate requirements.txt from blueprint.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
            
        Returns:
            Path to generated requirements.txt
        """
        self.logger.info("Generating requirements.txt from blueprint")
        
        # Extract tools from blueprint
        tools = blueprint.get('tools', {})
        
        # Create requirements content
        requirements = []
        for tool, details in tools.items():
            version = details.get('version', '')
            if version:
                requirements.append(f"{tool}>={version}")
            else:
                requirements.append(tool)
        
        # Add common dependencies for Lambda
        requirements.extend([
            "boto3>=1.26.0",
            "aws-lambda-powertools>=2.0.0"
        ])
        
        # Write requirements.txt
        requirements_path = os.path.join(build_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("\n".join(requirements))
        
        return requirements_path
    
    def _create_zip_archive(self, build_dir: str, output_path: str) -> None:
        """
        Create ZIP archive of the Lambda package.
        
        Args:
            build_dir: Path to build directory
            output_path: Path for the output ZIP file
            
        Raises:
            RuntimeError: If ZIP creation fails
        """
        self.logger.info(f"Creating Lambda ZIP archive: {output_path}")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Create ZIP archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files in the build directory to the ZIP
                for root, _, files in os.walk(build_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, build_dir)
                        zipf.write(file_path, arcname)
        
        except Exception as e:
            self.logger.error(f"ZIP archive creation failed: {str(e)}")
            raise RuntimeError(f"ZIP archive creation failed: {str(e)}")