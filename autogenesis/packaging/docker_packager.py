"""
Docker Packager for AutoGenesis.

This module handles the packaging of generated agent code into Docker containers.
"""

import logging
import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

class DockerPackager:
    """
    Packages agent code into Docker containers.
    
    Creates Docker images from generated agent code, with appropriate
    dependencies and configuration.
    """
    
    def __init__(self):
        """Initialize the Docker packager."""
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment for Dockerfile template
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
               tag: Optional[str] = None) -> str:
        """
        Package agent code into a Docker container.
        
        Args:
            scaffold_path: Path to the scaffolded project
            blueprint: Agent blueprint with tools and requirements
            tag: Optional tag for the Docker image
            
        Returns:
            String ID of the created Docker image
            
        Raises:
            RuntimeError: If Docker packaging fails
        """
        self.logger.info("Packaging agent as Docker container")
        
        # Generate tag if not provided
        if not tag:
            tag = f"{blueprint.get('name', 'agent').lower()}:latest"
        
        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            try:
                # Copy scaffold to build directory
                self._copy_scaffold(scaffold_path, build_dir)
                
                # Generate Dockerfile
                dockerfile_path = self._generate_dockerfile(build_dir, blueprint)
                
                # Build Docker image
                image_id = self._build_docker_image(build_dir, tag)
                
                self.logger.info(f"Docker image built successfully: {image_id}")
                return image_id
                
            except Exception as e:
                self.logger.error(f"Docker packaging failed: {str(e)}")
                raise RuntimeError(f"Docker packaging failed: {str(e)}")
    
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
    
    def _generate_dockerfile(self, build_dir: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate Dockerfile from blueprint.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
            
        Returns:
            Path to generated Dockerfile
        """
        self.logger.info("Generating Dockerfile")
        
        # Load Dockerfile template
        template = self.env.get_template("docker_template.Dockerfile.jinja")
        
        # Extract required information from blueprint
        agent_name = blueprint.get('name', 'agent')
        tools = blueprint.get('tools', {})
        
        # Determine system dependencies
        system_deps = []
        for tool, details in tools.items():
            if tool == "selenium":
                system_deps.extend(["chromium-browser", "chromium-driver"])
            elif tool in ["psycopg2", "psycopg2-binary"]:
                system_deps.extend(["libpq-dev", "gcc"])
            elif tool == "pandas" or tool == "numpy":
                system_deps.append("build-essential")
        
        # Create template context
        context = {
            "agent_name": agent_name,
            "python_version": "3.10",  # Default Python version
            "system_dependencies": system_deps,
            "tools": list(tools.keys()),
            "entry_point": self._determine_entry_point(blueprint),
            "environment_variables": blueprint.get('environment_variables', [])
        }
        
        # Render template
        dockerfile_content = template.render(**context)
        
        # Write Dockerfile
        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return dockerfile_path
    
    def _determine_entry_point(self, blueprint: Dict[str, Any]) -> str:
        """
        Determine the Docker entry point based on the blueprint.
        
        Args:
            blueprint: Agent blueprint
            
        Returns:
            String representing the entry point command
        """
        architecture_type = blueprint.get('architecture_type', 'modular')
        task_type = blueprint.get('task_type', 'agent')
        
        if architecture_type == "simple_script":
            return "python main.py"
        elif architecture_type == "service":
            return f"python {task_type}_service/main.py"
        else:
            return f"python -m {task_type}_agent.main"
    
    def _build_docker_image(self, build_dir: str, tag: str) -> str:
        """
        Build Docker image from Dockerfile.
        
        Args:
            build_dir: Path to build directory containing Dockerfile
            tag: Tag for the Docker image
            
        Returns:
            String ID of the created Docker image
            
        Raises:
            RuntimeError: If Docker build fails
        """
        self.logger.info(f"Building Docker image with tag: {tag}")
        
        try:
            # Run docker build command
            result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                cwd=build_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Extract image ID from last line of output
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                if "Successfully built" in line:
                    image_id = line.split("Successfully built")[1].strip()
                    return image_id
            
            # If we can't find image ID, use the tag
            return tag
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker build failed: {e.stderr}")
            raise RuntimeError(f"Docker build failed: {e.stderr}")