"""
Docker Manager for AutoGenesis.

This module handles Docker-related operations for creating sandbox
environments to test generated agents.
"""

import logging
import os
import subprocess
import tempfile
import shutil
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple

class DockerManager:
    """
    Manages Docker containers for agent testing.
    
    Creates and manages isolated Docker containers to provide
    sandbox environments for testing generated agents.
    """
    
    def __init__(self):
        """Initialize the Docker manager."""
        self.logger = logging.getLogger(__name__)
        self.containers = {}  # Map sandbox_id to container_id
    
    def create_sandbox(self, resource_requirements: Dict[str, Any]) -> str:
        """
        Create a Docker sandbox environment.
        
        Args:
            resource_requirements: Resource requirements for the container
            
        Returns:
            String sandbox ID
            
        Raises:
            RuntimeError: If Docker is not available or sandbox creation fails
        """
        self.logger.info("Creating Docker sandbox environment")
        
        # Check if Docker is available
        if not self._check_docker_available():
            raise RuntimeError("Docker is not available")
        
        # Generate sandbox ID
        sandbox_id = str(uuid.uuid4())
        
        # Create Docker container with resource limits
        try:
            # Define resource limits
            memory_limit = f"{resource_requirements.get('memory_mb', 512)}m"
            cpu_limit = resource_requirements.get('cpu_cores', 0.5)
            
            # Create container
            cmd = [
                "docker", "run",
                "-d",  # Detached mode
                "--name", f"autogenesis-sandbox-{sandbox_id}",
                "-m", memory_limit,
                "--cpus", str(cpu_limit),
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-e", "PYTHONUNBUFFERED=1",
                "--network", "host",  # Use host network for simplicity
                "-w", "/app",  # Working directory
                "python:3.10-slim",  # Base image
                "sleep", "infinity"  # Keep container running
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            container_id = result.stdout.strip()
            
            # Store container ID
            self.containers[sandbox_id] = container_id
            
            self.logger.info(f"Created sandbox {sandbox_id} with container {container_id}")
            
            # Install base dependencies
            self._install_base_dependencies(sandbox_id)
            
            return sandbox_id
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Docker sandbox creation failed: {e.stderr}")
            raise RuntimeError(f"Docker sandbox creation failed: {e.stderr}")
    
    def deploy_to_sandbox(self, sandbox_id: str, scaffold_path: str) -> None:
        """
        Deploy agent code to the sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            scaffold_path: Path to the scaffolded project
            
        Raises:
            RuntimeError: If deployment fails
        """
        self.logger.info(f"Deploying code to sandbox {sandbox_id}")
        
        if sandbox_id not in self.containers:
            raise RuntimeError(f"Sandbox {sandbox_id} does not exist")
        
        try:
            # Create a temporary tar archive of the scaffold
            temp_dir = tempfile.mkdtemp()
            tar_path = os.path.join(temp_dir, "scaffold.tar")
            
            # Create tar archive
            subprocess.run(
                ["tar", "-cf", tar_path, "-C", scaffold_path, "."],
                check=True,
                capture_output=True
            )
            
            # Copy tar to container
            container_name = f"autogenesis-sandbox-{sandbox_id}"
            subprocess.run(
                ["docker", "cp", tar_path, f"{container_name}:/app/scaffold.tar"],
                check=True,
                capture_output=True
            )
            
            # Extract tar in container
            subprocess.run(
                ["docker", "exec", container_name, "tar", "-xf", "/app/scaffold.tar", "-C", "/app"],
                check=True,
                capture_output=True
            )
            
            # Clean up
            subprocess.run(
                ["docker", "exec", container_name, "rm", "/app/scaffold.tar"],
                check=True,
                capture_output=True
            )
            shutil.rmtree(temp_dir)
            
            # Install requirements
            self._install_requirements(sandbox_id)
            
            self.logger.info(f"Code deployed to sandbox {sandbox_id}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Deployment to sandbox {sandbox_id} failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            raise RuntimeError(f"Deployment to sandbox failed: {str(e)}")
    
    def run_command(self, sandbox_id: str, command: List[str]) -> Tuple[int, str, str]:
        """
        Run a command in the sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            command: Command to run as a list of strings
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
            
        Raises:
            RuntimeError: If sandbox does not exist
        """
        self.logger.info(f"Running command in sandbox {sandbox_id}: {' '.join(command)}")
        
        if sandbox_id not in self.containers:
            raise RuntimeError(f"Sandbox {sandbox_id} does not exist")
        
        try:
            # Run command in container
            container_name = f"autogenesis-sandbox-{sandbox_id}"
            result = subprocess.run(
                ["docker", "exec", container_name] + command,
                capture_output=True,
                text=True
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command execution failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            return e.returncode if hasattr(e, 'returncode') else -1, "", str(e)
    
    def destroy_sandbox(self, sandbox_id: str) -> None:
        """
        Destroy a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            
        Raises:
            RuntimeError: If sandbox does not exist
        """
        self.logger.info(f"Destroying sandbox {sandbox_id}")
        
        if sandbox_id not in self.containers:
            raise RuntimeError(f"Sandbox {sandbox_id} does not exist")
        
        try:
            # Stop and remove container
            container_name = f"autogenesis-sandbox-{sandbox_id}"
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=True,
                capture_output=True
            )
            
            # Remove from containers dict
            del self.containers[sandbox_id]
            
            self.logger.info(f"Sandbox {sandbox_id} destroyed")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Sandbox destruction failed: {e.stderr}")
            raise RuntimeError(f"Sandbox destruction failed: {e.stderr}")
    
    def _check_docker_available(self) -> bool:
        """
        Check if Docker is available.
        
        Returns:
            Boolean indicating whether Docker is available
        """
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Docker is not available")
            return False
    
    def _install_base_dependencies(self, sandbox_id: str) -> None:
        """
        Install base dependencies in the sandbox.
        
        Args:
            sandbox_id: Sandbox ID
        """
        self.logger.info(f"Installing base dependencies in sandbox {sandbox_id}")
        
        try:
            # Run apt-get update and install basic dependencies
            container_name = f"autogenesis-sandbox-{sandbox_id}"
            
            # Update package lists
            subprocess.run(
                ["docker", "exec", container_name, "apt-get", "update", "-y"],
                check=True,
                capture_output=True
            )
            
            # Install base dependencies
            base_deps = ["git", "build-essential", "python3-dev", "python3-pip"]
            subprocess.run(
                ["docker", "exec", container_name, "apt-get", "install", "-y"] + base_deps,
                check=True,
                capture_output=True
            )
            
            # Upgrade pip
            subprocess.run(
                ["docker", "exec", container_name, "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )
            
            # Install basic Python packages
            basic_packages = ["pytest", "pytest-cov", "black", "flake8"]
            subprocess.run(
                ["docker", "exec", container_name, "pip", "install"] + basic_packages,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Base dependencies installed in sandbox {sandbox_id}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Base dependency installation failed: {e.stderr}")
            raise RuntimeError(f"Base dependency installation failed: {e.stderr}")
    
    def _install_requirements(self, sandbox_id: str) -> None:
        """
        Install project requirements in the sandbox.
        
        Args:
            sandbox_id: Sandbox ID
        """
        self.logger.info(f"Installing project requirements in sandbox {sandbox_id}")
        
        try:
            container_name = f"autogenesis-sandbox-{sandbox_id}"
            
            # Check if requirements.txt exists
            exit_code, _, _ = self.run_command(
                sandbox_id, 
                ["test", "-f", "requirements.txt"]
            )
            
            if exit_code == 0:
                # Install requirements
                subprocess.run(
                    ["docker", "exec", container_name, "pip", "install", "-r", "requirements.txt"],
                    check=True,
                    capture_output=True
                )
                self.logger.info(f"Project requirements installed in sandbox {sandbox_id}")
            else:
                self.logger.warning(f"No requirements.txt found in sandbox {sandbox_id}")
            
            # Install package in development mode
            exit_code, _, _ = self.run_command(
                sandbox_id, 
                ["test", "-f", "setup.py"]
            )
            
            if exit_code == 0:
                subprocess.run(
                    ["docker", "exec", container_name, "pip", "install", "-e", "."],
                    check=True,
                    capture_output=True
                )
                self.logger.info(f"Package installed in development mode in sandbox {sandbox_id}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Requirements installation failed: {e.stderr}")
            raise RuntimeError(f"Requirements installation failed: {e.stderr}")