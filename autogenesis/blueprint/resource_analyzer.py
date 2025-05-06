"""
Resource Analyzer for AutoGenesis.

This module analyzes the resources available on the host system
to inform resource-aware agent design decisions.
"""

import logging
import os
import platform
import subprocess
import json
import socket
from typing import Dict, Any, Optional, Tuple

class ResourceAnalyzer:
    """
    Analyzes host system resources to inform agent design decisions.
    
    Detects available CPU, memory, and other resources to optimize
    agent design for the specific environment.
    """
    
    def __init__(self):
        """Initialize the resource analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_host_resources(self) -> Dict[str, Any]:
        """
        Analyze the resources available on the host system.
        
        Returns:
            Dict containing resource specifications
        """
        self.logger.info("Analyzing host resources")
        
        # Initialize resource specs
        resource_specs = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        }
        
        # Attempt to get CPU and memory info
        cpu_info = self._get_cpu_info()
        if cpu_info:
            resource_specs.update(cpu_info)
        
        memory_info = self._get_memory_info()
        if memory_info:
            resource_specs.update(memory_info)
        
        # Check for Docker
        resource_specs["docker_available"] = self._check_docker_available()
        
        # Check for AWS environment
        resource_specs["is_aws"] = self._check_aws_environment()
        
        # Get network info
        resource_specs["hostname"] = socket.gethostname()
        try:
            resource_specs["ip_address"] = socket.gethostbyname(socket.gethostname())
        except Exception:
            resource_specs["ip_address"] = "unknown"
        
        self.logger.info(f"Resource analysis complete: {resource_specs}")
        return resource_specs
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information from the host system.
        
        Returns:
            Dict containing CPU information
        """
        cpu_info = {}
        
        try:
            if platform.system() == "Linux":
                # Using /proc/cpuinfo on Linux
                cpu_info = self._get_linux_cpu_info()
            elif platform.system() == "Darwin":
                # Using sysctl on macOS
                cpu_info = self._get_macos_cpu_info()
            elif platform.system() == "Windows":
                # Using WMI on Windows
                cpu_info = self._get_windows_cpu_info()
            else:
                # Fallback for unknown systems
                import multiprocessing
                cpu_info["cpu_cores"] = multiprocessing.cpu_count()
                
        except Exception as e:
            self.logger.error(f"Error getting CPU info: {str(e)}")
            # Fallback to multiprocessing
            try:
                import multiprocessing
                cpu_info["cpu_cores"] = multiprocessing.cpu_count()
            except Exception:
                cpu_info["cpu_cores"] = 1  # Default if we can't detect
        
        return cpu_info
    
    def _get_linux_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information from a Linux system.
        
        Returns:
            Dict containing CPU information
        """
        cpu_info = {}
        
        try:
            # Try to get information from /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            
            # Count physical processors
            physical_processors = len([line for line in cpuinfo.split('\n') if line.startswith('physical id')])
            cpu_info["physical_processors"] = physical_processors if physical_processors > 0 else 1
            
            # Count cores per processor
            cores_per_processor = len([line for line in cpuinfo.split('\n') if line.startswith('cpu cores')])
            if cores_per_processor > 0:
                cores_per_cpu = int(cpuinfo.split('\n')[cpuinfo.split('\n').index('cpu cores\t: ' + str(int(cores_per_processor / physical_processors))) + 1].split(': ')[1])
                cpu_info["cores_per_processor"] = cores_per_cpu
            
            # Total cores
            import multiprocessing
            cpu_info["cpu_cores"] = multiprocessing.cpu_count()
            
            # CPU model
            model_lines = [line for line in cpuinfo.split('\n') if line.startswith('model name')]
            if model_lines:
                cpu_info["cpu_model"] = model_lines[0].split(': ')[1]
        
        except Exception as e:
            self.logger.error(f"Error parsing Linux CPU info: {str(e)}")
            # Fallback
            import multiprocessing
            cpu_info["cpu_cores"] = multiprocessing.cpu_count()
        
        return cpu_info
    
    def _get_macos_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information from a macOS system.
        
        Returns:
            Dict containing CPU information
        """
        cpu_info = {}
        
        try:
            # Get CPU brand
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            cpu_info["cpu_model"] = brand
            
            # Get physical cores
            physical_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
            cpu_info["physical_cores"] = physical_cores
            
            # Get logical cores
            logical_cores = int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).decode().strip())
            cpu_info["cpu_cores"] = logical_cores
        
        except Exception as e:
            self.logger.error(f"Error getting macOS CPU info: {str(e)}")
            # Fallback
            import multiprocessing
            cpu_info["cpu_cores"] = multiprocessing.cpu_count()
        
        return cpu_info
    
    def _get_windows_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information from a Windows system.
        
        Returns:
            Dict containing CPU information
        """
        cpu_info = {}
        
        try:
            # Use wmic command
            cpu_output = subprocess.check_output("wmic cpu get Caption, NumberOfCores, NumberOfLogicalProcessors /format:list", shell=True).decode()
            
            # Parse output
            for line in cpu_output.split('\n'):
                if ":" in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "Caption":
                        cpu_info["cpu_model"] = value
                    elif key == "NumberOfCores":
                        cpu_info["physical_cores"] = int(value)
                    elif key == "NumberOfLogicalProcessors":
                        cpu_info["cpu_cores"] = int(value)
        
        except Exception as e:
            self.logger.error(f"Error getting Windows CPU info: {str(e)}")
            # Fallback
            import multiprocessing
            cpu_info["cpu_cores"] = multiprocessing.cpu_count()
        
        return cpu_info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information from the host system.
        
        Returns:
            Dict containing memory information
        """
        memory_info = {}
        
        try:
            if platform.system() == "Linux":
                # Using /proc/meminfo on Linux
                memory_info = self._get_linux_memory_info()
            elif platform.system() == "Darwin":
                # Using sysctl on macOS
                memory_info = self._get_macos_memory_info()
            elif platform.system() == "Windows":
                # Using WMI on Windows
                memory_info = self._get_windows_memory_info()
        
        except Exception as e:
            self.logger.error(f"Error getting memory info: {str(e)}")
            memory_info["memory_mb"] = 1024  # Default if we can't detect
        
        return memory_info
    
    def _get_linux_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information from a Linux system.
        
        Returns:
            Dict containing memory information
        """
        memory_info = {}
        
        try:
            # Try to get information from /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            # Parse total memory
            total_mem_line = [line for line in meminfo.split('\n') if line.startswith('MemTotal')][0]
            total_mem_kb = int(total_mem_line.split()[1])
            memory_info["memory_mb"] = total_mem_kb // 1024
            
            # Parse free memory
            free_mem_line = [line for line in meminfo.split('\n') if line.startswith('MemFree')][0]
            free_mem_kb = int(free_mem_line.split()[1])
            memory_info["free_memory_mb"] = free_mem_kb // 1024
            
            # Parse available memory (more accurate than free)
            avail_mem_line = [line for line in meminfo.split('\n') if line.startswith('MemAvailable')]
            if avail_mem_line:
                avail_mem_kb = int(avail_mem_line[0].split()[1])
                memory_info["available_memory_mb"] = avail_mem_kb // 1024
        
        except Exception as e:
            self.logger.error(f"Error parsing Linux memory info: {str(e)}")
            memory_info["memory_mb"] = 1024  # Default
        
        return memory_info
    
    def _get_macos_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information from a macOS system.
        
        Returns:
            Dict containing memory information
        """
        memory_info = {}
        
        try:
            # Get total physical memory
            total_mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            memory_info["memory_mb"] = int(total_mem) // (1024 * 1024)
            
            # Get memory usage with vm_stat
            vm_stat = subprocess.check_output(["vm_stat"]).decode().strip()
            
            # Parse vm_stat output
            page_size = 4096  # Default page size
            for line in vm_stat.split('\n'):
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                    break
            
            # Calculate free memory
            free_pages = 0
            for line in vm_stat.split('\n'):
                if "Pages free" in line:
                    free_pages = int(line.split(': ')[1].strip('\.'))
                    break
            
            memory_info["free_memory_mb"] = (free_pages * page_size) // (1024 * 1024)
        
        except Exception as e:
            self.logger.error(f"Error getting macOS memory info: {str(e)}")
            memory_info["memory_mb"] = 1024  # Default
        
        return memory_info
    
    def _get_windows_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information from a Windows system.
        
        Returns:
            Dict containing memory information
        """
        memory_info = {}
        
        try:
            # Use wmic command
            mem_output = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory /format:list", shell=True).decode()
            
            # Parse total memory
            for line in mem_output.split('\n'):
                if "=" in line:
                    key, value = line.split('=', 1)
                    if key.strip() == "TotalPhysicalMemory":
                        memory_info["memory_mb"] = int(value.strip()) // (1024 * 1024)
            
            # Get free memory
            mem_status = subprocess.check_output("wmic OS get FreePhysicalMemory /format:list", shell=True).decode()
            
            # Parse free memory
            for line in mem_status.split('\n'):
                if "=" in line:
                    key, value = line.split('=', 1)
                    if key.strip() == "FreePhysicalMemory":
                        memory_info["free_memory_mb"] = int(value.strip()) // 1024
        
        except Exception as e:
            self.logger.error(f"Error getting Windows memory info: {str(e)}")
            memory_info["memory_mb"] = 1024  # Default
        
        return memory_info
    
    def _check_docker_available(self) -> bool:
        """
        Check if Docker is available on the system.
        
        Returns:
            Boolean indicating whether Docker is available
        """
        try:
            result = subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_aws_environment(self) -> bool:
        """
        Check if running in an AWS environment.
        
        Returns:
            Boolean indicating whether running in AWS
        """
        # Check for common AWS environment variables
        aws_env_vars = ['AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'EC2_INSTANCE_ID']
        for var in aws_env_vars:
            if var in os.environ:
                return True
        
        # Check for EC2 metadata service
        try:
            # Try to connect to the EC2 metadata service (with timeout)
            socket.setdefaulttimeout(0.5)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("169.254.169.254", 80))
            return True
        except Exception:
            return False
        
        return False
    
    def analyze_blueprint_resources(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the resource requirements for a given blueprint.
        
        Args:
            blueprint (Dict[str, Any]): The blueprint to analyze
            
        Returns:
            Dict[str, Any]: Estimated resource requirements
        """
        self.logger.info(f"Analyzing resource requirements for blueprint: {blueprint.get('name', 'Unnamed')}")
        
        # Start with base requirements
        requirements = {
            'memory_mb': 128,  # Base memory requirement
            'cpu_cores': 0.25,  # Base CPU requirement
            'disk_mb': 50      # Base disk requirement
        }
        
        # Analyze tools in the blueprint
        if 'tools' in blueprint:
            for tool_name, tool_info in blueprint['tools'].items():
                tool_resources = self._get_tool_resource_requirements(tool_name, tool_info)
                
                # Add tool requirements
                requirements['memory_mb'] += tool_resources.get('memory_mb', 0)
                requirements['cpu_cores'] += tool_resources.get('cpu_cores', 0)
                requirements['disk_mb'] += tool_resources.get('disk_mb', 0)
        
        self.logger.info(f"Estimated resource requirements: {requirements}")
        return requirements

    def detect_host_resources(self) -> Dict[str, Any]:
        """
        Detect the resources available on the host machine.
        This is a wrapper around analyze_host_resources.
        
        Returns:
            Dict[str, Any]: Available host resources
        """
        return self.analyze_host_resources()