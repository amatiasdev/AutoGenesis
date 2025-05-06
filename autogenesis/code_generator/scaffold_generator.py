"""
Scaffold Generator for AutoGenesis.

This module handles the generation of the project scaffold structure
for the target agent based on the blueprint specifications.
"""

import logging
import os
import shutil
import tempfile
from typing import Dict, Any, List, Optional

class ScaffoldGenerator:
    """
    Generates project scaffold structure for the target agent.
    
    Creates the necessary directory structure and placeholder files
    based on the blueprint's architecture type and file structure.
    """
    
    def __init__(self):
        """Initialize the scaffold generator."""
        self.logger = logging.getLogger(__name__)
    
    def generate(self, blueprint: Dict[str, Any], output_dir: str = None) -> str:
        """
        Generate the project scaffold structure.
        
        Args:
            blueprint: Agent blueprint with architecture and file structure
            output_dir: Optional directory where the scaffold will be generated
            
        Returns:
            String path to the generated scaffold
        """
        self.logger.info(f"Generating project scaffold for architecture: {blueprint.get('architecture_type')}")
        
        # Use output_dir if provided, otherwise create a temporary directory
        if output_dir:
            scaffold_path = output_dir
            # Ensure the directory exists
            os.makedirs(scaffold_path, exist_ok=True)
        else:
            # Create temporary directory for scaffold
            scaffold_path = tempfile.mkdtemp(prefix=f"{blueprint.get('task_type', 'agent')}_")
        
        self.logger.info(f"Using scaffold directory at: {scaffold_path}")
        
        # Get file structure definition
        file_structure = blueprint.get('file_structure', {})
        
        # Create directory structure
        self._create_directory_structure(scaffold_path, file_structure)
        
        return scaffold_path
    
    def _create_directory_structure(self, 
                                  base_path: str, 
                                  structure: Dict[str, Any]) -> None:
        """
        Create directory structure recursively.
        
        Args:
            base_path: Base path for the structure
            structure: Dictionary describing the directory structure
        """
        # Process root level
        if 'root' in structure:
            root_structure = structure['root']
            
            # Create files
            for file_name in root_structure.get('files', []):
                file_path = os.path.join(base_path, file_name)
                self._create_placeholder_file(file_path)
            
            # Create directories
            for directory in root_structure.get('directories', []):
                dir_name = directory.get('name', '')
                if dir_name:
                    dir_path = os.path.join(base_path, dir_name)
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # Create files in this directory
                    for file_name in directory.get('files', []):
                        file_path = os.path.join(dir_path, file_name)
                        self._create_placeholder_file(file_path)
                    
                    # Process nested directories
                    for nested_dir in directory.get('directories', []):
                        nested_name = nested_dir.get('name', '')
                        if nested_name:
                            nested_path = os.path.join(dir_path, nested_name)
                            os.makedirs(nested_path, exist_ok=True)
                            
                            # Create files in nested directory
                            for nested_file in nested_dir.get('files', []):
                                nested_file_path = os.path.join(nested_path, nested_file)
                                self._create_placeholder_file(nested_file_path)
        
        self.logger.info("Directory structure created successfully")
    
    def _create_placeholder_file(self, file_path: str) -> None:
        """
        Create a placeholder file.
        
        Args:
            file_path: Path to the file to create
        """
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        # Create an empty file (will be filled by the code generator)
        with open(file_path, 'w') as f:
            f.write('')
        
        self.logger.debug(f"Created placeholder file at: {file_path}")