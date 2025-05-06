"""
Script Packager for AutoGenesis.

This module handles the packaging of generated agent code into executable Python scripts.
"""

import logging
import os
import subprocess
import tempfile
import shutil
import zipfile
from typing import Dict, Any, Optional, List
from pathlib import Path

class ScriptPackager:
    """
    Packages agent code into executable Python scripts.
    
    Creates standalone Python scripts or script packages with dependencies
    for easy distribution and execution.
    """
    
    def __init__(self):
        """Initialize the Script packager."""
        self.logger = logging.getLogger(__name__)
    
    def package(self, 
               scaffold_path: str, 
               blueprint: Dict[str, Any],
               output_path: Optional[str] = None) -> str:
        """
        Package agent code into an executable script package.
        
        Args:
            scaffold_path: Path to the scaffolded project
            blueprint: Agent blueprint with tools and requirements
            output_path: Optional path for the output package
            
        Returns:
            String path to the created script package
            
        Raises:
            RuntimeError: If script packaging fails
        """
        self.logger.info("Packaging agent as executable script")
        
        # Generate output path if not provided
        if not output_path:
            agent_name = blueprint.get('name', 'agent').lower()
            output_path = f"{agent_name}_script.zip"
        
        # Create temporary build directory
        with tempfile.TemporaryDirectory() as build_dir:
            try:
                # Copy scaffold to build directory
                self._copy_scaffold(scaffold_path, build_dir)
                
                # Make main script executable
                self._make_executable(build_dir, blueprint)
                
                # Create run script for simplified execution
                self._create_run_script(build_dir, blueprint)
                
                # Create ZIP archive
                self._create_zip_archive(build_dir, output_path)
                
                self.logger.info(f"Script package created successfully: {output_path}")
                return output_path
                
            except Exception as e:
                self.logger.error(f"Script packaging failed: {str(e)}")
                raise RuntimeError(f"Script packaging failed: {str(e)}")
    
    def _copy_scaffold(self, scaffold_path: str, build_dir: str) -> None:
        """
        Copy scaffold to build directory.
        
        Args:
            scaffold_path: Path to scaffold
            build_dir: Path to build directory
        """
        self.logger.info(f"Copying scaffold from {scaffold_path} to {build_dir}")
        
        # Copy all files except .git directory and __pycache__ directories
        for root, dirs, files in os.walk(scaffold_path):
            # Skip .git and __pycache__ directories
            dirs[:] = [d for d in dirs if d != ".git" and d != "__pycache__"]
            
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, scaffold_path)
                dst_path = os.path.join(build_dir, rel_path)
                
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
    
    def _make_executable(self, build_dir: str, blueprint: Dict[str, Any]) -> None:
        """
        Make main script executable.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
        """
        self.logger.info("Making main script executable")
        
        # Find main script based on architecture
        architecture_type = blueprint.get('architecture_type', 'modular')
        task_type = blueprint.get('task_type', 'agent')
        
        main_script_path = self._find_main_script(build_dir, architecture_type, task_type)
        
        if main_script_path and os.path.exists(main_script_path):
            # Add shebang line if not already present
            with open(main_script_path, 'r') as f:
                content = f.read()
            
            if not content.startswith("#!/usr/bin/env python"):
                with open(main_script_path, 'w') as f:
                    f.write("#!/usr/bin/env python3\n" + content)
            
            # Make executable
            os.chmod(main_script_path, 0o755)
            self.logger.info(f"Made {main_script_path} executable")
    
    def _find_main_script(self, build_dir: str, architecture_type: str, task_type: str) -> Optional[str]:
        """
        Find the main script based on architecture type.
        
        Args:
            build_dir: Path to build directory
            architecture_type: Type of architecture
            task_type: Type of task
            
        Returns:
            Path to main script or None if not found
        """
        if architecture_type == "simple_script":
            return os.path.join(build_dir, "main.py")
        elif architecture_type == "service":
            return os.path.join(build_dir, f"{task_type}_service", "main.py")
        else:
            # For modular and other architectures
            return os.path.join(build_dir, f"{task_type}_agent", "main.py")
    
    def _create_run_script(self, build_dir: str, blueprint: Dict[str, Any]) -> str:
        """
        Create a run script for simplified execution.
        
        Args:
            build_dir: Path to build directory
            blueprint: Agent blueprint
            
        Returns:
            Path to created run script
        """
        self.logger.info("Creating run script")
        
        # Create run script content based on architecture
        architecture_type = blueprint.get('architecture_type', 'modular')
        task_type = blueprint.get('task_type', 'agent')
        
        run_script_content = "#!/bin/bash\n\n"
        run_script_content += "# Activate virtual environment if it exists\n"
        run_script_content += "if [ -d 'venv' ]; then\n"
        run_script_content += "    source venv/bin/activate\n"
        run_script_content += "fi\n\n"
        
        # Add command to install dependencies
        run_script_content += "# Install dependencies if needed\n"
        run_script_content += "if [ ! -f '.dependencies_installed' ]; then\n"
        run_script_content += "    pip install -r requirements.txt\n"
        run_script_content += "    touch .dependencies_installed\n"
        run_script_content += "fi\n\n"
        
        # Add command to run the agent
        run_script_content += "# Run the agent\n"
        if architecture_type == "simple_script":
            run_script_content += "python main.py \"$@\"\n"
        elif architecture_type == "service":
            run_script_content += f"python {task_type}_service/main.py \"$@\"\n"
        else:
            run_script_content += f"python -m {task_type}_agent.main \"$@\"\n"
        
        # Write run script
        run_script_path = os.path.join(build_dir, "run.sh")
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        
        # Make executable
        os.chmod(run_script_path, 0o755)
        
        return run_script_path
    
    def _create_zip_archive(self, build_dir: str, output_path: str) -> None:
        """
        Create ZIP archive of the script package.
        
        Args:
            build_dir: Path to build directory
            output_path: Path for the output ZIP file
            
        Raises:
            RuntimeError: If ZIP creation fails
        """
        self.logger.info(f"Creating script package ZIP archive: {output_path}")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Create ZIP archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add README file with instructions
                readme_path = self._create_readme(build_dir)
                arcname = os.path.relpath(readme_path, build_dir)
                zipf.write(readme_path, arcname)
                
                # Add all files in the build directory to the ZIP
                for root, _, files in os.walk(build_dir):
                    for file in files:
                        if file != "README.md":  # Skip README as we've already added it
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, build_dir)
                            zipf.write(file_path, arcname)
        
        except Exception as e:
            self.logger.error(f"ZIP archive creation failed: {str(e)}")
            raise RuntimeError(f"ZIP archive creation failed: {str(e)}")
    
    def _create_readme(self, build_dir: str) -> str:
        """
        Create README file with usage instructions.
        
        Args:
            build_dir: Path to build directory
            
        Returns:
            Path to created README file
        """
        readme_content = """# Agent Script Package

This package contains an agent generated by AutoGenesis.

## Installation

1. Unzip the package to a directory of your choice.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
   
## Usage

You can run the agent using the provided run script:

```
./run.sh
```

Or directly with Python:

```
python -m main
```

## Configuration

The agent can be configured using the following:

* Environment variables
* Configuration files (config.yaml or config.json)

See the agent documentation for specific configuration options.

## License

This agent was generated by AutoGenesis.
"""
        
        # Write README file
        readme_path = os.path.join(build_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return readme_path