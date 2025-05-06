#!/usr/bin/env python3
"""
AutoGenesis - Agent Factory

A meta-agent that automates the creation, testing, and deployment of specialized software agents.
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from input_processing.nl_processor import NLProcessor
from input_processing.api_processor import APIProcessor
from input_processing.task_decomposer import TaskDecomposer
from blueprint.designer import BlueprintDesigner
from code_generator.scaffold_generator import ScaffoldGenerator
from code_generator.module_generator import ModuleGenerator
from code_generator.test_generator import TestGenerator
from sandbox.docker_manager import DockerManager
from sandbox.test_runner import TestRunner
from mpc.security_critic import SecurityCritic
from mpc.scalability_critic import ScalabilityCritic
from mpc.maintainability_critic import MaintainabilityCritic
from mpc.cost_estimator import CostEstimator
from packaging.docker_packager import DockerPackager
from packaging.lambda_packager import LambdaPackager
from packaging.script_packager import ScriptPackager
from utils.logging import setup_logging
from utils.aws_helper import DynamoDBLogger


class AutoGenesis:
    """
    Main class for the AutoGenesis (Agent Factory) meta-agent.
    
    This class orchestrates the entire workflow from requirement processing
    to agent deployment.
    """
    
    def __init__(self):
        """Initialize AutoGenesis with necessary components."""
        self.logger = logging.getLogger(__name__)
        self.agent_id = str(uuid.uuid4())
        
        # Initialize components
        self.nl_processor = NLProcessor()
        self.api_processor = APIProcessor()
        self.task_decomposer = TaskDecomposer()
        self.blueprint_designer = BlueprintDesigner()
        self.scaffold_generator = ScaffoldGenerator()
        self.module_generator = ModuleGenerator()
        self.test_generator = TestGenerator()
        self.docker_manager = DockerManager()
        self.test_runner = TestRunner()
        
        # MPC components
        self.security_critic = SecurityCritic()
        self.scalability_critic = ScalabilityCritic()
        self.maintainability_critic = MaintainabilityCritic()
        self.cost_estimator = CostEstimator()
        
        # Packaging components
        self.docker_packager = DockerPackager()
        self.lambda_packager = LambdaPackager()
        self.script_packager = ScriptPackager()
        
        # DynamoDB logger for MPC results
        self.dynamodb_logger = DynamoDBLogger()
        
        self.logger.info(f"AutoGenesis initialized with agent_id: {self.agent_id}")
    
    def process_nl_request(self, nl_text: str) -> Dict[str, Any]:
        """
        Process a natural language request and convert it to structured requirements.
        
        Args:
            nl_text: The natural language request text
            
        Returns:
            Dict containing structured requirements
        """
        self.logger.info(f"Processing NL request: {nl_text[:100]}...")
        
        # Process NL input
        processed_input = self.nl_processor.process(nl_text)
        
        # If clarification needed, this would typically involve interaction
        # For now, we'll assume the input is clear enough
        if processed_input.get('needs_clarification'):
            self.logger.warning("NL input needs clarification. In a real implementation, we would ask the user.")
            # In a real system, we would interact with the user here
        
        # Return structured requirements
        return processed_input.get('requirements', {})
    
    def process_api_request(self, api_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a structured API request.
        
        Args:
            api_payload: The API request payload
            
        Returns:
            Dict containing validated requirements
        """
        self.logger.info("Processing API request...")
        
        # Validate and process API input
        processed_input = self.api_processor.process(api_payload)
        
        return processed_input
    
    def generate_agent(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a target agent based on provided requirements.
        
        Args:
            requirements: Structured requirements for the target agent
            
        Returns:
            Dict containing paths to generated deployment packages
        """
        self.logger.info("Starting agent generation process...")
        
        # Step 1: Decompose task into atomic actions
        actions = self.task_decomposer.decompose(requirements)
        self.logger.info(f"Task decomposed into {len(actions)} atomic actions")
        
        # Step 2: Design agent blueprint
        blueprint = self.blueprint_designer.design(requirements, actions)
        self.logger.info(f"Blueprint designed with architecture type: {blueprint.get('architecture_type')}")
        
        # Step 3: Generate code scaffold
        scaffold_path = self.scaffold_generator.generate(blueprint)
        self.logger.info(f"Code scaffold generated at: {scaffold_path}")
        
        # Step 4: Generate module code
        code_paths = self.module_generator.generate(blueprint, scaffold_path, actions)
        self.logger.info(f"Generated {len(code_paths)} code modules")
        
        # Step 5: Generate tests
        test_paths = self.test_generator.generate(blueprint, scaffold_path, code_paths)
        self.logger.info(f"Generated {len(test_paths)} test files")
        
        # Step 6: Provision sandbox
        sandbox_id = self.docker_manager.create_sandbox(blueprint.get('resource_requirements', {}))
        self.logger.info(f"Created sandbox with ID: {sandbox_id}")
        
        # Step 7: Deploy to sandbox
        self.docker_manager.deploy_to_sandbox(sandbox_id, scaffold_path)
        
        # Step 8: Run tests in sandbox
        test_results = self.test_runner.run_tests(sandbox_id, test_paths)
        self.logger.info(f"Test results: {test_results.get('summary')}")
        
        # Step 9: Apply MPC
        mpc_results = self._apply_mpc(scaffold_path, blueprint, test_results)
        
        # Log MPC results to DynamoDB
        self.dynamodb_logger.log_mpc_results(
            self.agent_id, 
            scaffold_path, 
            blueprint,
            mpc_results
        )
        
        # If tests pass and MPC is satisfied (or refinements applied successfully)
        if test_results.get('passed', False) and mpc_results.get('satisfied', False):
            # Step 10: Package the agent
            packages = self._package_agent(scaffold_path, blueprint)
            
            self.logger.info(f"Agent generation successful. Packages: {packages}")
            return packages
        else:
            # In a more advanced implementation, attempt refinement
            self.logger.error("Agent generation failed due to test failures or MPC criteria not satisfied")
            return {"status": "failed", "reason": "Tests or MPC criteria not satisfied"}
    
    def _apply_mpc(self, 
                  scaffold_path: str, 
                  blueprint: Dict[str, Any],
                  test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Multi-Perspective Criticism to evaluate the agent.
        
        Args:
            scaffold_path: Path to the generated code
            blueprint: Agent blueprint
            test_results: Results from testing
            
        Returns:
            Dict containing MPC results
        """
        self.logger.info("Applying Multi-Perspective Criticism...")
        
        # Security evaluation
        security_results = self.security_critic.evaluate(scaffold_path)
        
        # Scalability evaluation
        scalability_results = self.scalability_critic.evaluate(
            scaffold_path, 
            blueprint.get('resource_requirements', {})
        )
        
        # Maintainability evaluation
        maintainability_results = self.maintainability_critic.evaluate(scaffold_path)
        
        # Cost estimation
        cost_results = self.cost_estimator.estimate(
            scaffold_path,
            blueprint.get('resource_requirements', {}),
            blueprint.get('expected_usage', {})
        )
        
        # Aggregate results
        mpc_results = {
            "security": security_results,
            "scalability": scalability_results,
            "maintainability": maintainability_results,
            "cost": cost_results,
            "timestamp": datetime.utcnow().isoformat(),
            "satisfied": all([
                security_results.get('passed', False),
                scalability_results.get('passed', False),
                maintainability_results.get('passed', False),
                cost_results.get('passed', False)
            ])
        }
        
        self.logger.info(f"MPC complete. Satisfied: {mpc_results['satisfied']}")
        return mpc_results
    
    def _package_agent(self, 
                      scaffold_path: str, 
                      blueprint: Dict[str, Any]) -> Dict[str, str]:
        """
        Package the agent in requested formats.
        
        Args:
            scaffold_path: Path to the generated code
            blueprint: Agent blueprint
            
        Returns:
            Dict with paths to packaged outputs
        """
        self.logger.info("Packaging agent...")
        
        packages = {}
        deployment_formats = blueprint.get('deployment_formats', ['script'])
        
        if 'docker' in deployment_formats:
            docker_image = self.docker_packager.package(scaffold_path, blueprint)
            packages['docker'] = docker_image
            
        if 'lambda' in deployment_formats:
            lambda_package = self.lambda_packager.package(scaffold_path, blueprint)
            packages['lambda'] = lambda_package
            
        if 'script' in deployment_formats:
            script_package = self.script_packager.package(scaffold_path, blueprint)
            packages['script'] = script_package
            
        self.logger.info(f"Agent packaged in formats: {list(packages.keys())}")
        return packages


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AutoGenesis - Agent Factory')
    
    # Add input mode argument
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['nl', 'api'], 
        default='nl',
        help='Input mode: natural language or API'
    )
    
    # Add input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Natural language input text')
    input_group.add_argument('--file', type=str, help='Path to input file (JSON for API mode, text for NL mode)')
    
    # Add output argument
    parser.add_argument('--output', type=str, help='Output directory for packages', default='./output')
    
    return parser.parse_args()


def main():
    """Main entry point for AutoGenesis."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Initialize AutoGenesis
        agent_factory = AutoGenesis()
        
        # Process input based on mode
        if args.text:
            input_content = args.text
        else:  # args.file
            with open(args.file, 'r') as f:
                input_content = f.read()
        
        # Process requirements based on mode
        if args.mode == 'nl':
            requirements = agent_factory.process_nl_request(input_content)
        else:  # args.mode == 'api'
            requirements = agent_factory.process_api_request(json.loads(input_content))
        
        # Generate agent
        result = agent_factory.generate_agent(requirements)
        
        # Output result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Error in AutoGenesis: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())