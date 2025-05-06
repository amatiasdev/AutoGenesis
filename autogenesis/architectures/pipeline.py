"""
Pipeline implementation for {{ agent_name }}.

This module defines the pipeline architecture for executing stages
in sequence or parallel as needed.
"""

import logging
import time
import traceback
from typing import Dict, Any, List, Optional, Union
import concurrent.futures

class Pipeline:
    """
    Pipeline implementation for executing stages in sequence or parallel.
    
    This class manages the execution flow of multiple stages, handling
    data passing between stages and error management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract pipeline-specific configuration
        self.pipeline_config = config.get('pipeline', {})
        self.max_retries = self.pipeline_config.get('max_retries', 3)
        self.retry_delay = self.pipeline_config.get('retry_delay_seconds', 5)
        self.parallel_stages = self.pipeline_config.get('parallel_stages', False)
        
        # Initialize stages dictionary
        self.stages = {}
        
        # Stage dependencies
        self.dependencies = {}
        
    def register_stage(self, stage_id: str, stage: Any, depends_on: Optional[List[str]] = None) -> None:
        """
        Register a stage in the pipeline.
        
        Args:
            stage_id: Unique identifier for the stage
            stage: Stage instance
            depends_on: List of stage IDs this stage depends on
        """
        self.logger.debug(f"Registering stage: {stage_id}")
        self.stages[stage_id] = stage
        
        # Register dependencies
        if depends_on:
            self.dependencies[stage_id] = depends_on
        else:
            self.dependencies[stage_id] = []
            
    def execute(self, 
               initial_data: Optional[Dict[str, Any]] = None, 
               start_stage: Optional[str] = None,
               end_stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            initial_data: Initial data to pass to the first stage
            start_stage: Optional stage ID to start execution from
            end_stage: Optional stage ID to end execution at
            
        Returns:
            Dict containing pipeline execution results
            
        Raises:
            ValueError: If stage dependencies cannot be resolved
        """
        self.logger.info("Executing pipeline")
        
        # Initialize pipeline execution
        execution_order = self._resolve_execution_order(start_stage, end_stage)
        stage_results = {}
        pipeline_data = initial_data or {}
        
        # Record start time
        start_time = time.time()
        
        # Execute stages in determined order
        try:
            if self.parallel_stages:
                stage_results = self._execute_parallel(execution_order, pipeline_data)
            else:
                stage_results = self._execute_sequential(execution_order, pipeline_data)
                
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return pipeline results
            return {
                "status": "success",
                "execution_time": execution_time,
                "results": stage_results,
                "final_data": pipeline_data
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "execution_time": execution_time,
                "results": stage_results
            }
            
    def _resolve_execution_order(self, 
                              start_stage: Optional[str] = None,
                              end_stage: Optional[str] = None) -> List[str]:
        """
        Resolve the execution order of stages based on dependencies.
        
        Args:
            start_stage: Optional stage ID to start execution from
            end_stage: Optional stage ID to end execution at
            
        Returns:
            List of stage IDs in execution order
            
        Raises:
            ValueError: If dependencies cannot be resolved
        """
        # If no stages registered, return empty list
        if not self.stages:
            return []
        
        # Get all stages
        all_stages = list(self.stages.keys())
        
        # If start_stage is specified, validate it exists
        if start_stage and start_stage not in all_stages:
            raise ValueError(f"Start stage '{start_stage}' not found in registered stages")
            
        # If end_stage is specified, validate it exists
        if end_stage and end_stage not in all_stages:
            raise ValueError(f"End stage '{end_stage}' not found in registered stages")
            
        # Topological sort to resolve dependencies
        execution_order = self._topological_sort()
        
        # Apply start_stage filter if specified
        if start_stage:
            start_index = execution_order.index(start_stage)
            execution_order = execution_order[start_index:]
            
        # Apply end_stage filter if specified
        if end_stage:
            end_index = execution_order.index(end_stage)
            execution_order = execution_order[:end_index + 1]
            
        return execution_order
        
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort on stages based on dependencies.
        
        Returns:
            List of stage IDs in dependency order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # Initialize result list
        result = []
        
        # All stages to process
        stages = list(self.stages.keys())
        
        # Track visited and in-progress states to detect cycles
        visited = set()
        temp = set()
        
        def visit(stage):
            """Helper function for DFS traversal."""
            if stage in temp:
                raise ValueError(f"Circular dependency detected involving stage: {stage}")
            if stage in visited:
                return
                
            temp.add(stage)
            
            # Visit dependencies first
            for dep in self.dependencies.get(stage, []):
                if dep not in self.stages:
                    raise ValueError(f"Stage '{stage}' depends on unknown stage '{dep}'")
                visit(dep)
                
            temp.remove(stage)
            visited.add(stage)
            result.append(stage)
            
        # Visit all stages
        for stage in stages:
            if stage not in visited:
                visit(stage)
                
        # Reverse result to get correct order
        return list(reversed(result))
        
    def _execute_sequential(self, 
                         execution_order: List[str],
                         pipeline_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Execute stages sequentially.
        
        Args:
            execution_order: List of stage IDs in execution order
            pipeline_data: Data to pass between stages
            
        Returns:
            Dict mapping stage IDs to their results
        """
        self.logger.info(f"Executing {len(execution_order)} stages sequentially")
        stage_results = {}
        
        for stage_id in execution_order:
            self.logger.info(f"Executing stage: {stage_id}")
            
            # Get stage instance
            stage = self.stages[stage_id]
            
            # Execute stage with retry logic
            result = self._execute_stage_with_retry(stage, pipeline_data)
            
            # Store stage result
            stage_results[stage_id] = result
            
            # If stage failed, halt pipeline
            if result.get("status") == "error":
                self.logger.error(f"Stage {stage_id} failed: {result.get('error')}")
                break
                
            # Update pipeline data with stage output
            if "data" in result:
                pipeline_data = result["data"]
                
        return stage_results
        
    def _execute_parallel(self, 
                       execution_order: List[str],
                       pipeline_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Execute stages in parallel where possible.
        
        Args:
            execution_order: List of stage IDs in dependency order
            pipeline_data: Data to pass between stages
            
        Returns:
            Dict mapping stage IDs to their results
        """
        self.logger.info(f"Executing {len(execution_order)} stages with parallelism")
        stage_results = {}
        
        # Group stages by dependency level
        levels = self._group_stages_by_level(execution_order)
        
        # Process each level in sequence
        for level, stage_ids in enumerate(levels):
            self.logger.info(f"Processing level {level} with {len(stage_ids)} stages")
            
            # For each level, execute stages in parallel
            level_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(stage_ids)) as executor:
                # Submit all stage executions
                future_to_stage = {
                    executor.submit(self._execute_stage_with_retry, self.stages[stage_id], pipeline_data): stage_id
                    for stage_id in stage_ids
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_stage):
                    stage_id = future_to_stage[future]
                    try:
                        result = future.result()
                        level_results[stage_id] = result
                    except Exception as e:
                        self.logger.error(f"Stage {stage_id} raised exception: {str(e)}")
                        level_results[stage_id] = {
                            "status": "error",
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
            
            # Update overall results
            stage_results.update(level_results)
            
            # Merge results from this level
            for stage_id, result in level_results.items():
                if result.get("status") == "error":
                    self.logger.error(f"Stage {stage_id} failed: {result.get('error')}")
                    # Don't halt pipeline for parallel execution, continue with other stages
                elif "data" in result:
                    # Merge data from this stage into pipeline data
                    pipeline_data.update(result["data"])
                    
        return stage_results
        
    def _group_stages_by_level(self, execution_order: List[str]) -> List[List[str]]:
        """
        Group stages by dependency level for parallel execution.
        
        Args:
            execution_order: List of stage IDs in dependency order
            
        Returns:
            List of lists, where each inner list contains stages that can be executed in parallel
        """
        levels = []
        visited = set()
        
        while len(visited) < len(execution_order):
            # Find stages whose dependencies are already visited
            current_level = []
            
            for stage_id in execution_order:
                if stage_id in visited:
                    continue
                    
                # Check if all dependencies are visited
                deps = self.dependencies.get(stage_id, [])
                if all(dep in visited for dep in deps):
                    current_level.append(stage_id)
                    
            # Add current level to levels
            levels.append(current_level)
            
            # Mark current level as visited
            visited.update(current_level)
            
        return levels
        
    def _execute_stage_with_retry(self, 
                               stage: Any,
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a stage with retry logic.
        
        Args:
            stage: Stage instance
            data: Input data for the stage
            
        Returns:
            Dict containing stage execution result
        """
        retries = 0
        
        while retries <= self.max_retries:
            try:
                # Execute stage
                result = stage.process(data)
                
                # If successful, return result
                if result.get("status") != "error":
                    return result
                    
                # If first attempt failed but retry is enabled, attempt retry
                retries += 1
                if retries <= self.max_retries:
                    self.logger.warning(f"Stage execution failed. Retrying ({retries}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Stage execution failed after {retries} retries")
                    return result
                    
            except Exception as e:
                # If exception occurred, log and retry
                retries += 1
                self.logger.error(f"Stage execution raised exception: {str(e)}")
                
                if retries <= self.max_retries:
                    self.logger.warning(f"Retrying ({retries}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Stage execution failed after {retries} retries")
                    return {
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
        # This should not be reached, but just in case
        return {
            "status": "error",
            "error": "Maximum retries reached",
            "traceback": ""
        }