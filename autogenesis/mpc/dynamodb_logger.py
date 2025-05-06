"""
DynamoDB Logger for AutoGenesis MPC.

This module handles logging Multi-Perspective Criticism (MPC) results
to DynamoDB for future learning and improvement.
"""

import logging
import os
import json
import time
import hashlib
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

class DynamoDBLogger:
    """
    Logs MPC results to DynamoDB for future learning.
    
    Stores criticism results, agent metadata, and recommendations
    to enable continuous improvement of the AutoGenesis system.
    """
    
    def __init__(self, table_name: str = None):
        """
        Initialize the DynamoDB logger.
        
        Args:
            table_name: Optional DynamoDB table name (defaults to environment variable)
        """
        self.logger = logging.getLogger(__name__)
        self.table_name = table_name or os.environ.get('AUTOGENESIS_MPC_TABLE', 'AutoGenesisMPCLogs')
        
        # Initialize boto3 client (lazy loading)
        self._dynamodb = None
        self._table = None
    
    def log_mpc_results(self,
                       agent_id: str,
                       code_path: str,
                       blueprint: Dict[str, Any],
                       mpc_results: Dict[str, Any]) -> bool:
        """
        Log MPC results to DynamoDB.
        
        Args:
            agent_id: ID of the agent being evaluated
            code_path: Path to the agent code
            blueprint: Agent blueprint
            mpc_results: MPC evaluation results
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Logging MPC results for agent {agent_id}")
        
        try:
            # Generate a hash of the code for tracking changes
            code_hash = self._hash_code_directory(code_path)
            
            # Prepare the log item
            log_item = {
                'critique_id': str(uuid.uuid4()),
                'agent_id': agent_id,
                'timestamp': datetime.utcnow().isoformat(),
                'code_hash': code_hash,
                'blueprint_summary': self._create_blueprint_summary(blueprint),
                'evaluation_summary': self._create_evaluation_summary(mpc_results),
                'findings': self._format_findings(mpc_results),
                'recommendations': self._format_recommendations(mpc_results),
                'scores': self._extract_scores(mpc_results),
                'environment_constraints': blueprint.get('resource_requirements', {})
            }
            
            # Write to DynamoDB
            success = self._write_to_dynamodb(log_item)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error logging MPC results: {e}")
            return False
    
    def _hash_code_directory(self, code_path: str) -> str:
        """
        Create a hash representing the code directory.
        
        Args:
            code_path: Path to the code directory
            
        Returns:
            String hash of the code contents
        """
        combined_hash = hashlib.sha256()
        
        try:
            for root, _, files in os.walk(code_path):
                for file in sorted(files):
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                            relative_path = os.path.relpath(file_path, code_path)
                            combined_hash.update(f"{relative_path}:{file_hash}".encode())
        
        except Exception as e:
            self.logger.error(f"Error hashing code directory: {e}")
            return "error_hashing_code"
        
        return combined_hash.hexdigest()
    
    def _create_blueprint_summary(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the blueprint for storage.
        
        Args:
            blueprint: Agent blueprint
            
        Returns:
            Dict containing blueprint summary
        """
        return {
            'agent_name': blueprint.get('name', ''),
            'task_type': blueprint.get('task_type', ''),
            'architecture_type': blueprint.get('architecture_type', ''),
            'tools': list(blueprint.get('tools', {}).keys()),
            'deployment_formats': blueprint.get('deployment_formats', [])
        }
    
    def _create_evaluation_summary(self, mpc_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the evaluation results.
        
        Args:
            mpc_results: MPC evaluation results
            
        Returns:
            Dict containing evaluation summary
        """
        # Extract overall results from each dimension
        security_passed = mpc_results.get('security', {}).get('passed', False)
        scalability_passed = mpc_results.get('scalability', {}).get('passed', False)
        maintainability_passed = mpc_results.get('maintainability', {}).get('passed', False)
        cost_passed = mpc_results.get('cost', {}).get('passed', False)
        
        # Count findings by severity
        findings_count = {
            'security': self._count_findings_by_severity(mpc_results.get('security', {}).get('findings', [])),
            'scalability': self._count_findings_by_severity(mpc_results.get('scalability', {}).get('findings', [])),
            'maintainability': self._count_findings_by_severity(mpc_results.get('maintainability', {}).get('findings', []))
        }
        
        return {
            'overall_passed': mpc_results.get('satisfied', False),
            'security_passed': security_passed,
            'scalability_passed': scalability_passed,
            'maintainability_passed': maintainability_passed,
            'cost_passed': cost_passed,
            'findings_count': findings_count,
            'total_findings': sum([
                sum(findings_count['security'].values()),
                sum(findings_count['scalability'].values()),
                sum(findings_count['maintainability'].values())
            ])
        }
    
    def _count_findings_by_severity(self, findings: list) -> Dict[str, int]:
        """
        Count findings by severity level.
        
        Args:
            findings: List of findings
            
        Returns:
            Dict mapping severity to count
        """
        counts = {'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        
        for finding in findings:
            severity = finding.get('severity', 'info').lower()
            if severity in counts:
                counts[severity] += 1
            else:
                counts['info'] += 1
        
        return counts
    
    def _format_findings(self, mpc_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format findings for storage in DynamoDB.
        
        Args:
            mpc_results: MPC evaluation results
            
        Returns:
            Dict containing formatted findings
        """
        # Extract findings from each dimension, keeping only essential info
        formatted_findings = {}
        
        for dimension in ['security', 'scalability', 'maintainability']:
            dimension_findings = []
            findings = mpc_results.get(dimension, {}).get('findings', [])
            
            for finding in findings:
                # Extract only necessary fields to reduce storage size
                formatted_finding = {
                    'type': finding.get('type', ''),
                    'severity': finding.get('severity', 'info'),
                    'description': finding.get('description', ''),
                    'file_path': finding.get('file_path', '')
                }
                
                if 'line_number' in finding:
                    formatted_finding['line_number'] = finding['line_number']
                
                dimension_findings.append(formatted_finding)
            
            formatted_findings[dimension] = dimension_findings
        
        return formatted_findings
    
    def _format_recommendations(self, mpc_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format recommendations for storage in DynamoDB.
        
        Args:
            mpc_results: MPC evaluation results
            
        Returns:
            Dict containing formatted recommendations
        """
        formatted_recommendations = {}
        
        for dimension in ['security', 'scalability', 'maintainability', 'cost']:
            recommendations = mpc_results.get(dimension, {}).get('recommendations', [])
            formatted_recommendations[dimension] = recommendations
        
        return formatted_recommendations
    
    def _extract_scores(self, mpc_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract scores from MPC results.
        
        Args:
            mpc_results: MPC evaluation results
            
        Returns:
            Dict containing dimension scores
        """
        scores = {}
        
        for dimension in ['security', 'scalability', 'maintainability']:
            scores[dimension] = mpc_results.get(dimension, {}).get('score', 0)
        
        # For cost, use estimated monthly cost
        cost_estimate = mpc_results.get('cost', {}).get('total_monthly_cost', 0)
        scores['cost'] = cost_estimate
        
        return scores
    
    def _write_to_dynamodb(self, item: Dict[str, Any]) -> bool:
        """
        Write log item to DynamoDB.
        
        Args:
            item: Log item to write
            
        Returns:
            Boolean indicating success
        """
        try:
            # Lazy load boto3 and DynamoDB resources
            if self._dynamodb is None:
                import boto3
                self._dynamodb = boto3.resource('dynamodb')
                self._table = self._dynamodb.Table(self.table_name)
            
            # Check if table exists
            self._ensure_table_exists()
            
            # Write item to DynamoDB
            response = self._table.put_item(Item=item)
            
            # Check response
            success = response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200
            if success:
                self.logger.info(f"Successfully logged MPC results to DynamoDB, critique_id: {item['critique_id']}")
            else:
                self.logger.error(f"Failed to log MPC results to DynamoDB: {response}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error writing to DynamoDB: {e}")
            return False
    
    def _ensure_table_exists(self) -> bool:
        """
        Ensure DynamoDB table exists, create if not.
        
        Returns:
            Boolean indicating if table exists or was created
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            try:
                # Check if table exists
                self._table.table_status
                return True
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Table doesn't exist, create it
                    self.logger.info(f"Table {self.table_name} does not exist, creating...")
                    
                    # Create DynamoDB client
                    dynamodb_client = boto3.client('dynamodb')
                    
                    # Create table
                    response = dynamodb_client.create_table(
                        TableName=self.table_name,
                        KeySchema=[
                            {
                                'AttributeName': 'critique_id',
                                'KeyType': 'HASH'  # Partition key
                            }
                        ],
                        AttributeDefinitions=[
                            {
                                'AttributeName': 'critique_id',
                                'AttributeType': 'S'
                            },
                            {
                                'AttributeName': 'agent_id',
                                'AttributeType': 'S'
                            },
                            {
                                'AttributeName': 'timestamp',
                                'AttributeType': 'S'
                            }
                        ],
                        GlobalSecondaryIndexes=[
                            {
                                'IndexName': 'AgentIndex',
                                'KeySchema': [
                                    {
                                        'AttributeName': 'agent_id',
                                        'KeyType': 'HASH'
                                    },
                                    {
                                        'AttributeName': 'timestamp',
                                        'KeyType': 'RANGE'
                                    }
                                ],
                                'Projection': {
                                    'ProjectionType': 'ALL'
                                },
                                'ProvisionedThroughput': {
                                    'ReadCapacityUnits': 5,
                                    'WriteCapacityUnits': 5
                                }
                            }
                        ],
                        ProvisionedThroughput={
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    )
                    
                    # Wait for table creation
                    waiter = dynamodb_client.get_waiter('table_exists')
                    waiter.wait(TableName=self.table_name)
                    
                    self.logger.info(f"Table {self.table_name} created successfully")
                    return True
                else:
                    # Other error
                    self.logger.error(f"Error checking DynamoDB table: {e}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error ensuring DynamoDB table exists: {e}")
            return False
    
    def get_agent_history(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get MPC history for a specific agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of records to return
            
        Returns:
            List of MPC history records
        """
        try:
            # Lazy load boto3 and DynamoDB resources
            if self._dynamodb is None:
                import boto3
                self._dynamodb = boto3.resource('dynamodb')
                self._table = self._dynamodb.Table(self.table_name)
            
            # Query the table using the AgentIndex
            response = self._table.query(
                IndexName='AgentIndex',
                KeyConditionExpression=boto3.dynamodb.conditions.Key('agent_id').eq(agent_id),
                ScanIndexForward=False,  # Sort in descending order (newest first)
                Limit=limit
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            self.logger.error(f"Error getting agent history: {e}")
            return []