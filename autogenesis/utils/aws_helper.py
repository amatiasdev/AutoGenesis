"""
AWS Helper utilities for AutoGenesis.

This module provides utilities for interacting with AWS services.
"""

import logging
import os
import json
import time
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

from utils.error_handler import handle_exceptions, AutoGenesisError


class AWSError(AutoGenesisError):
    """Exception raised for AWS-related errors."""
    pass


class AWSHelper:
    """
    Helper class for AWS operations.
    
    Provides utility methods for common AWS operations used in AutoGenesis.
    """
    
    def __init__(self, region: str = None):
        """
        Initialize the AWS helper.
        
        Args:
            region: AWS region (defaults to environment variable or us-east-1)
        """
        self.logger = logging.getLogger(__name__)
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        
        # Lazy-loaded boto3 clients
        self._s3 = None
        self._ec2 = None
        self._iam = None
        self._lambda = None
    
    @handle_exceptions(default_message="Failed to get EC2 instance metadata")
    def get_instance_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the current EC2 instance.
        
        Returns:
            Dict containing instance metadata
        
        Raises:
            AWSError: If metadata retrieval fails
        """
        try:
            import boto3
            import requests
            from botocore.exceptions import ClientError
            
            # First, check if we're running on EC2
            try:
                # Try to connect to instance metadata service with timeout
                response = requests.get(
                    "http://169.254.169.254/latest/meta-data/instance-id",
                    timeout=0.1
                )
                instance_id = response.text
            except (requests.RequestException, requests.Timeout):
                self.logger.info("Not running on EC2 or metadata service unavailable")
                return {
                    "is_ec2": False,
                    "instance_id": None,
                    "instance_type": None,
                    "region": self.region
                }
            
            # Get EC2 client
            if self._ec2 is None:
                self._ec2 = boto3.client('ec2', region_name=self.region)
            
            # Get instance details
            response = self._ec2.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            # Extract relevant metadata
            metadata = {
                "is_ec2": True,
                "instance_id": instance_id,
                "instance_type": instance.get('InstanceType'),
                "region": self.region,
                "vpc_id": instance.get('VpcId'),
                "subnet_id": instance.get('SubnetId'),
                "private_ip": instance.get('PrivateIpAddress'),
                "public_ip": instance.get('PublicIpAddress'),
                "availability_zone": instance.get('Placement', {}).get('AvailabilityZone')
            }
            
            # Get tags
            if 'Tags' in instance:
                metadata['tags'] = {tag['Key']: tag['Value'] for tag in instance['Tags']}
            
            return metadata
            
        except Exception as e:
            raise AWSError(f"Failed to get EC2 instance metadata: {str(e)}", details=str(e))
    
    @handle_exceptions(default_message="Failed to get EC2 instance specifications")
    def get_instance_specs(self) -> Dict[str, Any]:
        """
        Get specifications for the current EC2 instance or the local machine.
        
        Returns:
            Dict containing machine specifications
        """
        # First try to get EC2 instance specs
        try:
            metadata = self.get_instance_metadata()
            
            if metadata.get("is_ec2", False):
                instance_type = metadata.get("instance_type")
                
                # Get EC2 client
                if self._ec2 is None:
                    import boto3
                    self._ec2 = boto3.client('ec2', region_name=self.region)
                
                # Get instance type info
                response = self._ec2.describe_instance_types(InstanceTypes=[instance_type])
                instance_info = response['InstanceTypes'][0]
                
                return {
                    "cpu_cores": instance_info.get('VCpuInfo', {}).get('DefaultVCpus', 1),
                    "memory_mb": instance_info.get('MemoryInfo', {}).get('SizeInMiB', 1024),
                    "instance_type": instance_type,
                    "platform": "EC2"
                }
        except Exception as e:
            self.logger.warning(f"Failed to get EC2 instance specs: {str(e)}")
        
        # Fallback to local machine specs
        import os
        import platform
        import psutil
        
        # Get CPU cores
        try:
            cpu_cores = os.cpu_count() or psutil.cpu_count() or 1
        except Exception:
            cpu_cores = 1
        
        # Get memory
        try:
            memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        except Exception:
            memory_mb = 1024  # Default to 1GB
        
        return {
            "cpu_cores": cpu_cores,
            "memory_mb": int(memory_mb),
            "platform": platform.system(),
            "platform_version": platform.version()
        }
    
    @handle_exceptions(default_message="Failed to create IAM role")
    def create_iam_role(self, 
                       role_name: str,
                       description: str,
                       assume_role_policy: Dict[str, Any],
                       policy_arns: List[str] = None) -> Dict[str, Any]:
        """
        Create an IAM role with specified policies.
        
        Args:
            role_name: Name of the role
            description: Role description
            assume_role_policy: Trust policy document
            policy_arns: List of policy ARNs to attach
            
        Returns:
            Dict containing role details
            
        Raises:
            AWSError: If role creation fails
        """
        try:
            import boto3
            import json
            from botocore.exceptions import ClientError
            
            # Get IAM client
            if self._iam is None:
                self._iam = boto3.client('iam')
            
            # Convert assume_role_policy to JSON string if it's a dict
            if isinstance(assume_role_policy, dict):
                assume_role_policy = json.dumps(assume_role_policy)
            
            # Create role
            try:
                response = self._iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=assume_role_policy,
                    Description=description
                )
                
                role_arn = response['Role']['Arn']
                self.logger.info(f"Created IAM role: {role_arn}")
                
                # Attach policies if provided
                if policy_arns:
                    for policy_arn in policy_arns:
                        self._iam.attach_role_policy(
                            RoleName=role_name,
                            PolicyArn=policy_arn
                        )
                        self.logger.info(f"Attached policy {policy_arn} to role {role_name}")
                
                return {
                    "role_name": role_name,
                    "role_arn": role_arn,
                    "attached_policies": policy_arns or []
                }
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'EntityAlreadyExists':
                    # Role already exists, get it
                    response = self._iam.get_role(RoleName=role_name)
                    role_arn = response['Role']['Arn']
                    
                    self.logger.info(f"Using existing IAM role: {role_arn}")
                    
                    # Get attached policies
                    attached_policies = self._iam.list_attached_role_policies(RoleName=role_name)
                    current_policies = [p['PolicyArn'] for p in attached_policies.get('AttachedPolicies', [])]
                    
                    # Attach any missing policies
                    if policy_arns:
                        for policy_arn in policy_arns:
                            if policy_arn not in current_policies:
                                self._iam.attach_role_policy(
                                    RoleName=role_name,
                                    PolicyArn=policy_arn
                                )
                                self.logger.info(f"Attached policy {policy_arn} to role {role_name}")
                                current_policies.append(policy_arn)
                    
                    return {
                        "role_name": role_name,
                        "role_arn": role_arn,
                        "attached_policies": current_policies
                    }
                else:
                    raise
                    
        except Exception as e:
            raise AWSError(f"Failed to create IAM role: {str(e)}", details=str(e))
    
    @handle_exceptions(default_message="Failed to upload to S3")
    def upload_to_s3(self, 
                    local_path: str, 
                    bucket: str, 
                    key: str,
                    metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Upload a file to S3.
        
        Args:
            local_path: Path to local file
            bucket: S3 bucket name
            key: S3 object key
            metadata: Optional metadata to attach to the object
            
        Returns:
            Dict containing upload details
            
        Raises:
            AWSError: If upload fails
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Get S3 client
            if self._s3 is None:
                self._s3 = boto3.client('s3', region_name=self.region)
            
            # Upload file
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self._s3.upload_file(
                Filename=local_path,
                Bucket=bucket,
                Key=key,
                ExtraArgs=extra_args
            )
            
            # Get object URL
            object_url = f"s3://{bucket}/{key}"
            
            self.logger.info(f"Uploaded {local_path} to {object_url}")
            
            return {
                "bucket": bucket,
                "key": key,
                "url": object_url
            }
            
        except Exception as e:
            raise AWSError(f"Failed to upload to S3: {str(e)}", details=str(e))


class DynamoDBLogger:
    """
    Logger for storing data in DynamoDB.
    
    This is a simplified version - the full implementation is in dynamodb_logger.py
    """
    
    def __init__(self, table_name: str = None):
        """
        Initialize the DynamoDB logger.
        
        Args:
            table_name: Optional DynamoDB table name (defaults to environment variable)
        """
        self.logger = logging.getLogger(__name__)
        self.table_name = table_name or os.environ.get('AUTOGENESIS_MPC_TABLE', 'AutoGenesisMPCLogs')
    
    @handle_exceptions(default_message="Failed to log MPC results")
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
        # Just pass through to the main implementation
        from mpc.dynamodb_logger import DynamoDBLogger as FullDynamoDBLogger
        
        logger = FullDynamoDBLogger(self.table_name)
        return logger.log_mpc_results(agent_id, code_path, blueprint, mpc_results)