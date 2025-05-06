"""
Cost Estimator for AutoGenesis MPC.

This module evaluates the cost and resource usage aspects of generated agent code
as part of the Multi-Perspective Criticism (MPC) process.
"""

import logging
import os
import re
import ast
import json
from typing import Dict, Any, List, Set, Tuple, Optional

class CostEstimator:
    """
    Estimates cost and resource usage for the agent.
    
    Analyzes agent code to estimate runtime costs including compute, 
    memory, storage, and external service usage costs.
    """
    
    def __init__(self):
        """Initialize the cost estimator."""
        self.logger = logging.getLogger(__name__)
        
        # Define AWS service cost models (simplified)
        self.aws_cost_models = {
            "ec2": {
                "t3.micro": {"cpu": 2, "memory_gb": 1, "hourly_cost": 0.0104},
                "t3.small": {"cpu": 2, "memory_gb": 2, "hourly_cost": 0.0208},
                "t3.medium": {"cpu": 2, "memory_gb": 4, "hourly_cost": 0.0416},
                "t3.large": {"cpu": 2, "memory_gb": 8, "hourly_cost": 0.0832},
                "m5.large": {"cpu": 2, "memory_gb": 8, "hourly_cost": 0.096},
                "m5.xlarge": {"cpu": 4, "memory_gb": 16, "hourly_cost": 0.192},
                "c5.large": {"cpu": 2, "memory_gb": 4, "hourly_cost": 0.085},
                "r5.large": {"cpu": 2, "memory_gb": 16, "hourly_cost": 0.126}
            },
            "lambda": {
                "cost_per_request": 0.0000002,  # $0.20 per 1M requests
                "cost_per_gb_second": 0.0000166667  # $0.0000166667 per GB-second
            },
            "s3": {
                "storage_gb_month": 0.023,  # $0.023 per GB-month
                "get_request_per_1000": 0.0004,  # $0.0004 per 1,000 GET requests
                "put_request_per_1000": 0.005  # $0.005 per 1,000 PUT requests
            },
            "dynamodb": {
                "write_capacity_unit_hour": 0.00065,  # $0.00065 per WCU-hour
                "read_capacity_unit_hour": 0.00013,  # $0.00013 per RCU-hour
                "storage_gb_month": 0.25  # $0.25 per GB-month
            },
            "sqs": {
                "requests_per_million": 0.40  # $0.40 per 1M requests
            },
            "sns": {
                "publishes_per_million": 0.50  # $0.50 per 1M publishes
            },
            "api_gateway": {
                "requests_per_million": 3.50  # $3.50 per 1M requests
            }
        }
        
        # Define third-party service cost models (simplified)
        self.third_party_cost_models = {
            "stripe": {
                "per_transaction": 0.029  # 2.9% + $0.30, simplified
            },
            "twilio": {
                "sms_per_message": 0.0075  # $0.0075 per SMS
            },
            "sendgrid": {
                "emails_per_thousand": 1.00  # $1.00 per 1,000 emails
            },
            "cloudflare": {
                "requests_per_million": 0.50  # $0.50 per 1M requests
            }
        }
    
    def estimate(self, 
                code_path: str,
                resource_requirements: Dict[str, Any],
                expected_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate costs for the agent.
        
        Args:
            code_path: Path to the code to evaluate
            resource_requirements: Resource requirements for the agent
            expected_usage: Expected usage patterns
            
        Returns:
            Dict containing cost estimates
        """
        self.logger.info(f"Estimating costs for code at {code_path}")
        
        # Find all Python files
        python_files = self._find_python_files(code_path)
        self.logger.info(f"Found {len(python_files)} Python files to evaluate")
        
        # Analyze code for service usage
        service_usage = self._analyze_service_usage(python_files, expected_usage)
        
        # Compute estimations
        compute_costs = self._estimate_compute_costs(resource_requirements, expected_usage)
        storage_costs = self._estimate_storage_costs(service_usage, expected_usage)
        service_costs = self._estimate_service_costs(service_usage, expected_usage)
        
        # Calculate total costs
        total_monthly_cost = (
            compute_costs.get("monthly_cost", 0) +
            storage_costs.get("monthly_cost", 0) +
            service_costs.get("monthly_cost", 0)
        )
        
        # Determine if cost is acceptable
        cost_acceptable = total_monthly_cost <= 50  # $50/month is acceptable
        
        # Generate cost optimizations
        optimizations = self._generate_cost_optimizations(
            resource_requirements, 
            service_usage, 
            compute_costs, 
            storage_costs, 
            service_costs
        )
        
        return {
            "passed": cost_acceptable,
            "total_monthly_cost": total_monthly_cost,
            "hourly_cost": total_monthly_cost / 730,  # 730 hours in average month
            "compute_costs": compute_costs,
            "storage_costs": storage_costs,
            "service_costs": service_costs,
            "service_usage": service_usage,
            "cost_optimizations": optimizations
        }
    
    def _find_python_files(self, code_path: str) -> List[str]:
        """
        Find all Python files in the code path.
        
        Args:
            code_path: Path to search for Python files
            
        Returns:
            List of Python file paths
        """
        python_files = []
        
        # Walk through directory
        for root, _, files in os.walk(code_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_service_usage(self, python_files: List[str], expected_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code for service usage.
        
        Args:
            python_files: List of Python file paths
            
        Returns:
            Dict containing service usage information
        """
        service_usage = {
            "aws": {
                "s3": False,
                "dynamodb": False,
                "lambda": False,
                "sqs": False,
                "sns": False,
                "api_gateway": False
            },
            "third_party": {
                "stripe": False,
                "twilio": False,
                "sendgrid": False,
                "cloudflare": False
            },
            "database": {
                "usage": False,
                "type": None,
                "estimated_size_gb": 1  # Default estimate
            },
            "file_storage": {
                "usage": False,
                "estimated_size_gb": 1  # Default estimate
            },
            "network": {
                "usage": False,
                "estimated_requests_per_day": 1000  # Default estimate
            }
        }
        
        # Patterns to detect service usage
        service_patterns = {
            "aws": {
                "s3": r'boto3\..*[\'"]s3[\'"]|S3Client|S3Resource',
                "dynamodb": r'boto3\..*[\'"]dynamodb[\'"]|DynamoDBClient|Table\(',
                "lambda": r'boto3\..*[\'"]lambda[\'"]|LambdaClient|invoke\(',
                "sqs": r'boto3\..*[\'"]sqs[\'"]|SQSClient|send_message\(',
                "sns": r'boto3\..*[\'"]sns[\'"]|SNSClient|publish\(',
                "api_gateway": r'boto3\..*[\'"]apigateway[\'"]|APIGatewayClient'
            },
            "third_party": {
                "stripe": r'import\s+stripe|from\s+stripe|Stripe\(',
                "twilio": r'import\s+twilio|from\s+twilio|Client\(.*twilio',
                "sendgrid": r'import\s+sendgrid|from\s+sendgrid',
                "cloudflare": r'import\s+CloudFlare|from\s+CloudFlare'
            },
            "database": {
                "sqlite": r'import\s+sqlite3|from\s+sqlite3',
                "postgres": r'import\s+psycopg2|from\s+psycopg2',
                "mysql": r'import\s+mysql|from\s+mysql',
                "mongodb": r'import\s+pymongo|from\s+pymongo',
                "sqlalchemy": r'import\s+sqlalchemy|from\s+sqlalchemy'
            },
            "file_storage": {
                "local": r'open\(.*[\'"]w[\'"]|write\(|with\s+open\(',
                "csv": r'import\s+csv|from\s+csv|\.to_csv|csv\.writer',
                "excel": r'import\s+openpyxl|from\s+openpyxl|import\s+xlrd|from\s+xlrd',
                "json": r'json\.dump|json\.dumps'
            },
            "network": {
                "requests": r'import\s+requests|from\s+requests',
                "urllib": r'import\s+urllib|from\s+urllib',
                "http": r'import\s+http|from\s+http',
                "aiohttp": r'import\s+aiohttp|from\s+aiohttp'
            }
        }
        
        # Check each file for service usage
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check AWS services
                for service, pattern in service_patterns["aws"].items():
                    if re.search(pattern, content, re.IGNORECASE):
                        service_usage["aws"][service] = True
                
                # Check third-party services
                for service, pattern in service_patterns["third_party"].items():
                    if re.search(pattern, content, re.IGNORECASE):
                        service_usage["third_party"][service] = True
                
                # Check database usage
                for db_type, pattern in service_patterns["database"].items():
                    if re.search(pattern, content, re.IGNORECASE):
                        service_usage["database"]["usage"] = True
                        if not service_usage["database"]["type"]:
                            service_usage["database"]["type"] = db_type
                
                # Check file storage usage
                for storage_type, pattern in service_patterns["file_storage"].items():
                    if re.search(pattern, content, re.IGNORECASE):
                        service_usage["file_storage"]["usage"] = True
                
                # Check network usage
                for network_type, pattern in service_patterns["network"].items():
                    if re.search(pattern, content, re.IGNORECASE):
                        service_usage["network"]["usage"] = True
            
            except Exception as e:
                self.logger.error(f"Error analyzing service usage in {file_path}: {e}")
        
        # Estimate database size based on usage frequency
        if service_usage["database"]["usage"]:
            frequency = self._get_usage_frequency(expected_usage)
            if frequency == "high":
                service_usage["database"]["estimated_size_gb"] = 10
            elif frequency == "medium":
                service_usage["database"]["estimated_size_gb"] = 5
            
        # Estimate file storage size based on usage frequency
        if service_usage["file_storage"]["usage"]:
            frequency = self._get_usage_frequency(expected_usage)
            if frequency == "high":
                service_usage["file_storage"]["estimated_size_gb"] = 20
            elif frequency == "medium":
                service_usage["file_storage"]["estimated_size_gb"] = 5
        
        # Estimate network usage based on usage frequency
        if service_usage["network"]["usage"]:
            frequency = self._get_usage_frequency(expected_usage)
            if frequency == "high":
                service_usage["network"]["estimated_requests_per_day"] = 100000
            elif frequency == "medium":
                service_usage["network"]["estimated_requests_per_day"] = 10000
            else:
                service_usage["network"]["estimated_requests_per_day"] = 1000
        
        return service_usage
    
    def _get_usage_frequency(self, expected_usage: Dict[str, Any]) -> str:
        """
        Determine usage frequency based on expected usage.
        
        Args:
            expected_usage: Expected usage patterns
            
        Returns:
            String representing usage frequency: "high", "medium", or "low"
        """
        frequency = expected_usage.get("frequency", "on_demand")
        concurrent_users = expected_usage.get("concurrent_users", 1)
        data_volume_mb = expected_usage.get("data_volume_mb", 10)
        
        if (frequency in ["hourly", "daily"] and concurrent_users > 10) or data_volume_mb > 1000:
            return "high"
        elif (frequency in ["daily", "weekly"] and concurrent_users > 1) or data_volume_mb > 100:
            return "medium"
        else:
            return "low"
    
    def _estimate_compute_costs(self, 
                              resource_requirements: Dict[str, Any],
                              expected_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate compute costs based on resource requirements.
        
        Args:
            resource_requirements: Resource requirements for the agent
            expected_usage: Expected usage patterns
            
        Returns:
            Dict containing compute cost estimates
        """
        # Extract resource requirements
        memory_mb = resource_requirements.get("memory_mb", 512)
        cpu_cores = resource_requirements.get("cpu_cores", 0.5)
        
        # Convert memory to GB
        memory_gb = memory_mb / 1024
        
        # Determine usage hours based on frequency
        frequency = expected_usage.get("frequency", "on_demand")
        hours_per_month = self._calculate_monthly_hours(frequency)
        
        # Select appropriate EC2 instance type
        instance_type = self._select_instance_type(cpu_cores, memory_gb)
        hourly_cost = self.aws_cost_models["ec2"][instance_type]["hourly_cost"]
        
        # Calculate monthly cost
        monthly_cost = hourly_cost * hours_per_month
        
        # Determine if Lambda might be more cost-effective
        lambda_eligible = (
            memory_gb <= 10 and  # Lambda has up to 10GB memory
            frequency in ["on_demand", "hourly"] and  # Not continuously running
            expected_usage.get("expected_runtime_seconds", 60) < 900  # Less than 15 minutes runtime
        )
        
        lambda_cost = None
        if lambda_eligible:
            # Estimate Lambda costs
            lambda_cost = self._estimate_lambda_costs(
                memory_gb,
                expected_usage.get("expected_runtime_seconds", 60),
                self._calculate_monthly_invocations(frequency)
            )
        
        return {
            "instance_type": instance_type,
            "memory_gb": memory_gb,
            "cpu_cores": cpu_cores,
            "hours_per_month": hours_per_month,
            "hourly_cost": hourly_cost,
            "monthly_cost": monthly_cost,
            "lambda_eligible": lambda_eligible,
            "lambda_cost": lambda_cost
        }
    
    def _select_instance_type(self, cpu_cores: float, memory_gb: float) -> str:
        """
        Select appropriate EC2 instance type based on resource requirements.
        
        Args:
            cpu_cores: Required CPU cores
            memory_gb: Required memory in GB
            
        Returns:
            String representing selected EC2 instance type
        """
        # Define instance selection logic
        if cpu_cores <= 1 and memory_gb <= 0.5:
            return "t3.micro"
        elif cpu_cores <= 2 and memory_gb <= 1:
            return "t3.micro"
        elif cpu_cores <= 2 and memory_gb <= 2:
            return "t3.small"
        elif cpu_cores <= 2 and memory_gb <= 4:
            return "t3.medium"
        elif cpu_cores <= 2 and memory_gb <= 8:
            return "t3.large"
        elif cpu_cores <= 2 and memory_gb <= 16:
            return "r5.large"  # Memory-optimized
        elif cpu_cores <= 4 and memory_gb <= 8:
            return "c5.large"  # Compute-optimized
        elif cpu_cores <= 4 and memory_gb <= 16:
            return "m5.xlarge"
        else:
            return "m5.xlarge"  # Default to this for simplicity
    
    def _calculate_monthly_hours(self, frequency: str) -> float:
        """
        Calculate monthly runtime hours based on usage frequency.
        
        Args:
            frequency: Usage frequency
            
        Returns:
            Float representing monthly runtime hours
        """
        # Average month has 730 hours (365 days / 12 months * 24 hours)
        if frequency == "continuous":
            return 730  # Run continuously
        elif frequency == "hourly":
            return 730  # Likely running continuously for short tasks
        elif frequency == "daily":
            return 30  # 1 hour per day
        elif frequency == "weekly":
            return 4   # 1 hour per week
        elif frequency == "monthly":
            return 1   # 1 hour per month
        else:  # on_demand
            return 10  # Assume 10 hours per month on demand usage
    
    def _calculate_monthly_invocations(self, frequency: str) -> int:
        """
        Calculate monthly invocations based on usage frequency.
        
        Args:
            frequency: Usage frequency
            
        Returns:
            Integer representing estimated monthly invocations
        """
        if frequency == "continuous":
            return 44640  # 60 * 24 * 31 (once per minute)
        elif frequency == "hourly":
            return 744  # 24 * 31
        elif frequency == "daily":
            return 31
        elif frequency == "weekly":
            return 4
        elif frequency == "monthly":
            return 1
        else:  # on_demand
            return 30  # Assume 30 invocations per month
    
    def _estimate_lambda_costs(self, 
                             memory_gb: float, 
                             duration_seconds: int,
                             monthly_invocations: int) -> float:
        """
        Estimate AWS Lambda costs.
        
        Args:
            memory_gb: Memory allocation in GB
            duration_seconds: Function duration in seconds
            monthly_invocations: Number of monthly invocations
            
        Returns:
            Float representing estimated monthly Lambda cost
        """
        # Lambda pricing components
        request_cost = self.aws_cost_models["lambda"]["cost_per_request"] * monthly_invocations
        
        # Calculate GB-seconds (Lambda rounds up to nearest 1ms)
        rounded_duration_seconds = (duration_seconds * 1000 + 999) // 1000
        gb_seconds = memory_gb * rounded_duration_seconds * monthly_invocations
        compute_cost = self.aws_cost_models["lambda"]["cost_per_gb_second"] * gb_seconds
        
        total_cost = request_cost + compute_cost
        return total_cost
    
    def _estimate_storage_costs(self, 
                              service_usage: Dict[str, Any],
                              expected_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate storage costs.
        
        Args:
            service_usage: Service usage information
            expected_usage: Expected usage patterns
            
        Returns:
            Dict containing storage cost estimates
        """
        storage_costs = {
            "database": 0,
            "s3": 0,
            "local": 0,
            "total": 0,
            "monthly_cost": 0
        }
        
        # Database storage costs
        if service_usage["database"]["usage"]:
            db_size_gb = service_usage["database"]["estimated_size_gb"]
            
            # Different costs based on database type
            db_type = service_usage["database"]["type"]
            if db_type in ["postgres", "mysql"]:
                # RDS pricing (simplified)
                storage_costs["database"] = db_size_gb * 0.115  # $0.115 per GB-month
            elif db_type == "mongodb":
                # DocumentDB pricing (simplified)
                storage_costs["database"] = db_size_gb * 0.10  # $0.10 per GB-month
            elif db_type == "dynamodb":
                # DynamoDB pricing
                storage_costs["database"] = db_size_gb * self.aws_cost_models["dynamodb"]["storage_gb_month"]
            else:
                # Local database (SQLite) - no cost
                storage_costs["database"] = 0
        
        # S3 storage costs
        if service_usage["aws"]["s3"]:
            s3_size_gb = service_usage["file_storage"]["estimated_size_gb"]
            storage_costs["s3"] = s3_size_gb * self.aws_cost_models["s3"]["storage_gb_month"]
            
            # Estimate request costs
            monthly_requests = self._calculate_monthly_invocations(expected_usage.get("frequency", "on_demand")) * 5  # Assume 5 S3 operations per invocation
            get_request_cost = (monthly_requests * 0.7 / 1000) * self.aws_cost_models["s3"]["get_request_per_1000"]  # Assume 70% GET requests
            put_request_cost = (monthly_requests * 0.3 / 1000) * self.aws_cost_models["s3"]["put_request_per_1000"]  # Assume 30% PUT requests
            storage_costs["s3"] += get_request_cost + put_request_cost
        
        # Local file storage (included in EC2 instance cost)
        if service_usage["file_storage"]["usage"] and not service_usage["aws"]["s3"]:
            local_size_gb = service_usage["file_storage"]["estimated_size_gb"]
            # No additional cost for local storage unless it exceeds instance limits
            if local_size_gb > 30:  # Assume 30GB is included with instance
                excess_gb = local_size_gb - 30
                storage_costs["local"] = excess_gb * 0.10  # $0.10 per GB-month (EBS pricing simplified)
        
        # Calculate total storage costs
        storage_costs["total"] = (
            storage_costs["database"] +
            storage_costs["s3"] +
            storage_costs["local"]
        )
        
        storage_costs["monthly_cost"] = storage_costs["total"]
        
        return storage_costs
    
    def _estimate_service_costs(self, 
                              service_usage: Dict[str, Any],
                              expected_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate service costs (AWS and third-party).
        
        Args:
            service_usage: Service usage information
            expected_usage: Expected usage patterns
            
        Returns:
            Dict containing service cost estimates
        """
        service_costs = {
            "aws": {
                "sqs": 0,
                "sns": 0,
                "api_gateway": 0
            },
            "third_party": {
                "stripe": 0,
                "twilio": 0,
                "sendgrid": 0,
                "cloudflare": 0
            },
            "other": 0,
            "monthly_cost": 0
        }
        
        # Calculate monthly operations based on frequency
        frequency = expected_usage.get("frequency", "on_demand")
        monthly_operations = self._calculate_monthly_operations(frequency)
        
        # AWS service costs
        if service_usage["aws"]["sqs"]:
            # SQS pricing
            monthly_sqs_requests = monthly_operations * 2  # Assume 2 SQS operations per application operation
            service_costs["aws"]["sqs"] = (monthly_sqs_requests / 1000000) * self.aws_cost_models["sqs"]["requests_per_million"]
        
        if service_usage["aws"]["sns"]:
            # SNS pricing
            monthly_sns_publishes = monthly_operations * 0.5  # Assume 0.5 SNS publishes per application operation
            service_costs["aws"]["sns"] = (monthly_sns_publishes / 1000000) * self.aws_cost_models["sns"]["publishes_per_million"]
        
        if service_usage["aws"]["api_gateway"]:
            # API Gateway pricing
            monthly_api_requests = monthly_operations * 1.5  # Assume 1.5 API requests per application operation
            service_costs["aws"]["api_gateway"] = (monthly_api_requests / 1000000) * self.aws_cost_models["api_gateway"]["requests_per_million"]
        
        # Third-party service costs
        if service_usage["third_party"]["stripe"]:
            # Stripe pricing (transaction processing)
            monthly_transactions = monthly_operations * 0.1  # Assume 10% of operations involve payments
            average_transaction_amount = 50  # Assume $50 average transaction
            service_costs["third_party"]["stripe"] = monthly_transactions * self.third_party_cost_models["stripe"]["per_transaction"] * average_transaction_amount
        
        if service_usage["third_party"]["twilio"]:
            # Twilio pricing (SMS)
            monthly_sms = monthly_operations * 0.2  # Assume 20% of operations send SMS
            service_costs["third_party"]["twilio"] = monthly_sms * self.third_party_cost_models["twilio"]["sms_per_message"]
        
        if service_usage["third_party"]["sendgrid"]:
            # SendGrid pricing (email)
            monthly_emails = monthly_operations * 0.5  # Assume 50% of operations send emails
            service_costs["third_party"]["sendgrid"] = (monthly_emails / 1000) * self.third_party_cost_models["sendgrid"]["emails_per_thousand"]
        
        if service_usage["third_party"]["cloudflare"]:
            # Cloudflare pricing (requestsAI)
            monthly_cf_requests = monthly_operations * 2  # Assume 2 Cloudflare requests per application operation
            service_costs["third_party"]["cloudflare"] = (monthly_cf_requests / 1000000) * self.third_party_cost_models["cloudflare"]["requests_per_million"]
        
        # Network data transfer costs (simplified)
        if service_usage["network"]["usage"]:
            data_volume_mb = expected_usage.get("data_volume_mb", 10)
            data_volume_gb = data_volume_mb / 1024
            monthly_data_transfer_gb = data_volume_gb * monthly_operations
            
            # AWS data transfer out pricing (simplified)
            if monthly_data_transfer_gb <= 1:
                service_costs["other"] = 0  # First 1 GB free
            else:
                service_costs["other"] = (monthly_data_transfer_gb - 1) * 0.09  # $0.09 per GB after first 1 GB
        
        # Calculate total service costs
        aws_costs = sum(service_costs["aws"].values())
        third_party_costs = sum(service_costs["third_party"].values())
        other_costs = service_costs["other"]
        
        service_costs["monthly_cost"] = aws_costs + third_party_costs + other_costs
        
        return service_costs
    
    def _calculate_monthly_operations(self, frequency: str) -> int:
        """
        Calculate monthly operations based on usage frequency.
        
        Args:
            frequency: Usage frequency
            
        Returns:
            Integer representing estimated monthly operations
        """
        if frequency == "continuous":
            return 100000  # High number for continuous operation
        elif frequency == "hourly":
            return 24 * 30 * 10  # 10 operations per hour
        elif frequency == "daily":
            return 30 * 100  # 100 operations per day
        elif frequency == "weekly":
            return 4 * 250  # 250 operations per week
        elif frequency == "monthly":
            return 500  # 500 operations per month
        else:  # on_demand
            return 100  # Assume 100 operations per month
    
    def _generate_cost_optimizations(self, 
                                   resource_requirements: Dict[str, Any],
                                   service_usage: Dict[str, Any],
                                   compute_costs: Dict[str, Any],
                                   storage_costs: Dict[str, Any],
                                   service_costs: Dict[str, Any]) -> List[str]:
        """
        Generate cost optimization recommendations.
        
        Args:
            resource_requirements: Resource requirements for the agent
            service_usage: Service usage information
            compute_costs: Compute cost estimates
            storage_costs: Storage cost estimates
            service_costs: Service cost estimates
            
        Returns:
            List of cost optimization recommendations
        """
        optimizations = []
        
        # Compute optimizations
        if compute_costs["lambda_eligible"] and compute_costs["lambda_cost"] < compute_costs["monthly_cost"]:
            optimizations.append(
                f"Consider using AWS Lambda instead of EC2 to save approximately ${compute_costs['monthly_cost'] - compute_costs['lambda_cost']:.2f} per month"
            )
        
        # If using a larger instance than needed, suggest sizing down
        memory_utilization = resource_requirements.get("memory_mb", 512) / (self.aws_cost_models["ec2"][compute_costs["instance_type"]]["memory_gb"] * 1024)
        if memory_utilization < 0.5:
            optimizations.append(
                f"Instance {compute_costs['instance_type']} appears oversized (memory utilization ~{memory_utilization:.0%}). Consider using a smaller instance type."
            )
        
        # Storage optimizations
        if storage_costs["s3"] > 5:
            optimizations.append(
                "Consider implementing S3 lifecycle policies to move infrequently accessed data to cheaper storage tiers"
            )
        
        if storage_costs["database"] > 10:
            optimizations.append(
                "Database costs are significant. Consider optimizing schema, adding indexes, or implementing caching."
            )
        
        # Service optimizations
        aws_service_cost = sum(service_costs["aws"].values())
        if aws_service_cost > 10:
            optimizations.append(
                "AWS service costs are significant. Consider using reserved capacity or implementing request batching."
            )
        
        third_party_cost = sum(service_costs["third_party"].values())
        if third_party_cost > 10:
            optimizations.append(
                "Third-party service costs are significant. Consider negotiating volume discounts or finding alternative services."
            )
        
        # General optimizations
        optimizations.append(
            "Implement request caching to reduce API calls and improve performance"
        )
        
        optimizations.append(
            "Monitor usage patterns and adjust resources accordingly to optimize costs"
        )
        
        return optimizations