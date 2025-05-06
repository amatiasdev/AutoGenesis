"""
API Client Module Generator for AutoGenesis Agent Factory.

This module is responsible for generating API client modules for Target Agents.
"""

import logging
from typing import Dict, Any

from generators.base_module_generator import BaseModuleGenerator

# Set up logging
logger = logging.getLogger(__name__)

class ApiClientModuleGenerator(BaseModuleGenerator):
    """
    Generates API client modules for Target Agents.
    """
    
    def generate(self, module_name: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate an API client module.
        
        Args:
            module_name: Name of the module
            blueprint: Blueprint specifications
            
        Returns:
            Complete Python code for the API client module
        """
        # Get API details from blueprint if available
        source = blueprint.get('requirements', {}).get('source', {})
        source_type = source.get('type', '')
        source_value = source.get('value', '')
        
        # Start with the module header
        code = self.generate_module_header(
            module_name, 
            "API client for external services"
        )
        
        # Add API client code
        code += '''
import requests
import time
import json
from typing import Dict, Any, List, Optional, Union
import time

class RateLimiter:
    """
    Rate limiter for API requests.
    
    Controls the rate of requests to prevent exceeding API rate limits.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0.0
        
    def wait(self) -> None:
        """
        Wait if necessary to comply with rate limits.
        """
        if self.interval <= 0:
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.interval:
            # Need to wait to comply with rate limit
            sleep_time = self.interval - elapsed
            logger.debug(f"Rate limiter sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()


class APIClient:
    """
    Generic API client with rate limiting and error handling.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                requests_per_minute: int = 60):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication (optional)
            requests_per_minute: Maximum requests per minute
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.session = requests.Session()
        
        # Set up default headers
        self.session.headers.update({
            'User-Agent': 'AutoGenesis-Agent/1.0',
            'Content-Type': 'application/json'
        })
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.
        
        Args:
            response: Response object from requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.HTTPError: If response status code indicates an error
        """
        # Check for HTTP errors
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # Try to extract error details from JSON response
            error_detail = str(e)
            try:
                error_json = response.json()
                if isinstance(error_json, dict):
                    error_detail = error_json.get('error', {}).get('message', str(e))
                    if not error_detail:
                        error_detail = str(error_json)
            except:
                pass
                
            logger.error(f"API error: {error_detail} (Status: {response.status_code})")
            raise
        
        # Parse JSON response
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.warning(f"Response is not valid JSON: {response.text[:100]}...")
            return {'raw_content': response.text}
    
    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (will be appended to base_url)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.RequestException: For request-related errors
            ValueError: For invalid parameters
        """
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Construct full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Log request details (excluding sensitive information)
        log_kwargs = kwargs.copy()
        if 'headers' in log_kwargs and 'Authorization' in log_kwargs['headers']:
            log_kwargs['headers'] = log_kwargs['headers'].copy()
            log_kwargs['headers']['Authorization'] = '[REDACTED]'
            
        logger.debug(f"API Request: {method} {url} {json.dumps(log_kwargs)}")
        
        try:
            # Make the request
            response = self.session.request(method, url, **kwargs)
            
            # Handle response
            result = self._handle_response(response)
            return result
            
        except requests.Timeout:
            logger.error(f"Request timed out: {method} {url}")
            raise
        except requests.ConnectionError:
            logger.error(f"Connection error: {method} {url}")
            raise
        except Exception as e:
            logger.error(f"Request error: {method} {url} - {str(e)}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
           **kwargs) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response
        """
        return self.request('GET', endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, 
            json_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response
        """
        return self.request('POST', endpoint, data=data, json=json_data, **kwargs)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None,
           json_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response
        """
        return self.request('PUT', endpoint, data=data, json=json_data, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Parsed JSON response
        """
        return self.request('DELETE', endpoint, **kwargs)
'''

        # Add a specialized client if the source type is API
        if source_type == 'api' and source_value:
            # Extract domain from source value if it's a URL
            domain = source_value
            if '://' in source_value:
                domain = source_value.split('://', 1)[1].split('/', 1)[0]
                
            # Add a service-specific client class
            class_name = ''.join(word.capitalize() for word in domain.split('.')[0].split('-'))
            
            code += f'''

class {class_name}Client(APIClient):
    """
    Specialized client for {domain} API.
    """
    
    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 60):
        """
        Initialize the {domain} API client.
        
        Args:
            api_key: API key for authentication
            requests_per_minute: Maximum requests per minute
        """
        super().__init__(base_url="{source_value}", api_key=api_key, 
                         requests_per_minute=requests_per_minute)
    
    # Add specialized methods for the API here
    def get_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get data from the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Parsed API response
        """
        return self.get(endpoint, params=params)
    
    def send_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send data to the API.
        
        Args:
            endpoint: API endpoint
            data: Data to send
            
        Returns:
            Parsed API response
        """
        return self.post(endpoint, json_data=data)
'''
        
        return code