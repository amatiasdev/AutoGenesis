"""
Step Function Generator for AutoGenesis Agent Factory.

This module is responsible for generating step functions based on processing step definitions.
"""

import logging
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

class StepFunctionGenerator:
    """
    Generates step functions for Target Agents.
    """
    
    def generate_step_function(self, step_name: str, step: Dict[str, Any], tools: List[str]) -> str:
        """
        Generate a function for a processing step.
        
        Args:
            step_name: Name of the step
            step: Step definition
            tools: Available tools
            
        Returns:
            Python code for the step function
        """
        # Extract action and other parameters
        action = step.get('action', 'custom')
        
        # Start the function
        code = f'''def {step_name}(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the {step_name} step.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Result of the step execution
        
    Raises:
        Exception: If an error occurs during execution
    """
    logger.info(f"Starting {step_name}")
    result = {{"status": "success", "action": "{action}"}}
    
    try:
'''
        
        # Generate specific code based on action type
        if action == 'authenticate':
            code += self._generate_authenticate_step(step)
        elif action == 'extract_data' or action == 'scrape_data':
            code += self._generate_extract_data_step(step, tools)
        elif action == 'filter_data':
            code += self._generate_filter_data_step(step)
        elif action == 'transform_data':
            code += self._generate_transform_data_step(step)
        elif action == 'save_data':
            code += self._generate_save_data_step(step)
        else:
            # Generic processing step
            code += self._generate_generic_step()
        
        # Add error handling and return
        code += '''    except Exception as e:
        logger.error(f"Error in {step_name}: {str(e)}")
        logger.error(traceback.format_exc())
        result["status"] = "error"
        result["error"] = str(e)
        raise
        
    logger.info(f"Completed {step_name}")
    return result
'''
        
        return code
    
    def _generate_authenticate_step(self, step: Dict[str, Any]) -> str:
        """Generate authentication step code."""
        credentials_secret = step.get('credentials_secret_name', 'credentials')
        return f'''        # Authentication step
        credentials = config.get("{credentials_secret}")
        if not credentials:
            raise ValueError(f"Missing credentials for {credentials_secret}")
            
        logger.info(f"Authenticating with credentials from {credentials_secret}")
        # Implement authentication logic based on the target system
        # ...
        
        result["authenticated"] = True
        logger.info(f"Authentication successful")
'''
    
    def _generate_extract_data_step(self, step: Dict[str, Any], tools: List[str]) -> str:
        """Generate data extraction step code."""
        fields = step.get('fields', [])
        source_type = step.get('source_type', 'web')
        
        if source_type == 'web' and 'requests' in tools:
            code = f'''        # Extract data from web
        import requests
        from bs4 import BeautifulSoup
        
        url = config.get("source", {{}}).get("value")
        if not url:
            raise ValueError("Missing URL for data extraction")
            
        logger.info(f"Extracting data from URL: {{url}}")
        
        response = requests.get(url, headers={{"User-Agent": "AutoGenesis-Agent/1.0"}})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        extracted_data = []
        
        # Extract the specified fields
        # This is a simple example - adjust based on the actual site structure
        items = soup.select('div.item')  # Adjust selector based on page structure
        
        for item in items:
            item_data = {{}}
'''
            
            # Add extraction logic for each field
            for field in fields:
                code += f'''            # Extract {field}
            {field}_element = item.select_one('.{field}')  # Adjust selector
            item_data["{field}"] = {field}_element.text.strip() if {field}_element else ""
            
'''
            
            code += '''            extracted_data.append(item_data)
            
        result["data"] = extracted_data
        result["count"] = len(extracted_data)
        logger.info(f"Extracted {len(extracted_data)} items")
'''
            return code
            
        elif source_type == 'web' and 'selenium' in tools:
            code = f'''        # Extract data from web using Selenium
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        url = config.get("source", {{}}).get("value")
        if not url:
            raise ValueError("Missing URL for data extraction")
            
        logger.info(f"Extracting data from URL: {{url}} using Selenium")
        
        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            extracted_data = []
            
            # Extract the specified fields
            # This is a simple example - adjust based on the actual site structure
            items = driver.find_elements(By.CSS_SELECTOR, 'div.item')  # Adjust selector
            
            for item in items:
                item_data = {{}}
'''
            
            # Add extraction logic for each field
            for field in fields:
                code += f'''                # Extract {field}
                try:
                    {field}_element = item.find_element(By.CSS_SELECTOR, '.{field}')  # Adjust selector
                    item_data["{field}"] = {field}_element.text.strip()
                except:
                    item_data["{field}"] = ""
                
'''
            
            code += '''                extracted_data.append(item_data)
                
            result["data"] = extracted_data
            result["count"] = len(extracted_data)
            logger.info(f"Extracted {len(extracted_data)} items")
            
        finally:
            driver.quit()
'''
            return code
        
        elif source_type == 'api':
            code = f'''        # Extract data from API
        import requests
        
        api_url = config.get("source", {{}}).get("value")
        if not api_url:
            raise ValueError("Missing API URL for data extraction")
            
        # Get API credentials if needed
        api_key = config.get("credentials", {{}}).get("api_key")
        headers = {{"Content-Type": "application/json"}}
        
        if api_key:
            headers["Authorization"] = f"Bearer {{api_key}}"
            
        logger.info(f"Extracting data from API: {{api_url}}")
        
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the specified fields from the response
        # Adjust based on the actual API response structure
        extracted_data = []
        
        # Determine the items path in the response
        items = data
        if "results" in data:
            items = data["results"]
        elif "data" in data:
            items = data["data"]
        elif "items" in data:
            items = data["items"]
            
        if not isinstance(items, list):
            items = [items]
            
        for item in items:
            item_data = {{}}
'''
            
            # Add extraction logic for each field
            for field in fields:
                code += f'''            # Extract {field}
            if "{field}" in item:
                item_data["{field}"] = item["{field}"]
            else:
                item_data["{field}"] = ""
            
'''
            
            code += '''            extracted_data.append(item_data)
            
        result["data"] = extracted_data
        result["count"] = len(extracted_data)
        logger.info(f"Extracted {len(extracted_data)} items")
'''
            return code
        
        # Default case if no specific extraction method matched
        return '''        # Extract data (generic implementation)
        logger.info("Using generic data extraction")
        result["data"] = []
        result["count"] = 0
        logger.warning("Generic extraction not implemented - no data extracted")
'''
    
    def _generate_filter_data_step(self, step: Dict[str, Any]) -> str:
        """Generate data filtering step code."""
        criteria = step.get('criteria', '')
        
        return f'''        # Filter data based on criteria
        if "data" not in config:
            raise ValueError("Missing data to filter")
            
        logger.info(f"Filtering data with criteria: {criteria}")
        
        data = config["data"]
        filtered_data = []
        
        # Parse criteria into a filter function
        def matches_criteria(item):
            """Check if an item matches the filter criteria."""
            try:
                # This is a simplified parsing approach - in production you'd want a more robust parser
                if not "{criteria}":
                    return True
                    
                # Parse criteria like 'field contains value'
                if "contains" in "{criteria}":
                    field, value = "{criteria}".split("contains")
                    field = field.strip()
                    value = value.strip().strip("'").strip('"')
                    return field in item and value.lower() in str(item[field]).lower()
                    
                # Parse criteria like 'field = value'
                if "=" in "{criteria}":
                    field, value = "{criteria}".split("=")
                    field = field.strip()
                    value = value.strip().strip("'").strip('"')
                    
                    # Handle numeric comparisons
                    if value.isdigit():
                        return field in item and float(item[field]) == float(value)
                    else:
                        return field in item and item[field] == value
                        
                # Default: pass all items if criteria can't be parsed
                return True
                
            except Exception as e:
                logger.warning(f"Error evaluating criteria: {{str(e)}}")
                return True
        
        # Apply the filter to each item
        for item in data:
            if matches_criteria(item):
                filtered_data.append(item)
        
        result["data"] = filtered_data
        result["count"] = len(filtered_data)
        result["filtered_out"] = len(data) - len(filtered_data)
        logger.info(f"Filtered data from {{len(data)}} to {{len(filtered_data)}} items")
'''
    
    def _generate_transform_data_step(self, step: Dict[str, Any]) -> str:
        """Generate data transformation step code."""
        transformations = step.get('transformations', [])
        
        code = f'''        # Transform data
        import pandas as pd
        
        if "data" not in config:
            raise ValueError("Missing data to transform")
            
        logger.info("Transforming data")
        
        # Convert to pandas DataFrame for easier manipulation
        data = config["data"]
        df = pd.DataFrame(data)
        
        # Apply transformations
'''
        
        # Add transformation logic
        for i, transform in enumerate(transformations):
            transform_type = transform.get('type', '')
            
            if transform_type == 'rename':
                columns = transform.get('columns', {})
                columns_str = ', '.join([f"'{old}': '{new}'" for old, new in columns.items()])
                code += f'''        # Rename columns
        df = df.rename(columns={{{columns_str}}})
        logger.debug("Renamed columns")
'''
            
            elif transform_type == 'drop':
                columns = transform.get('columns', [])
                columns_str = ', '.join([f"'{col}'" for col in columns])
                code += f'''        # Drop columns
        df = df.drop(columns=[{columns_str}], errors='ignore')
        logger.debug("Dropped columns")
'''
            
            elif transform_type == 'calculate':
                column = transform.get('column', '')
                formula = transform.get('formula', '')
                code += f'''        # Calculate new column
        df['{column}'] = {formula}
        logger.debug("Calculated new column: {column}")
'''
        
        code += '''        # Convert back to list of dictionaries
        transformed_data = df.to_dict('records')
        
        result["data"] = transformed_data
        result["count"] = len(transformed_data)
        result["columns"] = list(df.columns)
        logger.info(f"Transformed data with {len(transformed_data)} rows and {len(df.columns)} columns")
'''
        return code
    
    def _generate_save_data_step(self, step: Dict[str, Any]) -> str:
        """Generate data saving step code."""
        format_type = step.get('format', 'csv')
        destination_type = step.get('destination_type', 'local_file')
        destination_value = step.get('destination_value', '')
        
        code = f'''        # Save data
        import pandas as pd
        
        if "data" not in config:
            raise ValueError("Missing data to save")
            
        logger.info("Saving data")
        
        # Convert to pandas DataFrame
        data = config["data"]
        df = pd.DataFrame(data)
        
        # Save based on destination type
'''
        
        if destination_type == 'local_file':
            code += f'''        # Save to local file
        output_path = "{destination_value}" or config.get("output", {{}}).get("destination_value", "output.{format_type}")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save based on format
        if "{format_type}" == "csv":
            df.to_csv(output_path, index=False)
        elif "{format_type}" == "json":
            df.to_json(output_path, orient='records')
        elif "{format_type}" == "excel" or "{format_type}" in ["xls", "xlsx"]:
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
            
        logger.info(f"Data saved to {{output_path}}")
        result["output_path"] = output_path
'''
        
        elif destination_type == 's3':
            code += f'''        # Save to S3
        import boto3
        from io import StringIO, BytesIO
        
        s3_path = "{destination_value}" or config.get("output", {{}}).get("destination_value")
        if not s3_path or not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {{s3_path}}")
        
        # Parse S3 path
        s3_path = s3_path.replace("s3://", "")
        bucket_name, s3_key = s3_path.split("/", 1)
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Save based on format
        if "{format_type}" == "csv":
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
        elif "{format_type}" == "json":
            json_buffer = StringIO()
            df.to_json(json_buffer, orient='records')
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json_buffer.getvalue(),
                ContentType='application/json'
            )
        elif "{format_type}" == "excel" or "{format_type}" in ["xls", "xlsx"]:
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=excel_buffer.getvalue(),
                ContentType=content_type
            )
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
            
        logger.info(f"Data saved to S3: {{s3_path}}")
        result["output_path"] = f"s3://{{s3_path}}"
'''
        
        return code
    
    def _generate_generic_step(self) -> str:
        """Generate generic processing step code."""
        return '''        # Generic processing step
        # TODO: Implement specific logic for this step
        
        # Example: Pass through data from previous step
        if "data" in config:
            result["data"] = config["data"]
            result["count"] = len(config["data"])
            logger.info(f"Processed {len(config['data'])} items")
        else:
            logger.warning("No data available for this step")
'''