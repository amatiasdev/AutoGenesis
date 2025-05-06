"""
Data Processor Module Generator for AutoGenesis Agent Factory.

This module is responsible for generating data processing modules for Target Agents.
"""

import logging
from typing import Dict, Any

from generators.base_module_generator import BaseModuleGenerator

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessorModuleGenerator(BaseModuleGenerator):
    """
    Generates data processing modules for Target Agents.
    """
    
    def generate(self, module_name: str, blueprint: Dict[str, Any]) -> str:
        """
        Generate a data processing module.
        
        Args:
            module_name: Name of the module
            blueprint: Blueprint specifications
            
        Returns:
            Complete Python code for the data processor module
        """
        # Extract data processing requirements
        output_format = blueprint.get('requirements', {}).get('output', {}).get('format', 'csv')
        processing_steps = blueprint.get('requirements', {}).get('processing_steps', [])
        
        # Start with the module header
        code = self.generate_module_header(
            module_name, 
            "Data processing module for transforming and analyzing data"
        )
        
        # Add imports based on output format
        if output_format == 'csv':
            code += 'import csv\n'
        
        code += '''
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class DataProcessor:
    """
    Processes and transforms data between various formats and structures.
    
    This class provides methods for filtering, transforming, and validating data.
    It supports various input and output formats like CSV, JSON, and dataframes.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        logger.info("Initializing DataProcessor")
    
    def load_data(self, source: Union[str, Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
                 format_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from various sources into a pandas DataFrame.
        
        Args:
            source: Data source (file path, dict, list, or DataFrame)
            format_type: Format type if source is a string path ('csv', 'json', etc.)
            
        Returns:
            Loaded data as a pandas DataFrame
            
        Raises:
            ValueError: If source format is invalid or unsupported
            FileNotFoundError: If source file doesn't exist
        """
        logger.info(f"Loading data from {type(source)}")
        
        # If source is already a DataFrame, return it
        if isinstance(source, pd.DataFrame):
            return source
            
        # If source is a dictionary, convert to DataFrame
        if isinstance(source, dict):
            return pd.DataFrame([source])
            
        # If source is a list of dictionaries, convert to DataFrame
        if isinstance(source, list) and all(isinstance(item, dict) for item in source):
            return pd.DataFrame(source)
            
        # If source is a string (file path), load based on format
        if isinstance(source, str):
            if not os.path.exists(source):
                logger.error(f"File not found: {source}")
                raise FileNotFoundError(f"File not found: {source}")
                
            # Determine format from file extension if not specified
            if format_type is None:
                _, ext = os.path.splitext(source)
                format_type = ext.lstrip('.').lower()
                
            # Load based on format
            if format_type == 'csv':
                logger.debug(f"Loading CSV file: {source}")
                return pd.read_csv(source)
            elif format_type in ['json', 'jsonl']:
                logger.debug(f"Loading JSON file: {source}")
                return pd.read_json(source, lines=(format_type == 'jsonl'))
            elif format_type == 'excel' or format_type in ['xls', 'xlsx']:
                logger.debug(f"Loading Excel file: {source}")
                return pd.read_excel(source)
            else:
                logger.error(f"Unsupported file format: {format_type}")
                raise ValueError(f"Unsupported file format: {format_type}")
        
        # If we get here, the source format is invalid
        logger.error(f"Invalid data source type: {type(source)}")
        raise ValueError(f"Invalid data source type: {type(source)}")
    
    def save_data(self, data: pd.DataFrame, destination: str, format_type: Optional[str] = None) -> str:
        """
        Save data to a file in the specified format.
        
        Args:
            data: Data to save as a pandas DataFrame
            destination: File path for the output
            format_type: Output format ('csv', 'json', etc.)
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If format is unsupported
        """
        logger.info(f"Saving data to {destination}")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Converting {type(data)} to DataFrame")
            data = self.load_data(data)
            
        # Determine format from file extension if not specified
        if format_type is None:
            _, ext = os.path.splitext(destination)
            format_type = ext.lstrip('.').lower()
            
        # Save based on format
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        
        if format_type == 'csv':
            logger.debug(f"Saving as CSV: {destination}")
            data.to_csv(destination, index=False)
        elif format_type == 'json':
            logger.debug(f"Saving as JSON: {destination}")
            data.to_json(destination, orient='records')
        elif format_type == 'jsonl':
            logger.debug(f"Saving as JSONL: {destination}")
            data.to_json(destination, orient='records', lines=True)
        elif format_type == 'excel' or format_type in ['xls', 'xlsx']:
            logger.debug(f"Saving as Excel: {destination}")
            data.to_excel(destination, index=False)
        else:
            logger.error(f"Unsupported output format: {format_type}")
            raise ValueError(f"Unsupported output format: {format_type}")
            
        logger.info(f"Data saved successfully to {destination}")
        return destination
    
    def filter_data(self, data: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter data based on specified criteria.
        
        Args:
            data: Data to filter as a DataFrame
            criteria: Filtering criteria (column-value pairs or expressions)
            
        Returns:
            Filtered DataFrame
            
        Example:
            >>> processor.filter_data(df, {"age": {"operator": ">", "value": 30}})
            >>> processor.filter_data(df, {"name": {"operator": "contains", "value": "Smith"}})
        """
        logger.info(f"Filtering data with criteria: {criteria}")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Converting {type(data)} to DataFrame")
            data = self.load_data(data)
            
        filtered_data = data.copy()
        
        for column, condition in criteria.items():
            if column not in filtered_data.columns:
                logger.warning(f"Column '{column}' not found in data. Skipping filter.")
                continue
                
            # Simple equality filter
            if not isinstance(condition, dict):
                logger.debug(f"Applying equality filter: {column} == {condition}")
                filtered_data = filtered_data[filtered_data[column] == condition]
                continue
                
            # Complex filter with operator
            operator = condition.get("operator", "==")
            value = condition.get("value")
            
            if operator == "==":
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == "!=":
                filtered_data = filtered_data[filtered_data[column] != value]
            elif operator == ">":
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == ">=":
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif operator == "<":
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == "<=":
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif operator == "contains":
                filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(str(value), na=False)]
            elif operator == "starts_with":
                filtered_data = filtered_data[filtered_data[column].astype(str).str.startswith(str(value), na=False)]
            elif operator == "ends_with":
                filtered_data = filtered_data[filtered_data[column].astype(str).str.endswith(str(value), na=False)]
            elif operator == "in":
                if isinstance(value, list):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    logger.warning(f"Value for 'in' operator must be a list. Skipping filter.")
            else:
                logger.warning(f"Unsupported operator: {operator}. Skipping filter.")
        
        logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} rows")
        return filtered_data
    
    def transform_data(self, data: pd.DataFrame, transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply a series of transformations to the data.
        
        Args:
            data: Data to transform as a DataFrame
            transformations: List of transformation specifications
            
        Returns:
            Transformed DataFrame
            
        Example:
            >>> transformations = [
            >>>     {"type": "rename", "columns": {"old_name": "new_name"}},
            >>>     {"type": "drop", "columns": ["unwanted_column"]},
            >>>     {"type": "calculate", "column": "total", "formula": "price * quantity"}
            >>> ]
            >>> processor.transform_data(df, transformations)
        """
        logger.info(f"Applying {len(transformations)} transformations to data")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Converting {type(data)} to DataFrame")
            data = self.load_data(data)
            
        transformed_data = data.copy()
        
        for i, transform in enumerate(transformations):
            transform_type = transform.get("type", "")
            logger.debug(f"Applying transformation {i+1}/{len(transformations)}: {transform_type}")
            
            try:
                if transform_type == "rename":
                    columns = transform.get("columns", {})
                    transformed_data = transformed_data.rename(columns=columns)
                    
                elif transform_type == "drop":
                    columns = transform.get("columns", [])
                    transformed_data = transformed_data.drop(columns=columns, errors='ignore')
                    
                elif transform_type == "select":
                    columns = transform.get("columns", [])
                    transformed_data = transformed_data[columns]
                    
                elif transform_type == "calculate":
                    column = transform.get("column", "")
                    formula = transform.get("formula", "")
                    
                    if column and formula:
                        # Replace column names with data references
                        for col in transformed_data.columns:
                            formula = formula.replace(col, f"transformed_data['{col}']")
                            
                        # Evaluate the formula
                        transformed_data[column] = eval(formula)
                    else:
                        logger.warning(f"Missing column or formula for calculate transformation")
                        
                elif transform_type == "convert":
                    column = transform.get("column", "")
                    dtype = transform.get("dtype", "")
                    
                    if column and dtype:
                        if dtype == "int":
                            transformed_data[column] = pd.to_numeric(transformed_data[column], errors='coerce').astype('Int64')
                        elif dtype == "float":
                            transformed_data[column] = pd.to_numeric(transformed_data[column], errors='coerce')
                        elif dtype == "str" or dtype == "string":
                            transformed_data[column] = transformed_data[column].astype(str)
                        elif dtype == "bool" or dtype == "boolean":
                            transformed_data[column] = transformed_data[column].astype(bool)
                        elif dtype == "date":
                            transformed_data[column] = pd.to_datetime(transformed_data[column], errors='coerce')
                        else:
                            logger.warning(f"Unsupported dtype: {dtype}")
                    else:
                        logger.warning(f"Missing column or dtype for convert transformation")
                        
                elif transform_type == "fill_na":
                    columns = transform.get("columns", transformed_data.columns.tolist())
                    value = transform.get("value", "")
                    
                    if isinstance(columns, str):
                        columns = [columns]
                        
                    for col in columns:
                        if col in transformed_data.columns:
                            transformed_data[col] = transformed_data[col].fillna(value)
                
                elif transform_type == "custom":
                    # Custom transformation using a lambda function
                    code = transform.get("code", "")
                    if code:
                        # Execute custom code (careful with security implications)
                        local_vars = {"df": transformed_data}
                        exec(code, {"pd": pd}, local_vars)
                        transformed_data = local_vars["df"]
                    else:
                        logger.warning("Missing code for custom transformation")
                
                else:
                    logger.warning(f"Unsupported transformation type: {transform_type}")
                    
            except Exception as e:
                logger.error(f"Error applying transformation {transform_type}: {str(e)}")
                logger.error(f"Transformation details: {transform}")
                # Continue with other transformations
        
        logger.info(f"Data transformation completed")
        return transformed_data
    
    def validate_data(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate as a DataFrame
            schema: Validation schema (column definitions and rules)
            
        Returns:
            Validation results with issues found
            
        Example:
            >>> schema = {
            >>>     "columns": {
            >>>         "age": {"type": "int", "min": 0, "max": 120},
            >>>         "email": {"type": "str", "pattern": r"[^@]+@[^@]+\\.[^@]+"}
            >>>     },
            >>>     "required": ["name", "email"]
            >>> }
            >>> processor.validate_data(df, schema)
        """
        logger.info("Validating data against schema")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.warning(f"Converting {type(data)} to DataFrame")
            data = self.load_data(data)
            
        # Initialize validation results
        validation_results = {
            "valid": True,
            "issues": [],
            "missing_columns": [],
            "type_errors": [],
            "constraint_violations": [],
            "row_count": len(data)
        }
        
        # Check required columns
        required_columns = schema.get("required", [])
        for column in required_columns:
            if column not in data.columns:
                validation_results["valid"] = False
                validation_results["missing_columns"].append(column)
                validation_results["issues"].append(f"Required column '{column}' is missing")
        
        # Validate column types and constraints
        column_schemas = schema.get("columns", {})
        for column, column_schema in column_schemas.items():
            if column not in data.columns:
                if column in required_columns:
                    # Already reported as missing required column
                    continue
                else:
                    # Optional column is missing, no issue
                    continue
            
            # Validate column type
            expected_type = column_schema.get("type", "str")
            
            # Check type based on pandas dtypes
            valid_type = True
            if expected_type == "int":
                valid_type = pd.api.types.is_numeric_dtype(data[column])
            elif expected_type == "float":
                valid_type = pd.api.types.is_float_dtype(data[column])
            elif expected_type == "str" or expected_type == "string":
                valid_type = pd.api.types.is_string_dtype(data[column]) or pd.api.types.is_object_dtype(data[column])
            elif expected_type == "bool" or expected_type == "boolean":
                valid_type = pd.api.types.is_bool_dtype(data[column])
            elif expected_type == "date":
                valid_type = pd.api.types.is_datetime64_dtype(data[column])
            
            if not valid_type:
                validation_results["valid"] = False
                validation_results["type_errors"].append(column)
                validation_results["issues"].append(f"Column '{column}' should be type '{expected_type}'")
            
            # Validate constraints
            # Numeric constraints
            if expected_type in ["int", "float"] and valid_type:
                # Check min value
                if "min" in column_schema:
                    min_value = column_schema["min"]
                    min_violations = data[data[column] < min_value]
                    if len(min_violations) > 0:
                        validation_results["valid"] = False
                        validation_results["constraint_violations"].append({
                            "column": column,
                            "constraint": "min",
                            "value": min_value,
                            "violation_count": len(min_violations)
                        })
                        validation_results["issues"].append(
                            f"Column '{column}' has {len(min_violations)} values below minimum {min_value}"
                        )
                
                # Check max value
                if "max" in column_schema:
                    max_value = column_schema["max"]
                    max_violations = data[data[column] > max_value]
                    if len(max_violations) > 0:
                        validation_results["valid"] = False
                        validation_results["constraint_violations"].append({
                            "column": column,
                            "constraint": "max",
                            "value": max_value,
                            "violation_count": len(max_violations)
                        })
                        validation_results["issues"].append(
                            f"Column '{column}' has {len(max_violations)} values above maximum {max_value}"
                        )
            
            # String pattern matching
            if expected_type in ["str", "string"] and valid_type and "pattern" in column_schema:
                import re
                pattern = column_schema["pattern"]
                pattern_re = re.compile(pattern)
                
                # Check each value against the pattern
                pattern_violations = data[~data[column].astype(str).str.match(pattern)]
                if len(pattern_violations) > 0:
                    validation_results["valid"] = False
                    validation_results["constraint_violations"].append({
                        "column": column,
                        "constraint": "pattern",
                        "value": pattern,
                        "violation_count": len(pattern_violations)
                    })
                    validation_results["issues"].append(
                        f"Column '{column}' has {len(pattern_violations)} values not matching pattern '{pattern}'"
                    )
        
        logger.info(f"Data validation completed. Valid: {validation_results['valid']}")
        if not validation_results["valid"]:
            logger.warning(f"Validation issues: {validation_results['issues']}")
            
        return validation_results
    
    def merge_data(self, left_data: pd.DataFrame, right_data: pd.DataFrame, 
                  on: Union[str, List[str]], how: str = "inner") -> pd.DataFrame:
        """
        Merge two datasets.
        
        Args:
            left_data: Left DataFrame
            right_data: Right DataFrame
            on: Column(s) to join on
            how: Join type ('inner', 'left', 'right', 'outer')
            
        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging data with {how} join on {on}")
        
        # Ensure both inputs are DataFrames
        if not isinstance(left_data, pd.DataFrame):
            left_data = self.load_data(left_data)
        if not isinstance(right_data, pd.DataFrame):
            right_data = self.load_data(right_data)
            
        # Perform merge
        result = pd.merge(left_data, right_data, on=on, how=how)
        
        logger.info(f"Merged data has {len(result)} rows and {len(result.columns)} columns")
        return result
    
    def group_data(self, data: pd.DataFrame, by: Union[str, List[str]], 
                  aggregations: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """
        Group data and apply aggregations.
        
        Args:
            data: Data to group as a DataFrame
            by: Column(s) to group by
            aggregations: Aggregation functions to apply to columns
            
        Returns:
            Grouped and aggregated DataFrame
            
        Example:
            >>> processor.group_data(df, by="category", 
            >>>                       aggregations={"price": ["mean", "sum"], "quantity": "sum"})
        """
        logger.info(f"Grouping data by {by} with aggregations: {aggregations}")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = self.load_data(data)
            
        # Convert aggregations to the format expected by pandas
        agg_dict = {}
        for column, aggs in aggregations.items():
            if isinstance(aggs, str):
                agg_dict[column] = [aggs]
            else:
                agg_dict[column] = aggs
                
        # Group and aggregate data
        grouped = data.groupby(by).agg(agg_dict)
        
        # Flatten multi-level column index if present
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
            
        # Reset index to make the groupby column(s) regular columns again
        result = grouped.reset_index()
        
        logger.info(f"Grouped data has {len(result)} rows and {len(result.columns)} columns")
        return result
'''
        
        return code