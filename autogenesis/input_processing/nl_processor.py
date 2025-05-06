"""
Natural Language Processor for AutoGenesis.

This module handles the processing of natural language input to extract structured
requirements for agent generation.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

import spacy
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


class NLProcessor:
    """
    Processes natural language input to extract structured requirements.
    
    Uses NLP techniques to identify intent, entities, and requirements from
    free-form text input.
    """
    
    def __init__(self):
        """Initialize the NL processor with required models and tools."""
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_lg")
            self.logger.info("Loaded spaCy model successfully")
        except Exception as e:
            self.logger.warning(f"Could not load spaCy model: {e}. Attempting to download...")
            try:
                import subprocess
                result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], 
                                    capture_output=True, timeout=120)
                if result.returncode == 0:
                    self.nlp = spacy.load("en_core_web_lg")
                    self.logger.info("Downloaded and loaded spaCy model successfully")
                else:
                    raise Exception(f"Download failed: {result.stderr.decode()}")
            except Exception as download_error:
                self.logger.warning(f"Could not download model: {download_error}. Using fallback...")
                try:
                    # Try smaller model
                    self.nlp = spacy.load("en_core_web_sm")
                    self.logger.info("Using smaller spaCy model as fallback")
                except Exception:
                    # Last resort: blank model
                    self.nlp = spacy.blank("en")
                    self.logger.warning("Using blank spaCy model - NLP capabilities will be limited")
        
        # Initialize transformer model for more complex understanding
        try:
            model_name = "google/flan-t5-base"  # We can use a more specialized model if needed
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.seq2seq = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
            self.logger.info(f"Loaded transformer model {model_name} successfully")
        except Exception as e:
            self.logger.error(f"Error loading transformer model: {e}")
            self.seq2seq = None
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process natural language input and extract structured requirements.
        
        Args:
            text: The natural language input text
            
        Returns:
            Dict containing structured requirements and metadata
        """
        self.logger.info(f"Processing NL input: {text[:100]}...")
        
        # Basic text preprocessing
        clean_text = self._preprocess_text(text)
        
        # Extract task type and primary goal
        task_type, goal = self._extract_task_type_and_goal(clean_text)
        
        # Extract entities (sources, destinations, tools)
        entities = self._extract_entities(clean_text)
        
        # Extract processing steps if mentioned
        steps = self._extract_processing_steps(clean_text)
        
        # Extract constraints
        constraints = self._extract_constraints(clean_text)
        
        # Check for ambiguities or missing critical information
        ambiguities, is_clear = self._check_for_ambiguities(
            task_type, goal, entities, steps, constraints
        )
        
        # Structure the requirements in the expected format
        requirements = self._structure_requirements(
            task_type, goal, entities, steps, constraints
        )
        
        result = {
            "requirements": requirements,
            "needs_clarification": not is_clear,
            "clarification_questions": ambiguities if not is_clear else [],
            "original_text": text,
            "extracted_metadata": {
                "task_type": task_type,
                "goal": goal,
                "entities": entities,
                "steps": steps,
                "constraints": constraints
            }
        }
        
        self.logger.info(f"NL processing complete. Needs clarification: {not is_clear}")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text for better NLP processing.
        
        Args:
            text: The raw input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add more preprocessing as needed
        
        return text
    
    def _extract_task_type_and_goal(self, text: str) -> Tuple[str, str]:
        """
        Extract the task type and primary goal from the text.
        
        Args:
            text: The preprocessed input text
            
        Returns:
            Tuple of (task_type, goal)
        """
        # Use transformer model if available
        if self.seq2seq:
            # Generate a structured query
            query = f"Extract the task type and main goal from: {text}\nTask type:"
            response = self.seq2seq(query, max_length=50)
            task_type_text = response[0]['generated_text']
            
            query = f"What is the main goal or objective in this task: {text}\nMain goal:"
            response = self.seq2seq(query, max_length=100)
            goal_text = response[0]['generated_text']
            
            # Clean up responses
            task_type = task_type_text.strip()
            goal = goal_text.strip()
            
            # Map to predefined task types
            task_type = self._map_to_task_type(task_type)
        else:
            # Fallback to rule-based extraction
            task_type = "unknown"
            goal = text[:100]  # Just use first 100 chars as goal
            
            # Check for task type keywords
            task_type_keywords = {
                "web_scraping": ["scrape", "extract from website", "crawler", "scraper"],
                "data_processing": ["process data", "transform data", "analyze data", "compute", "calculate"],
                "api_integration": ["api", "integrate with", "connect to service", "webhook"],
                "file_processing": ["process files", "parse files", "read files", "convert files"],
                "automation": ["automate", "schedule", "routine", "repeatedly", "periodic"]
            }
            
            for t_type, keywords in task_type_keywords.items():
                if any(kw.lower() in text.lower() for kw in keywords):
                    task_type = t_type
                    break
        
        self.logger.debug(f"Extracted task type: {task_type}, goal: {goal[:50]}...")
        return task_type, goal
    
    def _map_to_task_type(self, extracted_type: str) -> str:
        """Map extracted task type to predefined types."""
        predefined_types = [
            "web_scraping", "data_processing", "file_processing", 
            "api_integration", "ui_automation", "email_processing",
            "data_extraction", "etl", "reporting", "notification"
        ]
        
        # Try direct match first
        for p_type in predefined_types:
            if p_type.lower() in extracted_type.lower():
                return p_type
        
        # Fallback mapping based on keywords
        keywords_map = {
            "web_scraping": ["scrape", "crawl", "extract from web", "html"],
            "data_processing": ["process", "transform", "clean", "analyze", "filter"],
            "api_integration": ["api", "integration", "service", "endpoint", "webhook"],
            "file_processing": ["file", "document", "pdf", "excel", "csv"],
            "ui_automation": ["ui", "interface", "click", "automate browser", "selenium"],
            "email_processing": ["email", "outlook", "gmail", "message", "mail"]
        }
        
        for p_type, keywords in keywords_map.items():
            if any(kw.lower() in extracted_type.lower() for kw in keywords):
                return p_type
        
        # Default if no match
        return "data_processing"
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities like sources, destinations, and tools from text.
        
        Args:
            text: The preprocessed input text
            
        Returns:
            Dict of extracted entities
        """
        entities = {
            "sources": [],
            "destinations": [],
            "tools": []
        }
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract URLs, file paths, and services
        for ent in doc.ents:
            if ent.label_ == "URL" or ent.label_ == "ORG":
                # Check if it appears to be a source or destination based on context
                context_start = max(0, ent.start - 5)
                context_end = min(len(doc), ent.end + 5)
                context = doc[context_start:context_end].text.lower()
                
                if any(w in context for w in ["from", "source", "input", "read"]):
                    entities["sources"].append({
                        "type": "url" if ent.label_ == "URL" else "service",
                        "value": ent.text
                    })
                elif any(w in context for w in ["to", "destination", "output", "write", "save"]):
                    entities["destinations"].append({
                        "type": "url" if ent.label_ == "URL" else "service",
                        "value": ent.text
                    })
                else:
                    # Default to source if unclear
                    entities["sources"].append({
                        "type": "url" if ent.label_ == "URL" else "service",
                        "value": ent.text
                    })
        
        # Extract file formats and potential tool preferences
        file_formats = re.findall(r'\b(csv|json|excel|pdf|xml|yaml|txt|html)\b', text.lower())
        for fmt in file_formats:
            # Check if it's being used as input or output
            if f"from {fmt}" in text.lower() or f"read {fmt}" in text.lower():
                entities["sources"].append({"type": "file", "format": fmt})
            elif f"to {fmt}" in text.lower() or f"save as {fmt}" in text.lower() or f"write {fmt}" in text.lower():
                entities["destinations"].append({"type": "file", "format": fmt})
        
        # Extract potential tool preferences
        tools = ["pandas", "selenium", "beautifulsoup", "requests", "scrapy", 
                 "openpyxl", "tensorflow", "sklearn", "pytorch", "flask"]
        for tool in tools:
            if tool.lower() in text.lower():
                entities["tools"].append(tool)
        
        # Attempt to extract S3 buckets or other AWS resources
        s3_buckets = re.findall(r's3://[\w.-]+', text)
        for bucket in s3_buckets:
            entities["destinations"].append({"type": "s3", "value": bucket})
        
        self.logger.debug(f"Extracted entities: {json.dumps(entities)}")
        return entities
    
    def _extract_processing_steps(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract processing steps from the text.
        
        Args:
            text: The preprocessed input text
            
        Returns:
            List of processing steps
        """
        steps = []
        
        # Use the transformer for step extraction if available
        if self.seq2seq:
            query = f"List the processing steps needed for this task, one per line: {text}"
            response = self.seq2seq(query, max_length=200)
            steps_text = response[0]['generated_text']
            
            # Parse the generated steps
            for line in steps_text.strip().split('\n'):
                if line and not line.isspace():
                    # Extract action and parameters
                    action_match = re.match(r'(.*?)(?:\s*\((.*)\))?$', line.strip())
                    if action_match:
                        action = action_match.group(1).strip().lower()
                        params_str = action_match.group(2) if action_match.group(2) else ""
                        
                        # Convert to snake_case
                        action = re.sub(r'\s+', '_', action)
                        
                        # Create step with action
                        step = {"action": action}
                        
                        # Add parameters if any
                        if params_str:
                            params = {}
                            param_matches = re.findall(r'(\w+)=([^,]+)(?:,|$)', params_str)
                            for param_name, param_value in param_matches:
                                params[param_name] = param_value.strip()
                            
                            if params:
                                step.update(params)
                        
                        steps.append(step)
        else:
            # Simple rule-based extraction
            # Look for numbered lists or bullet points
            step_patterns = [
                r'\d+\.\s*(.*?)(?=\d+\.|$)',  # Numbered lists: 1. Step one 2. Step two
                r'[-•*]\s*(.*?)(?=[-•*]|$)'    # Bullet points: - Step one - Step two
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    for match in matches:
                        step_text = match.strip()
                        if step_text:
                            # Basic parsing to get action
                            action = step_text.split(' ')[0].lower()
                            steps.append({"action": action, "description": step_text})
        
        self.logger.debug(f"Extracted {len(steps)} processing steps")
        return steps
    
    def _extract_constraints(self, text: str) -> Dict[str, Any]:
        """
        Extract constraints like timing, frequency, formats from text.
        
        Args:
            text: The preprocessed input text
            
        Returns:
            Dict of constraints
        """
        constraints = {}
        
        # Extract rate limits
        rate_limit_patterns = [
            r'(?:rate limit|limit|max|maximum)\s+(\d+)\s+(?:per|\/)\s+(second|minute|hour|day)',
            r'(\d+)\s+(?:requests|calls)\s+(?:per|\/)\s+(second|minute|hour|day)'
        ]
        
        for pattern in rate_limit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value, unit = matches[0]
                key = f"rate_limit_per_{unit}"
                constraints[key] = int(value)
        
        # Extract scheduling/timing
        schedule_patterns = [
            r'(daily|weekly|monthly|hourly|every day|every hour|every minute)'
        ]
        
        for pattern in schedule_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                schedule = matches[0].lower()
                if schedule == "daily":
                    constraints["run_schedule"] = "daily@00:00"
                elif schedule == "hourly":
                    constraints["run_schedule"] = "hourly"
                elif schedule == "weekly":
                    constraints["run_schedule"] = "weekly@monday"
                elif schedule == "monthly":
                    constraints["run_schedule"] = "monthly@1"
                else:
                    constraints["run_schedule"] = schedule
        
        # Extract specific time for schedule
        time_patterns = [
            r'at\s+(\d{1,2}(?::\d{2})?)\s*([ap]m)?',
            r'(\d{1,2}(?::\d{2}))\s*([ap]m)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                time_str, period = matches[0]
                # Parse time to 24-hour format
                if ":" not in time_str:
                    time_str += ":00"
                
                hour, minute = map(int, time_str.split(':'))
                if period and period.lower() == "pm" and hour < 12:
                    hour += 12
                
                time_formatted = f"{hour:02d}:{minute:02d}"
                
                # Update existing schedule or create new one
                if "run_schedule" in constraints:
                    if "@" in constraints["run_schedule"]:
                        base_schedule = constraints["run_schedule"].split('@')[0]
                        constraints["run_schedule"] = f"{base_schedule}@{time_formatted}"
                    else:
                        constraints["run_schedule"] += f"@{time_formatted}"
                else:
                    constraints["run_schedule"] = f"daily@{time_formatted}"
        
        # Extract max runtime
        runtime_patterns = [
            r'(?:runtime|run time|execution time|timeout|time out)\s*(?:of)?\s*(\d+)\s*(second|minute|hour|day)',
            r'(?:complete|finish|run)\s+(?:in|within)\s+(\d+)\s*(second|minute|hour|day)'
        ]
        
        for pattern in runtime_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value, unit = matches[0]
                # Convert to seconds
                multiplier = {
                    "second": 1,
                    "minute": 60,
                    "hour": 3600,
                    "day": 86400
                }
                constraints["max_runtime_seconds"] = int(value) * multiplier.get(unit, 1)
        
        self.logger.debug(f"Extracted constraints: {constraints}")
        return constraints
    
    def _check_for_ambiguities(self,
                              task_type: str,
                              goal: str,
                              entities: Dict[str, Any],
                              steps: List[Dict[str, Any]],
                              constraints: Dict[str, Any]) -> Tuple[List[str], bool]:
        """
        Check for ambiguities or missing information that requires clarification.
        
        Args:
            task_type: Extracted task type
            goal: Extracted goal
            entities: Extracted entities
            steps: Extracted processing steps
            constraints: Extracted constraints
            
        Returns:
            Tuple of (list of clarification questions, is_clear flag)
        """
        questions = []
        
        # Check for critical missing information based on task type
        if task_type == "unknown":
            questions.append("What type of task do you need to automate? (e.g., web scraping, data processing, API integration)")
        
        # Check for missing sources/destinations
        if not entities.get("sources"):
            questions.append("What is the source of data for this task? (e.g., a website URL, API, file)")
        
        if not entities.get("destinations"):
            questions.append("Where should the output of this task be stored? (e.g., CSV file, database, API)")
        
        # For web scraping, check for missing details
        if task_type == "web_scraping" and entities.get("sources"):
            has_authentication = any("authenticate" in str(step.get("action", "")).lower() for step in steps)
            if not has_authentication:
                questions.append("Does the website require authentication? If yes, how should the agent authenticate?")
            
            has_data_extraction = any("extract" in str(step.get("action", "")).lower() for step in steps)
            if not has_data_extraction:
                questions.append("What specific data elements need to be extracted from the website?")
        
        # For API integration, check for missing details
        if task_type == "api_integration" and entities.get("sources"):
            has_authentication = any("authenticate" in str(step.get("action", "")).lower() for step in steps)
            if not has_authentication:
                questions.append("Does the API require authentication? If yes, what authentication method is used?")
        
        # Check for missing scheduling information if it seems like a recurring task
        if "automate" in goal.lower() or "schedule" in goal.lower() or "periodic" in goal.lower():
            if not constraints.get("run_schedule"):
                questions.append("How often should this task run? (e.g., daily, hourly, weekly)")
        
        is_clear = len(questions) == 0
        return questions, is_clear
    
    def _structure_requirements(self,
                               task_type: str,
                               goal: str,
                               entities: Dict[str, Any],
                               steps: List[Dict[str, Any]],
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure the extracted information into the format expected by the Blueprint Designer.
        
        Args:
            task_type: Extracted task type
            goal: Extracted goal
            entities: Extracted entities
            steps: Extracted processing steps
            constraints: Extracted constraints
            
        Returns:
            Dict containing structured requirements
        """
        # Build source object
        source = None
        if entities.get("sources"):
            source_entity = entities["sources"][0]  # Take first source for now
            source = {
                "type": source_entity.get("type", "unknown"),
                "value": source_entity.get("value", "")
            }
            
            if "format" in source_entity:
                source["format"] = source_entity["format"]
        
        # Build output object
        output = None
        if entities.get("destinations"):
            dest_entity = entities["destinations"][0]  # Take first destination for now
            output = {
                "format": dest_entity.get("format", "csv"),
                "destination_type": dest_entity.get("type", "local_file"),
                "destination_value": dest_entity.get("value", "output.csv")
            }
        
        # Build requirements structure
        requirements = {
            "task_type": task_type,
            "description": goal,
            "requirements": {
                "source": source,
                "processing_steps": steps,
                "output": output,
                "constraints": constraints,
                "preferred_tools": entities.get("tools", []),
                "deployment_format": ["docker", "script"]  # Default deployment formats
            }
        }
        
        return requirements