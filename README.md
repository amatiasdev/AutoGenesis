# AutoGenesis (Agent Factory)

AutoGenesis is a meta-agent designed to automate the creation, testing, and deployment of specialized software agents. It accepts task requirements via natural language, API calls, or UI commands, designs an appropriate agent architecture, generates functional Python code, validates the agent in a sandbox environment, and outputs a deployable package.

## Overview

The AutoGenesis system accepts input in the form of natural language descriptions or structured API requests, and then:

1. Processes and understands the requirements
2. Decomposes the task into sub-processes and atomic actions
3. Designs an agent architecture and selects appropriate tools
4. Generates functional Python code
5. Tests the agent in a sandbox environment
6. Evaluates the generated agent using Multi-Perspective Criticism (MPC)
7. Packages the agent for deployment

## Key Features

- **Natural Language Understanding**: Accept and parse task requirements in natural language
- **Automated Agent Design**: Intelligently select tools and design agent architecture
- **Code Generation**: Generate well-structured, well-documented Python code
- **Sandbox Testing**: Test generated agents in an isolated Docker environment
- **Multi-Perspective Criticism (MPC)**: Evaluate agents for security, scalability, maintainability, and cost
- **Self-Improvement**: Learn from MPC results to improve future agent generation
- **Flexible Deployment**: Package agents as Docker containers, AWS Lambda functions, or standalone scripts

## System Architecture

The AutoGenesis system consists of the following main components:

- **Input Processing**: Processes natural language or API input and clarifies requirements
- **Blueprint Design**: Creates a blueprint for the agent architecture and tool selection
- **Code Generation**: Generates Python code based on the blueprint
- **Sandbox Testing**: Tests the generated code in an isolated Docker environment
- **Multi-Perspective Criticism (MPC)**: Evaluates the agent across multiple criteria
- **Agent Packaging**: Packages the agent for deployment

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker
- AWS credentials (if using AWS services)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autogenesis.git
   cd autogenesis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export AWS_REGION=us-east-1
   export AUTOGENESIS_MPC_TABLE=AutoGenesisMPCLogs
   export LOG_LEVEL=INFO
   ```

### Usage

#### Using Natural Language Input

To generate an agent using natural language input:

```bash
python autogenesis/main.py --mode nl --text "Create an agent that scrapes product information from Amazon and saves it to a CSV file."
```

Alternatively, you can provide the natural language input in a file:

```bash
python autogenesis/main.py --mode nl --file requirements.txt
```

#### Using API Input

To generate an agent using structured API input:

```bash
python autogenesis/main.py --mode api --file api_request.json
```

Where `api_request.json` follows the format specified in the API documentation.

#### Output

The output of the agent generation process will be placed in the `./output` directory by default. You can specify a different output directory using the `--output` parameter:

```bash
python autogenesis/main.py --mode nl --text "..." --output /path/to/output
```

## API Documentation

### Agent Generation API

**Endpoint:** `/generate_agent` (when deployed as a service)

**Method:** POST

**Payload Schema:**

```json
{
  "task_type": "web_scraping", 
  "user_description_nl": "Optional: User's original NL request for logging/context",
  "requirements": {
    "source": { 
      "type": "url", 
      "value": "example.com/products"
    },
    "processing_steps": [ 
      {"action": "authenticate", "credentials_secret_name": "example-creds"},
      {"action": "extract_data", "fields": ["name", "price", "description"]}
    ],
    "output": { 
      "format": "csv", 
      "destination_type": "s3", 
      "destination_value": "s3://my-bucket/example-data.csv"
    },
    "constraints": { 
      "rate_limit_per_minute": 5,
      "run_schedule": "daily@3am", 
      "max_runtime_seconds": 3600 
    },
    "preferred_tools": ["selenium", "pandas"], 
    "deployment_format": ["docker", "lambda"] 
  }
}
```

## Multi-Perspective Criticism (MPC)

AutoGenesis uses Multi-Perspective Criticism (MPC) to evaluate generated agents across several dimensions:

1. **Security**: Checks for common security vulnerabilities and issues
2. **Scalability**: Evaluates the agent's ability to handle increasing load and data volume
3. **Maintainability**: Assesses code quality, documentation, and adherence to best practices
4. **Cost**: Estimates operational costs and resource utilization

MPC results are stored in DynamoDB for future learning and improvement.

## Architecture Types

AutoGenesis can generate agents with different architectural patterns:

1. **Simple Script**: Single script implementation for simple tasks
2. **Modular**: Modular implementation with separate modules for different functions
3. **Pipeline**: Data pipeline architecture for sequential processing steps
4. **Service**: Service-based architecture with API endpoints
5. **Event-driven**: Event-driven architecture for handling asynchronous operations

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black autogenesis/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- This project was inspired by the need for automated agent generation for common automation tasks.
- Special thanks to all contributors who have helped shape this project.