# CodeAct Multi-Agent Discovery

A LangGraph application that provides a code-acting agent with mathematical and computational capabilities using PyodideSandbox.

## Project Structure

```
code_act_discovery/
├── codeact_multi_agent/        # Main application package
│   ├── __init__.py            # Package initialization
│   ├── graph.py               # Main agent graph definition
│   └── utils.py               # Utility functions
├── .env                       # Environment variables (create this)
├── requirements.txt           # Python dependencies
├── langgraph.json            # LangGraph configuration
└── README.md                 # This file
```

## Features

- **Code Execution**: Uses PyodideSandbox for safe code execution
- **Mathematical Tools**: Built-in math functions (add, multiply, divide, etc.)
- **Multi-Model Support**: Configurable with OpenAI models
- **Async Support**: Windows-compatible async execution

## Setup for Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```env
   # OpenAI API Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Anthropic API Configuration (optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # LangSmith Configuration (optional, for tracing)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=code-act-discovery
   ```

4. Test locally:
   ```bash
   langgraph dev
   ```

## Deployment to LangGraph Cloud

This project is configured for deployment to LangGraph Cloud (managed SaaS).

### Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **LangGraph CLI**: Install with `pip install langgraph-cli`
3. **Local Testing**: Ensure `langgraph dev` works locally

### Configuration Files

- **`langgraph.json`**: Defines the deployment configuration
- **`requirements.txt`**: Specifies Python dependencies
- **`.env`**: Environment variables (create this file)

### Deployment Steps

1. **Push to GitHub**: Ensure your code is in a GitHub repository

2. **LangSmith Setup**: 
   - Go to [LangSmith UI](https://smith.langchain.com/)
   - Install LangChain's `hosted-langserve` GitHub app

3. **Create Deployment**:
   - Navigate to `LangGraph Platform` in LangSmith
   - Click `+ New Deployment`
   - Select your GitHub repository
   - Set config file path: `langgraph.json`
   - Choose deployment type (Development/Production)
   - Add environment variables (API keys)

4. **Monitor**: Track build logs and deployment status in the LangSmith UI

## Environment Variables

Required for deployment:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (if using Anthropic models)
- `LANGCHAIN_API_KEY`: For LangSmith tracing (optional)

## Dependencies

The project uses these key dependencies:

- `langgraph-codeact`: CodeAct agent implementation
- `langchain[anthropic]`: LangChain with Anthropic support
- `langchain-sandbox`: PyodideSandbox for code execution
- `langchain-openai`: OpenAI integration
- `langchain-core`: Core LangChain functionality

## Usage

Once deployed, your agent can:

1. Execute Python code safely in a sandbox environment
2. Perform mathematical calculations
3. Handle complex computational tasks
4. Maintain conversation context

## Available Tools

The agent has access to these mathematical functions:

- `add(a, b)`: Addition
- `multiply(a, b)`: Multiplication
- `divide(a, b)`: Division
- `subtract(a, b)`: Subtraction
- `sin(a)`: Sine function
- `cos(a)`: Cosine function
- `radians(a)`: Convert degrees to radians
- `exponentiation(a, b)`: Power function
- `sqrt(a)`: Square root
- `ceil(a)`: Ceiling function

## License

This project is configured for deployment on LangGraph Cloud. 