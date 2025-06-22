# pip install langgraph-codeact "langchain[anthropic]"
import asyncio
import inspect
import uuid
import tempfile
import sys
from typing import Any, Callable, Tuple

from langchain.chat_models import init_chat_model
from langchain_sandbox import PyodideSandbox

from langgraph_codeact import create_codeact

# -------------------------------------------------
# MUST be the first asyncio-related call on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# -------------------------------------------------

# Define the type for eval functions
EvalFunction = Callable[[str, dict[str, Any]], Tuple[str, dict[str, Any]]]


def create_pyodide_eval_fn(sandbox: PyodideSandbox) -> EvalFunction:
    """Create an eval_fn that uses PyodideSandbox.
    """

    def sync_eval_fn(
        code: str, _locals: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        # Run the async function in a new event loop
        return asyncio.run(async_eval_fn(code, _locals))
    
    async def async_eval_fn(
        code: str, _locals: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        # Create a wrapper function that will execute the code and return locals
        wrapper_code = f"""
def execute():
    try:
        # Execute the provided code
{chr(10).join("        " + line for line in code.strip().split(chr(10)))}
        return locals()
    except Exception as e:
        return {{"error": str(e)}}

execute()
"""
        
        # Define all tools directly in the execution context
        tools_code = """
import math

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def subtract(a, b):
    return a - b

def sin(a):
    return math.sin(a)

def cos(a):
    return math.cos(a)

def radians(a):
    return math.radians(a)

def exponentiation(a, b):
    return a ** b

def sqrt(a):
    return math.sqrt(a)

def ceil(a):
    return math.ceil(a)
"""
        
        # Add other variables from _locals
        context_setup = ""
        for key, value in _locals.items():
            if not callable(value):
                context_setup += f"\n{key} = {repr(value)}"

        # Combine everything
        full_code = tools_code + context_setup + "\n\n" + wrapper_code

        try:
            # Execute the code and get the result
            response = await sandbox.execute(code=full_code)

            # Check if execution was successful
            if response.stderr:
                return f"Error during execution: {response.stderr}", {}

            # Get the output from stdout
            output = (
                response.stdout
                if response.stdout
                else "<Code ran, no output printed to stdout>"
            )
            result = response.result

            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                return f"Error during execution: {result['error']}", {}

            # Get the new variables by comparing with original locals
            # Exclude tool names from new variables
            tool_names = {"add", "multiply", "divide", "subtract", "sin", "cos", "radians", "exponentiation", "sqrt", "ceil", "math"}
            new_vars = {
                k: v
                for k, v in result.items()
                if k not in _locals and not k.startswith("_") and k not in tool_names
            }
            return output, new_vars

        except Exception as e:
            return f"Error during PyodideSandbox execution: {repr(e)}", {}

    return sync_eval_fn


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


def sin(a: float) -> float:
    """Take the sine of a number."""
    import math

    return math.sin(a)


def cos(a: float) -> float:
    """Take the cosine of a number."""
    import math

    return math.cos(a)


def radians(a: float) -> float:
    """Convert degrees to radians."""
    import math

    return math.radians(a)


def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b


def sqrt(a: float) -> float:
    """Take the square root of a number."""
    import math

    return math.sqrt(a)


def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    import math

    return math.ceil(a)


tools = [
    add,
    multiply,
    divide,
    subtract,
    sin,
    cos,
    radians,
    exponentiation,
    sqrt,
    ceil,
]

model = init_chat_model("o4-mini-2025-04-16", model_provider="openai")

# Create a temporary directory for sandbox sessions
sessions_dir = tempfile.mkdtemp()
sandbox = PyodideSandbox(sessions_dir=sessions_dir, allow_net=True)
eval_fn = create_pyodide_eval_fn(sandbox)
code_act = create_codeact(model, tools, eval_fn)
agent = code_act.compile()

