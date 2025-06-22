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

# Import tools - handle both relative and absolute imports
try:
    from .utils import tools
except ImportError:
    from codeact_multi_agent.utils import tools

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
import statistics

# Basic arithmetic
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def subtract(a, b):
    return a - b

# Trigonometric functions
def sin(a):
    return math.sin(a)

def cos(a):
    return math.cos(a)

def tan(a):
    return math.tan(a)

def asin(a):
    return math.asin(a)

def acos(a):
    return math.acos(a)

def atan(a):
    return math.atan(a)

def atan2(y, x):
    return math.atan2(y, x)

# Angle conversion
def radians(a):
    return math.radians(a)

def degrees(a):
    return math.degrees(a)

# Exponential and logarithmic
def exp(a):
    return math.exp(a)

def log(a, base=math.e):
    return math.log(a, base)

def log10(a):
    return math.log10(a)

def log2(a):
    return math.log2(a)

def ln(a):
    return math.log(a)

# Power and root functions
def exponentiation(a, b):
    return a ** b

def sqrt(a):
    return math.sqrt(a)

def cbrt(a):
    return a ** (1/3) if a >= 0 else -((-a) ** (1/3))

def nth_root(a, n):
    return a ** (1/n) if a >= 0 else -((-a) ** (1/n))

# Rounding and integer functions
def ceil(a):
    return math.ceil(a)

def floor(a):
    return math.floor(a)

def round_to(a, decimals=0):
    return round(a, decimals)

def absolute(a):
    return abs(a)

# Statistical functions
def mean(numbers):
    return statistics.mean(numbers)

def median(numbers):
    return statistics.median(numbers)

def mode(numbers):
    return statistics.mode(numbers)

def stdev(numbers):
    return statistics.stdev(numbers)

def variance(numbers):
    return statistics.variance(numbers)

# Physics constants (in SI units)
def gravitational_constant():
    return 6.67430e-11

def speed_of_light():
    return 299792458

def planck_constant():
    return 6.62607015e-34

def boltzmann_constant():
    return 1.380649e-23

def avogadro_number():
    return 6.02214076e23

def electron_charge():
    return 1.602176634e-19

def electron_mass():
    return 9.1093837015e-31

def proton_mass():
    return 1.67262192369e-27

# Unit conversions
def celsius_to_kelvin(celsius):
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

def meters_to_feet(meters):
    return meters * 3.28084

def feet_to_meters(feet):
    return feet / 3.28084

def kg_to_pounds(kg):
    return kg * 2.20462

def pounds_to_kg(pounds):
    return pounds / 2.20462

# Advanced mathematical functions
def factorial(n):
    return math.factorial(n)

def gcd(a, b):
    return math.gcd(a, b)

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Geometry functions
def circle_area(radius):
    return math.pi * radius ** 2

def circle_circumference(radius):
    return 2 * math.pi * radius

def sphere_volume(radius):
    return (4/3) * math.pi * radius ** 3

def sphere_surface_area(radius):
    return 4 * math.pi * radius ** 2

def cylinder_volume(radius, height):
    return math.pi * radius ** 2 * height

def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def distance_3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

# Numerical methods
def derivative_numerical(func_values, h):
    derivatives = []
    for i in range(len(func_values)):
        if i == 0:
            deriv = (func_values[i+1] - func_values[i]) / h
        elif i == len(func_values) - 1:
            deriv = (func_values[i] - func_values[i-1]) / h
        else:
            deriv = (func_values[i+1] - func_values[i-1]) / (2 * h)
        derivatives.append(deriv)
    return derivatives

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
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


# Tools are now imported from utils.py

model = init_chat_model("o4-mini-2025-04-16", model_provider="openai")

# Create a temporary directory for sandbox sessions
sessions_dir = tempfile.mkdtemp()

# Try to create PyodideSandbox with error handling for Deno dependency
try:
    sandbox = PyodideSandbox(sessions_dir=sessions_dir, allow_net=True)
    eval_fn = create_pyodide_eval_fn(sandbox)
    print("✅ PyodideSandbox initialized successfully with Deno support")
except RuntimeError as e:
    if "Deno is not installed" in str(e):
        print("⚠️  Deno not available, falling back to basic Python execution")
        # Create a fallback eval function that uses exec() instead of PyodideSandbox
        def create_fallback_eval_fn() -> EvalFunction:
            def sync_eval_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                try:
                    # Create a safe execution environment
                    safe_globals = {
                        '__builtins__': {
                            'len': len, 'str': str, 'int': int, 'float': float,
                            'print': print, 'range': range, 'list': list,
                            'dict': dict, 'tuple': tuple, 'set': set,
                        },
                        'math': __import__('math'),
                        'statistics': __import__('statistics'),
                    }
                    # Add all tools from utils
                    for tool in tools:
                        safe_globals[tool.__name__] = tool
                    safe_globals.update(_locals)
                    
                    # Capture stdout
                    from io import StringIO
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()
                    
                    # Execute the code
                    exec(code, safe_globals)
                    
                    # Restore stdout and get output
                    sys.stdout = old_stdout
                    output = captured_output.getvalue()
                    
                    # Get new variables
                    new_vars = {k: v for k, v in safe_globals.items() 
                               if k not in _locals and not k.startswith('_') and not callable(v)}
                    
                    return output or "<Code executed successfully>", new_vars
                except Exception as exec_error:
                    return f"Error: {str(exec_error)}", {}
            return sync_eval_fn
        
        eval_fn = create_fallback_eval_fn()
    else:
        raise e

code_act = create_codeact(model, tools, eval_fn)
agent = code_act.compile()

