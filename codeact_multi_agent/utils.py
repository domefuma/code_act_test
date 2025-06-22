import math
from typing import List, Tuple, Union
import statistics


# Basic arithmetic functions
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


# Trigonometric functions
def sin(a: float) -> float:
    """Take the sine of a number (in radians)."""
    return math.sin(a)


def cos(a: float) -> float:
    """Take the cosine of a number (in radians)."""
    return math.cos(a)


def tan(a: float) -> float:
    """Take the tangent of a number (in radians)."""
    return math.tan(a)


def asin(a: float) -> float:
    """Take the arcsine of a number, returns radians."""
    return math.asin(a)


def acos(a: float) -> float:
    """Take the arccosine of a number, returns radians."""
    return math.acos(a)


def atan(a: float) -> float:
    """Take the arctangent of a number, returns radians."""
    return math.atan(a)


def atan2(y: float, x: float) -> float:
    """Two-argument arctangent function, useful for converting Cartesian to polar coordinates."""
    return math.atan2(y, x)


# Angle conversion
def radians(a: float) -> float:
    """Convert degrees to radians."""
    return math.radians(a)


def degrees(a: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(a)


# Exponential and logarithmic functions
def exp(a: float) -> float:
    """Calculate e raised to the power of a number."""
    return math.exp(a)


def log(a: float, base: float = math.e) -> float:
    """Calculate logarithm of a number. Default is natural log (base e)."""
    return math.log(a, base)


def log10(a: float) -> float:
    """Calculate base-10 logarithm of a number."""
    return math.log10(a)


def log2(a: float) -> float:
    """Calculate base-2 logarithm of a number."""
    return math.log2(a)


def ln(a: float) -> float:
    """Calculate natural logarithm (base e) of a number."""
    return math.log(a)


# Power and root functions
def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b


def sqrt(a: float) -> float:
    """Take the square root of a number."""
    return math.sqrt(a)


def cbrt(a: float) -> float:
    """Calculate the cube root of a number."""
    return a ** (1/3) if a >= 0 else -((-a) ** (1/3))


def nth_root(a: float, n: float) -> float:
    """Calculate the nth root of a number."""
    return a ** (1/n) if a >= 0 else -((-a) ** (1/n))


# Rounding and integer functions
def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    return math.ceil(a)


def floor(a: float) -> float:
    """Round a number down to the nearest integer."""
    return math.floor(a)


def round_to(a: float, decimals: int = 0) -> float:
    """Round a number to specified decimal places."""
    return round(a, decimals)


def absolute(a: float) -> float:
    """Get the absolute value of a number."""
    return abs(a)


# Statistical functions
def mean(numbers: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return statistics.mean(numbers)


def median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    return statistics.median(numbers)


def mode(numbers: List[float]) -> float:
    """Calculate the mode of a list of numbers."""
    return statistics.mode(numbers)


def stdev(numbers: List[float]) -> float:
    """Calculate the standard deviation of a list of numbers."""
    return statistics.stdev(numbers)


def variance(numbers: List[float]) -> float:
    """Calculate the variance of a list of numbers."""
    return statistics.variance(numbers)


# Physics constants (in SI units)
def gravitational_constant() -> float:
    """Universal gravitational constant G in m³/(kg⋅s²)."""
    return 6.67430e-11


def speed_of_light() -> float:
    """Speed of light in vacuum in m/s."""
    return 299792458


def planck_constant() -> float:
    """Planck constant in J⋅s."""
    return 6.62607015e-34


def boltzmann_constant() -> float:
    """Boltzmann constant in J/K."""
    return 1.380649e-23


def avogadro_number() -> float:
    """Avogadro's number in mol⁻¹."""
    return 6.02214076e23


def electron_charge() -> float:
    """Elementary charge in coulombs."""
    return 1.602176634e-19


def electron_mass() -> float:
    """Electron rest mass in kg."""
    return 9.1093837015e-31


def proton_mass() -> float:
    """Proton rest mass in kg."""
    return 1.67262192369e-27


# Unit conversions
def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return celsius * 9/5 + 32


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * 3.28084


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet / 3.28084


def kg_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462


def pounds_to_kg(pounds: float) -> float:
    """Convert pounds to kilograms."""
    return pounds / 2.20462


# Advanced mathematical functions
def factorial(n: int) -> int:
    """Calculate the factorial of a number."""
    return math.factorial(n)


def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two numbers."""
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of two numbers."""
    return abs(a * b) // math.gcd(a, b)


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


# Geometry functions
def circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * radius ** 2


def circle_circumference(radius: float) -> float:
    """Calculate the circumference of a circle given its radius."""
    return 2 * math.pi * radius


def sphere_volume(radius: float) -> float:
    """Calculate the volume of a sphere given its radius."""
    return (4/3) * math.pi * radius ** 3


def sphere_surface_area(radius: float) -> float:
    """Calculate the surface area of a sphere given its radius."""
    return 4 * math.pi * radius ** 2


def cylinder_volume(radius: float, height: float) -> float:
    """Calculate the volume of a cylinder given its radius and height."""
    return math.pi * radius ** 2 * height


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Calculate the Euclidean distance between two 3D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Numerical methods
def derivative_numerical(func_values: List[float], h: float) -> List[float]:
    """Calculate numerical derivative using finite differences."""
    derivatives = []
    for i in range(len(func_values)):
        if i == 0:
            # Forward difference
            deriv = (func_values[i+1] - func_values[i]) / h
        elif i == len(func_values) - 1:
            # Backward difference
            deriv = (func_values[i] - func_values[i-1]) / h
        else:
            # Central difference
            deriv = (func_values[i+1] - func_values[i-1]) / (2 * h)
        derivatives.append(deriv)
    return derivatives


def linear_interpolation(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Perform linear interpolation between two points."""
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


# List all available tools
tools = [
    # Basic arithmetic
    add, multiply, divide, subtract,
    
    # Trigonometric functions
    sin, cos, tan, asin, acos, atan, atan2,
    
    # Angle conversion
    radians, degrees,
    
    # Exponential and logarithmic
    exp, log, log10, log2, ln,
    
    # Power and root functions
    exponentiation, sqrt, cbrt, nth_root,
    
    # Rounding and integer functions
    ceil, floor, round_to, absolute,
    
    # Statistical functions
    mean, median, mode, stdev, variance,
    
    # Physics constants
    gravitational_constant, speed_of_light, planck_constant,
    boltzmann_constant, avogadro_number, electron_charge,
    electron_mass, proton_mass,
    
    # Unit conversions
    celsius_to_kelvin, kelvin_to_celsius,
    fahrenheit_to_celsius, celsius_to_fahrenheit,
    meters_to_feet, feet_to_meters,
    kg_to_pounds, pounds_to_kg,
    
    # Advanced mathematical functions
    factorial, gcd, lcm, fibonacci, is_prime,
    
    # Geometry functions
    circle_area, circle_circumference,
    sphere_volume, sphere_surface_area, cylinder_volume,
    distance_2d, distance_3d,
    
    # Numerical methods
    derivative_numerical, linear_interpolation,
]



