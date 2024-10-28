#!/usr/bin/env python3

"""
Advanced Scientific Calculator with Unlimited Precision

This calculator provides arbitrary-precision arithmetic and supports a wide range of 
mathematical functions, including trigonometric, logarithmic, and exponential functions. 
It features a menu-driven interface, comprehensive error handling, and logging capabilities.

Technical Details:

* Calculations:
    - Uses the `mpmath` library for arbitrary-precision arithmetic.
    - Handles special functions (trigonometric, logarithmic, etc.) with optimized algorithms.
    - Manages precision dynamically to ensure accuracy.
    - Includes mechanisms to handle large numbers in trigonometric calculations, 
        maintaining quadrant correctness.

* Formatting:
    - Formats results with appropriate precision, including scientific notation for 
        very large or small numbers.
    - Handles special cases like repeating decimals (e.g., 1/3) to provide user-friendly output.

* Test Cases and Error Handling:
    - Includes a comprehensive suite of test cases to verify the accuracy of calculations 
        across different precision levels.
    - Implements robust error handling for invalid input, division by zero, and other 
        potential exceptions.
    - Provides informative error messages to the user.

* Logging:
    - Logs events, errors, and test results to JSON files for debugging and analysis.
    - Uses custom JSON formatters for structured log output.

* Speed Benchmarking:
    - Measures and reports the execution time of calculations.
    - Tracks function execution times for performance analysis.

To-do:
- Fix and improve `report_function_timings` - create a new JSON file for timings.
- Get rid of everything that's not being used.
- See if the application can be reduced in code or simplified without changing 
    features or functionality.
- Create a unified JSON logging object or use a third-party logging library.
- Refactor.
"""

# --- Imports ---
import os, sys, subprocess, venv, signal, logging, json, traceback, re  # Importing necessary modules
from datetime import datetime  # For timestamping logs
from time import time  # For timing operations

import mpmath as mp  # Importing the mpmath library for arbitrary-precision arithmetic
from mpmath import mpf, mpc, inf, nan, j as mpj  # Importing specific mpmath functions and constants

# --- Configuration ---
history_file = 'calculator_history.json'  # File to store calculation history
console_log = 'calculator_console.json'  # File for console logs
test_log = 'calculator_tests.json'  # File for test results
function_timings = {}  # Dictionary to store function execution times

# --- Exception Classes ---

class CalculationError(Exception):
    """Custom exception for calculation errors."""
    pass

class TimeoutError(Exception):
    """Custom exception for calculation timeouts."""
    pass

def timeout_handler(signum, frame):
    """Handler for signal.SIGALRM timeout."""
    raise TimeoutError("Calculation timed out")

# --- Logging Setup ---

class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format."""

    def format(self, record):
        """Format the log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f'),  # Format timestamp
            "level": record.levelname,  # Log level (e.g., DEBUG, INFO, ERROR)
            "type": getattr(record, 'log_type', 'general'),  # Log type (e.g., function_call, test_result)
        }
        
        if isinstance(record.msg, dict):  # If the log message is a dictionary, add it to the log entry
            log_entry.update(record.msg)
        else:
            log_entry["message"] = record.getMessage()  # Otherwise, add the message as a string
        
        if record.exc_info:  # If there's exception information, add it to the log entry
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)  # Return the log entry as a JSON string

def setup_logging():
    """Set up loggers for console and test logs."""

    def setup_logger(name, log_file):
        """Configure a logger with a file handler."""
        logger = logging.getLogger(name)  # Get a logger with the specified name
        logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG (log everything)
        handler = logging.FileHandler(log_file, mode='w')  # Create a file handler to write logs to the specified file
        handler.setFormatter(JsonFormatter())  # Set the custom JSON formatter for the handler
        logger.handlers.clear()
        logger.addHandler(handler)  # Add the handler to the logger
        
        # Initialize the log file with a valid JSON structure
        with open(log_file, 'w') as f:
            json.dump({"logs": []}, f)  # Write an empty JSON array to the file
        
        return logger  # Return the configured logger

    console_logger = setup_logger('console', console_log)  # Set up the console logger
    test_logger = setup_logger('test', test_log)  # Set up the test logger

    return console_logger, test_logger  # Return both loggers

console_logger, test_logger = setup_logging()  # Initialize the loggers

def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        """Wrapper function to log function call details."""
        start_time = time()  # Record the start time of the function call
        custom_log(console_logger, logging.DEBUG, {  # Log the function call start
            "log_type": "function_call",
            "function": func.__name__,
            "status": "started",
            "args": str(args),
            "kwargs": str(kwargs)
        })
        try:
            result = func(*args, **kwargs)  # Call the original function
            end_time = time()  # Record the end time of the function call
            elapsed_time = end_time - start_time  # Calculate elapsed time
            custom_log(console_logger, logging.DEBUG, {  # Log the function call completion
                "log_type": "function_call",
                "function": func.__name__,
                "status": "completed",
                "elapsed_time": f"{elapsed_time:.6f} seconds"
            })
            return result  # Return the result of the original function
        except Exception as e:
            custom_log(console_logger, logging.ERROR, {  # Log the function call error
                "log_type": "function_call",
                "function": func.__name__,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise  # Re-raise the exception
    return wrapper  # Return the wrapper function

def log_test_result(operation, expression, result, elapsed_time=None, error=None):
    """Log the result of a test case. Overloaded"""
    log_entry = {
        "log_type": "test_result",
        "test": operation,
        "expression": expression,
        "result": result if not error else None,
        "error": error if error else None,
        "elapsed_time": f"{elapsed_time:.6f} seconds" if elapsed_time is not None else None
    }
    custom_log(test_logger, logging.INFO, log_entry)  # Log the test result to the test logger
    custom_log(console_logger, logging.INFO, log_entry)  # Log the test result to the console logger

def log_test_result(test_name, expression, expected_output, actual_output, elapsed_time, result):
    """Log the result of a test case. Overloaded"""
    log_entry = {
        "log_type": "test_result",
        "test_name": test_name,
        "expression": expression,
        "answer_output": expected_output,
        "actual_output": actual_output,
        "result": result,
        "elapsed_time": f"{elapsed_time:.6f} seconds" if elapsed_time is not None else None
    }
    custom_log(test_logger, logging.INFO, log_entry)  # Log the test result to the test logger
    custom_log(console_logger, logging.INFO, log_entry)  # Log the test result to the console logger

def log_to_file(logger, log_entry):
    """Append a log entry to the specified log file."""
    log_file = logger.handlers[0].baseFilename  # Get the filename from the logger's handler
    with open(log_file, 'r+') as f:  # Open the log file in read/write mode
        data = json.load(f)  # Load the existing JSON data from the file
        data["logs"].append(json.loads(log_entry))  # Append the new log entry to the "logs" array
        f.seek(0)  # Go back to the beginning of the file
        json.dump(data, f, indent=2)  # Write the updated JSON data back to the file with indentation
        f.truncate()  # Truncate the file in case the new data is shorter than the old data

def custom_log(logger, level, message):
    """Log a message with the custom JSON formatter."""
    record = logger.makeRecord(logger.name, level, "", 0, message, None, None)  # Create a log record
    log_entry = JsonFormatter().format(record)  # Format the log record using the custom JSON formatter
    log_to_file(logger, log_entry)  # Append the formatted log entry to the log file

@log_function_call
def clear_all_logs():
    """Clear all log files."""
    log_files = ['calculator_console.json', 'calculator_tests.json', history_file]  # List of log files to clear
    for file in log_files:  # Iterate through the log files
        try:
            with open(file, 'w') as f:  # Open the file in write mode (this clears the file)
                if file == history_file:
                    json.dump([], f)  # Initialize history file as an empty list
                else:
                    json.dump({"logs": []}, f)  # Initialize other log files with an empty "logs" array
            custom_log(console_logger, logging.INFO, f"Cleared {file}")  # Log that the file was cleared
        except Exception as e:
            custom_log(console_logger, logging.ERROR, f"Error clearing {file}: {str(e)}")  # Log any errors during clearing
    custom_log(console_logger, logging.INFO, "All log files have been cleared.")  # Log that all files were cleared
    print("\nAll log files have been cleared.")  # Print a message to the console

@log_function_call
def report_function_timings():
    """Generate and log a report of function execution times."""
    custom_log(console_logger, logging.INFO, "Generating function execution time report")  # Log the start of report generation
    print("\nFunction Execution Time Report:")
    print("--------------------------------")
    total_time = sum(function_timings.values())  # Calculate the total execution time
    timing_report = []  # Initialize an empty list to store timing data
    for func_name, elapsed_time in sorted(function_timings.items(), key=lambda x: x[1], reverse=True):  # Iterate through function timings, sorted by time
        percentage = (elapsed_time / total_time) * 100  # Calculate the percentage of total time for each function
        print(f"{func_name}: {elapsed_time:.6f} seconds ({percentage:.2f}%)")  # Print the function name, time, and percentage
        timing_report.append({  # Add the function timing data to the list
            "function": func_name,
            "time": f"{elapsed_time:.6f} seconds",
            "percentage": f"{percentage:.2f}%"
        })
    print(f"\nTotal execution time: {total_time:.6f} seconds")  # Print the total execution time
    
    log_entry = {  # Create a log entry for the timing report
        "log_type": "function_execution_time_report",
        "function_timings": timing_report,
        "total_execution_time": f"{total_time:.6f} seconds"
    }
    custom_log(console_logger, logging.INFO, log_entry)  # Log the timing report

@log_function_call
def create_venv():
    """Create a virtual environment for the calculator."""
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calculator_venv")  # Construct the path for the virtual environment
    if not os.path.exists(venv_path):  # Check if the virtual environment already exists
        custom_log(console_logger, logging.INFO, "Creating virtual environment...")  # Log the creation of the virtual environment
        venv.create(venv_path, with_pip=True)  # Create the virtual environment with pip
    return venv_path  # Return the path to the virtual environment

@log_function_call
def get_venv_python(venv_path):
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":  # For Windows
        return os.path.join(venv_path, "Scripts", "python.exe")  # Return the path to python.exe
    return os.path.join(venv_path, "bin", "python")  # For Linux/macOS

@log_function_call
def install_libraries(venv_python):
    """Install required libraries in the virtual environment."""
    required_libraries = ['sympy', 'mpmath']  # List of required libraries
    for library in required_libraries:  # Iterate through the required libraries
        custom_log(console_logger, logging.INFO, f"Installing {library}...")  # Log the installation attempt
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", library])  # Install the library using pip
            custom_log(console_logger, logging.INFO, f"Successfully installed {library}")  # Log successful installation
        except subprocess.CalledProcessError as e:
            custom_log(console_logger, logging.ERROR, f"Failed to install {library}: {str(e)}")  # Log installation failure
            raise  # Re-raise the exception

def setup_unlimited_precision():
    """Set up mpmath for unlimited precision calculations."""
    try:
        mp.dps = 1000  # Increase default precision to 1000 significant digits
        custom_log(console_logger, logging.INFO, f"Precision set to {mp.dps} decimal places")  # Log the precision setting
    except Exception as e:
        custom_log(console_logger, logging.ERROR, f"Error setting precision: {str(e)}")  # Log any errors during precision setup
        mp.dps = 10  # Fallback to a lower but still high precision
    custom_log(console_logger, logging.INFO, f"Final precision: {mp.dps} decimal places")  # Log the final precision

@log_function_call
def handle_special_functions(expr_string):
    """Handle special functions in the expression string."""
    custom_log(console_logger, logging.DEBUG, f"Processing expression: {expr_string}")  # Log the original expression
    special_funcs = [
        'pi', 'e', 'inf', 'nan', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'log', 'sqrt', 'factorial', 'zeta', 'abs', 'exp', 'log10', 'root'
    ]  # List of special functions to handle
    for func in special_funcs:  # Iterate through the special functions
        expr_string = expr_string.replace(func + '(', func + '(')  # Add spaces around function names
    expr_string = expr_string.replace('I', 'j')  # Replace imaginary unit 'I' with 'j' for mpmath
    custom_log(console_logger, logging.DEBUG, f"Processed expression: {expr_string}")  # Log the processed expression
    return expr_string  # Return the modified expression string

def handle_large_trig_number(func, x, precision):
    """Handle trigonometric calculations of large numbers with correct quadrants."""
    def reduce_to_range(x, pi2):
        """Reduce number to [-π, π] range with high precision."""
        # Use high precision for division and floor
        with mp.workprec(precision + 40):  # Increase precision for intermediate calculations
            quotient = mp.floor(x / pi2 + 0.5)  # Add 0.5 for rounding to nearest multiple of pi2
            reduced = x - quotient * pi2  # Reduce the angle to the range [-π, π]
            return reduced, int(quotient % 4)  # Quadrant is determined by the remainder
    with mp.workprec(precision + 40):
        pi2 = mp.mpf(2) * mp.pi  # Calculate 2π with high precision
        reduced_x, quadrant = reduce_to_range(x, pi2)  # Reduce the angle to the range [-π, π]
        
        # For very large numbers, use even higher precision
        if abs(x) > 1e15:  # If the input is extremely large
            with mp.workprec(precision + 100):  # Increase precision even further
                reduced_x, quadrant = reduce_to_range(x, pi2)  # Reduce the angle again with higher precision
        
        # Compute the function value with the reduced angle
        result = func(reduced_x)  # Calculate the trigonometric function with the reduced angle
        
        # Adjust the sign of the result based on the quadrant and function
        if func.__name__ == 'sin':  # For sine function
            if quadrant in (2, 3):  # Sin is negative in Q2 and Q3
                result = -result  # Change the sign if necessary
        elif func.__name__ == 'cos':  # For cosine function
            if quadrant in (1, 2):  # Cos is negative in Q1 and Q2
                result = -result  # Change the sign if necessary
        elif func.__name__ == 'tan':  # For tangent function
            if quadrant in (2, 4):  # Tan is negative in Q2 and Q4
                result = -result  # Change the sign if necessary
        
        return mp.chop(result, precision)  # Return result with correct precision

def calculate(expr_string, source="user_input"):
    """
    Evaluate a mathematical expression with error handling and logging.

    Args:
        expr_string (str): The expression to evaluate.
        source (str, optional): The source of the calculation (e.g., 'user_input', 'test_case'). 
                                Defaults to "user_input".

    Returns:
        tuple: A tuple containing the formatted result (or an error message) and a boolean 
                indicating whether an error occurred.
    """
    try:
        start_time = time()  # Record the start time of the calculation
        interpreted = handle_special_functions(expr_string)  # Process special functions in the expression
        
        # Create wrappers for trig functions to handle large numbers
        def wrap_trig(func):
            """Wrapper to handle large numbers for trigonometric functions."""
            def wrapper(x):  # No need for **kwargs here
                if abs(x) > 1e5:  # If the input is large
                    with mp.workprec(mp.dps + 40):  # Increase precision for the calculation
                        result = handle_large_trig_number(func, x, mp.dps)  # Call the function with large number handling
                        return mp.mpf(result)  # Return the result as an mpf
                return func(x)  # Otherwise, call the function directly
            wrapper.__name__ = func.__name__  # Ensure the wrapper has the same name as the original function
            return wrapper  # Return the wrapper function
            
        # Set up namespace with wrapped functions and constants for evaluation
        mpmath_namespace = {  # Create a namespace for evaluating the expression
            'mp': mp,  # Include the mpmath module
            'sin': wrap_trig(mp.sin),  # Include the wrapped trigonometric functions
            'cos': wrap_trig(mp.cos),
            'tan': wrap_trig(mp.tan),
            'asin': mp.asin,  # Include other mpmath functions
            'acos': mp.acos,
            'atan': mp.atan,
            'log': mp.log,
            'sqrt': mp.sqrt,
            'factorial': mp.factorial,
            'zeta': mp.zeta,
            'abs': mp.fabs,  # Use mp.fabs for absolute value
            'exp': mp.exp,
            'log10': mp.log10,
            'root': mp.root,
            'pi': mp.pi,  # Include mathematical constants
            'e': mp.e,
            'inf': mp.inf,
            'NaN': mp.nan,
            'j': mp.j,
            'mpf': mp.mpf,  # Include mpmath types
            'mpc': mp.mpc,
        }
        
        # Check for division by zero
        if re.search(r'/\s*0(?![.\d])', interpreted):  # Check if there's division by zero in the expression
            raise ValueError("division by zero")  # Raise a ValueError if division by zero is detected
        
        try:
            # Compile and validate code
            code = compile(interpreted, '<string>', 'eval')  # Compile the expression string into bytecode
            for name in code.co_names:  # Check if all variables in the expression are defined in the namespace
                if name not in mpmath_namespace:
                    raise NameError(f"name '{name}' is not defined")  # Raise a NameError if an undefined variable is found
            
            # Evaluate with precision control
            with mp.workprec(mp.dps):  # Set the precision for the calculation
                # print(f"Debugging - Expression: {interpreted}")  # Debug print statement
                # print(f"Debugging - Precision: {mp.dps}")  # Debug print statement
                result = eval(code, {"__builtins__": None}, mpmath_namespace)  # Evaluate the expression using the namespace
                # print(f"Debugging - Raw Result: {result}")  # Debug print statement
                
                if mp.isnan(result):  # Check if the result is NaN
                    raise ValueError("Result is not a number")  # Raise a ValueError if the result is NaN
                if mp.isinf(result):  # Check if the result is infinite
                    raise ValueError("Result is infinite")  # Raise a ValueError if the result is infinite
                
                formatted_result = format_result(result)  # Format the result
                
                end_time = time()  # Record the end time of the calculation
                elapsed_time = end_time - start_time  # Calculate elapsed time
                
                custom_log(console_logger, logging.INFO, {  # Log the calculation completion
                    "calculation_completed": True,
                    "result": formatted_result,
                    "elapsed_time": f"{elapsed_time:.6f} seconds"
                })
                
                return formatted_result, False  # Return the formatted result and a flag indicating no error
                
        except SyntaxError:  # Catch syntax errors in the expression
            raise ValueError("Error in calculation: invalid syntax (<string>, line 1)")  # Raise a ValueError with a descriptive message
        except ZeroDivisionError:  # Catch division by zero errors
            raise ValueError("division by zero")  # Raise a ValueError
            
    except ValueError as e:  # Catch ValueErrors (e.g., invalid input, division by zero)
        error_msg = f"Error: {str(e)}"  # Format the error message
        custom_log(console_logger, logging.ERROR, {  # Log the error
            "calculation_error": True,
            "error": error_msg
        })
        return error_msg, True  # Return the error message and a flag indicating an error occurred
    except Exception as e:  # Catch any other exceptions
        error_msg = f"Error: {str(e)}"  # Format the error message
        custom_log(console_logger, logging.ERROR, {  # Log the error
            "calculation_error": True,
            "error": error_msg
        })
        return error_msg, True  # Return the error message and a flag indicating an error occurred
    
def custom_round(x, precision):
    """Rounds x to the nearest multiple of 10^-precision."""
    multiplier = mp.power(10, precision)  # Calculate the multiplier for rounding
    rounded_x = mp.nint(x * multiplier) / multiplier  # Round to nearest integer and adjust back
    return rounded_x  # Return the rounded value

def format_result(result, precision=None):
    """
    Format the result of a calculation with appropriate precision and handling of special cases.

    Args:
        result (str, mp.mpf, mp.mpc, float, int): The result to format.
        precision (int, optional): The number of decimal places to display. Defaults to None, 
                                    which uses the current precision set in mpmath.

    Returns:
        str: The formatted result as a string.
    """
    if precision is None:  # If precision is not specified
        precision = mp.dps  # Use the current precision set in mpmath

    try:
        if isinstance(result, str):  # If the result is already a string, return it
            return result

        if isinstance(result, (mp.mpf, float,int)):  # If the result is a real number
            if mp.isinf(result):  # Check if the result is infinite
                return "Infinity" if result > 0 else "-Infinity"  # Return "Infinity" or "-Infinity"
            if mp.isnan(result):  # Check if the result is NaN
                return "NaN"  # Return "NaN"

            with mp.workdps(precision):  # Enforce precision using context manager
                # Special case for fractions like 1/3
                if abs(abs(result) - abs(mp.mpf('1') / 3)) < mp.power(10, -precision):  # Check if the result is close to 1/3
                    if precision <= 9:  # If precision is low, use mp.nstr for formatting
                        # Round within the specified precision context
                        result = mp.nstr(mp.fdiv(1, 3), precision)  # Format the result as a string
                    else:  # If precision is high, return a repeating decimal representation
                        return "0.3(3)" if result > 0 else "-0.3(3)"  # Return "0.3(3)" or "-0.3(3)"

                # Format with specified precision
                formatted = mp.nstr(result, precision, min_fixed=-inf, max_fixed=inf)  # Format the result as a string with specified precision

            # Handle scientific notation
            if 'e' in formatted.lower():  # Check if the result is in scientific notation
                mantissa, exp = formatted.lower().split('e')  # Split the result into mantissa and exponent
                exp_val = int(exp)  # Convert the exponent to an integer
                if abs(exp_val) <= 3:  # Convert to regular notation for small exponents
                    return str(mp.nstr(result, precision))  # Format the result without scientific notation
                # Keep scientific notation for large exponents
                return f"{mantissa.rstrip('0').rstrip('.')}e{exp_val}"  # Format the result with scientific notation

            # Remove trailing zeros and decimal point if whole number
            if '.' in formatted:  # Check if the result has a decimal point
                formatted = formatted.rstrip('0').rstrip('.')  # Remove trailing zeros and decimal point
                if not formatted:  # If the formatted result is empty, return "0"
                    return "0"  # Return "0"

            return formatted  # Return the formatted result

        elif isinstance(result, mp.mpc):  # If the result is a complex number
            real_part = format_result(result.real, precision)  # Format the real part
            imag_part = format_result(abs(result.imag), precision)  # Format the imaginary part

            if abs(result.real) < mp.power(10, -precision):  # If the real part is very small
                return f"{imag_part}i" if result.imag >= 0 else f"-{imag_part}i"  # Return only the imaginary part
            else:  # Otherwise, return both real and imaginary parts
                return f"{real_part}+{imag_part}i" if result.imag >= 0 else f"{real_part}-{imag_part}i"  # Return the formatted complex number

        return str(result)  # If the result is of any other type, convert it to a string and return

    except Exception as e:
        custom_log(console_logger, logging.ERROR, f"Error in format_result: {str(e)}")  # Log any errors during formatting
        return str(result)  # Return the result as a string in case of errors

def compare_results(answer, actual, rel_tolerance=1e-15):
    """
    Compare the expected answer with the actual result of a calculation,
    handling different data types and potential errors.

    Args:
        answer (str or numeric): The expected answer.
        actual (str or numeric): The actual calculated result.
        rel_tolerance (float, optional): The relative tolerance for numeric comparisons. 
                                        Defaults to 1e-15.

    Returns:
        tuple: A tuple containing a boolean indicating whether the comparison passed and a 
                string with details about the comparison.
    """
    custom_log(console_logger, logging.DEBUG, f"Comparing results: answer={answer}, actual={actual}")  # Log the comparison attempt
    
    # If we're comparing error messages, do exact string comparison
    if isinstance(answer, str) and answer.startswith("Error:"):  # If the expected answer is an error message
        result = answer == actual  # Check if the actual result matches the expected error message
        custom_log(console_logger, logging.DEBUG, f"Error message comparison result: {result}")  # Log the result of the comparison
        return result, "Error message comparison"  # Return the result and a description of the comparison
    elif isinstance(actual, str) and actual.startswith("Error:"):  # If the actual result is an error message
        if isinstance(answer, str) and answer.startswith("Error:"):  # If the expected answer is also an error message
            result = answer == actual  # Check if the error messages match exactly
            return result, "Error message comparison"  # Return the result and a description of the comparison
        return False, "Got error when expecting numeric result"  # Otherwise, the comparison fails
    
    try:
        def parse_result(s):
            """Parse a result string into a numeric type if possible."""
            if isinstance(s, str):  # If the input is a string
                # Handle repeating decimals
                if '(' in s and ')' in s:  # If the string contains a repeating decimal pattern
                    non_repeating, repeating = s.split('(')  # Split the string into non-repeating and repeating parts
                    repeating = repeating.rstrip(')')  # Remove the closing parenthesis
                    full_precision = f"{non_repeating}{repeating * 10}"  # Repeat the pattern 10 times
                    return mp.mpf(full_precision)  # Convert the expanded string to mp.mpf
                try:
                    return mp.mpf(s)  # Attempt conversion to mp.mpf
                except ValueError:
                    return s  # If conversion fails, return the original string
            else:
                return s  # If the input is not a string, return it as is

        answer_value = parse_result(answer.strip("'"))  # Remove extra quotes before parsing
        actual_value = parse_result(actual)  # Parse the actual result

        # Determine the precision of the answer value
        answer_precision = len(str(answer_value).split('.')[-1]) if isinstance(answer_value, (mp.mpf, float)) else len(str(answer)) # Get precision from the answer string
        
        # Round both values to the precision of the answer value
        rounded_answer = mp.mpf(mp.nstr(answer_value, answer_precision))  # Convert rounded string back to mp.mpf
        rounded_actual = mp.mpf(mp.nstr(actual_value, answer_precision))  # Convert rounded string back to mp.mpf

        if isinstance(answer_value, mp.mpc) or isinstance(actual_value, mp.mpc):  # If either result is a complex number
            answer_real, answer_imag = mp.re(answer_value), mp.im(answer_value)  # Get real and imaginary parts of the expected answer
            actual_real, actual_imag = mp.re(actual_value), mp.im(actual_value)  # Get real and imaginary parts of the actual result
            real_close = mp.almosteq(answer_real, actual_real, rel_eps=rel_tolerance)  # Compare real parts
            imag_close = mp.almosteq(answer_imag, actual_imag, rel_eps=rel_tolerance)  # Compare imaginary parts
            result = real_close and imag_close  # The comparison passes if both real and imaginary parts are close
            details = f"Complex comparison: real close: {real_close}, imag close: {imag_close}"  # Description of the comparison
        else:  # If both results are real numbers
            result = rounded_answer == rounded_actual  # Compare the rounded values
            if result:
                details = "Exact string match after rounding"  # Description of the comparison
            else:
                rel_diff = abs(answer_value - actual_value) / abs(answer_value) if answer_value != 0 else abs(actual_value)  # Calculate relative difference
                details = f"Numeric comparison: relative difference {rel_diff}"  # Description of the comparison

        custom_log(console_logger, logging.DEBUG, f"Comparison result: {result}")  # Log the result of the comparison
        custom_log(console_logger, logging.DEBUG, f"Answer (rounded): {rounded_answer}")
        custom_log(console_logger, logging.DEBUG, f"Actual (rounded): {rounded_actual}")  # Log the rounded actual result
        
        return result, details  # Return the result and a description of the comparison
    except ValueError as ve:
        # Handle cases where parsing fails (e.g., comparing "0.3(3)" with a full decimal expansion)
        if '(' in str(answer) and ')' in str(answer):  # If the expected answer contains a repeating decimal pattern
            non_repeating, repeating = str(answer).split('(')  # Split the answer into non-repeating and repeating parts
            repeating = repeating.rstrip(')')  # Remove the closing parenthesis
            pattern = f"{non_repeating}{repeating}"  # Construct the repeating pattern
            if str(actual).startswith(pattern):  # Check if the actual result starts with the repeating pattern
                return True, "Repeating decimal match"  # If it does, the comparison passes
        return False, f"ValueError: {str(ve)}"  # Otherwise, the comparison fails
    except Exception as e:
        custom_log(console_logger, logging.ERROR, f"Error in result comparison: {str(e)}")  # Log any errors during comparison
        return False, f"Comparison error: {str(e)}"  # Return a failure result and the error message

def setup_test_cases():
    """
    Set up test cases for the calculator.

    Returns:
        dict: A dictionary of test cases, where the keys are test names and the values are tuples 
              containing the expression, an optional function to generate the expected output, and 
              an optional string representation of the expected output.
    """

    # --- Helper Functions for Test Cases ---

    def precision_test(precision):
        """Test the precision of the calculator."""
        mp.dps = precision
        result = mp.fdiv(1, 3, prec=precision)  # Calculate 1/3 with the specified precision
        return result  # Return the raw result without manual rounding

    def high_order_root(precision):
        """Test calculation of a high-order root."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec to set precision for root
            result = mp.root(2, 100)  # Calculate the 100th root of 2
        return result  # Return the result

    def trigonometric_precision(precision):
        """Test trigonometric calculations with specified precision."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec to set precision for power
            result = mp.fadd(mp.power(mp.sin(mp.pi/6, prec=precision), 5), 
                            mp.power(mp.cos(mp.pi/6, prec=precision), 5), 
                            prec=precision)  # Calculate sin(pi/6)^5 + cos(pi/6)^5
        return result  # Return the result
    
    def inverse_trigonometric(precision):
        """Test inverse trigonometric functions."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec to ensure precision
            # Calculate each component separately for clarity
            asin_val = mp.asin(mp.mpf('0.5'))  # Calculate asin(0.5)
            acos_val = mp.acos(mp.mpf('0.5'))  # Calculate acos(0.5)
            atan_val = mp.atan(mp.mpf('0.5'))  # Calculate atan(0.5)
            
            # Add them together using fadd for maximum precision
            first_sum = mp.fadd(asin_val, acos_val, prec=precision)  # Add asin and acos
            final_sum = mp.fadd(first_sum, atan_val, prec=precision)  # Add the previous sum and atan
            
            return final_sum  # Return the final result

    def natural_log_of_large_number(precision):
        """Test natural logarithm of a large number."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for power and log
            result = mp.log(mp.power(10, 100))  # Calculate log(10^100)
        return result  # Return the result

    def log_of_fraction_to_various_bases(precision):
        """Test logarithm of a fraction to different bases."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec to ensure precision
            result = mp.fadd(mp.fadd(mp.log(mp.mpf('0.5', prec=precision), 2), 
                                    mp.log(mp.mpf('0.5', prec=precision), 3),
                                    prec=precision),  # Add the first two logs
                            mp.log(mp.mpf('0.5', prec=precision), 10),
                            prec=precision)  # Then add the third log
        return result  # Return the result

    def trig_function_of_large_number(precision):
        """Test trigonometric function of a large number."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for power
            result = mp.sin(mp.power(10, 10), prec=precision)  # Calculate sin(10^10)
        return result  # Return the result

    def nested_logarithms(precision):
        """Test nested logarithmic calculations."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for power and nested logs
            result = mp.log(mp.log(mp.log(mp.power(10, 100))))  # No prec for individual log calls
        return result  # Return the result

    def complex_trig_expression(precision):
        """Test a complex trigonometric expression."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for power
            result = mp.fsub(mp.fadd(mp.power(mp.sin(mp.pi/4, prec=precision), 5),
                                        mp.power(mp.cos(mp.pi/4, prec=precision), 5),
                                        prec=precision), 1, prec=precision)  # Calculate (sin(pi/4)^5 + cos(pi/4)^5) - 1
        return result  # Return the result

    def extreme_logarithm(precision):
        """Test logarithm of an extremely large number."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for power
            result = mp.log(mp.power(10, 1000))  # Calculate log(10^1000)
        return result  # Return the result

    def ultra_precise_transcendental_number(precision):
        """Test calculation of a transcendental number with high precision."""
        mp.dps = precision  # Set the precision for the calculation
        return mp.zeta(3, prec=precision)  # Calculate zeta(3) with the specified precision

    def natural_logarithm(precision):
        """Test natural logarithm calculation."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for log
            result = mp.log(10)  # Calculate log(10)
        return result
    # Return the result

    def logarithm_base_10(precision):
        """Test logarithm base 10 calculation."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for log10
            result = mp.log10(100)  # Calculate log10(100)
        return result  # Return the result

    def logarithm_base_2(precision):
        """Test logarithm base 2 calculation."""
        mp.dps = precision  # Set the precision for the calculation
        with mp.workprec(precision):  # Use workprec for log with base argument
            result = mp.log(8, 2)  # Calculate log2(8)
        return result  # Return the result

    def zeta_function(precision):
        """Test the Riemann zeta function."""
        mp.dps = precision  # Set the precision for the calculation
        return mp.zeta(2, prec=precision)  # Calculate zeta(2) with the specified precision

    # --- Test Cases ---

    test_cases = {  # Dictionary of test cases
        "High Order Root": ("root(2, 100)", high_order_root, "1.0069555500567188088"),  # Test case for high-order root
        "Trigonometric Precision": ("sin(pi/6)**5 + cos(pi/6)**5", trigonometric_precision, "0.51838928962874673881"),  # Test case for trigonometric precision
        "Inverse Trigonometric": ("asin(0.5) + acos(0.5) + atan(0.5)", inverse_trigonometric, "1.5"),  # Test case for inverse trigonometric functions
        "Natural Log of Large Number": ("log(10**100)", natural_log_of_large_number, "230.25850929940456840"),  # Test case for natural log of a large number
        "Log of Fraction to Various Bases": ("log(0.5, 2) + log(0.5, 3) + log(0.5, 10)", log_of_fraction_to_various_bases, "-1.9319597492354386323"),  # Test case for log of a fraction to various bases
        "Nested Logarithms": ("log(log(log(10**100)))", nested_logarithms, "1.6936324749842330549"),  # Test case for nested logarithms
        "Complex Trig Expression": ("sin(pi/4)**5 + cos(pi/4)**5 - 1", complex_trig_expression, "-0.64644660940672623780"),  # Test case for a complex trigonometric expression
        "Extreme Logarithm": ("log(10**1000)", extreme_logarithm, "2302.5850929940456840"),  # Test case for extreme logarithm
        "Ultra-Precise Transcendental Number": ("zeta(3)", ultra_precise_transcendental_number, "1.2020569031595942854"),  # Test case for ultra-precise transcendental number
        "Natural Logarithm": ("log(10)", natural_logarithm, "2.3025850929940456840"),  # Test case for natural logarithm
        "Logarithm base 10": ("log10(100)", logarithm_base_10, "2.0"),  # Test case for logarithm base 10
        "Logarithm base 2": ("log(8, 2)", logarithm_base_2, "3.0"),  # Test case for logarithm base 2
        "Zeta Function": ("zeta(2)", zeta_function, "1.6449340668482264365"),  # Test case for zeta function
    }

    global error_test_cases  # Declare error_test_cases as global
    error_test_cases = {  # Dictionary of error test cases
        "Division by Zero": ("1 / 0", lambda: "Error: division by zero", "Error: division by zero"),  # Test case for division by zero
        "Invalid Input": ("2 + *3", lambda: "Error: Error in calculation: invalid syntax (<string>, line 1)", "Error: Error in calculation: invalid syntax (<string>, line 1)"),  # Test case for invalid input
        "Undefined Operation": ("0 ** 0", lambda: mp.mpf(1), "1.0"),  # Test case for undefined operation
        }

    return {**test_cases, **error_test_cases}  # Return a dictionary combining both test cases and error test cases

def test_calculator():
    """Run test cases and log the results."""
    custom_log(test_logger, logging.INFO, {"log_type": "test_suite", "message": "Starting calculator tests"})  # Log the start of the test suite
    custom_log(console_logger, logging.INFO, {"log_type": "test_suite", "message": "Starting calculator tests"})  # Log to console as well

    all_test_cases = setup_test_cases()  # Get the test cases
    test_results = {}  # Initialize a dictionary to store the test results
    total_start_time = time()  # Record the start time of the test suite
    total_tests = 0  # Initialize counters for total tests, passed tests, and failed tests
    passed_tests = 0
    failed_tests = 0

    precisions = [1, 2, 3, 5, 6, 9, 100, 1000]  # List of precisions to test with

    for test_name, (expression, expected_output_func, expected_text) in all_test_cases.items():  # Iterate through the test cases
        test_results[test_name] = {  # Initialize a dictionary to store the results for the current test case
            "expression": expression,
            "results": []
        }

        for precision in precisions:  # Iterate through the precisions
            mp.dps = precision  # Set the current precision
            start_time = time()  # Record the start time of the test
            actual_output, is_error = calculate(expression, source="test_case")  # Calculate the result
            end_time = time()  # Record the end time of the test
            elapsed_time = end_time - start_time  # Calculate the elapsed time

            if callable(expected_output_func):  # If the expected output is a function
                if test_name in error_test_cases:  # If the test case is an error test case
                    answer = format_result(expected_output_func())  # Call the function to get the expected output
                else:  # Otherwise, call the function with the current precision
                    answer = format_result(expected_output_func(precision))  # Call the function to get the expected output
            else:  # If the expected output is not a function
                answer = expected_text  # Use the provided expected text

            # Format the results *before* logging
            formatted_answer = format_result(answer)  # Format the expected answer
            formatted_actual_output = format_result(actual_output)  # Format the actual output

            actual_output = actual_output.strip("'")  # Remove any extra quotes from the actual output

            compare_result, comparison_details = compare_results(answer, actual_output)  # Compare the expected and actual outputs

            test_results[test_name]["results"].append({  # Append the test result to the list of results for the current test case
                "precision": precision,
                "answer": formatted_answer,  # Log the formatted answer
                "actual": formatted_actual_output,  # Log the formatted actual output
                "result": "Passed" if compare_result else "Failed",  # Log whether the test passed or failed
                "comparison_details": comparison_details,  # Log details about the comparison
                "elapsed_time": f"{elapsed_time:.6f}"
            })

            if is_error:  # If the calculation resulted in an error
                test_results[test_name]["results"][-1]["error"] = actual_output  # Add the error message
                total_tests += 1  # Increment the total tests counter
            if compare_result:  # If the comparison passed
                passed_tests += 1  # Increment the passed tests counter
            else:  # If the comparison failed
                failed_tests += 1  # Increment the failed tests counter

    total_end_time = time()  # Record the end time of the test suite
    total_elapsed_time = total_end_time - total_start_time  # Calculate the total elapsed time

    # Log the consolidated test results
    custom_log(test_logger, logging.INFO, {  # Log the test results to the test logger
        "log_type": "test_results",
        "results": test_results
    })

    # Log and print summary
    summary = {  # Create a summary of the test results
        "log_type": "test_summary",
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_elapsed_time": f"{total_elapsed_time:.6f} seconds"
    }

    custom_log(test_logger, logging.INFO, summary)  # Log the summary to the test logger
    custom_log(console_logger, logging.INFO, summary)  # Log the summary to the console logger

    print(f"Tests completed. Results saved to calculator_tests.json")  # Print a message indicating completion of the tests
    print(f"Total tests: {total_tests}")  # Print the total number of tests
    print(f"Passed tests: {passed_tests}")  # Print the number of passed tests
    print(f"Failed tests: {failed_tests}")  # Print the number of failed tests
    print(f"Total elapsed time: {total_elapsed_time:.6f} seconds")  # Print the total elapsed time

    # Report failed tests
    if failed_tests > 0:  # If there were any failed tests
        print("\nFailed tests:")  # Print a header for the failed tests section
        for test_name, test_data in test_results.items():  # Iterate through the test results
            for result in test_data["results"]:  # Iterate through the results for each test case
                if result["result"] == "Failed":  # If the test failed
                    print(f"  {test_name} (Precision: {result['precision']}):")  # Print the test name and precision
                    print(f"    Expression: {test_data['expression']}")  # Print the expression
                    print(f"    Answer: {result['answer']}")  # Print the expected answer
                    print(f"    Actual: {result['actual']}")  # Print the actual output
                    print(f"    Comparison details: {result['comparison_details']}")  # Print details about the comparison
                    if "error" in result:  # If there was an error message
                        print(f"    Error: {result['error']}")  # Print the error message
                    print()  # Print a blank line

    # Investigate precision issues
    precision_failures = [  # Create a list of failed tests with precision information
        f"{test_name} (Precision: {result['precision']})"
        for test_name, test_data in test_results.items()
        for result in test_data["results"]
        if result["result"] == "Failed"
    ]

    if precision_failures:  # If there were any precision failures
        failure_message = f"The following tests failed: {', '.join(precision_failures)}"  # Create a message listing the failed tests
        custom_log(console_logger, logging.WARNING, {  # Log a warning about the precision failures
            "log_type": "precision_investigation",
            "message": failure_message,
            "failed_tests": precision_failures
        })
        print(f"\nWarning: {failure_message}")  # Print the warning message

    return test_results  # Return the test results

@log_function_call
def display_menu():
    """Display the calculator menu."""
    menu = """
\033[95m╔═════════════════════════════════════════════╗\033[0m
\033[95m║\033[0m       \033[92mAdvanced Scientific Calculator  \033[0m      \033[95m║\033[0m
\033[95m╠═════════════════════════════════════════════╣\033[0m
\033[95m║\033[0m \033[92mSupported Operations:\033[0m                       \033[95m║\033[0m
\033[95m║\033[0m  Addition (+)            │   x + y          \033[95m║\033[0m
\033[95m║\033[0m  Subtraction (-)         │   x - y          \033[95m║\033[0m
\033[95m║\033[0m  Multiplication (*)      │   x * y          \033[95m║\033[0m
\033[95m║\033[0m  Division (/)            │   x / y          \033[95m║\033[0m
\033[95m║\033[0m  Exponentiation (**)     │   x ** y         \033[95m║\033[0m
\033[95m║\033[0m  Square Root (sqrt)      │   sqrt(x)        \033[95m║\033[0m
\033[95m║\033[0m  Modulo (%)              │   x % y          \033[95m║\033[0m
\033[95m║\033[0m  nth Root (root)         │   root(x, n)     \033[95m║\033[0m
\033[95m║\033[0m  Sine (sin)              │   sin(x)         \033[95m║\033[0m
\033[95m║\033[0m  Cosine (cos)            │   cos(x)         \033[95m║\033[0m
\033[95m║\033[0m  Tangent (tan)           │   tan(x)         \033[95m║\033[0m
\033[95m║\033[0m  Arcsine (asin)          │   asin(x)        \033[95m║\033[0m
\033[95m║\033[0m  Arccosine (acos)        │   acos(x)        \033[95m║\033[0m
\033[95m║\033[0m  Arctangent (atan)       │   atan(x)        \033[95m║\033[0m
\033[95m║\033[0m  Natural Logarithm (log) │   log(x)         \033[95m║\033[0m
\033[95m║\033[0m  Logarithm base 10       │   log10(x)       \033[95m║\033[0m
\033[95m║\033[0m  Logarithm base n        │   log(x, n)      \033[95m║\033[0m
\033[95m║\033[0m  Exponential (exp)       │   exp(x)         \033[95m║\033[0m
\033[95m║\033[0m  Factorial (!)           │   factorial(x)   \033[95m║\033[0m
\033[95m║\033[0m  Absolute Value (abs)    │   abs(x)         \033[95m║\033[0m
\033[95m║\033[0m  Pi Constant (pi)        │   pi             \033[95m║\033[0m
\033[95m║\033[0m  Euler's Constant (e)    │   E              \033[95m║\033[0m
\033[95m║\033[0m  Zeta Function           │   zeta(x)        \033[95m║\033[0m
\033[95m║\033[0m  Binomial Coefficient    │   binomial(n, k) \033[95m║\033[0m
\033[95m║\033[0m  Parentheses             │   (x + y) * z    \033[95m║\033[0m
\033[95m╠═════════════════════════════════════════════╣\033[0m
\033[95m║\033[0m \033[92mOptions:\033[0m                                    \033[95m║\033[0m
\033[95m║\033[0m   t.  Run Tests                             \033[95m║\033[0m
\033[95m║\033[0m   p.  Change Precision                      \033[95m║\033[0m
\033[95m║\033[0m   v.  View Precision                        \033[95m║\033[0m
\033[95m║\033[0m   h.  View History                          \033[95m║\033[0m
\033[95m║\033[0m   c.  Clear History                         \033[95m║\033[0m
\033[95m║\033[0m   l.  Clear All Logs                        \033[95m║\033[0m
\033[95m║\033[0m   x.  Exit                                  \033[95m║\033[0m
\033[95m╚═════════════════════════════════════════════╝\033[0m
"""  # The menu string with ANSI escape codes for formatting
    print(menu)  # Print the menu
    custom_log(console_logger, logging.DEBUG, "Displayed menu")  # Log that the menu was displayed

@log_function_call
def display_calculation_history(num_entries=10):
    """Display the recent calculation history."""
    try:
        with open(history_file, 'r') as file:  # Open the history file in read mode
            history = json.load(file)  # Load the history from the file
    except FileNotFoundError:  # If the history file is not found
        custom_log(console_logger, logging.WARNING, "No calculation history found.")  # Log a warning
        print("No calculation history found.")  # Print a message to the console
        return  # Return from the function
    except json.JSONDecodeError:  # If there's an error decoding the JSON data
        custom_log(console_logger, logging.ERROR, "Error decoding calculation history file.")  # Log an error
        print("Error reading calculation history. The file may be corrupted.")  # Print an error message
        return  # Return from the function

    history = history[-num_entries:]  # Get the last n entries
    for entry in reversed(history):  # Iterate through the history entries in reverse order
        print("\n" + "="*50)  # Print a separator
        print(f"\nTimestamp: {entry['timestamp']}")  # Print the timestamp
        print(f"Source: {entry['source']}")  # Print the source of the calculation
        print(f"Input: {entry['input']}")  # Print the input expression
        print(f"Parse: {entry['parse']}")  # Print the parsed expression
        if entry['result']:  # If there's a result, print it
            print(f"Result: {entry['result']}")
        if entry['error']:  # If there's an error, print it
            print(f"Error: {entry['error']}")
        if entry['elapsed_time']:  # If there's an elapsed time, print it
            print(f"Elapsed Time: {entry['elapsed_time']}")
        print(f"Precision: {entry['precision']} decimal places")  # Print the precision
    print("="*50)  # Print a separator
    custom_log(console_logger, logging.INFO, f"Displayed {len(history)} history entries")  # Log the number of history entries displayed

@log_function_call
def clear_calculation_history():
    """Clear the calculation history file."""
    try:
        os.remove(history_file)  # Try to remove the history file
        custom_log(console_logger, logging.INFO, "Calculation history has been cleared.")  # Log that the history was cleared
        print("\nCalculation history has been cleared.")  # Print a message to the console
    except FileNotFoundError:  # If the history file is not found
        custom_log(console_logger, logging.WARNING, "No calculation history file found.")  # Log a warning
        print("No calculation history file found.")  # Print a message to the console

@log_function_call
def change_precision():
    """Change the precision for calculations."""
    while True:  # Keep asking for input until a valid precision is entered
        try:
            new_precision = int(input("Enter new precision (number of decimal places): "))  # Get the new precision from the user
            if new_precision < 1:  # If the precision is less than 1
                custom_log(console_logger, logging.WARNING, f"Attempted to set invalid precision: {new_precision}")  # Log a warning
                print("Precision must be at least 1 decimal place.")  # Print an error message
            else:  # If the precision is valid
                mp.dps = new_precision  # Set the new precision in mpmath
                custom_log(console_logger, logging.INFO, f"Precision changed to {mp.dps} decimal places")  # Log the precision change
                print(f"\nPrecision successfully changed to {mp.dps} decimal places.\n")  # Print a success message
                break  # Exit the loop
        except ValueError:  # If the input is not a valid integer
            custom_log(console_logger, logging.WARNING, "Invalid input for precision change")  # Log a warning
            print("Invalid input. Please enter a positive integer.")  # Print an error message

@log_function_call
def view_precision():
    """Display the current precision."""
    custom_log(console_logger, logging.INFO, f"Current precision: {mp.dps} decimal places")  # Log the current precision
    print(f"Current precision: {mp.dps} decimal places")  # Print the current precision

@log_function_call
def main():
    """Main function to run the calculator."""
    custom_log(console_logger, logging.INFO, {"message": "Starting Advanced Scientific Calculator with Infinite Precision"})  # Log the start of the calculator
    print("\n\nWelcome to the Advanced Scientific Calculator with Infinite Precision")  # Print a welcome message
    setup_unlimited_precision()  # Set up unlimited precision
    
    user_input_time = 0  # Initialize a variable to track user input time
    
    while True:  # Main loop of the calculator
        display_menu()  # Display the menu
        input_start_time = time()  # Record the start time of user input
        choice = input("\033[40;93mEnter expression or option: \033[0m").strip().lower()  # Get user input
        input_end_time = time()  # Record the end time of user input
        user_input_time += input_end_time - input_start_time  # Calculate and add to user input time
        
        custom_log(console_logger, logging.INFO, {"user_input": choice})  # Log the user input
        
        if choice == 'x':  # If the user chooses to exit
            custom_log(console_logger, logging.INFO, {"message": "Exiting calculator"})  # Log the exit
            break  # Exit the loop
        elif choice == 't':  # If the user chooses to run tests
            custom_log(console_logger, logging.INFO, {"message": "Running tests"})  # Log that the tests are running
            test_results = test_calculator()  # Run the tests
        elif choice == 'h':  # If the user chooses to view history
            try:
                num_entries = int(input("How many recent calculations would you like to see? "))  # Get the number of entries to display
                custom_log(console_logger, logging.INFO, {"action": "view_history", "num_entries": num_entries})  # Log the action and number of entries
                display_calculation_history(num_entries)  # Display the history
            except ValueError:  # If the input is not a valid integer
                custom_log(console_logger, logging.WARNING, "Invalid input for history entries")  # Log a warning
                print("Invalid input. Please enter a positive integer.")  # Print an error message
        elif choice == 'c':  # If the user chooses to clear history
            custom_log(console_logger, logging.INFO, {"action": "clear_history"})  # Log the action
            clear_calculation_history()  # Clear the history
        elif choice == 'l':  # If the user chooses to clear all logs
            custom_log(console_logger, logging.INFO, {"action": "clear_all_logs"})  # Log the action
            clear_all_logs()  # Clear all logs
        elif choice == 'p':  # If the user chooses to change precision
            custom_log(console_logger, logging.INFO, {"action": "change_precision"})  # Log the action
            change_precision()  # Change the precision
        elif choice == 'v':  # If the user chooses to view precision
            custom_log(console_logger, logging.INFO, {"action": "view_precision"})  # Log the action
            view_precision()  # View the precision
        else:  # If the user enters an expression
            try:
                custom_log(console_logger, logging.INFO, {  # Log the calculation attempt
                    "action": "calculating",
                    "expression": choice,
                    "precision": mp.dps
                })
                signal.signal(signal.SIGALRM, timeout_handler)  # Set up a signal handler for timeouts
                signal.alarm(10)  # Set a 10-second timeout for calculations
                start_time = time()  # Record the start time of the calculation
                result, is_error = calculate(choice)  # Calculate the result
                end_time = time()  # Record the end time of the calculation
                signal.alarm(0)  # Disable the alarm
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                
                if is_error:  # If there was an error during calculation
                    print(f"\033[31m{result}\033[0m")  # Print the error message in red
                else:  # If the calculation was successful
                    print(f"\033[92mResult: {result}\033[0m")  # Print the result in green
                
                print(f"Calculation time: {elapsed_time:.6f} seconds")  # Print the calculation time
                print(f"Precision: {mp.dps} decimal places")  # Print the precision
                custom_log(console_logger, logging.INFO, {  # Log the calculation result
                    "calculation_result": result,
                    "calculation_time": f"{elapsed_time:.6f} seconds",
                    "precision": mp.dps
                })
            except TimeoutError:  # If the calculation timed out
                print("\033[31mError: Calculation timed out\033[0m")  # Print a timeout error message in red
                custom_log(console_logger, logging.ERROR, {"error": "Calculation timed out"})  # Log the timeout error
            except CalculationError as ce:  # If there was a calculation error
                print(f"\033[31mCalculation Error: {str(ce)}\033[0m")  # Print the calculation error message in red
                custom_log(console_logger, logging.ERROR, {"error": f"Calculation Error: {str(ce)}"})  # Log the calculation error
            except Exception as e:  # If there was any other error
                print(f"\033[31mError: {e}\033[0m")  # Print the error message in red
                custom_log(console_logger, logging.ERROR, {"error": f"Calculation error: {e}"})  # Log the error
                custom_log(console_logger, logging.ERROR, {"traceback": traceback.format_exc()})  # Log the traceback

    print("Thank you for using the Advanced Scientific Calculator with Infinite Precision!")  # Print a thank you message
    custom_log(console_logger, logging.INFO, {"message": "Advanced Scientific Calculator session ended"})  # Log the end of the session
    
    # Report function timings
    report_function_timings()  # Report the function timings
    print(f"\nTotal user input time: {user_input_time:.6f} seconds")  # Print the total user input time
    custom_log(console_logger, logging.INFO, {"total_user_input_time": f"{user_input_time:.6f} seconds"})  # Log the total user input time

if __name__ == "__main__":  # If the script is run directly
    try:
        custom_log(console_logger, logging.INFO, "Initializing calculator environment")  # Log the initialization of the environment
        venv_path = create_venv()  # Create a virtual environment
        venv_python = get_venv_python(venv_path)  # Get the path to the Python executable in the virtual environment
        install_libraries(venv_python)  # Install the required libraries
        main()  # Run the main function
    except Exception as e:  # If there's an exception during initialization or main execution
        custom_log(console_logger, logging.CRITICAL, f"Fatal error in main execution: {str(e)}")  # Log a critical error
        custom_log(console_logger, logging.CRITICAL, {"traceback": traceback.format_exc()})  # Log the traceback
        print(f"\033[31mA critical error occurred. Please check the log file for details.\033[0m")  # Print an error message in red
    finally:  # Always execute this block
        custom_log(console_logger, logging.INFO, "Calculator application terminated")  # Log the termination of the application
        logging.shutdown()  # Shut down the logging system
