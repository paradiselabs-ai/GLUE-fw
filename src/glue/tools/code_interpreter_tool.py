"""
Code interpreter tool for the GLUE framework.

This module implements a code interpreter tool that can execute code in various languages,
currently supporting Python. The tool provides a sandboxed environment for code execution
with timeout handling and state persistence between executions. Timeout handling is done by
the parent execute method, and a TimeoutError can be raised if code execution times out.
"""
import io
import sys
import time
import asyncio
import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional, Union
from contextlib import redirect_stdout, redirect_stderr

from .tool_base import Tool, ToolConfig, ToolPermission
from ..core.types import AdhesiveType

logger = logging.getLogger(__name__)

class CodeLanguage(str, Enum):
    """Supported code languages for interpretation"""
    PYTHON = "python"
    # Future languages can be added here

class CodeInterpreterTool(Tool):
    """Tool for executing code in various languages"""
    
    def __init__(
        self,
        name: str = "code_interpreter",
        description: str = "Execute code in various programming languages with state persistence",
        config: Optional[ToolConfig] = None
    ):
        """
        Initialize the code interpreter tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            config: Tool configuration
        """
        # Create default tool config if not provided
        if config is None:
            config = ToolConfig(
                required_permissions={ToolPermission.EXECUTE}
            )
        
        super().__init__(name, description, config)
        
        # Runtime state
        self._namespace = {}
    
    async def initialize(self, instance_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the tool.
        
        Args:
            instance_data: Optional instance data for initialization
        """
        await super().initialize(instance_data)
        
        # Initialize the namespace for code execution
        self._namespace = {}
        
        # If instance data is provided, load the namespace
        if instance_data and "namespace" in instance_data:
            # Only load safe types from the namespace
            for key, value in instance_data["namespace"].items():
                if isinstance(value, (str, int, float, bool, list, dict, tuple, set)):
                    self._namespace[key] = value
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code in the specified language.
        
        Args:
            input_data: Dictionary containing 'language' and 'code'
            
        Returns:
            Dictionary containing the execution result
            
        Raises:
            ValueError: If the language or code is missing or invalid
            asyncio.TimeoutError: If code execution times out
        """
        language = None
        code = None

        # Check if input_data is a string (assume Python code)
        if isinstance(input_data, str):
            code = input_data
            language = CodeLanguage.PYTHON
            logger.debug("CodeInterpreterTool received raw string input, assuming Python code.")
        # Check if input_data is a dictionary
        elif isinstance(input_data, dict):
            language_str = input_data.get("language")
            code = input_data.get("code")

            if not language_str:
                 # Try to infer language if only code is provided in dict
                 if code:
                     logger.warning("Language not specified, defaulting to Python.")
                     language = CodeLanguage.PYTHON
                 else:
                    return {"success": False, "error": "Code is required"}
            else:
                # Convert string to enum if needed
                if isinstance(language_str, str):
                    try:
                        language = CodeLanguage(language_str.lower())
                    except ValueError:
                        return {"success": False, "error": f"Unsupported language: {language_str}"}
                else:
                     # Assume it's already CodeLanguage enum
                     language = language_str 
        else:
            return {"success": False, "error": "Invalid input_data format. Expected string or dict."}

        # Execute the code in the appropriate language
        if language == CodeLanguage.PYTHON:
            # The timeout is handled by the parent execute method
            # We don't need to catch TimeoutError here as it will be propagated
            return await self._execute_python(code)
        else:
            return {
                "success": False,
                "error": f"Unsupported language: {language}"
            }
    
    async def _execute_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing the execution result
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Track execution time
        start_time = time.time()
        
        # Execute the code
        try:
            # Redirect stdout and stderr
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code in the namespace
                exec_globals = {
                    "__builtins__": __builtins__,
                    **self._namespace
                }
                
                # Execute the code and get the last expression value
                code_obj = compile(code, "<string>", "exec")
                exec(code_obj, exec_globals)
                
                # Update the namespace with the new variables
                self._namespace.update(exec_globals)
                
                # Try to get the result of the last expression
                result = None
                try:
                    # Compile the last line as an expression to get its value
                    last_line = code.strip().split('\n')[-1]
                    if last_line and not last_line.strip().startswith(('#', 'import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with')):
                        try:
                            expr_code = compile(last_line, "<string>", "eval")
                            result = eval(expr_code, exec_globals)
                        except:
                            # If the last line is not a valid expression, ignore it
                            pass
                except:
                    # If there's an error getting the last expression, ignore it
                    pass
                
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Get stdout and stderr
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            return {
                "success": True,
                "result": str(result) if result is not None else "",
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time
            }
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Get stdout and stderr
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Get the error traceback
            error_traceback = traceback.format_exc()
            
            # Include the error type in the error message
            error_type = type(e).__name__
            error_message = f"{error_type}: {str(e)}"
            
            return {
                "success": False,
                "error": error_message,
                "traceback": error_traceback,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time
            }

    async def cleanup(self) -> None:
        """Clean up the tool resources"""
        # Clear the namespace
        self._namespace.clear()
        
        # Call the parent cleanup method
        await super().cleanup()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary.
        
        Returns:
            A dictionary representation of the tool
        """
        result = super().to_dict()
        
        # Add code interpreter specific information
        result["supported_languages"] = [lang.value for lang in CodeLanguage]
        
        return result
