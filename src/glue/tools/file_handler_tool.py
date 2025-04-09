"""
File handler tool for the GLUE framework.

This module implements a file handler tool that can perform various file operations
such as reading, writing, appending, deleting, and listing files.
"""
import os
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from .tool_base import Tool, ToolConfig, ToolPermission

logger = logging.getLogger(__name__)

class FileOperation(str, Enum):
    """Types of supported file operations"""
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    EXISTS = "exists"
    LIST = "list"

class FileHandlerTool(Tool):
    """Tool for handling file operations"""
    
    def __init__(
        self,
        name: str = "file_handler",
        description: str = "Perform file operations such as reading, writing, and listing files",
        config: Optional[ToolConfig] = None
    ):
        """
        Initialize the file handler tool.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            config: Tool configuration
        """
        # Create default tool config if not provided
        if config is None:
            config = ToolConfig(
                required_permissions={ToolPermission.READ, ToolPermission.WRITE}
            )
        
        super().__init__(name, description, config)
    
    async def initialize(self, instance_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the tool.
        
        Args:
            instance_data: Optional instance data for initialization
        """
        # No special initialization needed, just call the parent method
        await super().initialize(instance_data)
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file operation.
        
        Args:
            input_data: Dictionary containing 'operation' and 'file_path' and possibly 'content'
            
        Returns:
            Dictionary containing the operation result
            
        Raises:
            ValueError: If the operation or file_path is missing or invalid
        """
        # Handle string input by assuming it's a file path for a read operation
        if isinstance(input_data, str):
            # If input is a string, assume it's a file path for a read operation
            input_data = {
                "operation": "read",
                "file_path": input_data
            }
            logger.info(f"Converted string input to read operation for path: {input_data['file_path']}")
        
        # Get the operation
        operation_str = input_data.get("operation")
        if not operation_str:
            raise ValueError("File operation is required")
        
        # Convert string to enum if needed
        if isinstance(operation_str, str):
            try:
                operation = FileOperation(operation_str.lower())
            except ValueError:
                raise ValueError(f"Invalid file operation: {operation_str}")
        else:
            operation = operation_str
        
        # Get the file path
        file_path = input_data.get("file_path")
        if not file_path:
            raise ValueError("File path is required")
        
        # Execute the appropriate operation
        try:
            if operation == FileOperation.READ:
                return await self._read_file(file_path)
            elif operation == FileOperation.WRITE:
                content = input_data.get("content")
                if content is None:
                    raise ValueError("Content is required for write operation")
                return await self._write_file(file_path, content)
            elif operation == FileOperation.APPEND:
                content = input_data.get("content")
                if content is None:
                    raise ValueError("Content is required for append operation")
                return await self._append_file(file_path, content)
            elif operation == FileOperation.DELETE:
                return await self._delete_file(file_path)
            elif operation == FileOperation.EXISTS:
                return await self._file_exists(file_path)
            elif operation == FileOperation.LIST:
                return await self._list_directory(file_path)
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
        except Exception as e:
            logger.error(f"Error during file operation {operation} on {file_path}: {str(e)}")
            return {
                "success": False,
                "operation": operation,
                "file_path": file_path,
                "error": str(e)
            }
    
    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Dictionary containing the file content
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "operation": FileOperation.READ,
                "file_path": file_path,
                "error": f"File not found: {file_path}"
            }
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return {
            "success": True,
            "operation": FileOperation.READ,
            "file_path": file_path,
            "content": content
        }
    
    async def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write to a file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            Dictionary containing the result of the operation
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return {
            "success": True,
            "operation": FileOperation.WRITE,
            "file_path": file_path
        }
    
    async def _append_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append to a file.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append to the file
            
        Returns:
            Dictionary containing the result of the operation
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "operation": FileOperation.APPEND,
                "file_path": file_path,
                "error": f"File not found: {file_path}"
            }
        
        with open(file_path, 'a') as f:
            f.write(content)
        
        return {
            "success": True,
            "operation": FileOperation.APPEND,
            "file_path": file_path
        }
    
    async def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Dictionary containing the result of the operation
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "operation": FileOperation.DELETE,
                "file_path": file_path,
                "error": f"File not found: {file_path}"
            }
        
        os.remove(file_path)
        
        return {
            "success": True,
            "operation": FileOperation.DELETE,
            "file_path": file_path
        }
    
    async def _file_exists(self, file_path: str) -> Dict[str, Any]:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Dictionary containing whether the file exists
        """
        exists = os.path.exists(file_path)
        
        return {
            "success": True,
            "operation": FileOperation.EXISTS,
            "file_path": file_path,
            "exists": exists
        }
    
    async def _list_directory(self, directory: str) -> Dict[str, Any]:
        """
        List the contents of a directory.
        
        Args:
            directory: Path to the directory to list
            
        Returns:
            Dictionary containing the list of files in the directory
        """
        if not os.path.exists(directory):
            return {
                "success": False,
                "operation": FileOperation.LIST,
                "directory": directory,
                "error": f"Directory not found: {directory}"
            }
        
        if not os.path.isdir(directory):
            return {
                "success": False,
                "operation": FileOperation.LIST,
                "directory": directory,
                "error": f"Not a directory: {directory}"
            }
        
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        
        return {
            "success": True,
            "operation": FileOperation.LIST,
            "directory": directory,
            "files": files
        }
    
    async def cleanup(self) -> None:
        """Clean up the tool resources"""
        # No special cleanup needed, just call the parent method
        await super().cleanup()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary.
        
        Returns:
            A dictionary representation of the tool
        """
        result = super().to_dict()
        return result
