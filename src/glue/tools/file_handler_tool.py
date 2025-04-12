"""
File handler tool for the GLUE framework.

This module implements a file handler tool that can perform various file operations
such as reading, writing, appending, deleting, and listing files.
"""
import os
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

from .tool_base import ToolConfig, ToolPermission
from .pydantic_validated_tool import PydanticValidatedTool

logger = logging.getLogger(__name__)

class FileOperation(str, Enum):
    """Types of supported file operations"""
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    EXISTS = "exists"
    LIST = "list"

class FileHandlerInput(BaseModel):
    """Input schema for file handler tool"""
    operation: FileOperation = Field(description="File operation to perform")
    path: str = Field(description="Path to the file or directory")
    content: Optional[str] = Field(default=None, description="Content to write or append to the file")
    recursive: Optional[bool] = Field(default=False, description="Whether to perform the operation recursively")

class FileHandlerOutput(BaseModel):
    """Output schema for file handler tool"""
    success: bool = Field(description="Whether the operation was successful")
    operation: str = Field(description="Operation that was performed")
    path: str = Field(description="Path to the file or directory")
    content: Optional[str] = Field(default=None, description="Content read from the file")
    files: Optional[List[str]] = Field(default=None, description="List of files in the directory")
    exists: Optional[bool] = Field(default=None, description="Whether the file exists")
    message: Optional[str] = Field(default=None, description="Additional information about the operation")

class FileHandlerTool(PydanticValidatedTool):
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
        
        # Initialize the PydanticValidatedTool with our schemas
        super().__init__(
            name, 
            description, 
            input_schema=FileHandlerInput,
            output_schema=FileHandlerOutput,
            config=config
        )
    
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
            input_data: Dictionary containing operation details
            
        Returns:
            Dictionary containing the operation result
            
        Raises:
            ValueError: If the operation is not supported or required parameters are missing
        """
        # Input validation is already handled by PydanticValidatedTool
        operation_str = input_data["operation"]
        path = input_data["path"]
        content = input_data.get("content")
        recursive = input_data.get("recursive", False)
        
        # Normalize the path
        path = os.path.expanduser(path)
        
        # Check if the operation is valid
        try:
            operation = FileOperation(operation_str)
        except ValueError:
            return {
                "success": False,
                "operation": operation_str,
                "path": path,
                "message": f"Unsupported operation: {operation_str}"
            }
        
        # Execute the operation
        if operation == FileOperation.READ:
            return await self._read_file(path)
        elif operation == FileOperation.WRITE:
            return await self._write_file(path, content)
        elif operation == FileOperation.APPEND:
            return await self._append_file(path, content)
        elif operation == FileOperation.DELETE:
            return await self._delete_file(path)
        elif operation == FileOperation.EXISTS:
            return await self._file_exists(path)
        elif operation == FileOperation.LIST:
            return await self._list_files(path, recursive)
        else:
            # This should never happen due to the enum validation
            return {
                "success": False,
                "operation": operation_str,
                "path": path,
                "message": f"Unsupported operation: {operation_str}"
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
                "path": file_path,
                "message": f"File not found: {file_path}"
            }
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return {
            "success": True,
            "operation": FileOperation.READ,
            "path": file_path,
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
            "path": file_path
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
                "path": file_path,
                "message": f"File not found: {file_path}"
            }
        
        with open(file_path, 'a') as f:
            f.write(content)
        
        return {
            "success": True,
            "operation": FileOperation.APPEND,
            "path": file_path
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
                "path": file_path,
                "message": f"File not found: {file_path}"
            }
        
        os.remove(file_path)
        
        return {
            "success": True,
            "operation": FileOperation.DELETE,
            "path": file_path
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
            "path": file_path,
            "exists": exists
        }
    
    async def _list_files(self, directory: str, recursive: bool) -> Dict[str, Any]:
        """
        List the contents of a directory.
        
        Args:
            directory: Path to the directory to list
            recursive: Whether to list recursively
            
        Returns:
            Dictionary containing the list of files in the directory
        """
        if not os.path.exists(directory):
            return {
                "success": False,
                "operation": FileOperation.LIST,
                "path": directory,
                "message": f"Directory not found: {directory}"
            }
        
        if not os.path.isdir(directory):
            return {
                "success": False,
                "operation": FileOperation.LIST,
                "path": directory,
                "message": f"Not a directory: {directory}"
            }
        
        if recursive:
            files = []
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            files = [os.path.join(directory, f) for f in os.listdir(directory)]
        
        return {
            "success": True,
            "operation": FileOperation.LIST,
            "path": directory,
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
