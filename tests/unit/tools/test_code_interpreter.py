#!/usr/bin/env python
"""
Simple test script for the code interpreter tool.
"""
import os
import sys
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the tool classes
from glue.tools.tool_base import Tool, ToolConfig, ToolPermission
from glue.core.types import AdhesiveType

# Try to import the code interpreter tool
try:
    from glue.tools.code_interpreter_tool import CodeInterpreterTool, CodeLanguage
    print("Successfully imported CodeInterpreterTool")
    
    # Create an instance
    tool = CodeInterpreterTool()
    print(f"Created tool: {tool.name}")
    
    # Test initialization
    async def test_init():
        await tool.initialize()
        print(f"Initialized tool: {tool._initialized}")
    
    # Run the async function
    asyncio.run(test_init())
    
    print("Test completed successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
