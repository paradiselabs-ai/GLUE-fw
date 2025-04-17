"""
Few-shot prompting implementation for tool calling in GLUE framework.

This module provides functionality to enhance tool calling accuracy through few-shot examples.
It dynamically generates example conversations that demonstrate correct tool usage patterns
to help models learn proper tool invocation syntax and parameter formatting.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set, cast
import inspect

from glue.core.types import Message  

logger = logging.getLogger("glue.core.few_shot_tool_calling")

class FewShotExampleGenerator:
    """
    Generates few-shot examples for tool calling.
    
    This class helps create example conversations that demonstrate correct tool usage
    to the model. These examples are inserted in the conversation history to improve
    the model's ability to correctly format and make tool calls.
    
    The generator can create examples for various tool types including file handlers,
    web search tools, code interpreters, and communication tools. It supports multiple
    tool call formats to accommodate different LLM preferences.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the few-shot example generator.
        
        Args:
            config: Optional configuration dictionary with the following options:
                - max_examples: Maximum number of examples to generate (default: 5)
                - formats: List of formats to include ("function", "json", "yaml", default: all)
                - include_explanation: Whether to include explanations (default: True)
                - custom_examples: Dictionary of predefined examples by tool name
        """
        self.config = config or {}
        self.max_examples = self.config.get("max_examples", 5)
        self.formats = self.config.get("formats", ["function", "json", "yaml"])
        self.include_explanation = self.config.get("include_explanation", True)
        self.custom_examples = self.config.get("custom_examples", {})
        
    def generate_examples(self, tools: List[Dict[str, Any]]) -> List[Message]:
        """
        Generate few-shot examples for the given tools.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            List of message objects containing example conversations
        """
        if not tools:
            return []
        
        examples: List[Message] = []
        
        # Process tools by priority (based on presence in tool registry and relevance)
        prioritized_tools = self._prioritize_tools(tools)
        
        # Generate examples for at most max_examples tools
        for tool in prioritized_tools[:self.max_examples]:
            # Check for custom example first
            if tool.get("name") in self.custom_examples:
                user_msg, asst_msg = self._get_custom_example(tool)
                examples.append(Message(role="user", content=user_msg))
                examples.append(Message(role="assistant", content=asst_msg))
                continue
                
            # Generate user message asking for something that would need this tool
            user_message, query_type = self._generate_user_message(tool)
            
            # Generate assistant message with correct tool calling
            assistant_message = self._generate_assistant_message(tool, query_type)
            
            # Add the example conversation
            examples.append(Message(role="user", content=user_message))
            examples.append(Message(role="assistant", content=assistant_message))
        
        return examples
    
    def _prioritize_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize tools based on relevance and complexity.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Prioritized list of tool definitions
        """
        # Define priority categories
        essential_tools = {"file_handler", "web_search", "code_interpreter", "communicate"}
        
        # Sort tools: essential tools first, then alphabetically by name
        def priority_key(tool: Dict[str, Any]) -> Tuple[int, str]:
            tool_name = tool.get("name", "").lower()
            is_essential = any(essential in tool_name for essential in essential_tools)
            return (0 if is_essential else 1, tool_name)
            
        return sorted(tools, key=priority_key)
    
    def _get_custom_example(self, tool: Dict[str, Any]) -> Tuple[str, str]:
        """
        Get a custom example for a specific tool.
        
        Args:
            tool: Tool definition
            
        Returns:
            Tuple containing user message and assistant message
        """
        tool_name = tool.get("name", "")
        custom = self.custom_examples.get(tool_name, {})
        return custom.get("user", f"Can you use the {tool_name} tool?"), custom.get("assistant", "")
    
    def _generate_user_message(self, tool: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a user message that would prompt the use of a specific tool.
        
        Args:
            tool: Tool definition
            
        Returns:
            Tuple containing the user message string and the query type
        """
        tool_name = tool.get("name", "").lower()
        tool_description = tool.get("description", "").lower()
        
        # File Handler tool
        if any(x in tool_name or x in tool_description for x in ["file", "read", "write"]):
            if "write" in tool_name or "write" in tool_description:
                return "Can you create a new file called notes.txt with the text 'Meeting notes from today'?", "file_write"
            else:
                return "Can you read the content of README.md?", "file_read"
                
        # Web Search tool
        elif any(x in tool_name or x in tool_description for x in ["search", "web", "retrieve", "lookup"]):
            return "Can you search for the latest developments in quantum computing?", "search"
            
        # Code Interpreter tool
        elif any(x in tool_name or x in tool_description for x in ["code", "execute", "interpret", "run"]):
            return "Can you execute this Python code: `print('Hello ' + 'world')`?", "code"
            
        # Communicate tool
        elif any(x in tool_name or x in tool_description for x in ["communicate", "message", "chat", "email"]):
            return "Can you ask the research team about climate change data?", "communicate"
            
        # Generic fallback based on tool name
        else:
            return f"I need help with {tool_name}. What can you tell me about it?", "generic"
    
    def _generate_assistant_message(self, tool: Dict[str, Any], query_type: str) -> str:
        """
        Generate an assistant message with the correct tool call.
        
        Args:
            tool: Tool definition
            query_type: Type of query to generate response for
            
        Returns:
            Assistant message string with tool call
        """
        tool_name = tool.get("name", "")
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        
        # Build a sample parameters object with valid values
        params = self._generate_parameters(properties, required, query_type)
        
        # Format the tool call in different styles to accommodate various models
        response = "I'll help you with that."
        
        if "function" in self.formats:
            # Function call syntax (for models like OpenAI)
            response += "\n\n```\n"
            params_str = ", ".join([f"{k}={json.dumps(v)}" for k, v in params.items()])
            response += f"{tool_name}({params_str})\n"
            response += "```"
        
        if "json" in self.formats:
            # JSON format (for models like Claude and others)
            response += "\n\n```json\n"
            response += "{\n"
            response += f'  "tool": "{tool_name}",\n'
            response += '  "parameters": {\n'
            for i, (k, v) in enumerate(params.items()):
                # Format the value as a JSON string
                value_str = json.dumps(v)
                comma = "," if i < len(params) - 1 else ""
                response += f'    "{k}": {value_str}{comma}\n'
            response += "  }\n"
            response += "}\n"
            response += "```"
        
        if "yaml" in self.formats:
            # Tool call format with YAML-like syntax
            response += "\n\n```tool_call\n"
            response += f"tool_name: {tool_name}\n"
            response += "parameters:\n"
            for k, v in params.items():
                # Format the value appropriately for YAML-like syntax
                if isinstance(v, str):
                    value_str = f'"{v}"'
                else:
                    value_str = json.dumps(v)
                response += f'  {k}: {value_str}\n'
            response += "```"
        
        # Add explanation if configured
        if self.include_explanation:
            response += "\n\n"
            if query_type == "file_read":
                response += "This will read the content of the README.md file."
            elif query_type == "file_write":
                response += "This will create a new file named notes.txt with the specified content."
            elif query_type == "search":
                response += "This will search for information about the latest developments in quantum computing."
            elif query_type == "code":
                response += "This will execute the Python code to print 'Hello world'."
            elif query_type == "communicate":
                response += "This will send a message to the research team asking about climate change data."
            else:
                response += f"This will use the {tool_name} tool with the parameters you requested."
            
        return response
    
    def _generate_parameters(self, properties: Dict[str, Any], required: List[str], query_type: str) -> Dict[str, Any]:
        """
        Generate sample parameters for a tool based on query type.
        
        Args:
            properties: Tool parameter properties
            required: List of required parameters
            query_type: Type of query to generate parameters for
            
        Returns:
            Dictionary of parameter names to example values
        """
        params: Dict[str, Any] = {}
        
        # Set parameters based on tool type and query
        if query_type == "file_read":
            self._add_if_exists(params, properties, "path", "README.md")
            self._add_if_exists(params, properties, "file_path", "README.md") 
            self._add_if_exists(params, properties, "action", "read")
        
        elif query_type == "file_write":
            self._add_if_exists(params, properties, "path", "notes.txt")
            self._add_if_exists(params, properties, "file_path", "notes.txt")
            self._add_if_exists(params, properties, "content", "Meeting notes from today")
            self._add_if_exists(params, properties, "action", "write")
        
        elif query_type == "search":
            self._add_if_exists(params, properties, "query", "latest developments in quantum computing")
            self._add_if_exists(params, properties, "search_query", "latest developments in quantum computing")
            self._add_if_exists(params, properties, "num_results", 5)
            self._add_if_exists(params, properties, "max_results", 5)
        
        elif query_type == "code":
            self._add_if_exists(params, properties, "code", "print('Hello ' + 'world')")
            self._add_if_exists(params, properties, "language", "python")
            
        elif query_type == "communicate":
            self._add_if_exists(params, properties, "target_type", "team")
            self._add_if_exists(params, properties, "target_name", "research")
            self._add_if_exists(params, properties, "message", "What can you tell me about climate change data?")
        
        # Add any required parameters that we haven't set yet
        self._ensure_required_parameters(params, properties, required)
            
        return params
    
    def _add_if_exists(self, params: Dict[str, Any], properties: Dict[str, Any], 
                      param_name: str, value: Any) -> None:
        """
        Add a parameter to the params dict if it exists in properties.
        
        Args:
            params: Parameters dictionary to update
            properties: Tool parameter properties
            param_name: Name of the parameter to check
            value: Value to set if parameter exists
        """
        if param_name in properties:
            params[param_name] = value
    
    def _ensure_required_parameters(self, params: Dict[str, Any], 
                                   properties: Dict[str, Any], 
                                   required: List[str]) -> None:
        """
        Ensure all required parameters are present in the params dictionary.
        
        Args:
            params: Parameters dictionary to update
            properties: Tool parameter properties
            required: List of required parameter names
        """
        for param_name in required:
            if param_name not in params:
                param_info = properties.get(param_name, {})
                param_type = param_info.get("type", "string")
                enum_values = param_info.get("enum", [])
                
                # Set a default value based on the parameter type
                if param_type == "string":
                    params[param_name] = enum_values[0] if enum_values else f"sample_{param_name}"
                elif param_type in ("integer", "number"):
                    params[param_name] = 1
                elif param_type == "boolean":
                    params[param_name] = True
                elif param_type == "array":
                    params[param_name] = []
                elif param_type == "object":
                    params[param_name] = {}


def add_few_shot_examples(
    messages: List[Union[Message, Dict[str, Any]]], 
    tools: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> List[Union[Message, Dict[str, Any]]]:
    """
    Add few-shot examples to a list of messages to help the model with tool calling.
    
    This function generates example conversations demonstrating proper tool usage
    and inserts them into the message history at the appropriate position.
    
    Args:
        messages: Original list of messages in either Message objects or dicts
        tools: List of tools that may be used
        config: Optional configuration for few-shot examples with options:
            - max_examples: Maximum number of examples to include
            - formats: List of formats to include ("function", "json", "yaml")
            - include_explanation: Whether to include explanations
            - custom_examples: Dictionary of predefined examples by tool name
            - position: Where to insert examples ("start", "after_system", default: "after_system")
        
    Returns:
        Messages list with few-shot examples inserted
    """
    # If no tools, no examples needed
    if not tools:
        return messages
    
    # Get position configuration
    position = config.get("position", "after_system") if config else "after_system"
        
    # Generate examples
    example_generator = FewShotExampleGenerator(config)
    examples = example_generator.generate_examples(tools)
    
    if not examples:
        return messages
    
    # Insert examples at the right position based on configuration
    if position == "start":
        enhanced_messages = examples + messages
    else:  # "after_system" (default)
        # Check if there's a system message
        has_system = False
        system_idx = -1
        
        for i, msg in enumerate(messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "system":
                has_system = True
                system_idx = i
                break
        
        # Insert examples after system message or at beginning
        if has_system:
            # Insert after system message
            enhanced_messages = messages[:system_idx+1] + examples + messages[system_idx+1:]
        else:
            # Insert at beginning
            enhanced_messages = examples + messages
    
    # Log the enhancement
    logger.info(f"Added {len(examples)} few-shot examples for {len(tools)} tools")
    
    return enhanced_messages 