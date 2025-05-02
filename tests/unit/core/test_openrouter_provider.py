"""
Tests for the OpenRouter provider implementation in the GLUE framework.

This module contains tests for the OpenRouter provider class that handles
communication with the OpenRouter API for model interactions.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from glue.core.schemas import Message, ToolCall, ToolResult, ModelConfig
from glue.core.model import BaseModel, ModelProvider
from glue.core.providers.openrouter import OpenRouterProvider
from glue.core.types import AdhesiveType

# Mock the openai module since we don't have it installed
@pytest.fixture(autouse=True)
def mock_openai_import():
    """Mock the openai import to avoid ImportError"""
    with patch.dict('sys.modules', {'openai': MagicMock()}):
        yield


class TestOpenRouterProvider:
    """Tests for the OpenRouterProvider class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # This config matches the DSL syntax from the example GLUE app
        self.config = ModelConfig(
            name="researcher",
            provider=ModelProvider.OPENROUTER,
            model="meta-llama/llama-3.1-405b-instruct:free",
            temperature=0.7,
            max_tokens=2048,
            description="Research different topics and subjects online."
        )
        
        # Mock the _initialize_client method to prevent actual API client creation
        with patch.object(OpenRouterProvider, '_initialize_client'):
            self.model = BaseModel(self.config)
            self.provider = OpenRouterProvider(self.model)
            
            # Mock the client
            self.provider.client = MagicMock()
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self):
        """Test generating a basic response without tools"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        
        self.provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [
            Message(role="user", content="Hello, world!")
        ]
        
        response = await self.provider.generate_response(messages)
        
        assert response == "Test response"
        self.provider.client.chat.completions.create.assert_called_once()
        
        # Verify the call arguments
        call_args = self.provider.client.chat.completions.create.call_args[1]
        assert call_args["model"] == "meta-llama/llama-3.1-405b-instruct:free"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 2048
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Hello, world!"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_tools(self):
        """Test generating a response with tools"""
        # Mock the API response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = '{"query": "test query"}'
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        
        self.provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [
            Message(role="user", content="Search for something")
        ]
        
        # This matches the tool definition in the example GLUE app
        tools = [
            {
                "name": "web_search", 
                "description": "Search the web",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"}
                }
            }
        ]
        
        response = await self.provider.generate_response(messages, tools)
        
        # Verify that the response contains the tool calls
        assert isinstance(response, dict)
        assert "tool_calls" in response
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["id"] == "call_1"
        assert response["tool_calls"][0]["name"] == "web_search"
        assert response["tool_calls"][0]["arguments"] == {"query": "test query"}
        
        # Verify the API call
        self.provider.client.chat.completions.create.assert_called_once()
        call_args = self.provider.client.chat.completions.create.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 1
        assert call_args["tools"][0]["name"] == "web_search"
    
    @pytest.mark.asyncio
    async def test_api_key_from_environment(self):
        """Test that the API key is retrieved from the environment"""
        # Save the original environment
        original_env = os.environ.get("OPENROUTER_API_KEY")
        
        try:
            # Set the environment variable
            os.environ["OPENROUTER_API_KEY"] = "test_api_key"
            
            # Create a new provider with a mock for the client initialization
            mock_openai = MagicMock()
            with patch('openai.AsyncOpenAI', return_value=mock_openai) as mock_openai_class:
                config = ModelConfig(
                    name="researcher",
                    provider=ModelProvider.OPENROUTER,
                    model="meta-llama/llama-3.1-405b-instruct:free",
                    temperature=0.7
                )
                
                # Mock the BaseModel._initialize_client method
                with patch.object(BaseModel, '_initialize_client'):
                    model = BaseModel(config)
                    
                    # Create the provider directly to test API key handling
                    with patch.object(OpenRouterProvider, '_initialize_client') as mock_init:
                        provider = OpenRouterProvider(model)
                        
                        # Call the method we're testing
                        api_key = provider._get_api_key()
                        
                        # Verify the API key was retrieved from the environment
                        assert api_key == "test_api_key"
                
        finally:
            # Restore the original environment
            if original_env is not None:
                os.environ["OPENROUTER_API_KEY"] = original_env
            else:
                if "OPENROUTER_API_KEY" in os.environ:
                    del os.environ["OPENROUTER_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_api_key_from_dotenv(self):
        """Test that the API key is loaded from a .env file"""
        # Save the original environment
        original_env = os.environ.get("OPENROUTER_API_KEY")
        
        try:
            # Clear the environment variable
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            
            # Create a temporary .env file
            with open(".env.test", "w") as f:
                f.write("OPENROUTER_API_KEY=dotenv_test_key\n")
            
            # Mock the dotenv module
            mock_dotenv = MagicMock()
            
            # Mock the import of dotenv
            with patch.dict('sys.modules', {'dotenv': mock_dotenv}):
                # Mock the load_dotenv function
                mock_dotenv.load_dotenv = MagicMock(return_value=True)
                
                # Mock os.path.exists to return True for .env
                with patch('os.path.exists', return_value=True):
                    # Mock os.environ to simulate dotenv loading the key
                    with patch.dict('os.environ', {}, clear=True):
                        # After the provider calls load_dotenv, we want the key to be available
                        def mock_load_dotenv_effect(*args, **kwargs):
                            os.environ["OPENROUTER_API_KEY"] = "dotenv_test_key"
                            return True
                        
                        mock_dotenv.load_dotenv.side_effect = mock_load_dotenv_effect
                        
                        # Create a new provider
                        config = ModelConfig(
                            name="researcher",
                            provider=ModelProvider.OPENROUTER,
                            model="meta-llama/llama-3.1-405b-instruct:free",
                            temperature=0.7
                        )
                        
                        # Mock the BaseModel._initialize_client method
                        with patch.object(BaseModel, '_initialize_client'):
                            model = BaseModel(config)
                            
                            # Create the provider directly to test API key handling
                            with patch.object(OpenRouterProvider, '_initialize_client'):
                                provider = OpenRouterProvider(model)
                                
                                # Call the method we're testing
                                api_key = provider._get_api_key()
                                
                                # Verify the API key was loaded from the .env file
                                assert api_key == "dotenv_test_key"
                                mock_dotenv.load_dotenv.assert_called_once()
                
        finally:
            # Clean up the test .env file
            if os.path.exists(".env.test"):
                os.remove(".env.test")
            
            # Restore the original environment
            if original_env is not None:
                os.environ["OPENROUTER_API_KEY"] = original_env
            else:
                if "OPENROUTER_API_KEY" in os.environ:
                    del os.environ["OPENROUTER_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_process_tool_calls(self):
        """Test processing tool calls"""
        # Create a tool call that matches the web_search tool from the example
        tool_calls = [
            ToolCall(
                tool_id="call_1",
                name="web_search",
                arguments={"query": "test query"}
            )
        ]
        
        # Mock the tool executor
        tool_executor = AsyncMock(return_value=ToolResult(
            tool_name="web_search",
            result="Search results for test query",
            adhesive=AdhesiveType.GLUE,
            metadata={}
        ))
        
        # Process the tool calls
        results = await self.provider.process_tool_calls(tool_calls, tool_executor)
        
        # Verify the tool executor was called
        tool_executor.assert_called_once_with(tool_calls[0])
        
        # Verify the results
        assert len(results) == 1
        assert results[0].tool_name == "web_search"
        assert results[0].result == "Search results for test query"
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test handling API errors"""
        # Mock an API error
        self.provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        messages = [
            Message(role="user", content="Hello, world!")
        ]
        
        # Test that the error is properly caught and raised
        with pytest.raises(Exception, match="API Error"):
            await self.provider.generate_response(messages)
    
    @pytest.mark.asyncio
    async def test_development_mode(self):
        """Test that development mode allows operation without an API key"""
        # Create a config with development mode enabled
        config = ModelConfig(
            name="researcher",
            provider=ModelProvider.OPENROUTER,
            model="meta-llama/llama-3.1-405b-instruct:free",
            temperature=0.7
        )
        
        # Add development flag to the model
        with patch.object(BaseModel, '_initialize_client'):
            model = BaseModel(config)
            model.development = True
            
            # Mock the environment to ensure no API key is present
            with patch.dict(os.environ, {}, clear=True):
                # Mock the dotenv module to prevent loading from .env file
                mock_dotenv = MagicMock()
                with patch.dict('sys.modules', {'dotenv': mock_dotenv}):
                    # Mock os.path.exists to return False for .env files
                    with patch('os.path.exists', return_value=False):
                        # Mock load_dotenv to do nothing
                        mock_dotenv.load_dotenv = MagicMock(return_value=False)
                        
                        # Create a provider and test the _get_api_key method
                        with patch.object(OpenRouterProvider, '_initialize_client'):
                            provider = OpenRouterProvider(model)
                            
                            # This should return a dummy key and not raise an error
                            api_key = provider._get_api_key()
                            assert api_key == "dummy_key_for_development"
