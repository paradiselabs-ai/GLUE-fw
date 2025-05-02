"""
Tests for the Google Gemini provider implementation in the GLUE framework.

This module contains tests for the Gemini provider class that handles
communication with the Google Generative AI API for model interactions.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from glue.core.schemas import Message, ToolCall, ToolResult, ModelConfig
from glue.core.model import BaseModel, ModelProvider
from glue.core.types import AdhesiveType

# We'll need to mock the google.generativeai module
@pytest.fixture(autouse=True)
def mock_google_genai_import():
    """Mock the google.generativeai import to avoid ImportError"""
    with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
        yield


class TestGeminiProvider:
    """Tests for the GeminiProvider class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # This config matches the DSL syntax for a Gemini model
        self.config = ModelConfig(
            name="gemini",
            provider=ModelProvider.GEMINI,
            model="gemini-1.5-pro",
            temperature=0.7,
            max_tokens=2048,
            description="Google's Gemini model for general-purpose tasks."
        )
        
        # Import the provider here to use the mocked google.generativeai
        from glue.core.providers.gemini import GeminiProvider
        
        # Mock the _initialize_client method to prevent actual API client creation
        with patch.object(GeminiProvider, '_initialize_client'):
            self.model = BaseModel(self.config)
            self.provider = GeminiProvider(self.model)
            
            # Mock the client
            self.provider.client = MagicMock()
            self.provider.genai = MagicMock()
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self):
        """Test generating a basic response without tools"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Test response"
        mock_response.candidates[0].content.role = "model"
        
        # Set up the mock for generate_content
        self.provider.client.generate_content = AsyncMock(return_value=mock_response)
        
        messages = [
            Message(role="user", content="Hello, world!")
        ]
        
        response = await self.provider.generate_response(messages)
        
        assert response == "Test response"
        self.provider.client.generate_content.assert_called_once()
        
        # Since we're using MagicMock, we can't directly assert on the role
        # Just verify that generate_content was called with the right number of arguments
        call_args = self.provider.client.generate_content.call_args[0][0]
        assert len(call_args) == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_with_tools(self):
        """Test generating a response with tools"""
        # Mock the API response with tool calls
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [MagicMock()]
        
        # Mock function call in the response
        function_call = {
            "name": "web_search",
            "args": {
                "query": "test query"
            }
        }
        mock_response.candidates[0].content.parts[0].function_call = function_call
        mock_response.candidates[0].content.role = "model"
        
        # Set up the mock for generate_content
        self.provider.client.generate_content = AsyncMock(return_value=mock_response)
        
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
        assert response["tool_calls"][0]["name"] == "web_search"
        assert response["tool_calls"][0]["arguments"] == {"query": "test query"}
        
        # Verify the API call
        self.provider.client.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_key_from_environment(self):
        """Test that the API key is retrieved from the environment"""
        # Save the original environment
        original_env = os.environ.get("GOOGLE_API_KEY")
        
        try:
            # Set the environment variable
            os.environ["GOOGLE_API_KEY"] = "test_api_key"
            
            # Import the provider here to use the mocked google.generativeai
            from glue.core.providers.gemini import GeminiProvider
            
            # Create a new provider with a mock for the client initialization
            with patch.object(GeminiProvider, '_initialize_client'):
                config = ModelConfig(
                    name="gemini",
                    provider=ModelProvider.GEMINI,
                    model="gemini-1.5-pro",
                    temperature=0.7
                )
                
                # Mock the BaseModel._initialize_client method
                with patch.object(BaseModel, '_initialize_client'):
                    model = BaseModel(config)
                    
                    # Create the provider directly to test API key handling
                    provider = GeminiProvider(model)
                    
                    # Call the method we're testing
                    api_key = provider._get_api_key()
                    
                    # Verify the API key was retrieved from the environment
                    assert api_key == "test_api_key"
                
        finally:
            # Restore the original environment
            if original_env is not None:
                os.environ["GOOGLE_API_KEY"] = original_env
            else:
                if "GOOGLE_API_KEY" in os.environ:
                    del os.environ["GOOGLE_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_api_key_from_dotenv(self):
        """Test that the API key is loaded from a .env file"""
        # Save the original environment
        original_env = os.environ.get("GOOGLE_API_KEY")
        
        try:
            # Clear the environment variable
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]
            
            # Import the provider here to use the mocked google.generativeai
            from glue.core.providers.gemini import GeminiProvider
            
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
                            os.environ["GOOGLE_API_KEY"] = "dotenv_test_key"
                            return True
                        
                        mock_dotenv.load_dotenv.side_effect = mock_load_dotenv_effect
                        
                        # Create a provider with mocked initialization
                        with patch.object(GeminiProvider, '_initialize_client'):
                            config = ModelConfig(
                                name="gemini",
                                provider=ModelProvider.GEMINI,
                                model="gemini-1.5-pro",
                                temperature=0.7
                            )
                            
                            # Mock the BaseModel._initialize_client method
                            with patch.object(BaseModel, '_initialize_client'):
                                model = BaseModel(config)
                                provider = GeminiProvider(model)
                                
                                # Call the method we're testing
                                api_key = provider._get_api_key()
                                
                                # Verify the API key was loaded from dotenv
                                assert api_key == "dotenv_test_key"
                                mock_dotenv.load_dotenv.assert_called_once()
        finally:
            # Restore the original environment
            if original_env is not None:
                os.environ["GOOGLE_API_KEY"] = original_env
            else:
                if "GOOGLE_API_KEY" in os.environ:
                    del os.environ["GOOGLE_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_development_mode(self):
        """Test that the provider works in development mode without an API key"""
        # Import the provider here to use the mocked google.generativeai
        from glue.core.providers.gemini import GeminiProvider
        
        # Save the original environment
        original_env = os.environ.get("GOOGLE_API_KEY")
        
        try:
            # Clear the environment variable
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]
            
            # Create a provider with development mode enabled
            with patch.object(GeminiProvider, '_initialize_client'):
                config = ModelConfig(
                    name="gemini",
                    provider=ModelProvider.GEMINI,
                    model="gemini-1.5-pro",
                    temperature=0.7,
                    development=True  # Enable development mode
                )
                
                # Mock the BaseModel._initialize_client method
                with patch.object(BaseModel, '_initialize_client'):
                    model = BaseModel(config)
                    model.development = True
                    provider = GeminiProvider(model)
                    
                    # Call the method we're testing
                    api_key = provider._get_api_key()
                    
                    # Verify a dummy key is returned in development mode
                    assert api_key == "dummy_key_for_development"
        finally:
            # Restore the original environment
            if original_env is not None:
                os.environ["GOOGLE_API_KEY"] = original_env
            else:
                if "GOOGLE_API_KEY" in os.environ:
                    del os.environ["GOOGLE_API_KEY"]
    
    @pytest.mark.asyncio
    async def test_process_tool_calls(self):
        """Test processing tool calls"""
        # Import the provider here to use the mocked google.generativeai
        from glue.core.providers.gemini import GeminiProvider
        
        # Create tool calls
        tool_calls = [
            ToolCall(
                tool_id="call_1",
                name="web_search",
                arguments={"query": "test query"}
            )
        ]
        
        # Create a mock tool executor
        async def mock_tool_executor(tool_call):
            return ToolResult(
                tool_call_id=tool_call.tool_id,
                content="Search results for 'test query'"
            )
        
        # Create the provider
        with patch.object(GeminiProvider, '_initialize_client'):
            provider = GeminiProvider(self.model)
            
            # Process the tool calls
            results = await provider.process_tool_calls(tool_calls, mock_tool_executor)
            
            # Verify the results
            assert len(results) == 1
            assert results[0].tool_call_id == "call_1"
            assert results[0].content == "Search results for 'test query'"
