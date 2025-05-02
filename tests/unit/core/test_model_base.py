"""
Tests for the base model class in the GLUE framework.

This module contains tests for the base model class that provides abstraction
over different AI model providers and handles tool usage capabilities.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from glue.core.schemas import Message, ToolCall, ToolResult, ModelConfig
from glue.core.model import BaseModel, ModelProvider
from glue.core.types import AdhesiveType

class TestModelProvider:
    """Tests for the ModelProvider enum"""
    
    def test_valid_providers(self):
        """Test that valid providers are accepted"""
        assert ModelProvider.OPENROUTER == "openrouter"
        assert ModelProvider.ANTHROPIC == "anthropic"
        assert ModelProvider.OPENAI == "openai"
        assert ModelProvider.CUSTOM == "custom"
    
    def test_invalid_provider(self):
        """Test that invalid providers are rejected"""
        with pytest.raises(ValueError):
            ModelProvider("invalid_provider")


class TestBaseModel:
    """Tests for the BaseModel class"""
    
    def test_initialization(self):
        """Test that a model can be initialized with valid configuration"""
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048
        )
        
        model = BaseModel(config)
        
        assert model.name == "test_model"
        assert model.provider == ModelProvider.OPENAI
        assert model.model == "gpt-4"
        assert model.temperature == 0.7
        assert model.max_tokens == 2048
        assert model.api_key is None
    
    def test_initialization_with_api_key(self):
        """Test that a model can be initialized with an API key"""
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            api_key="test_api_key"
        )
        
        model = BaseModel(config)
        
        assert model.api_key == "test_api_key"
    
    def test_initialization_with_api_params(self):
        """Test that a model can be initialized with API parameters"""
        api_params = {"top_p": 0.9, "frequency_penalty": 0.5}
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            api_params=api_params
        )
        
        model = BaseModel(config)
        
        assert model.api_params == api_params
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test that a model can generate a response"""
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4"
        )
        
        model = BaseModel(config)
        # Mock the _generate_response method
        model._generate_response = AsyncMock(return_value="Test response")
        
        messages = [
            Message(role="user", content="Hello, world!")
        ]
        
        response = await model.generate_response(messages)
        
        assert response == "Test response"
        model._generate_response.assert_called_once_with(messages, None)
    
    @pytest.mark.asyncio
    async def test_generate_response_with_tools(self):
        """Test that a model can generate a response with tools"""
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4"
        )
        
        model = BaseModel(config)
        # Mock the _generate_response method
        model._generate_response = AsyncMock(return_value="Test response with tools")
        
        messages = [
            Message(role="user", content="Search for something")
        ]
        
        tools = [
            {"name": "search", "description": "Search the web"}
        ]
        
        response = await model.generate_response(messages, tools)
        
        assert response == "Test response with tools"
        model._generate_response.assert_called_once_with(messages, tools)
    
    @pytest.mark.asyncio
    async def test_process_tool_calls(self):
        """Test that a model can process tool calls"""
        config = ModelConfig(
            name="test_model",
            provider=ModelProvider.OPENAI,
            model="gpt-4"
        )
        
        model = BaseModel(config)
        
        tool_calls = [
            ToolCall(
                tool_id="call_1",
                name="search",
                arguments={"query": "test query"}
            )
        ]
        
        # Mock the tool execution
        tool_executor = AsyncMock(return_value=ToolResult(
            tool_name="search",
            result="Search results for test query",
            adhesive=AdhesiveType.GLUE,
            metadata={}
        ))
        
        results = await model.process_tool_calls(tool_calls, tool_executor)
        
        assert len(results) == 1
        assert results[0].tool_name == "search"
        assert results[0].result == "Search results for test query"
        tool_executor.assert_called_once_with(tool_calls[0])
    
    def test_get_provider_class(self):
        """Test that the correct provider class is returned"""
        # Test with OpenAI provider
        with patch("glue.core.providers.openai.OpenAIProvider") as mock_openai:
            with patch("glue.core.model.importlib.import_module") as mock_import:
                mock_import.return_value = type('module', (), {'OpenAIProvider': mock_openai})
                
                config = ModelConfig(
                    name="test_model",
                    provider=ModelProvider.OPENAI,
                    model="gpt-4"
                )
                model = BaseModel(config)
                
                # Verify that the OpenAIProvider was called to create the provider instance
                mock_openai.assert_called_once_with(model)
        
        # Test with Anthropic provider
        with patch("glue.core.providers.anthropic.AnthropicProvider") as mock_anthropic:
            with patch("glue.core.model.importlib.import_module") as mock_import:
                mock_import.return_value = type('module', (), {'AnthropicProvider': mock_anthropic})
                
                config = ModelConfig(
                    name="test_model",
                    provider=ModelProvider.ANTHROPIC,
                    model="claude-3-opus"
                )
                model = BaseModel(config)
                
                # Verify that the AnthropicProvider was called to create the provider instance
                mock_anthropic.assert_called_once_with(model)
        
        # Test with OpenRouter provider
        with patch("glue.core.providers.openrouter.OpenRouterProvider") as mock_openrouter:
            with patch("glue.core.model.importlib.import_module") as mock_import:
                mock_import.return_value = type('module', (), {'OpenRouterProvider': mock_openrouter})
                
                config = ModelConfig(
                    name="test_model",
                    provider=ModelProvider.OPENROUTER,
                    model="anthropic/claude-3-opus"
                )
                model = BaseModel(config)
                
                # Verify that the OpenRouterProvider was called to create the provider instance
                mock_openrouter.assert_called_once_with(model)
        
        # Test with custom provider
        with patch("glue.core.model.importlib.import_module") as mock_import:
            mock_custom_provider = MagicMock()
            mock_module = MagicMock()
            mock_module.CustomProvider = mock_custom_provider
            mock_import.return_value = mock_module
            
            config = ModelConfig(
                name="test_model",
                provider=ModelProvider.CUSTOM,
                model="custom-model",
                provider_class="custom.provider.CustomProvider"
            )
            model = BaseModel(config)
            
            # Check that import_module was called with the right module path
            mock_import.assert_any_call("custom.provider")
            # Verify that the CustomProvider was called to create the provider instance
            mock_custom_provider.assert_called_once_with(model)
