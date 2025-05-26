import pytest
from agno.agent import Agent as AgnoAgent
from agno.memory.v2 import Memory as AgnoV2Memory
from agno.models.base import Model as AgnoModel, ModelResponse
from agno.models.message import Message
from typing import Any, List, Optional

from glue.core.agent import GlueAgent
# Placeholder for GlueModel and other necessary imports, will be refined as we build
# from glue.core.model import GlueModel 

@pytest.mark.asyncio
async def test_glue_agent_initialization():
    """Tests basic initialization of a GlueAgent and its Agno components."""
    # Minimal GlueModel configuration for now
    # glue_model = GlueModel(name="test_model", provider="test_provider", role="test_role")
    
    # For now, let's assume GlueAgent takes a name and perhaps a config or model
    # This will evolve as we define GlueAgent's constructor
    glue_agent_name = "TestGlueAgent"
    
    # Expected: GlueAgent constructor will need to be defined
    # For TDD, we expect this to fail until GlueAgent is implemented
    try:
        glue_agent = GlueAgent(name=glue_agent_name) # This line will likely cause the initial fail
    except NameError: # Catching NameError if GlueAgent is not yet defined
        pytest.fail("GlueAgent class not found. Please define it in src/glue/core/agent.py")
    except TypeError as e:
        pytest.fail(f"GlueAgent constructor raised TypeError: {e}. Check __init__ signature.")

    assert isinstance(glue_agent, GlueAgent), "Object is not an instance of GlueAgent"
    assert glue_agent.name == glue_agent_name, "GlueAgent name was not set correctly"
    
    # Check for internal AgnoAgent
    assert hasattr(glue_agent, '_agno_agent'), "GlueAgent should have an internal '_agno_agent' attribute"
    assert isinstance(glue_agent._agno_agent, AgnoAgent), "'_agno_agent' is not an instance of AgnoAgent"
    
    # Check for Agno V2 Memory
    # Assuming GlueAgent initializes or receives an AgnoV2Memory instance
    assert hasattr(glue_agent, '_memory'), "GlueAgent should have an internal '_memory' attribute for Agno V2 Memory"
    assert isinstance(glue_agent._memory, AgnoV2Memory), "'_memory' is not an instance of AgnoV2Memory"
    
    # Further checks once GlueModel is integrated:
    # assert glue_agent._agno_agent.model is not None, "AgnoAgent's model should be configured"
    # assert glue_agent._memory.model is not None, "AgnoV2Memory's model should be configured for manager/summarizer"

    # Placeholder for cleanup if needed
    # await glue_agent.cleanup() # If GlueAgent has async cleanup


# Minimal concrete AgnoModel for testing purposes
class ConcreteTestAgnoModel(AgnoModel):
    def __init__(self, name="test_concrete_model", id="test_model_id", **kwargs):
        super().__init__(name=name, id=id, **kwargs)
        self.provider = "test_provider"
        self.api_key_name = "TEST_API_KEY"

    def prepare_model_messages(self, messages: List[Message], **kwargs) -> list:
        return [msg.model_dump() for msg in messages]

    async def invoke(self, messages: list, **kwargs) -> ModelResponse:
        return ModelResponse(content="test_response", model_name=self.name, cost=0.0, usage=None, raw_response=None)

    async def ainvoke(self, messages: list, **kwargs) -> ModelResponse:
        return await self.invoke(messages, **kwargs)

    async def ainvoke_stream(self, messages: list, **kwargs):
        yield ModelResponse(content="test_stream_chunk", model_name=self.name, cost=0.0, usage=None, raw_response=None)

    def invoke_stream(self, messages: list, **kwargs):
        yield ModelResponse(content="test_stream_chunk", model_name=self.name, cost=0.0, usage=None, raw_response=None)

    def parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(content=str(response), model_name=self.name, cost=0.0, usage=None, raw_response=response)

    def parse_provider_response_delta(self, delta: Any, **kwargs) -> Optional[ModelResponse]:
        if delta:
            return ModelResponse(content=str(delta), model_name=self.name, cost=0.0, usage=None, raw_response=delta, is_delta=True)
        return None

    def parse_response(self, response: Any, **kwargs) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(content=str(response), model_name=self.name, cost=0.0, usage=None, raw_response=response)

    def get_api_key(self) -> Optional[str]:
        return "test_key_value"


@pytest.mark.asyncio
async def test_glue_agent_configures_agno_memory_with_model():
    """Tests that GlueAgent passes its agno_model to its AgnoV2Memory instance."""
    glue_agent_name = "TestMemoryAgent"
    test_model = ConcreteTestAgnoModel()

    glue_agent = GlueAgent(name=glue_agent_name, agno_model=test_model)

    assert glue_agent._memory is not None, "GlueAgent's _memory should be initialized."
    assert isinstance(glue_agent._memory, AgnoV2Memory), "_memory is not an instance of AgnoV2Memory"
    assert glue_agent._memory.model == test_model, \
        "AgnoV2Memory instance within GlueAgent should be configured with the provided agno_model."
