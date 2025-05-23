from glue.core.providers.provider_base import ProviderBase
from glue.utils.logger import get_logger

logger = get_logger(__name__)

class TestProvider(ProviderBase):
    """A dummy provider for testing purposes, named to match BaseModel loading."""

    def __init__(self, model): 
        super().__init__(model.config) 
        self.model_instance = model 
        logger.info(f"TestProvider initialized for model: {model.name}")
        if not hasattr(self, 'client'): 
            self.client = None 


    def _initialize_client(self):
        """
        Initializes any client specific to this provider.
        For TestProvider, there's no external client.
        """
        logger.info("TestProvider: No specific client to initialize.")
        # self.client is already set to None or by super() if applicable

    def get_available_models(self):
        """Returns a list of dummy model names for testing."""
        return ["test_model_variant_1", "test_model_variant_2"]

    async def agenerate_response(self, messages, tools=None, tool_choice=None, stream=False, **kwargs):
        """Dummy async response generation for testing."""
        logger.info("TestProvider: agenerate_response called (dummy implementation)")
        return {
            "content": "This is a dummy async response from TestProvider.",
            "tool_calls": None,
            "finish_reason": "stop"
        }

    def generate_response(self, messages, tools=None, tool_choice=None, stream=False, **kwargs):
        """Dummy sync response generation for testing."""
        logger.info("TestProvider: generate_response called (dummy implementation)")
        return {
            "content": "This is a dummy sync response from TestProvider.",
            "tool_calls": None,
            "finish_reason": "stop"
        }

    def count_tokens(self, text: str) -> int:
        """Dummy token counting for testing."""
        logger.info(f"TestProvider: count_tokens called for text: '{text[:50]}...' (dummy implementation)")
        return len(text.split()) 
