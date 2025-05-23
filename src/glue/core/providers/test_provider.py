from ..base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)

class TestProviderProvider(BaseProvider):
    """A dummy provider for testing purposes."""

    def __init__(self, model):
        super().__init__(model)
        logger.info(f"TestProviderProvider initialized for model: {model.name}")

    def _initialize_client(self):
        # No client to initialize for the test provider
        logger.info("TestProviderProvider: No client to initialize.")
        self.client = None # Explicitly set to None or a dummy object if needed

    def get_available_models(self):
        # Return a dummy list or an empty list
        return ["test_model_variant_1", "test_model_variant_2"]

    async def agenerate_response(self, messages, tools=None, tool_choice=None, stream=False, **kwargs):
        logger.info("TestProviderProvider: agenerate_response called (dummy implementation)")
        # Return a dummy response structure if needed by calling code, otherwise NotImplementedError
        raise NotImplementedError("TestProviderProvider.agenerate_response is a dummy and not implemented.")

    def generate_response(self, messages, tools=None, tool_choice=None, stream=False, **kwargs):
        logger.info("TestProviderProvider: generate_response called (dummy implementation)")
        # Return a dummy response structure if needed by calling code, otherwise NotImplementedError
        raise NotImplementedError("TestProviderProvider.generate_response is a dummy and not implemented.")

    def count_tokens(self, text: str) -> int:
        logger.info(f"TestProviderProvider: count_tokens called for text: '{text[:50]}...' (dummy implementation)")
        return len(text.split()) # A very naive token count for testing
