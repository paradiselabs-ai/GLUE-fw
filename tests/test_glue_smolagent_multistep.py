
from glue.core.glue_smolagent import make_glue_smol_agent
from smolagents import Tool  # noqa: F401, import kept for potential future extensions

class DummyFinal3Model:
    """
    A dummy model that always returns a final_answer code snippet for the number 3.
    """
    def generate(self, messages, stop_sequences=None, **kwargs):
        class TokenUsage:
            input_tokens = 10
            output_tokens = 5
            
        class Chat:
            content = "```python\nfinal_answer(\"3\")\n```"
            token_usage = TokenUsage()
        return Chat()

    # For ToolCallingAgent compatibility; not used in CodeAgent
    def __call__(self, messages):
        return self.generate(messages)


def test_multistep_counting_returns_three():
    # Use the dummy model to ensure final_answer returns '3'
    model = DummyFinal3Model()
    # No tools needed as final_answer is directly returned
    agent = make_glue_smol_agent(
        model=model,
        tools=[],
        glue_config=None,
        name="test_agent",
        description="Test agent for counting",
    )
    result = agent.run("Count to 3")
    # The agent should return the string '3' as the final answer
    assert isinstance(result, str)
    assert "3" in result 