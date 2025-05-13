# tests/unit/core/adapters/agno/test_glue_message_adapter.py
import pytest

# Assume GLUE's Message structure is imported (or defined for simplicity)
from glue.core.types import Message as GlueMessage

# Import the adapter function (which doesn't exist yet)
from glue.core.adapters.agno.adapter import adapt_glue_message_to_agno, AgnoMessage

@pytest.mark.unit
def test_adapt_glue_message_to_agno():
    """
    Test adapting a GLUE Message object to an Agno-compatible format.
    (This test will fail initially due to the missing adapter function).
    """
    # 1. Arrange: Create a sample GLUE message
    glue_msg = GlueMessage(
        role="user",
        content="Hello from GLUE!",
        metadata={"timestamp": "2024-01-01T12:00:00Z", "source": "glue_test"}
    )

    # Expected Agno-like output (adjust based on actual Agno structure)
    expected_sender = "user"
    expected_content = "Hello from GLUE!"
    expected_metadata = {"timestamp": "2024-01-01T12:00:00Z", "source": "glue_test"}

    # 2. Act: Try to adapt the message (This line will cause the initial failure)
    agno_msg = adapt_glue_message_to_agno(glue_msg)

    # 3. Assert: Check if the adapted message matches expectations
    assert isinstance(agno_msg, AgnoMessage)
    assert agno_msg.sender == expected_sender
    assert agno_msg.content == expected_content
    assert agno_msg.metadata == expected_metadata

    # Temporarily assert False to represent the failing state (RED phase)
    # assert False, "adapt_glue_message_to_agno function not implemented yet"
