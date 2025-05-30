import pytest
from glue.core.types import Message

def normalize_message_content(input_content):
    message_content = input_content
    if isinstance(message_content, dict) and "content" in message_content:
        message_content = message_content["content"]
    if isinstance(message_content, list):
        text_value = None
        for part in message_content:
            if isinstance(part, dict):
                val = part.get('text') or part.get('content')
                if isinstance(val, str) and val.strip():
                    text_value = val.strip()
                    break
        if text_value is not None:
            message_content = text_value
        else:
            message_content = ""
    if not isinstance(message_content, str):
        message_content = str(message_content) if message_content is not None else ""
    return message_content

@pytest.mark.parametrize("input_content,expected", [
    ("hello", "hello"),
    ({"content": "hi there"}, "hi there"),
    ([{"type": "text", "text": "greetings"}], "greetings"),
    ([{"type": "text", "text": ""}, {"type": "text", "text": "not empty"}], "not empty"),
    ([{"type": "text", "text": ""}], ""),
    ([{"type": "text", "text": ""}, {"type": "content", "content": "alt"}], "alt"),
    ([{"type": "text", "text": ""}], ""),
])
def test_message_content_normalization(input_content, expected):
    message_content = normalize_message_content(input_content)
    msg = Message(role="user", content=message_content)
    assert msg.content == expected

def test_direct_normalization():
    # Directly test the normalization utility for the problematic input
    assert normalize_message_content([{'type': 'text', 'text': ''}]) == ""
    assert normalize_message_content([{'type': 'text', 'text': 'foo'}]) == "foo"
    assert normalize_message_content([{'type': 'text', 'text': ''}, {'type': 'text', 'text': 'bar'}]) == "bar"
