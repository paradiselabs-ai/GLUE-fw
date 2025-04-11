import re
import json
from typing import Optional, Dict, Any

# Improved regex to find JSON block
# This handles basic nested structures better with a non-greedy approach
# It matches from the first opening brace to the last closing brace
JSON_EXTRACT_REGEX = re.compile(
    r"```(?:json)?\s*(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})\s*```|(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})", 
    re.DOTALL
)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object (dictionary) from a string.
    Handles JSON provided either as a bare object {...} or fenced
    with ```json ... ``` or ``` ... ```.
    Has improved handling of nested JSON structures.
    
    Args:
        text: The string possibly containing a JSON object.
    
    Returns:
        The extracted dictionary if valid JSON is found, otherwise None.
    """
    if not text:
        return None
        
    # First try the improved regex for better nested structure handling
    match = JSON_EXTRACT_REGEX.search(text)
    if not match:
        # As a fallback for complex cases, try a simple approach:
        # Find the first { and then try to parse progressively larger substrings
        start_idx = text.find('{')
        if start_idx >= 0:
            for end_idx in range(len(text), start_idx, -1):
                try:
                    substr = text[start_idx:end_idx]
                    if substr.count('{') != substr.count('}'):
                        continue  # Skip unbalanced braces
                    data = json.loads(substr)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return None
        
    # The regex has two capturing groups: one for fenced JSON, one for bare.
    # Extract the content from whichever group matched.
    json_str = match.group(1) or match.group(2)
    if not json_str:
        return None
        
    try:
        # Clean potential leading/trailing whitespace within the captured string
        json_str_cleaned = json_str.strip()
        data = json.loads(json_str_cleaned)
        if isinstance(data, dict):
            return data
        else:
            # Ensure we only return dictionaries, not lists or primitives
            return None
    except json.JSONDecodeError:
        # If regex match failed to provide valid JSON, try the fallback approach
        start_idx = text.find('{')
        if start_idx >= 0:
            for end_idx in range(len(text), start_idx, -1):
                try:
                    substr = text[start_idx:end_idx]
                    if substr.count('{') != substr.count('}'):
                        continue  # Skip unbalanced braces
                    data = json.loads(substr)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return None

# Example usage (for testing/demonstration)
if __name__ == '__main__':
    test_cases = [
        ("Some text ```json\n{\"key\": \"value\"}\n``` more text", {"key": "value"}),
        ("Bare json {\"a\": 1, \"b\": [2, 3]} here", {"a": 1, "b": [2, 3]}),
        ("No json here", None),
        ("Invalid json ```{\"key\": \"value\",}```", None),
        ("Fenced ```\n{\"another\": true}\n```", {"another": True}),
        ("Mixed text and {\"nested\": {\"deep\": 42}} and more.", {"nested": {"deep": 42}}),
        ("```\n[1, 2, 3]\n```", None),  # Should not match lists
        ("Just a string", None),
        ("", None),
        (None, None),
        ("```json\n{\n  \"tool_name\": \"web_search\",\n  \"arguments\": {\n    \"query\": \"latest advancements in large language models\"\n  }\n}\n```", 
         {"tool_name": "web_search", "arguments": {"query": "latest advancements in large language models"}}),
        ("The model decided to call a tool: {\"tool_name\": \"code_interpreter\", \"arguments\": {\"code\": \"print('hello')\"}}", 
         {"tool_name": "code_interpreter", "arguments": {"code": "print('hello')"}}),
        # Additional test cases for complex structures
        ("Text {\"nested\": {\"objects\": {\"with\": \"multiple\", \"levels\": [1, 2, {\"of\": \"nesting\"}]}}} end", 
         {"nested": {"objects": {"with": "multiple", "levels": [1, 2, {"of": "nesting"}]}}}),
        ("JSON with string containing braces {\"code\": \"function() { return true; }\"}", 
         {"code": "function() { return true; }"}),
    ]
    
    for i, (text, expected) in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        print(f"Input: {repr(text)}")
        result = extract_json(str(text))  # Cast to str for the None case
        print(f"Output: {result}")
        print(f"Expected: {expected}")
        print(f"Match: {result == expected}")
        print("-" * 20)