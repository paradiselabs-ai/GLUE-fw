import re
import json
from typing import Optional, Dict, Any
import logging

# Configure logging for the utility
logger = logging.getLogger(__name__)

# Regex specifically for fenced JSON blocks (```json ... ``` or ``` ... ```)
# Prioritizes these as they are more likely the intended tool call output
FENCED_JSON_REGEX = re.compile(r"```(?:json)?\s*(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})\s*```", re.DOTALL)

# Regex for potentially bare JSON object (less prioritized)
# Improved to handle basic nesting
BARE_JSON_REGEX = re.compile(r"(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})", re.DOTALL)

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object (dictionary) from a string.
    Prioritizes JSON fenced with ```json ... ``` or ``` ... ```.
    If no valid fenced JSON is found, it looks for the largest valid bare JSON object.
    Returns only dictionaries, ignores lists or primitives.

    Args:
        text: The string possibly containing a JSON object.

    Returns:
        The extracted dictionary if valid JSON is found, otherwise None.
    """
    if not isinstance(text, str) or not text:
        logger.debug("extract_json received non-string or empty input.")
        return None

    logger.debug(f"Attempting to extract JSON from text: {text[:500]}...")

    # 1. Prioritize fenced JSON blocks
    fenced_matches = FENCED_JSON_REGEX.findall(text)
    if fenced_matches:
        logger.debug(f"Found {len(fenced_matches)} potential fenced JSON blocks.")
        for match_str in reversed(fenced_matches):
            try:
                # Clean potential leading/trailing whitespace within the captured string
                json_str_cleaned = match_str.strip()
                data = json.loads(json_str_cleaned)
                if isinstance(data, dict):
                    logger.debug(f"Successfully parsed fenced JSON: {data}")
                    return data
                else:
                    logger.debug(f"Fenced block parsed, but was not a dict: {type(data)}")
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse potential fenced JSON block: {e}. Content: {match_str[:100]}...")
        logger.debug("None of the fenced blocks yielded a valid JSON dictionary.")

    # 2. If no valid fenced JSON found, look for bare JSON objects
    # Find all potential bare JSON objects
    bare_matches = BARE_JSON_REGEX.findall(text)
    if bare_matches:
        logger.debug(f"Found {len(bare_matches)} potential bare JSON objects. Will try parsing the longest first.")
        # Sort by length descending - largest valid JSON is often the intended one
        bare_matches.sort(key=len, reverse=True)
        for match_str in bare_matches:
            try:
                json_str_cleaned = match_str.strip()
                # Basic sanity check for balanced braces before attempting full parse
                if json_str_cleaned.count('{') != json_str_cleaned.count('}'):
                    logger.debug(f"Skipping bare match due to unbalanced braces: {json_str_cleaned[:100]}...")
                    continue
                data = json.loads(json_str_cleaned)
                if isinstance(data, dict):
                    logger.debug(f"Successfully parsed bare JSON: {data}")
                    return data
                else:
                    logger.debug(f"Bare JSON parsed, but was not a dict: {type(data)}")
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse potential bare JSON: {e}. Content: {match_str[:100]}...")
        logger.debug("None of the bare JSON matches yielded a valid dictionary.")

    # 3. Fallback: Simple first '{' to last '}' - less reliable
    # This is kept as a last resort but is less likely to be correct with messy input
    try:
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if 0 <= first_brace < last_brace:
            potential_json = text[first_brace : last_brace + 1]
            logger.debug(f"Fallback: trying first {{ to last }}: {potential_json[:100]}...")
            data = json.loads(potential_json)
            if isinstance(data, dict):
                 logger.debug(f"Successfully parsed fallback JSON: {data}")
                 return data
    except json.JSONDecodeError:
        logger.debug("Fallback first/last brace parsing failed.")
        pass # Ignore if this fails

    logger.debug("Could not extract any valid JSON dictionary from the text.")
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