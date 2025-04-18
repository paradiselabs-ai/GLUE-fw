import re
import json
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Regex for fenced JSON (non-greedy match)
FENCED_JSON_REGEX = re.compile(
    r'```json\s*(.*?)\s*```',
    re.DOTALL | re.IGNORECASE
)
# Add partial fence detection (e.g., missing closing fence)
PARTIAL_FENCED_JSON_REGEX = re.compile(
    r'```json\s*(.*?)\s*(?=```|\\Z)',  # Look for closing fence or end of string
    re.DOTALL | re.IGNORECASE
)

# Common prefixes to strip
UNWANTED_PREFIXES = [r"<\|\w+\|>"]

# Valid tool names in GLUE framework
VALID_TOOL_NAMES = {"web_search", "file_handler", "code_interpreter", "communicate", "search_web"}

def strip_unwanted_prefixes(text: str) -> str:
    """
    Removes unwanted prefixes and excessive whitespace from the input text.

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string.
    """
    if not text:
        return ""
    for prefix in UNWANTED_PREFIXES:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    return text.strip()

def normalize_glue_tool_call(data: Dict[str, Any], strict: bool = False) -> Optional[Dict[str, Any]]:
    """
    Normalizes a tool call dictionary to GLUE format: {"tool_name": ..., "arguments": ...}.
    Converts alternative formats like {"type": "function", "name": ..., "parameters": ...}.

    Args:
        data: The input dictionary to normalize.
        strict: If True, invalid tool names return None; otherwise, return raw JSON.

    Returns:
        The normalized dictionary, raw JSON, or None if invalid in strict mode.
    """
    if not isinstance(data, dict):
        logger.debug("Input is not a dictionary")
        return None

    if "type" in data and data.get("type") == "function" and "name" in data and "parameters" in data:
        tool_name = data["name"]
        if tool_name not in VALID_TOOL_NAMES:
            logger.error(f"Invalid tool name: {tool_name}. Expected one of {VALID_TOOL_NAMES}")
            return None if strict else data
        logger.debug(f"Normalizing function-type tool call: {tool_name}")
        return {"tool_name": tool_name, "arguments": data["parameters"]}
    elif "tool_name" in data and "arguments" in data:
        tool_name = data["tool_name"]
        if tool_name not in VALID_TOOL_NAMES:
            logger.error(f"Invalid tool name: {tool_name}. Expected one of {VALID_TOOL_NAMES}")
            return None if strict else data
        logger.debug(f"Valid tool call: {tool_name}")
        return data
    logger.debug("Not a tool call format")
    return data if not strict else None

@contextmanager
def log_context(description: str):
    """
    Context manager to log entry and exit of a parsing operation.

    Args:
        description: Description of the operation.
    """
    logger.debug(f"Start: {description}")
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {description}: {e}")
        raise
    finally:
        logger.debug(f"End: {description}")

def extract_json_from_fenced(text: str) -> Optional[str]:
    """
    Extracts a JSON string from a fenced block, including partial JSON.

    Args:
        text: The input string to search.

    Returns:
        The extracted JSON string if found, otherwise None.
    """
    with log_context("extract_json_from_fenced"):
        match = FENCED_JSON_REGEX.search(text)
        if match:
            json_str = match.group(1)
            logger.debug(f"Fenced JSON found")
            return json_str
        partial_match = PARTIAL_FENCED_JSON_REGEX.search(text)
        if partial_match:
            json_str = partial_match.group(1)
            logger.debug(f"Partial fenced JSON found")
            return json_str
        logger.debug("No fenced JSON")
        return None

def extract_json_from_bare(text: str, max_length: int = 10000) -> Optional[str]:
    """
    Extracts a JSON string from a bare JSON object by finding balanced curly braces.

    Args:
        text: The input string to search.
        max_length: Maximum length to process for performance.

    Returns:
        The extracted JSON string if found, otherwise None.
    """
    with log_context("extract_json_from_bare"):
        start_idx = text.find("{")
        if start_idx < 0:
            logger.debug("No opening brace")
            return None

        brace_count = 0
        in_quotes = False
        truncated_text = text[start_idx:min(len(text), start_idx + max_length)]
        for end_idx, char in enumerate(truncated_text):
            if char == '"' and (end_idx == 0 or truncated_text[end_idx - 1] != "\\"):
                in_quotes = not in_quotes
            if not in_quotes:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
            if brace_count == 0 and end_idx > 0:
                json_str = truncated_text[:end_idx + 1]
                logger.debug(f"Bare JSON found")
                return json_str
        logger.debug("No balanced JSON within max_length")
        return truncated_text

def close_braces(json_str: str) -> str:
    """
    Balances and closes any open braces/brackets in a JSON string.
    
    Args:
        json_str: The JSON string to balance.
        
    Returns:
        A string with balanced braces/brackets.
    """
    stack = []
    in_quotes = False
    
    for i, c in enumerate(json_str):
        if c == '"' and (i == 0 or json_str[i - 1] != "\\"):
            in_quotes = not in_quotes
        if not in_quotes:
            if c in ('{', '['):
                stack.append('}' if c == '{' else ']')
            elif c in ('}', ']'):
                if stack and stack[-1] == c:
                    stack.pop()
                else:
                    # Mismatched closing brace - this is harder to fix
                    pass
    
    # Add missing closing braces/brackets
    while stack:
        json_str += stack.pop()
        
    return json_str

def fix_typos(json_str: str) -> str:
    """
    Fixes common typos in JSON keys.
    
    Args:
        json_str: The JSON string to fix.
        
    Returns:
        A string with fixed typos.
    """
    typos = {
        "parame": "parameters",
        "target_type": "target_type",  # Add known typo mappings
        "arguement": "arguments",
        "argumets": "arguments",
        "tool_call": "tool_name",
        "toolname": "tool_name",
        "toolcall": "tool_name",
        "functio": "function",
        "paramter": "parameter"
    }
    
    for typo, correction in typos.items():
        json_str = json_str.replace(f'"{typo}"', f'"{correction}"')
        
    return json_str

def attempt_json_recovery(json_str: str) -> Optional[str]:
    """
    Attempts to recover a valid JSON string from a malformed one by cleaning artifacts,
    balancing braces, removing trailing commas, and escaping control characters.

    Args:
        json_str: The malformed JSON string.

    Returns:
        The corrected JSON string if recovery is successful, otherwise None.
    """
    with log_context("attempt_json_recovery"):
        original = json_str
        json_str = json_str.strip()

        # Step 1: Clean up fenced JSON artifacts (e.g., trailing ``` and newlines)
        json_str = re.sub(r"```.*$", "", json_str, flags=re.DOTALL)
        json_str = re.sub(r"\n+$", "", json_str)
        json_str = json_str.strip()

        # Step 2: Remove trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # Step 3: Escape control characters
        control_char_map = {
            '\n': r'\n',
            '\r': r'\r',
            '\t': r'\t',
            '\b': r'\b',
            '\f': r'\f'
        }
        for char, escape in control_char_map.items():
            json_str = json_str.replace(char, escape)

        # Step 4: Fix unclosed quotes
        if json_str.count('"') % 2 != 0:
            json_str = json_str.rsplit('"', 1)[0] + '"'

        # Step 5: Balance braces and brackets
        stack = []
        in_quotes = False
        cleaned_json = []
        i = 0
        while i < len(json_str):
            char = json_str[i]
            cleaned_json.append(char)
            if char == '"' and (i == 0 or json_str[i - 1] != "\\"):
                in_quotes = not in_quotes
            if not in_quotes:
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char in "}]":
                    if stack and stack[-1] == char:
                        stack.pop()
                    else:
                        cleaned_json.pop()  # Remove extra }
            i += 1
        json_str = "".join(cleaned_json)

        # Add closing braces/brackets
        while stack:
            json_str += stack.pop()

        # Step 6: Try parsing after initial recovery
        try:
            json.loads(json_str)
            logger.debug(f"Recovered JSON")
            return json_str
        except json.JSONDecodeError as e:
            logger.debug(f"Initial recovery failed: {e}")

        # Step 7: Remove trailing incomplete tokens (e.g., partial fields)
        json_str = re.sub(r",?\s*[^{}[\],]*$", "", json_str)

        # Step 8: Try parsing after cleanup
        try:
            json.loads(json_str)
            logger.debug(f"Recovered JSON after cleanup")
            return json_str
        except json.JSONDecodeError as e:
            logger.debug(f"Final recovery failed: {e}")
            
        # Step 9: Handle truncated JSON by closing braces/brackets
        json_str = close_braces(json_str)
        
        # Step 10: Fix common typos
        json_str = fix_typos(json_str)
        
        # Final attempt to parse
        try:
            json.loads(json_str)
            logger.debug(f"Recovered JSON after typo fixes")
            return json_str
        except json.JSONDecodeError as e:
            logger.debug(f"Recovery with typo fixes failed: {e}")
            return None

def extract_json(
    text: str,
    max_length: int = 10000,
    strict_tool_calls: bool = False,
    glue_compatibility: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a string, handling bare JSON, fenced JSON, and prefixes.
    Normalizes GLUE tool calls and supports generic JSON parsing.

    Args:
        text: The string possibly containing a JSON object.
        max_length: Maximum length to process for performance.
        strict_tool_calls: If True, reject invalid tool calls; otherwise, return raw JSON.
        glue_compatibility: If True, return raw tool call dictionary for GLUE.

    Returns:
        A dictionary with parsed JSON, raw tool call, or None.
    """
    with log_context("extract_json"):
        if not text or not isinstance(text, str):
            logger.debug("Empty or non-string input")
            return None

        # Clean input
        cleaned_text = strip_unwanted_prefixes(text)
        if not cleaned_text:
            logger.debug("Cleaned text is empty")
            return None

        # Extract JSON string
        json_str = extract_json_from_fenced(cleaned_text) or extract_json_from_bare(cleaned_text, max_length)
        if not json_str:
            logger.debug("No JSON found")
            return None
        logger.debug(f"Extracted JSON: {json_str}...")

        # Parse JSON
        def parse_json(s: str) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(s)
                if not isinstance(data, dict):
                    logger.debug("Parsed data is not a dictionary")
                    return None
                normalized = normalize_glue_tool_call(data, strict=strict_tool_calls)
                if normalized and ("tool_name" in normalized or "tool_name" in data):
                    logger.info(f"Normalized tool call: {normalized}")
                    return normalized if glue_compatibility else {"data": normalized, "tool_call": True}
                elif strict_tool_calls and normalized is None:
                    logger.debug("Not a valid tool call in strict mode")
                    return None
                else:
                    logger.info(f"Parsed generic JSON: {data}")
                    return data if glue_compatibility else {"data": data, "tool_call": False}
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return None

        result = parse_json(json_str)
        if result:
            return result

        # Attempt recovery
        recovered_json = attempt_json_recovery(json_str)
        if recovered_json:
            result = parse_json(recovered_json)
            if result:
                return result
        return None