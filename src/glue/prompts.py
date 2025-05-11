"""
GLUE Framework Prompts

This module contains prompt templates for team leads and team members.
These can be customized for specific use cases without modifying core GLUE files.

PROMPT TYPES:
-------------
1. Static Prompts: These are fixed templates with placeholders, used directly in the system.
   Examples: TEAM_LEAD_SYSTEM_PROMPT, TEAM_MEMBER_SYSTEM_PROMPT

2. Dynamic Prompts: These are templates that are filled with runtime content
   and often combined with additional context at execution time.
   Examples: REASONING_PROMPT, PLANNING_PROMPT

When interactive mode is enabled:
1. The "Adaptive User Interaction" responsibility is added to the Core Responsibilities
2. The "Interaction Protocols" section is included in the prompt


You need to redifine tools and their arguments to be used in the tool. Also add more tools as needed. Keep the prompts concise and to the point.
persist_knowledge, request_clarification, flag_task_issue, broadcast_query_to_leads.

## request_clarification
Description: Requests clarification from the user for a specific task.
Parameters:
- query: (required) The specific question or clarification needed.
Usage:
{
  "tool_name": "request_clarification",
  "arguments": {
    "query": "Please clarify the scope of the task."
  }
}


## persist_knowledge
Description: Adds verified, high-confidence information to the team's persistent knowledge base. Only use this after rigorous validation, typically associated with 'Glue' adhesive outputs.
Parameters:
- knowledge_entry: (required) The verified piece of information or data to be stored.
- source_task_id: (required) The task ID from which this knowledge was derived and verified.
- keywords: (optional) Relevant keywords for indexing and retrieval.
Usage:
{
  "tool_name": "persist_knowledge",
  "arguments": {
    "knowledge_entry": "Verified market analysis indicates Competitor Y holds 30% market share.",
    "source_task_id": "{team_name_slug}-task-0001",
    "keywords": ["market analysis", "competitor y", "market share", "product x"]
  }
}

## broadcast_query_to_leads
Description: Broadcasts a query to other team leads for clarification or collaboration.
Parameters:
- query: (required) The query or question to be broadcasted.
- target_leads: (required) A list of team lead names to whom the query should be broadcasted.
- source_task_id: (required) The task ID from which this query originated.
Usage:
    {
    "tool_name": "broadcast_query_to_leads",
    "arguments": {
    "query": "Regarding task market-analysis-team-task-0007: Clarification needed on Competitive Analysis scope. Should it exclusively feature North American competitors, or include both EU & NA competitors with NA priority? Source instructions mention EU focus for main report but NA priority for competitor section.",
    "target_leads": ["Marketing_Team_Lead", "Sales_Team_Lead"],
    "source_task_id": "market-analysis-team-task-0007"
    }
}
"""

import logging

logger = logging.getLogger(__name__)

#################################################################
# DYNAMIC PROMPTS - Combined with runtime content
#################################################################

# Communication Prompts (dynamic - combined with actual message content)
TEAM_COMMUNICATION_PROMPT = """
You are communicating with the {target_team} team.

Context:
{communication_context}

Previous Communication:
{previous_communication}

Your Message:
"""

# Reasoning Prompts (dynamic - used in agent_loop.py with observations)
REASONING_PROMPT = """
Given the following observations:

{observations}

And your current understanding:

{thoughts}

Please reason through the next steps to achieve the following goal:

{goal}
"""

# Planning Prompts (dynamic - used in agent_loop.py with thoughts and tools)
PLANNING_PROMPT = """
Based on your reasoning, please create a plan to accomplish the following goal:

{goal}

Your plan should include:
- Specific actions to take
- Tools to use
- Timeline for completion
- Success criteria

Current context:
{context}

Your current thoughts:
{formatted_thoughts}

Available tools:
{available_tools}
"""

#################################################################
# FORMATTING COMPONENTS - Used within dynamic prompts
#################################################################

# Tool Format Descriptions - Used to format each tool
TOOL_FORMAT_DESCRIPTION = """
Tool: {tool_name}
Description: {tool_description}
Parameters:
{param_descriptions}
"""

# Parameter Format Description - Used to format each parameter
PARAMETER_FORMAT_DESCRIPTION = (
    """  - {param_name} ({param_type}, {required}): {param_desc}"""
)

# Team Members Format
MEMBERS_IN_LEAD_TEAM_FORMAT = """
Team members in your team you're in charge of:
{models_list}
"""

# Teams Available to Lead Format
TEAMS_AVAILABLE_TO_LEAD_FORMAT = """
Teams visible to you:
{teams_list}
"""

# Observations Format - Used to format observations in reasoning/planning
OBSERVATIONS_FORMAT = """
OBSERVATIONS:
{formatted_observations}
"""

# Thoughts Format - Used to format thoughts in reasoning/planning
THOUGHTS_FORMAT = """
THOUGHTS:
{formatted_thoughts}
"""

# Results Format - Used to format results in reasoning/planning
RESULTS_FORMAT = """
RESULTS:
{formatted_results}
"""

#################################################################
# FORMATTING FUNCTIONS
#################################################################

def format_team_lead_tool_usage_prompt(tool_name: str) -> str:
    """Format a tool usage prompt for team leads."""
    # Using a dictionary lookup for better maintainability
    tool_formats = {
        "delegate_task": lambda: """
## delegate_task
Description:
The delegate_task tool allows you to assign a sub-task to a specific agent within your team.
Parameters:
- target_agent_id: (required) ID of the agent to delegate the task to
- task_description: (required) Detailed description of the task
- parent_task_id: (required) ID of the parent task
- context_keys: (optional) List of context keys for additional background
- required_artifacts: (optional) List of artifacts required
Usage:
{
  "tool_name": "delegate_task",
  "arguments": {
    "target_agent_id": "agent_id",
    "parent_task_id": "parent_id",
    "task_description": "Describe the sub-task...",
    "context_keys": [],
    "required_artifacts": []
  }
}
""",
        "web_search": lambda: """
## web_search
Description:
The web_search tool performs a web search to find information or verify facts relevant to the assigned task. It takes a query as an argument and returns a list of results. The query should be a question that can be answered by the web search, concise, to the point, and no more than 100 words.
Parameters:
- query: (required) The search query string.
Usage:
{
  "tool_name": "web_search",
  "arguments": {
    "query": "a list of the top 10 competitors of product x"
  }
}
""",
        "file_handler": lambda: """
## file_handler
Description:
The file_handler tool allows you to read, write, append, delete, and list files. It is a powerful tool that can be used to manage files on a wide range of topics.
Parameters:
- action: (required) The file operation to perform (e.g., 'read', 'write', 'append', 'delete', 'list').
- path: (required) The path to the file or directory.
- content: (optional) The content to write or append (required for 'write', 'append').
Usage:
(Example for reading a file)
{
  "tool_name": "file_handler",
  "arguments": {
    "action": "read",
    "path": "path/to/your/file.txt"
  }
}
(Example for writing to a file)
{
  "tool_name": "file_handler",
  "arguments": {
    "action": "write",
    "path": "path/to/new/file.log",
    "content": "Log entry: Task started."
  }
}
""",
        "code_interpreter": lambda: """
## code_interpreter
Description:
The code_interpreter tool allows you to execute code in Python. It is a powerful tool that can be used to execute code on a wide range of topics.
Parameters:
- code: (required) The code snippet to execute.
Usage:
{
  "tool_name": "code_interpreter",
  "arguments": {
    "code": "import math\\nradius = 5\\narea = math.pi * radius**2\\nprint(f'The area is: {area}')"
  }
}
""",
    }

    formatter_func = tool_formats.get(tool_name)
    if formatter_func:
        # Escape curly braces in the result
        return _escape_curly_braces(formatter_func())
    else:
        logging.warning(f"No formatter found for tool: {tool_name}")
        return ""  # Return empty string for unknown tools


def format_team_member_tool_usage_prompt(tool_name: str) -> str:
    """Format a tool usage prompt for team members."""
    # Using a dictionary lookup for better maintainability
    tool_formats = {
        "report_task_completion": lambda: """
## report_task_completion
Description:
The report_task_completion tool signals that a task is complete and returns results to the lead. The system auto-injects the task_id, calling_agent_id, and calling_team, so you must NOT include them in your JSON arguments.
Parameters:
- status: (required) Completion status, one of 'success', 'failure', or 'escalation'.
- detailed_answer: (required) Detailed answer of the task results.
- artifact_keys: (optional) List of artifact keys produced during the task.
- failure_reason: (optional) Concise explanation for why the task failed or needs escalation (only include if status is 'failure' or 'escalation').
Usage (include only these arguments, in 'arguments'):
{
  "tool_name": "report_task_completion",
  "arguments": {
    "status": "success",
    "detailed_answer": "Detailed answer of the task results...",
    "artifact_keys": [],
    "failure_reason": null
  }
}
""",
        "web_search": lambda: """
## web_search
Description:
The web_search tool performs a web search to find information or verify facts relevant to the assigned task. It takes a query as an argument and returns a list of results. The query should be a question that can be answered by the web search, concise, to the point, and no more than 100 words.
Parameters:
- query: (required) The search query string.
Usage:
{
  "tool_name": "web_search",
  "arguments": {
    "query": "a list of the top 10 competitors of product x"
  }
}
""",
        "file_handler": lambda: """
## file_handler
Description:
The file_handler tool allows you to read, write, append, delete, and list files. It is a powerful tool that can be used to manage files on a wide range of topics.
Parameters:
- action: (required) The file operation to perform (e.g., 'read', 'write', 'append', 'delete', 'list').
- path: (required) The path to the file or directory.
- content: (optional) The content to write or append (required for 'write', 'append').
Usage:
(Example for reading a file)
{
  "tool_name": "file_handler",
  "arguments": {
    "action": "read",
    "path": "path/to/your/file.txt"
  }
}
(Example for writing to a file)
{
  "tool_name": "file_handler",
  "arguments": {
    "action": "write",
    "path": "path/to/new/file.log",
    "content": "Log entry: Task started."
  }
}
""",
        "code_interpreter": lambda: """
## code_interpreter
Description:
The code_interpreter tool allows you to execute code in Python. It is a powerful tool that can be used to execute code on a wide range of topics.
Parameters:
- code: (required) The code snippet to execute.
Usage:
{
  "tool_name": "code_interpreter",
  "arguments": {
    "code": "import math\\nradius = 5\\narea = math.pi * radius**2\\nprint(f'The area is: {area}')"
  }
}
""",
    }

    formatter_func = tool_formats.get(tool_name)
    if formatter_func:
        # Escape curly braces in the result
        return _escape_curly_braces(formatter_func())
    else:
        logging.warning(f"No formatter found for tool: {tool_name}")
        return ""  # Return empty string for unknown tools


# Dynamic prompt formatters
def format_team_structure(models_list: str = "", teams_list: str = "") -> str:
    """Format instructions for the team structure component.

    Args:
        models_list: String containing formatted list of models (empty string if none)
        teams_list: String containing formatted list of teams (empty string if none)

    Returns:
        Formatted team structure
    """
    # Initialize instructions to avoid UnboundLocalError
    instructions = ""
    # Add models and teams information
    additional_info = ""

    if models_list:
        additional_info += MEMBERS_IN_LEAD_TEAM_FORMAT.format(models_list=models_list)

    if teams_list:
        additional_info += TEAMS_AVAILABLE_TO_LEAD_FORMAT.format(teams_list=teams_list)

    # If we have additional info, escape it and append to the instructions
    if additional_info:
        instructions += "\n" + _escape_curly_braces(additional_info)

    return instructions


def _escape_curly_braces(text: str, preserve_placeholders: list = None) -> str:
    """Escape curly braces in a string for safe use with str.format().

    Args:
        text: Text that may contain curly braces
        preserve_placeholders: Optional list of placeholder names to preserve

    Returns:
        Text with curly braces escaped (doubled), preserving specified placeholders
    """
    if preserve_placeholders is None:
        # Default placeholders to preserve
        preserve_placeholders = [
            "team_name",
            "team_name_slug",
            "role_description",
            "additional_context",
            "interactive_protocols",
            "adaptive_user_interaction",
            "user_given_role",
            "agent_role_slug",
            "tools_given_prompts_for_this_lead",
            "tools_given_prompts_for_this_agent",
        ]

    # Temporarily replace placeholders we want to preserve
    placeholder_map = {}
    for ph in preserve_placeholders:
        placeholder = "{" + ph + "}"
        placeholder_token = f"__PRESERVE_PLACEHOLDER_{ph}__"
        placeholder_map[placeholder] = placeholder_token
        text = text.replace(placeholder, placeholder_token)

    # Escape all remaining curly braces
    text = text.replace("{", "{{").replace("}", "}}")

    # Restore preserved placeholders
    for placeholder, token in placeholder_map.items():
        text = text.replace(token, placeholder)

    return text
