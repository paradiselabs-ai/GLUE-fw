# src/glue/core/glue_smolagent.py
from smolagents import CodeAgent, InferenceClientModel, Tool
from typing import Optional, Dict, Any, List
from ..core.glue_memory_adapters import GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from ..core.types import AdhesiveType
import logging
import types
from jinja2 import Environment, BaseLoader, Undefined

logger = logging.getLogger(__name__)
# Add or update the default system prompt template for GlueSmolAgent
DEFAULT_SYSTEM_PROMPT_TEMPLATE = '''
# ReAct Agent System Prompt

You are a ReAct agent designed to solve tasks through systematic reasoning and action. Follow the strict **Thought → Action → Observation** cycle until you reach a final answer.

## Core Process

### 1. Thought
Before each action, provide detailed reasoning that includes:
- **Current Understanding**: What do you know about the task/problem?
- **Analysis**: What have you learned from previous observations?
- **Strategy**: What approach will you take and why?
- **Next Step**: What specific action will move you closer to the solution?
- **Alternatives**: What backup plans do you have if this approach fails?

**For ambiguous or unclear tasks**: If a `user_input` tool is available, use it to clarify the request. Otherwise, make reasonable assumptions based on context and proceed with the most likely interpretation.

### 2. Action
Execute **exactly ONE** tool using proper Python syntax:
```python
tool_name(parameter="value", parameter2="value2")
```
- Use only tools from the "Available Tools" list
- Parameters must be simple values (string, number, boolean)
- No JSON objects or complex data structures

### 3. Observation
State the exact result returned by the action. Do not interpret or summarize unless the result is extremely long.

### 4. Repeat
Continue the cycle until you have sufficient information for a complete answer.

### 5. Final Answer
When ready, use **ONLY** the `final_answer` tool:
```python
final_answer(answer="Your complete and concise answer here.")
```

## Available Tools
{%- for tool in tools.values() %}
- **{{ tool.name }}**: {{ tool.description }}
  - Parameters: {{ tool.inputs.keys() | join(', ') }}
  - Usage: `{{ tool.name }}({% for param, meta in tool.inputs.items() %}{{ param }}="description"{% if not loop.last %}, {% endif %}{% endfor %})`
{%- endfor %}

{%- if managed_agents and managed_agents.values() | list %}

## Team Members
{%- for agent in managed_agents.values() %}
- **{{ agent.name }}**: {{ agent.description }}
  - Usage: `{{ agent.name }}(task="comprehensive description of what you need")`
{%- endfor %}
{%- endif %}

## Authorized Python Imports
{{authorized_imports}}

## Critical Rules

1. **Start with Thought**: Always begin with detailed reasoning
2. **One Action Rule**: Execute only ONE tool per Action step
3. **Use Listed Tools Only**: Don't invent tools or use unauthorized imports
4. **Proper Syntax**: Tool calls must use exact Python function syntax
5. **No Premature Answers**: Never state the final answer before using `final_answer()`
6. **Handle Ambiguity**: For unclear requests, use `user_input()` if available to clarify, otherwise make reasonable assumptions
7. **Error Recovery**: If a tool fails, explain why and try alternative approaches
8. **Efficient Flow**: Once you have sufficient information, proceed directly to `final_answer()` - avoid unnecessary follow-up questions
9. **Stay Focused**: Don't expand the conversation beyond what's needed to complete the task

## Task Approach Strategy

**For Simple/Clear Tasks**: 
- Proceed directly with the most efficient approach
- Minimize unnecessary steps
- Once you have the information needed, use `final_answer()` immediately

**For Ambiguous Tasks**:
- If `user_input()` is available, use it to clarify requirements early
- If not available, make reasonable assumptions based on context and proceed with the most likely interpretation
- Avoid making wild guesses - stick to reasonable interpretations

**For Complex Tasks**:
- Break into logical sub-problems
- Use multiple tools systematically
- Verify results when possible
- Avoid unnecessary expansion of scope

## Example: Good ReAct Flow

**Thought:** The user said "hey" which is a casual greeting. This appears to be an incomplete or ambiguous request. Since I have access to a `user_input` tool, I should ask for clarification about what specific task or information they need help with.

**Action:**
```python
user_input(question="Hello! How can I help you today? What would you like to know or discuss?")
```

**Observation:** User responds: "I'd like to define llms"

**Thought:** Now I have a clear request. The user wants to define "llms" which most likely refers to "Large Language Models" given the context. Since this is a straightforward definitional request that I can answer from my knowledge base, I don't need to search or use additional tools. I'll provide a comprehensive but concise definition.

**Action:**
```python
final_answer(answer="Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like language. They use transformer architecture and neural networks to process context, answer questions, complete tasks, and engage in conversations. Examples include GPT, Claude, and Llama models.")
```

### Alternative Example (without user_input available):

**Thought:** The user said "hey" which is a casual greeting but appears incomplete. Since I don't have a `user_input` tool available, I need to make a reasonable assumption about their intent. "Hey" could be a greeting expecting a response, or they might have sent an incomplete message. The most reasonable approach is to provide a friendly greeting and indicate I'm ready to help with any questions or tasks.

**Action:**
```python
final_answer(answer="Hello! I'm here to help with any questions or tasks you might have. Please let me know what you'd like to know or discuss.")
```

## Common Mistakes to Avoid

❌ **Multiple tools in one action:**
```python
search_web(query="topic")
user_input(question="clarification")  # WRONG - two tools
```

❌ **Stating answer before final_answer:**
```python
# WRONG
Thought: The answer is Paris.
Action: final_answer(answer="Paris")
```

❌ **Insufficient reasoning:**
```python
# WRONG - too brief
Thought: I'll search for this.
```

❌ **Complex parameter formats:**
```python
# WRONG
tool_name(param={"key": "value"})
```

---

**Remember**: Quality reasoning in your Thought section is crucial. Explain your logic, consider alternatives, and connect each step to your overall goal. Begin every response with "Thought:"
'''

# Add a lean template for sub-agents
SUB_AGENT_SYSTEM_PROMPT_TEMPLATE = '''
# **YOUR ROLE & TASK**

You are {{ agent_name }}. Your task is to use your available tools to successfully complete the specific assignment you've been given.

# **YOUR RESOURCES**

**1. Available Tools:**
{{tool_descriptions}}

**2. Authorized Imports (for code execution if applicable):**
{{authorized_imports}}

**3. Team Context (Managed Agents Overview - For Your Awareness):**
# This section provides context about the broader team structure, even if you do not delegate.
{{managed_agents_description}}

# **HOW TO OPERATE**

1.  **Understand Your Assignment:** Carefully review the task given to you.
2.  **Utilize Your Tools:**
    *   Select the best tool(s) to accomplish your task.
    *   *You are a specialized agent. Focus on using your tools; do not delegate tasks further.*
3.  **Handle Tool Issues:**
    *   If a tool call encounters an error, analyze the 'Observation:'.
    *   Adjust your approach, correct tool arguments if needed, and retry if it seems logical.
4.  **Provide Your Result:**
    *   Once you have completed your task or determined you cannot proceed, clearly state your reasoning and then use the `final_answer` tool.
    *   **Format:**
        Thoughts: <Your brief reasoning for the action or final outcome>
        Code:
        ```python
        # Example: result = your_tool_name(input="value")
        final_answer("<Your complete answer or result for the assigned task>")
        ```
    *   If unrecoverable errors prevent task completion, use `final_answer` to explain the issue and what you attempted.

# **YOUR ASSIGNED TASK**

{{user_task}}

---
{{ agent_name }}, please proceed with your assigned task.
'''

class GlueSmolAgent(CodeAgent):
    """
    A wrapper around smolagents CodeAgent that integrates GLUE-specific features
    such as adhesives and custom memory persistence hooks.
    """

    def __init__(
        self,
        model: InferenceClientModel,
        tools: list,
        planning_interval: int = 1,
        system_prompt_override: str = "",
        glue_config: Optional[Dict[str, Any]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        managed_agents_dict: Optional[Dict[str, Any]] = None,
        team_name_for_prompt: Optional[str] = "team",
        **kwargs
    ):
        self.glue_config = glue_config or {}
        self._agent_name_for_prompt = name or "Agent"
        self._team_name_for_prompt = team_name_for_prompt
        self._initial_managed_agents_dict_for_template_choice = managed_agents_dict or {}
        # Removed basicConfig and local logger; use module-level logger

        # Ensure name and description attributes exist before initializing prompt templates
        self.name = name
        self.description = description

        is_lead_agent_initial_check = bool(self._initial_managed_agents_dict_for_template_choice)
        if system_prompt_override:
            self._chosen_system_prompt_template_str = system_prompt_override
            logger.info(f"[GLUE] Agent '{self._agent_name_for_prompt}' will use custom system prompt (override string).")
        elif is_lead_agent_initial_check:
            self._chosen_system_prompt_template_str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
            logger.info(f"[GLUE] Agent '{self._agent_name_for_prompt}' will use LEAD agent system prompt template string (based on initial dict for '{name}').")
        else:
            self._chosen_system_prompt_template_str = SUB_AGENT_SYSTEM_PROMPT_TEMPLATE
            logger.info(f"[GLUE] Agent '{self._agent_name_for_prompt}' will use SUB-AGENT system prompt template string (based on initial dict for '{name}').")

        smolagents_tools: List[Tool] = []
        for t_spec in tools:
            valid_tool: Optional[Tool] = None
            tool_name_for_log = "unknown_tool"

            if isinstance(t_spec, Tool):
                valid_tool = t_spec
                tool_name_for_log = getattr(valid_tool, 'name', tool_name_for_log)
            elif hasattr(t_spec, '_glue_tool_schema') and callable(t_spec):
                schema = t_spec._glue_tool_schema  # type: ignore
                func_name = getattr(t_spec, '__name__', 'dynamic_tool')
                tool_name_for_log = func_name
                description_str = schema.get('description') or getattr(t_spec, '__doc__', f'Executes the {func_name} tool.') or ''
                inputs_dict = schema.get('inputs', {})
                output_type_val = schema.get('output_type', 'Any')
                output_type_str = str(output_type_val) if not isinstance(output_type_val, str) else output_type_val

                input_keys = list(inputs_dict.keys())
                args_str = ', '.join(input_keys)
                
                ns: Dict[str, Any] = {}
                exec_globals = {'t_spec_func': t_spec}
                forward_method_code = f"def forward(self, {args_str}):\n    return t_spec_func({args_str})"
                
                try:
                    exec(forward_method_code, exec_globals, ns)
                    forward_method = ns['forward']
                except Exception as e:
                    logger.error(f"Failed to create forward method for {func_name}: {e}")
                    logger.error(f"Code was: {forward_method_code}")
                    continue 

                attrs = {
                    'name': func_name,
                    'description': description_str,
                    'inputs': inputs_dict,
                    'output_type': output_type_str, 
                    'forward': forward_method,
                }
                ToolClass = type(f"{func_name}_GlueFunctionTool", (Tool,), attrs)
                valid_tool = ToolClass()
            else:
                logger.error(f"Tool spec {t_spec} for agent '{name}' is not a SmolAgents Tool or a GLUE function tool with _glue_tool_schema. Skipping.")
                continue
            
            if valid_tool:
                original_tool_callable = valid_tool.__call__
                def create_logged_call(tool_instance, original_callable):
                    def logged_call(*args, **kwargs):
                        tool_name_call = getattr(tool_instance, 'name', 'unnamed_tool_instance')
                        # Log tool execution start
                        logging.getLogger("glue.smolagent").debug(
                            f"[GLUE TOOL EXEC] Calling {tool_name_call} with args={args}, kwargs={kwargs}"
                        )
                        try:
                            result = original_callable(*args, **kwargs)
                            # Log tool execution result
                            logging.getLogger("glue.smolagent").debug(
                                f"[GLUE TOOL EXEC] {tool_name_call} returned: {str(result)[:500]}..."
                            )
                            return result
                        except Exception as e:
                            error_message = f"Error executing tool {tool_name_call}: {type(e).__name__}: {str(e)}"
                            logging.getLogger("glue.smolagent").error(
                                f"[GLUE TOOL EXEC ERROR] {error_message}"
                            )
                            raise
                    return logged_call

                valid_tool.__call__ = types.MethodType(create_logged_call(valid_tool, original_tool_callable), valid_tool)
                smolagents_tools.append(valid_tool)
                logger.debug(f"Processed and wrapped GLUE tool: {tool_name_for_log} for agent '{name}'")
        
        if 'add_base_tools' not in kwargs:
            kwargs['add_base_tools'] = True
            logger.debug(f"Agent '{self._agent_name_for_prompt}': 'add_base_tools' set to True by default.")

        # Inject raw system prompt template into prompt_templates for CodeAgent
        try:
            from smolagents.agents import EMPTY_PROMPT_TEMPLATES, PromptTemplates
            prompt_templates_obj: PromptTemplates = PromptTemplates(
                system_prompt=self._chosen_system_prompt_template_str,
                planning=EMPTY_PROMPT_TEMPLATES['planning'],
                managed_agent=EMPTY_PROMPT_TEMPLATES['managed_agent'],
                final_answer=EMPTY_PROMPT_TEMPLATES['final_answer'],
            )
            kwargs['prompt_templates'] = prompt_templates_obj
        except ImportError as e:
            logger.warning(f"Agent '{self._agent_name_for_prompt}': failed to import PromptTemplates or EMPTY_PROMPT_TEMPLATES: {e}")

        kwargs_for_super = kwargs.copy() 
        if 'managed_agents' in kwargs_for_super:
            logger.warning(f"Agent '{name}': 'managed_agents' was found in general kwargs. It will be overridden by the explicit 'managed_agents_dict' parameter. Removing from kwargs for super().__init__ call.")
            del kwargs_for_super['managed_agents']
        
        managed_agents_list_for_super = []
        if managed_agents_dict and isinstance(managed_agents_dict, dict):
            managed_agents_list_for_super = list(managed_agents_dict.values())
            logger.debug(f"Agent '{name}': Converting managed_agents_dict to list for superclass: {[getattr(a, 'name', 'N/A') for a in managed_agents_list_for_super]}")
        
        super().__init__(
            tools=smolagents_tools,
            model=model,
            additional_authorized_imports=additional_authorized_imports,
            planning_interval=planning_interval,
            name=name,
            description=description,
            managed_agents=managed_agents_list_for_super,
            **kwargs_for_super,
        )
        
        self.name = name
        self.description = description
        self.user_task = "" 

        self._init_glue_features()
        final_managed_agents_in_self = getattr(self, 'managed_agents', {})
        final_tools_in_self = getattr(self, 'tools', {})
        logger.info(
            f"[GLUE] Finished initializing GlueSmolAgent '{self.name}'. "
            f"Actual tools from self.tools: {[t_name for t_name in final_tools_in_self.keys()] if final_tools_in_self else 'None'}. "
            f"Actual managed_agents from self.managed_agents: {[ma_name for ma_name in final_managed_agents_in_self.keys()] if final_managed_agents_in_self else 'None'}"
        )


    def initialize_system_prompt(self) -> str:
        # Removed basicConfig and local logger; use module-level logger
        current_managed_agents_from_self = getattr(self, 'managed_agents', {})
        is_actually_lead = bool(current_managed_agents_from_self)
        
        template_str_to_render: str
        if hasattr(self, '_chosen_system_prompt_template_str'):
            initial_choice_was_override = self._chosen_system_prompt_template_str not in [DEFAULT_SYSTEM_PROMPT_TEMPLATE, SUB_AGENT_SYSTEM_PROMPT_TEMPLATE]
            if initial_choice_was_override:
                template_str_to_render = self._chosen_system_prompt_template_str
                logger.debug(f"Agent '{self.name}' (in initialize_system_prompt): Using explicit system_prompt_override string.")
            elif is_actually_lead:
                template_str_to_render = DEFAULT_SYSTEM_PROMPT_TEMPLATE
                logger.debug(f"Agent '{self.name}' (in initialize_system_prompt): Using LEAD agent template because self.managed_agents is populated: {list(current_managed_agents_from_self.keys())}.")
            else:
                template_str_to_render = SUB_AGENT_SYSTEM_PROMPT_TEMPLATE
                logger.debug(f"Agent '{self.name}' (in initialize_system_prompt): Using SUB-AGENT template because self.managed_agents is empty.")
        else:
            logger.warning(f"Agent '{self.name}': _chosen_system_prompt_template_str not found, determining prompt purely on lead status.")
            template_str_to_render = DEFAULT_SYSTEM_PROMPT_TEMPLATE if is_actually_lead else SUB_AGENT_SYSTEM_PROMPT_TEMPLATE

        if not template_str_to_render:
            logger.error(f"Agent '{self.name}': System prompt template string is empty. Fallback.")
            return "You are a helpful assistant. Please solve the task."

        env = Environment(loader=BaseLoader(), undefined=Undefined)
        template = env.from_string(template_str_to_render)

        # Build tool descriptions
        tool_descs = []
        if hasattr(self, 'tools') and self.tools:
            for tool_name, tool_obj in self.tools.items():
                desc = getattr(tool_obj, 'description', f"Tool named '{tool_name}'.")
                inputs_schema = getattr(tool_obj, 'inputs', None)
                if inputs_schema and isinstance(inputs_schema, dict):
                    args_desc_parts = []
                    for arg_name, arg_info in inputs_schema.items():
                        arg_type = arg_info.get('type', 'unknown') if isinstance(arg_info, dict) else 'unknown'
                        arg_desc_text = arg_info.get('description', '') if isinstance(arg_info, dict) else ''
                        args_desc_parts.append(f"{arg_name} ({arg_type}): {arg_desc_text}")
                    if args_desc_parts:
                        desc += " Args: " + "; ".join(args_desc_parts)
                tool_descs.append(f"- {tool_name}: {desc}")
        tool_descriptions_str = "\n".join(tool_descs) if tool_descs else "No tools available."

        # Build managed agents descriptions
        managed_agents_desc_parts = []
        if current_managed_agents_from_self and isinstance(current_managed_agents_from_self, dict):
            for agent_name, agent_instance in current_managed_agents_from_self.items():
                agent_desc = getattr(agent_instance, 'description', "Managed agent.")
                managed_agents_desc_parts.append(f"- {agent_name}: {agent_desc}")
        managed_agents_description_str = "\n".join(managed_agents_desc_parts) if managed_agents_desc_parts else "You have no managed agents to delegate to."

        # Determine display name
        agent_name_for_prompt_display = self._agent_name_for_prompt
        if is_actually_lead:
            agent_name_for_prompt_display = f"{self._team_name_for_prompt}_LeadAgent ({self._agent_name_for_prompt})"
        else:
            agent_name_for_prompt_display = f"{self._team_name_for_prompt}_SubAgent ({self._agent_name_for_prompt})"

        # Ensure task is set
        user_task_for_prompt = getattr(self, "user_task", "") or "No specific task assigned yet. Please await instructions or analyze the situation."

        context = {
            "agent_name": self._agent_name_for_prompt,
            "team_name": self._team_name_for_prompt,
            "agent_name_or_team_lead": agent_name_for_prompt_display,
            "tool_descriptions": tool_descriptions_str,
            "authorized_imports": str(getattr(self, "authorized_imports", [])),
            "managed_agents_description": managed_agents_description_str,
            "managed_agents": current_managed_agents_from_self,
            "user_task": user_task_for_prompt,
            "tools": self.tools,
        }

        try:
            return template.render(**context)
        except Exception as e:
            logger.error(f"[GLUE] Error rendering system prompt for agent '{self.name}': {e}. Using raw template.")
            return template_str_to_render

    def _init_glue_features(self):
        adhesives = set()
        if hasattr(self, 'glue_config') and self.glue_config and 'adhesives' in self.glue_config:
            adhesives = set(self.glue_config['adhesives'])
        elif hasattr(self.model, 'adhesives'): 
            adhesives = set(self.model.adhesives) # type: ignore
        if not adhesives: 
            adhesives = {AdhesiveType.GLUE}
        self.memories: Dict[str, Any] = {}
        team_id_for_mem = getattr(self, 'team_id', self._team_name_for_prompt) 
        # Ensure team_id_for_mem is a string for GLUEPersistentAdapter
        if not isinstance(team_id_for_mem, str):
            logger.warning(f"Agent '{self.name}': team_id_for_mem was not a string (was {type(team_id_for_mem)}), defaulting to 'default_team_id_fallback'.")
            team_id_for_mem = "default_team_id_fallback"
        
        for adhesive in adhesives:
            if adhesive == AdhesiveType.GLUE:
                self.memories['glue'] = GLUEPersistentAdapter(team_id=team_id_for_mem, memory_dir='memory')
            elif adhesive == AdhesiveType.VELCRO:
                self.memories['velcro'] = VELCROSessionAdapter()
            elif adhesive == AdhesiveType.TAPE:
                self.memories['tape'] = TAPEEphemeralAdapter()
        if 'glue' in self.memories:
            self.memory = self.memories['glue']
        elif 'velcro' in self.memories:
            self.memory = self.memories['velcro']
        elif 'tape' in self.memories:
            self.memory = self.memories['tape']
        else:
            # Ensure team_id_for_mem is a string for the fallback GLUEPersistentAdapter
            fallback_team_id = team_id_for_mem
            if not isinstance(team_id_for_mem, str): # This check is redundant due to the earlier check, but kept for safety here
                 logger.warning(f"Agent '{self.name}': team_id_for_mem was not a string for fallback (was {type(team_id_for_mem)}), using 'default_team_id_fallback_else'.")
                 fallback_team_id = "default_team_id_fallback_else"
            self.memory = GLUEPersistentAdapter(team_id=fallback_team_id, memory_dir='memory')
            logging.warning(f"Agent '{self.name}' had no specific memory adapter; defaulted to GLUEPersistentAdapter.")
        logger.info(f"[GLUE] Agent '{self.name}' memory adapters: {list(self.memories.keys())}, default: {type(self.memory).__name__ if self.memory else None}")

    def debug_print_agent_memory(self): 
        logger.debug(f"Agent '{getattr(self, 'name', 'unnamed')}' memory adapters:")
        for k, v in getattr(self, 'memories', {}).items():
            logger.debug(f"  - {k}: {type(v).__name__}")
        logger.debug(f"Default memory: {type(getattr(self, 'memory', None)).__name__ if getattr(self, 'memory', None) else None}")

    def debug_print_tool_schemas(self): 
        logger.debug(f"Tool schemas for agent '{getattr(self, 'name', 'unnamed')}':")
        for tool_name, tool_instance in getattr(self, 'tools', {}).items():
            description = getattr(tool_instance, 'description', 'N/A')
            inputs = getattr(tool_instance, 'inputs', 'N/A')
            output_type = getattr(tool_instance, 'output_type', 'N/A')
            logger.debug(f"  - Name: {tool_name}")
            logger.debug(f"    Description: {description}")
            logger.debug(f"    Inputs: {inputs}")
            logger.debug(f"    Output type: {output_type}")

    # Removed _generate_messages method as it was causing an error with super()
    # and its functionality for refreshing system prompt is handled by initialize_system_prompt
    # or would be better placed in an overridden write_memory_to_messages if dynamic updates are needed.

    # Add method to ensure interpreter exists for flow and tool injection
    def force_interpreter(self):
        """Ensure interpreter and interpreter_globals exist for code execution and flow function injection."""
        if hasattr(self, 'python_executor'):
            interp = self.python_executor
        else:
            interp = type('Interpreter', (), {})()
        if not hasattr(interp, 'globals'):
            setattr(interp, 'globals', {})
        self.interpreter = interp
        self.interpreter_globals = interp.globals # type: ignore

    # Update run method to call force_interpreter and inject tools into interpreter.globals
    def run(self, task: str, stream: bool = False, reset: bool = True, images: Optional[List[Any]] = None, additional_args: Optional[Dict[str, Any]] = None, max_steps: Optional[int] = None, **kwargs):
        self.user_task = task
        logger.info(f"Agent '{self.name}' starting run for task: {task}")
        self.force_interpreter()
        # Pass along all parameters to the super().run call, including the new ones
        return super().run(task=task, stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)


def make_glue_smol_agent(
    *, 
    model: InferenceClientModel, 
    tools: list, 
    glue_config: Optional[Dict[str, Any]], 
    name: str, 
    description: str, 
    managed_agents_dict: Optional[Dict[str, Any]] = None, 
    team_name: str ="default_team", 
    **kwargs
):
    if not name or not isinstance(name, str): 
        logger.error("All agents must have a non-empty string 'name' attribute.")
        raise ValueError("All agents must have a non-empty string 'name' attribute.")
    if not description or not isinstance(description, str):
        logger.error(f"Agent '{name}' must have a non-empty string 'description' attribute.")
        raise ValueError(f"Agent '{name}' must have a non-empty string 'description' attribute.")

    for t_idx, t_spec in enumerate(tools): 
        if not (isinstance(t_spec, Tool) or (callable(t_spec) and hasattr(t_spec, '_glue_tool_schema'))):
            logger.error(f"Tool at index {t_idx} for agent '{name}' ('{t_spec}') is not a valid SmolAgents Tool or GLUE function tool.")

    system_prompt_override = None
    if hasattr(model, 'smol_config') and isinstance(model.smol_config, dict): # type: ignore
        system_prompt_override = model.smol_config.get('system_prompt', None) # type: ignore

    if managed_agents_dict:
        for ma_name, ma_instance in managed_agents_dict.items():
            if not isinstance(ma_instance, CodeAgent): 
                logger.error(f"Managed agent '{ma_name}' for lead '{name}' is not a valid agent instance. Type: {type(ma_instance)}")
                # This is a critical error for smolagents, which expects agent instances.
                # Depending on strictness, you might want to raise an error or clear the dict.
                # For now, we'll let it pass to smolagents and see if it handles it or errors out,
                # but this is a point of potential failure if not actual agent instances.

    try:
        agent = GlueSmolAgent(
            model=model,
            tools=tools, 
            glue_config=glue_config,
            name=name,
            description=description,
            managed_agents_dict=managed_agents_dict, 
            team_name_for_prompt=team_name, 
            system_prompt_override=system_prompt_override or "",
            **kwargs 
        )
        logger.info(f"Successfully created GlueSmolAgent '{name}'.")
        return agent
    except Exception as e:
        logger.error(f"Error creating GlueSmolAgent '{name}': {e}")
        logger.exception(e) 
        raise