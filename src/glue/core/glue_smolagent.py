# src/glue/core/glue_smolagent.py
from smolagents import CodeAgent, InferenceClientModel, Tool
from typing import Optional, Dict, Any, List
from ..core.glue_memory_adapters import GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from ..core.types import AdhesiveType
import logging
import types
from jinja2 import Environment, BaseLoader, Undefined
from datetime import datetime

logger = logging.getLogger(__name__)
# Add or update the default system prompt template for GlueSmolAgent
DEFAULT_SYSTEM_PROMPT_TEMPLATE = '''
# ReAct Agent System Prompt

You are a ReAct agent designed to solve tasks through systematic reasoning and action. Follow the strict **Thought -> Action -> Observation** cycle until you reach a final answer.

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
When ready, use **ONLY** the `final_answer` tool, and always output in the following format:

Thoughts: <Your brief reasoning for the action or final outcome>
Code:
```py
final_answer(answer="Your complete and concise answer here.")
```
<end_code>

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
<end_code>

### Alternative Example (without user_input available):

**Thought:** The user said "hey" which is a casual greeting but appears incomplete. Since I don't have a `user_input` tool available, I need to make a reasonable assumption about their intent. "Hey" could be a greeting expecting a response, or they might have sent an incomplete message. The most reasonable approach is to provide a friendly greeting and indicate I'm ready to help with any questions or tasks.

**Action:**
```python
final_answer(answer="Hello! I'm here to help with any questions or tasks you might have. Please let me know what you'd like to know or discuss.")
```
<end_code>

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

**Remember**: Quality reasoning in your Thought section is crucial. Explain your logic, consider alternatives, and connect each step to your overall goal. Begin every response with "Thought:" and always output your code in the required format:

Thoughts: <reasoning>
Code:
```py
# your code here
```
<end_code>

---
**IMPORTANT:**
If you are unable to answer, encounter an error, or must apologize, you must still output a valid code block in the required format. For example:

Thoughts: I cannot answer because the request is unclear or not possible.
Code:
```py
final_answer(answer="I'm sorry, but I cannot answer this request as stated.")
```
<end_code>
Never output plain text or apologies outside of this format.
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
        ```py
        # Example: result = your_tool_name(input="value")
        final_answer("<Your complete answer or result for the assigned task>")
        ```
        <end_code>
    *   If unrecoverable errors prevent task completion, use `final_answer` to explain the issue and what you attempted, always in the required code block format.

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
        
        # Adhesive policy management
        self._next_adhesive_override: Optional[AdhesiveType] = None
        self._adhesive_policy_config: Dict[str, Any] = {}
        self._adhesive_rationale_log: List[Dict[str, Any]] = []
        
        # Store model reference for adhesive system
        self.model = model
        
        # Store model compatibility info separately to avoid modifying the model object
        self._model_adhesive_compatibility = [AdhesiveType.GLUE, AdhesiveType.VELCRO, AdhesiveType.TAPE]
        logger.debug(f"Set adhesive compatibility for model: {[a.value for a in self._model_adhesive_compatibility]}")
        
        # Initialize or set team for adhesive system
        if not hasattr(self, '_team_initialized'):
            from ..core.teams import Team
            team_name = team_name_for_prompt or "DefaultTeam"
            agent_name = name or "DefaultAgent"
            self.team = Team(name=team_name, description=f"Team for agent {agent_name}")
            self._team_initialized = True

        # Initialize adhesive system if not provided
        if not hasattr(self, 'adhesive_system'):
            from .adhesive import AdhesiveSystem
            self.adhesive_system = AdhesiveSystem()

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
                def create_adhesive_aware_call(tool_instance, original_callable, agent_ref):
                    def adhesive_aware_logged_call(*args, **kwargs):
                        tool_name_call = getattr(tool_instance, 'name', 'unnamed_tool_instance')
                        
                        # Log tool execution start
                        logging.getLogger("glue.smolagent").debug(
                            f"[GLUE TOOL EXEC] Calling {tool_name_call} with args={args}, kwargs={kwargs}"
                        )
                        
                        try:
                            # Execute the tool
                            result = original_callable(*args, **kwargs)
                            
                            # Log tool execution result
                            logging.getLogger("glue.smolagent").debug(
                                f"[GLUE TOOL EXEC] {tool_name_call} returned: {str(result)[:500]}..."
                            )
                            
                            # Agent-controlled adhesive binding
                            try:
                                # Create context for adhesive decision
                                context = {
                                    "args": args,
                                    "kwargs": kwargs,
                                    "result_type": type(result).__name__,
                                    "result_length": len(str(result)) if result else 0,
                                    "team_context": agent_ref._is_team_context() if hasattr(agent_ref, '_is_team_context') else False
                                }
                                
                                # Let agent choose adhesive
                                chosen_adhesive = agent_ref.choose_adhesive(tool_name_call, result, context)
                                
                                # Create ToolResult for binding
                                from ..core.simple_schemas import ToolResult
                                tool_result = ToolResult(
                                    tool_name=tool_name_call,
                                    result=result,
                                    metadata={"agent": agent_ref.name, "timestamp": datetime.now().isoformat()},
                                    timestamp=datetime.now().isoformat()
                                )
                                
                                # Bind result with chosen adhesive using custom compatibility check
                                from ..core.adhesive import bind_tool_result
                                
                                # Create a custom model compatibility check
                                def custom_check_compatibility(model, adhesive_type):
                                    # Always return True for our agent since we manage compatibility internally
                                    return True
                                
                                # Temporarily patch the compatibility check
                                import glue.core.adhesive as adhesive_module
                                original_check = adhesive_module.check_adhesive_compatibility
                                adhesive_module.check_adhesive_compatibility = custom_check_compatibility
                                
                                try:
                                    bind_tool_result(
                                        system=agent_ref.adhesive_system,
                                        team=agent_ref.team,
                                        model=agent_ref.model,
                                        tool_result=tool_result,
                                        adhesive_type=chosen_adhesive
                                    )
                                    
                                    logger.info(f"[ADHESIVE BINDING] Agent '{agent_ref.name}': Bound tool '{tool_name_call}' result with {chosen_adhesive.value} adhesive")
                                    
                                finally:
                                    # Restore original compatibility check
                                    adhesive_module.check_adhesive_compatibility = original_check
                                
                            except Exception as binding_error:
                                logger.error(f"[ADHESIVE BINDING ERROR] Failed to bind result for tool {tool_name_call}: {binding_error}")
                                # Continue execution even if binding fails
                            
                            return result
                            
                        except Exception as e:
                            error_message = f"Error executing tool {tool_name_call}: {type(e).__name__}: {str(e)}"
                            logging.getLogger("glue.smolagent").error(
                                f"[GLUE TOOL EXEC ERROR] {error_message}"
                            )
                            raise
                    return adhesive_aware_logged_call

                valid_tool.__call__ = types.MethodType(create_adhesive_aware_call(valid_tool, original_tool_callable, self), valid_tool)
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
        
        # Wrap all tools (including base tools) with adhesive functionality
        self._wrap_all_tools_with_adhesive()
        
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

    def _wrap_all_tools_with_adhesive(self):
        """
        Wrap all tools (including base tools added by smolagents) with adhesive-aware functionality.
        This method is called after superclass initialization to ensure all tools are wrapped.
        """
        if not hasattr(self, 'tools') or not self.tools:
            logger.debug(f"Agent '{self.name}': No tools to wrap with adhesive functionality")
            return
            
        wrapped_count = 0
        for tool_name, tool_instance in self.tools.items():
            # Check if tool is already wrapped (has our custom adhesive wrapper)
            if hasattr(tool_instance, '_adhesive_wrapped'):
                logger.debug(f"Agent '{self.name}': Tool '{tool_name}' already wrapped, skipping")
                continue
                
            # Store original callable
            original_tool_callable = tool_instance.__call__
            
            # Create adhesive-aware wrapper
            def create_adhesive_aware_call(tool_inst, original_callable, agent_ref):
                def adhesive_aware_logged_call(*args, **kwargs):
                    tool_name_call = getattr(tool_inst, 'name', 'unnamed_tool_instance')
                    
                    # Log tool execution start
                    logging.getLogger("glue.smolagent").debug(
                        f"[GLUE TOOL EXEC] Calling {tool_name_call} with args={args}, kwargs={kwargs}"
                    )
                    
                    try:
                        # Execute the tool
                        result = original_callable(*args, **kwargs)
                        
                        # Log tool execution result
                        logging.getLogger("glue.smolagent").debug(
                            f"[GLUE TOOL EXEC] {tool_name_call} returned: {str(result)[:500]}..."
                        )
                        
                        # Agent-controlled adhesive binding
                        try:
                            # Create context for adhesive decision
                            context = {
                                "args": args,
                                "kwargs": kwargs,
                                "result_type": type(result).__name__,
                                "result_length": len(str(result)) if result else 0,
                                "team_context": agent_ref._is_team_context() if hasattr(agent_ref, '_is_team_context') else False
                            }
                            
                            # Let agent choose adhesive
                            chosen_adhesive = agent_ref.choose_adhesive(tool_name_call, result, context)
                            
                            # Create ToolResult for binding
                            from ..core.simple_schemas import ToolResult
                            tool_result = ToolResult(
                                tool_name=tool_name_call,
                                result=result,
                                metadata={"agent": agent_ref.name, "timestamp": datetime.now().isoformat()},
                                timestamp=datetime.now().isoformat()
                            )
                            
                            # Bind result with chosen adhesive using custom compatibility check
                            from ..core.adhesive import bind_tool_result
                            
                            # Create a custom model compatibility check
                            def custom_check_compatibility(model, adhesive_type):
                                # Always return True for our agent since we manage compatibility internally
                                return True
                            
                            # Temporarily patch the compatibility check
                            import glue.core.adhesive as adhesive_module
                            original_check = adhesive_module.check_adhesive_compatibility
                            adhesive_module.check_adhesive_compatibility = custom_check_compatibility
                            
                            try:
                                bind_tool_result(
                                    system=agent_ref.adhesive_system,
                                    team=agent_ref.team,
                                    model=agent_ref.model,
                                    tool_result=tool_result,
                                    adhesive_type=chosen_adhesive
                                )
                                
                                logger.info(f"[ADHESIVE BINDING] Agent '{agent_ref.name}': Bound tool '{tool_name_call}' result with {chosen_adhesive.value} adhesive")
                                
                            finally:
                                # Restore original compatibility check
                                adhesive_module.check_adhesive_compatibility = original_check
                            
                        except Exception as binding_error:
                            logger.error(f"[ADHESIVE BINDING ERROR] Failed to bind result for tool {tool_name_call}: {binding_error}")
                            # Continue execution even if binding fails
                        
                        return result
                        
                    except Exception as e:
                        error_message = f"Error executing tool {tool_name_call}: {type(e).__name__}: {str(e)}"
                        logging.getLogger("glue.smolagent").error(
                            f"[GLUE TOOL EXEC ERROR] {error_message}"
                        )
                        raise
                
                return adhesive_aware_logged_call

            # Apply the wrapper
            tool_instance.__call__ = types.MethodType(
                create_adhesive_aware_call(tool_instance, original_tool_callable, self), 
                tool_instance
            )
            
            # Mark as wrapped to avoid double wrapping
            tool_instance._adhesive_wrapped = True
            wrapped_count += 1
            logger.debug(f"Agent '{self.name}': Wrapped tool '{tool_name}' with adhesive functionality")
        
        logger.info(f"[GLUE] Agent '{self.name}': Wrapped {wrapped_count} tools with adhesive functionality")

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

    # Agent-controlled adhesive policy methods
    def choose_adhesive(self, tool_name: str, result: Any = None, context: Optional[Dict[str, Any]] = None) -> AdhesiveType:
        """
        Choose the appropriate adhesive type for a tool result based on agent policy.
        
        Args:
            tool_name: Name of the tool being executed
            result: The result from tool execution (optional)
            context: Additional context for adhesive decision (optional)
            
        Returns:
            AdhesiveType: The chosen adhesive type
        """
        context = context or {}
        
        # Check for explicit override first
        if self._next_adhesive_override is not None:
            chosen = self._next_adhesive_override
            rationale = f"Using explicit override: {chosen.value}"
            self._log_adhesive_decision(tool_name, chosen, rationale, context)
            self._next_adhesive_override = None  # Clear after use
            return chosen
        
        # Check for tool-specific policy
        if tool_name in self._adhesive_policy_config:
            chosen = self._adhesive_policy_config[tool_name]
            rationale = f"Using tool-specific policy: {chosen.value}"
            self._log_adhesive_decision(tool_name, chosen, rationale, context)
            return chosen
        
        # Apply default policy based on agent capabilities and context
        available_adhesives = set()
        if hasattr(self, 'memories'):
            available_adhesives = set(self.memories.keys())
        
        # Convert string keys to AdhesiveType for comparison
        available_types = set()
        for mem_key in available_adhesives:
            if mem_key == 'glue':
                available_types.add(AdhesiveType.GLUE)
            elif mem_key == 'velcro':
                available_types.add(AdhesiveType.VELCRO)
            elif mem_key == 'tape':
                available_types.add(AdhesiveType.TAPE)
        
        # Default policy: GLUE > VELCRO > TAPE (team tools > model tools > ephemeral)
        if AdhesiveType.GLUE in available_types and self._is_team_context():
            chosen = AdhesiveType.GLUE
            rationale = "Default policy: GLUE for team-wide persistence"
        elif AdhesiveType.VELCRO in available_types:
            chosen = AdhesiveType.VELCRO
            rationale = "Default policy: VELCRO for session persistence"
        elif AdhesiveType.TAPE in available_types:
            chosen = AdhesiveType.TAPE
            rationale = "Default policy: TAPE for ephemeral use"
        else:
            # Fallback to GLUE if no memories configured
            chosen = AdhesiveType.GLUE
            rationale = "Fallback: GLUE (no adhesive memories configured)"
        
        self._log_adhesive_decision(tool_name, chosen, rationale, context)
        return chosen
    
    def set_next_adhesive(self, adhesive_type: AdhesiveType, rationale: Optional[str] = None):
        """
        Set the adhesive type for the next tool execution.
        
        Args:
            adhesive_type: The adhesive type to use for the next tool call
            rationale: Optional explanation for the choice
        """
        self._next_adhesive_override = adhesive_type
        log_msg = f"Agent '{self.name}': Set next adhesive override to {adhesive_type.value}"
        if rationale:
            log_msg += f" (rationale: {rationale})"
        logger.info(log_msg)
    
    def get_adhesive_policy(self) -> Dict[str, Any]:
        """
        Get the current adhesive policy configuration.
        
        Returns:
            Dict containing current policy settings
        """
        return {
            "next_override": self._next_adhesive_override.value.lower() if self._next_adhesive_override else None,
            "tool_specific_policies": {k: v.value.lower() for k, v in self._adhesive_policy_config.items()},
            "available_adhesives": list(self.memories.keys()) if hasattr(self, 'memories') else [],
            "recent_decisions": self._adhesive_rationale_log[-10:]  # Last 10 decisions
        }
    
    def set_tool_adhesive_policy(self, tool_name: str, adhesive_type: AdhesiveType):
        """
        Set a specific adhesive policy for a particular tool.
        
        Args:
            tool_name: Name of the tool
            adhesive_type: Adhesive type to use for this tool
        """
        self._adhesive_policy_config[tool_name] = adhesive_type
        logger.info(f"Agent '{self.name}': Set adhesive policy for tool '{tool_name}' to {adhesive_type.value}")
    
    def _is_team_context(self) -> bool:
        """Check if the agent is operating in a team context."""
        return bool(getattr(self, 'managed_agents', {})) or len(getattr(self, '_initial_managed_agents_dict_for_template_choice', {})) > 0
    
    def _log_adhesive_decision(self, tool_name: str, adhesive_type: AdhesiveType, rationale: str, context: Dict[str, Any]):
        """Log an adhesive decision for debugging and transparency."""
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "adhesive_type": adhesive_type.value,
            "rationale": rationale,
            "context": context,
            "agent_name": self.name
        }
        self._adhesive_rationale_log.append(decision_entry)
        
        # Keep only last 100 decisions to prevent memory growth
        if len(self._adhesive_rationale_log) > 100:
            self._adhesive_rationale_log = self._adhesive_rationale_log[-100:]
        
        logger.debug(f"[ADHESIVE DECISION] Agent '{self.name}': {tool_name} -> {adhesive_type.value} ({rationale})")

    # Advanced adhesive API methods
    def run_tool_with_adhesive(self, tool_name: str, adhesive_type: AdhesiveType, *args, **kwargs) -> Any:
        """
        Run a tool with a specific adhesive type, bypassing the agent's policy.
        
        Args:
            tool_name: Name of the tool to execute
            adhesive_type: Specific adhesive type to use
            *args, **kwargs: Arguments to pass to the tool
            
        Returns:
            The tool result after binding with specified adhesive
        """
        # Find the tool - self.tools is a dictionary (tool_name -> tool_object)
        tool_instance = None
        if hasattr(self, 'tools') and self.tools:
            # First try direct lookup by name (most efficient)
            if tool_name in self.tools:
                tool_instance = self.tools[tool_name]
            else:
                # Fallback: iterate through all tools to check name attributes
                for name, tool_obj in self.tools.items():
                    if hasattr(tool_obj, 'name') and tool_obj.name == tool_name:
                        tool_instance = tool_obj
                        break
                    elif hasattr(tool_obj, '__name__') and tool_obj.__name__ == tool_name:
                        tool_instance = tool_obj
                        break
        
        if tool_instance is None:
            available_tools = list(self.tools.keys()) if hasattr(self, 'tools') and self.tools else []
            raise ValueError(f"Tool '{tool_name}' not found in agent's toolset. Available tools: {available_tools}")
        
        # Set override for this specific execution
        old_override = self._next_adhesive_override
        self._next_adhesive_override = adhesive_type
        
        try:
            # Execute the tool through normal mechanism to leverage existing wrapper
            if hasattr(tool_instance, '__call__'):
                result = tool_instance(*args, **kwargs)
            else:
                # For Tool objects, need to call their execution method
                result = tool_instance(*args, **kwargs)
            
            # The wrapper should have handled the binding automatically
            return result
            
        finally:
            # Restore previous override state
            self._next_adhesive_override = old_override
    
    def get_bound_result(self, result_id: str, adhesive_type: Optional[AdhesiveType] = None) -> Any:
        """
        Retrieve a previously bound result from adhesive storage.
        
        Args:
            result_id: The ID of the result to retrieve
            adhesive_type: Optional specific adhesive to search (searches all if None)
            
        Returns:
            The bound result if found, None otherwise
        """
        if not hasattr(self, 'memories'):
            logger.warning(f"Agent '{self.name}' has no adhesive memories configured")
            return None
        
        adhesive_types_to_search = []
        if adhesive_type:
            # Search specific adhesive type
            adhesive_key = adhesive_type.value.lower()
            if adhesive_key in self.memories:
                adhesive_types_to_search = [adhesive_key]
        else:
            # Search all available adhesives
            adhesive_types_to_search = list(self.memories.keys())
        
        for adhesive_key in adhesive_types_to_search:
            try:
                memory_adapter = self.memories[adhesive_key]
                if hasattr(memory_adapter, 'get') and callable(memory_adapter.get):
                    result = memory_adapter.get(result_id)
                    if result is not None:
                        logger.debug(f"Retrieved result '{result_id}' from {adhesive_key} adhesive")
                        return result
                else:
                    logger.warning(f"Memory adapter for '{adhesive_key}' does not support get() operation")
            except Exception as e:
                logger.warning(f"Error retrieving result '{result_id}' from {adhesive_key}: {e}")
        
        logger.debug(f"Result '{result_id}' not found in any adhesive storage")
        return None
    
    def clear_adhesive_storage(self, adhesive_type: Optional[AdhesiveType] = None):
        """
        Clear adhesive storage for debugging or cleanup.
        
        Args:
            adhesive_type: Specific adhesive to clear, or None to clear all
        """
        if not hasattr(self, 'memories'):
            logger.warning(f"Agent '{self.name}' has no adhesive memories configured")
            return
        
        adhesive_types_to_clear = []
        if adhesive_type:
            adhesive_key = adhesive_type.value.lower()
            if adhesive_key in self.memories:
                adhesive_types_to_clear = [adhesive_key]
        else:
            adhesive_types_to_clear = list(self.memories.keys())
        
        for adhesive_key in adhesive_types_to_clear:
            try:
                memory_adapter = self.memories[adhesive_key]
                if hasattr(memory_adapter, 'clear') and callable(memory_adapter.clear):
                    memory_adapter.clear()
                    logger.info(f"Cleared {adhesive_key} adhesive storage for agent '{self.name}'")
                else:
                    logger.warning(f"Memory adapter for '{adhesive_key}' does not support clear() operation")
            except Exception as e:
                logger.error(f"Error clearing {adhesive_key} adhesive storage: {e}")
    
    def get_adhesive_rationale_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the adhesive decision rationale log for debugging and analysis.
        
        Args:
            limit: Maximum number of recent entries to return (None for all)
            
        Returns:
            List of adhesive decision entries
        """
        if limit is None:
            return self._adhesive_rationale_log.copy()
        else:
            return self._adhesive_rationale_log[-limit:] if limit > 0 else []

    # Agent negotiation methods for collaborative adhesive selection
    def suggest_adhesive_for_task(self, task_description: str, preferred_adhesive: AdhesiveType, rationale: str) -> Dict[str, Any]:
        """
        Suggest an adhesive type for a collaborative task.
        
        Args:
            task_description: Description of the task requiring adhesive choice
            preferred_adhesive: The agent's preferred adhesive type
            rationale: Reasoning for the preference
            
        Returns:
            Dict containing the suggestion details
        """
        suggestion = {
            "agent_name": self.name,
            "task_description": task_description,
            "preferred_adhesive": preferred_adhesive.value.upper(),
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
            "agent_capabilities": {
                "available_adhesives": list(self.memories.keys()) if hasattr(self, 'memories') else [],
                "is_team_lead": self._is_team_context(),
                "tools_count": len(self.tools)
            }
        }
        
        logger.info(f"Agent '{self.name}' suggests {preferred_adhesive.value} for task: {task_description}")
        logger.debug(f"Suggestion rationale: {rationale}")
        
        return suggestion
    
    def evaluate_adhesive_suggestions(self, suggestions: List[Dict[str, Any]], task_context: Optional[Dict[str, Any]] = None) -> AdhesiveType:
        """
        Evaluate multiple adhesive suggestions and choose the best one.
        
        Args:
            suggestions: List of adhesive suggestions from various agents
            task_context: Additional context about the task
            
        Returns:
            The chosen adhesive type based on evaluation
        """
        if not suggestions:
            logger.warning("No adhesive suggestions provided, using default policy")
            return self.choose_adhesive("collaborative_task", context=task_context)
        
        # Count preferences and weight by agent capabilities
        adhesive_scores = {
            AdhesiveType.GLUE: 0,
            AdhesiveType.VELCRO: 0,
            AdhesiveType.TAPE: 0
        }
        
        total_weight = 0
        for suggestion in suggestions:
            # Parse adhesive type
            suggested_type = None
            for adhesive_type in AdhesiveType:
                if adhesive_type.value == suggestion["preferred_adhesive"]:
                    suggested_type = adhesive_type
                    break
            
            if suggested_type is None:
                logger.warning(f"Invalid adhesive type in suggestion: {suggestion['preferred_adhesive']}")
                continue
            
            # Calculate weight based on agent capabilities
            weight = 1  # Base weight
            capabilities = suggestion.get("agent_capabilities", {})
            
            # Team leads have higher weight
            if capabilities.get("is_team_lead", False):
                weight += 2
            
            # Agents with more tools have slightly higher weight
            tools_count = capabilities.get("tools_count", 0)
            weight += min(tools_count / 10, 1)  # Max +1 bonus for tool count
            
            # Agents with the suggested adhesive available have higher weight
            available_adhesives = capabilities.get("available_adhesives", [])
            if suggested_type.value.lower() in available_adhesives:
                weight += 1
            
            adhesive_scores[suggested_type] += weight
            total_weight += weight
        
        # Choose the adhesive with highest score
        chosen_adhesive = max(adhesive_scores.items(), key=lambda x: x[1])[0]
        
        # Log the decision
        rationale = f"Collaborative choice: {chosen_adhesive.value} (score: {adhesive_scores[chosen_adhesive]:.1f}/{total_weight:.1f})"
        self._log_adhesive_decision("collaborative_task", chosen_adhesive, rationale, task_context or {})
        
        logger.info(f"Agent '{self.name}' chose {chosen_adhesive.value} from {len(suggestions)} suggestions")
        return chosen_adhesive
    
    def negotiate_adhesive_with_peers(self, peer_agents: List['GlueSmolAgent'], task_description: str, my_preference: AdhesiveType, my_rationale: str) -> AdhesiveType:
        """
        Negotiate adhesive choice with peer agents for a collaborative task.
        
        Args:
            peer_agents: List of other agents to negotiate with
            task_description: Description of the collaborative task
            my_preference: This agent's preferred adhesive
            my_rationale: This agent's reasoning for the preference
            
        Returns:
            The negotiated adhesive type
        """
        # Collect suggestions from all agents including self
        suggestions = []
        
        # Add own suggestion
        own_suggestion = self.suggest_adhesive_for_task(task_description, my_preference, my_rationale)
        suggestions.append(own_suggestion)
        
        # Collect suggestions from peers
        for peer in peer_agents:
            if hasattr(peer, 'suggest_adhesive_for_task'):
                try:
                    # Have peer suggest based on their default policy
                    peer_preference = peer.choose_adhesive("negotiation_task")
                    peer_rationale = f"Based on {peer.name}'s default policy and capabilities"
                    peer_suggestion = peer.suggest_adhesive_for_task(task_description, peer_preference, peer_rationale)
                    suggestions.append(peer_suggestion)
                except Exception as e:
                    logger.warning(f"Could not get suggestion from peer '{peer.name}': {e}")
            else:
                logger.warning(f"Peer agent '{peer.name}' does not support adhesive negotiation")
        
        # Evaluate all suggestions
        chosen_adhesive = self.evaluate_adhesive_suggestions(suggestions, {"task": task_description, "negotiation": True})
        
        logger.info(f"Negotiation complete: chose {chosen_adhesive.value} for task '{task_description}'")
        return chosen_adhesive

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
        logger.debug(f"[GlueSmolAgent.run] Received task: {task!r}")
        # --- PATCH: Robust fix for empty user input bug ---
        # Only unwrap dict['content'] if it is a string or a list of dicts with non-empty 'text'.
        if isinstance(task, dict) and 'content' in task:
            fixed_task = task.get('content')
            # If content is a string, use it directly
            if isinstance(fixed_task, str):
                logger.debug(f"[GlueSmolAgent.run] PATCH: Extracted string content from dict: {fixed_task!r}")
                logger.debug(f"[GlueSmolAgent.run] About to call super().run with: {fixed_task!r}")
                result = super().run(task=str(fixed_task), stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
            # If content is a list of dicts, try to extract the first non-empty 'text' value
            elif isinstance(fixed_task, list):
                text_value = None
                for part in fixed_task:
                    if isinstance(part, dict):
                        val = part.get('text') or part.get('content')
                        if isinstance(val, str) and val.strip():
                            text_value = val.strip()
                            break
                if text_value:
                    logger.debug(f"[GlueSmolAgent.run] PATCH: Extracted non-empty text from list: {text_value!r}")
                    logger.debug(f"[GlueSmolAgent.run] About to call super().run with: {text_value!r}")
                    result = super().run(task=str(text_value), stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
                else:
                    logger.debug("[GlueSmolAgent.run] PATCH: Content list had no non-empty text, falling back to original logic.")
                    result = super().run(task="", stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
            else:
                logger.debug(f"[GlueSmolAgent.run] About to call super().run with: {fixed_task!r}")
                result = super().run(task=str(fixed_task) if fixed_task is not None else "", stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
        elif isinstance(task, str) and not task.strip():
            logger.info(f"[GlueSmolAgent.run] Skipping empty or blank user input: {task!r}")
            return "[info] No input provided. Please enter a non-empty message."
        elif isinstance(task, str):
            logger.debug(f"[GlueSmolAgent.run] PATCH: Passing string task directly: {task!r}")
            logger.debug(f"[GlueSmolAgent.run] About to call super().run with: {task!r}")
            result = super().run(task=task, stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
        else:
            logger.debug(f"[GlueSmolAgent.run] About to call super().run with: {task!r}")
            result = super().run(task=str(task) if task is not None else "", stream=stream, reset=reset, images=images, additional_args=additional_args, max_steps=max_steps, **kwargs)
        # --- END PATCH ---

        # --- PATCH: Unwrap output if it is a list of dicts with only empty 'text' or 'content' ---
        if isinstance(result, list) and all(isinstance(part, dict) and (not (part.get('text') or part.get('content'))) for part in result):
            logger.debug("[GlueSmolAgent.run] Output was a list of dicts with only empty text/content; returning empty string.")
            return ""
        return result

    def write_memory_to_messages(self, summary_mode: bool | None = False) -> list:
        """
        Ensure the first user message is always the raw user input, not a TaskStep with 'New task:'.
        """
        messages = []
        # Add system prompt as first message if present
        system_prompt_step = getattr(self.memory, 'system_prompt', None)
        if system_prompt_step:
            messages.extend(system_prompt_step.to_messages(summary_mode=summary_mode))
        # Add the real user input as the first user message
        if hasattr(self, 'user_task') and self.user_task:
            messages.append({"role": "user", "content": self.user_task})
        # Add the rest of the memory steps, skipping the first TaskStep (which would duplicate the user input)
        steps = self.memory.steps
        skip_first_taskstep = False
        for step in steps:
            # Only skip the first TaskStep
            if not skip_first_taskstep and hasattr(step, 'task'):
                skip_first_taskstep = True
                continue
            messages.extend(step.to_messages(summary_mode=summary_mode))
        return messages

def make_glue_smol_agent(
    *,
    model: InferenceClientModel,
    tools: list,
    glue_config: Optional[Dict[str, Any]],
    name: str,
    description: str,
    managed_agents_dict: Optional[Dict[str, Any]] = None,
    team_name: str = "default_team",
    **kwargs
):
    """Factory function to create a GlueSmolAgent with all required arguments."""
    return GlueSmolAgent(
        model=model,
        tools=tools,
        glue_config=glue_config,
        name=name,
        description=description,
        managed_agents_dict=managed_agents_dict,
        team_name_for_prompt=team_name,
        **kwargs
    )