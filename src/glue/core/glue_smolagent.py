from smolagents import CodeAgent, InferenceClientModel, Tool
from typing import Optional, Dict, Any
from glue.core.glue_memory_adapters import GLUEPersistentAdapter, VELCROSessionAdapter, TAPEEphemeralAdapter
from glue.core.types import AdhesiveType
import logging
import types

class FunctionTool(Tool):
    """
    Smolagents-compliant wrapper for function-based tools.
    Requires explicit 'inputs' and 'output_type' to match Smolagents schema requirements.
    """
    def __init__(self, func, *, name=None, description=None, inputs=None, output_type=None):
        if inputs is None or output_type is None:
            raise ValueError("FunctionTool requires explicit 'inputs' and 'output_type' arguments.")
        self.func = func
        # Skip forward signature validation for function-based tools
        self.skip_forward_signature_validation = True
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Tool: {self.name}"
        self.inputs = inputs
        self.output_type = output_type
        super().__init__(self.name, self.description, self.inputs, self.output_type)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Add or update the default system prompt template for GlueSmolAgent
DEFAULT_SYSTEM_PROMPT_TEMPLATE = '''
# **ROLE AND OBJECTIVE**

You are {{ team_name }}_LeadAgent, the strategic and efficient lead for the "{{ team_name }}" team.
Your primary objective is to successfully complete the user's task by effectively orchestrating your team of managed agents and utilizing available tools. You are responsible for planning, delegating, and synthesizing information to provide a final, comprehensive answer.

# **SUB-AGENT EXECUTION INSTRUCTIONS**
- If you are a sub-agent (i.e., you have no managed agents), do not delegate further.
- After reasoning, output your result by calling the `final_answer` tool with a single string argument, using the following format:
  Thoughts: <brief reasoning>
  Code:
  ```python
  final_answer("<your answer>")
  ```
  
# **TEAM RESOURCES & STRUCTURE**

**1. Tools Available:**
{{tool_descriptions}}

**2. Authorized Imports (for code execution if applicable):**
{{authorized_imports}}

**3. Managed Agents & Team Hierarchy:**
*This defines your direct and indirect reports. You can delegate tasks to any agent listed here.*

{% macro render_hierarchy(agent, level=0) -%}
  {%- set indent = '  ' * level %}
  {{ indent }}- {{ agent.name }}: {{ agent.description }}
  {%- if agent.managed_agents %}
    {%- for sub_name, sub_agent in agent.managed_agents.items() %}
      {{ render_hierarchy(sub_agent, level+1) }}
    {%- endfor %}
  {%- endif %}
{%- endmacro %}

{% for name, agent in managed_agents.items() %}
  {{ render_hierarchy(agent, 0) }}
{% endfor %}

# **OPERATING INSTRUCTIONS**

**1. Planning & Strategy:**
   - Before acting or delegating, first analyze the user's task.
   - Formulate a high-level plan. Identify key steps and which agent(s) or tool(s) are best suited for each step.
   - *Think: What needs to be done? Who is the best agent/tool for each part? In what order?*

**2. Delegation:**
   - You **must** delegate tasks to your managed agents as appropriate.
   - If a managed agent is a subteam lead, you can delegate a broader task to them, and they will manage further delegation within their subteam.
   - When delegating:
     - Clearly state the name of the agent you are delegating to.
     - Provide **specific, clear, and complete instructions** for the sub-task. Include all necessary context they might need from the original user query or previous steps.
     - Define the expected output or result from the delegate.
     - After your reasoning, format the delegation command as a Python code block using the following pattern:
       Thoughts: <brief reasoning about the delegation>
       Code:
       ```python
       assistant_<agent_name>("<task description>")
       ```

**3. Tool Usage:**
   - If a task or sub-task can be accomplished directly by you using one of the "Tools Available," you may use it.
   - Prefer delegation if an agent has specialized skills or tools better suited for the task.
   - For sub-agents (agents without managed agents), do not delegate further. Instead, produce your result by calling the `final_answer` tool with a single string argument.
     After your reasoning, format your output as follows:
     Thoughts: <brief reasoning>
     Code:
     ```python
     final_answer("<your answer>")
     ```

**4. Synthesizing Information & Final Answer:**
   - Monitor the progress of delegated tasks.
   - Once all necessary information has been gathered and sub-tasks are completed by your managed agents, synthesize the results.
   - Provide a **single, comprehensive, and final answer** to the user that directly addresses their original request. Do not provide incremental updates unless the task requires it or you are asked for clarification.
   - If the task cannot be completed, explain why.

# **USER TASK**

{{user_task}}

---
Now, {{team_name}}_LeadAgent, please begin by outlining your plan to address the user task.
'''

# Add a lean template for sub-agents
SUB_AGENT_SYSTEM_PROMPT_TEMPLATE = '''
# **ROLE AND OBJECTIVE**

You are {{team_name}}_Agent, a sub-agent responsible for completing the assigned task.

# **SUB-AGENT EXECUTION INSTRUCTIONS**
- Do not delegate further.
- After reasoning, output your result by calling the `final_answer` tool with a single string argument, using this format:
  Thoughts: <brief reasoning>
  Code:
  ```python
  final_answer("<your answer>")
  ```

# **USER TASK**
{{user_task}}
'''

class GlueSmolAgent(CodeAgent):
    """
    A wrapper around smolagents CodeAgent that integrates GLUE-specific features
    such as adhesives and custom memory persistence hooks.
    """
    # Use the intuitive, hierarchy-aware system prompt by default
    prompt_templates = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        # ... other templates can be added here ...
    }

    def __init__(
        self,
        model: InferenceClientModel,
        tools: list,
        planning_interval: int = 1,
        system_prompt: str = "",
        glue_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        # Default empty managed_agents for sub-agents
        self.managed_agents = {}
        self.glue_config = glue_config or {}
        logger = logging.getLogger("glue.smolagent")

        # Prepare tools for smolagents
        smolagents_tools = []
        for t in tools:
            try:
                if isinstance(t, Tool):
                    # Already a Smolagents Tool
                    if t.name in ("delegate_task", "report_task_completion"):
                        orig_call = t.__call__
                        def logged_call(*args, **kwargs):
                            logger.info(f"[GLUE] Tool '{t.name}' called with args={args}, kwargs={kwargs}")
                            try:
                                return orig_call(*args, **kwargs)
                            except Exception as e:
                                logger.error(f"Error in tool '{t.name}': {e}")
                                raise
                        t.__call__ = logged_call
                    smolagents_tools.append(t)
                elif hasattr(t, 'name') and callable(getattr(t, '__call__', None)) and hasattr(t, 'inputs') and hasattr(t, 'output_type'):
                    # GLUE tool object: wrap its __call__ method
                    smolagents_tools.append(FunctionTool(
                        t.__call__,
                        name=t.name,
                        description=getattr(t, 'description', None),
                        inputs=t.inputs,
                        output_type=t.output_type,
                    ))
                elif hasattr(t, 'forward') and hasattr(t, '_glue_tool_schema'):
                    # Function-based GLUE tool with a forward method and explicit schema
                    schema = t._glue_tool_schema
                    smolagents_tools.append(FunctionTool(
                        t.forward,
                        name=getattr(t, '__name__', None) or getattr(t, 'name', None),
                        description=None,
                        inputs=schema.get('inputs'),
                        output_type=schema.get('output_type'),
                    ))
                elif callable(t) and hasattr(t, '_glue_tool_schema'):
                    # Plain function with explicit schema
                    schema = t._glue_tool_schema
                    smolagents_tools.append(FunctionTool(
                        t,
                        name=getattr(t, '__name__', None),
                        description=None,
                        inputs=schema.get('inputs'),
                        output_type=schema.get('output_type'),
                    ))
                else:
                    logger.error(f"Tool '{repr(t)}' is not a valid SmolAgents Tool or function.")
                    raise ValueError(f"Tool '{repr(t)}' is not a valid SmolAgents Tool or function.")
            except Exception as e:
                logger.error(f"Error wrapping tool '{repr(t)}': {e}")
                raise

        # Call the base constructor with the tools
        super().__init__(
            tools=smolagents_tools,
            model=model,
            planning_interval=planning_interval,
            name=name,
            description=description,
            **kwargs,
        )
        # Restore user-provided name and description as super().__init__ may override them
        self.name = name
        self.description = description

        # Handle system prompt override if provided
        if system_prompt:
            self.prompt_templates["system_prompt"] = system_prompt
            logger.info(f"[GLUE] Agent '{getattr(self, 'name', 'unnamed')}' using custom system prompt (override).")
        else:
            # Override CodeAgent default with our custom template
            self.prompt_templates["system_prompt"] = DEFAULT_SYSTEM_PROMPT_TEMPLATE
            logger.info(f"[GLUE] Agent '{getattr(self, 'name', 'unnamed')}' using DEFAULT_SYSTEM_PROMPT_TEMPLATE.")

        # Initialize GLUE-specific features (adhesives, memory hooks)
        self._init_glue_features()

    def _init_glue_features(self):
        """
        Setup GLUE-specific features: adhesives (GLUE, VELCRO, TAPE) and
        memory persistence hooks.
        """
        # Attach memory adapters based on adhesives
        adhesives = set()
        # Try to get adhesives from glue_config or self.model (if it's a GLUE Model)
        if hasattr(self, 'glue_config') and self.glue_config and 'adhesives' in self.glue_config:
            adhesives = set(self.glue_config['adhesives'])
        elif hasattr(self.model, 'adhesives'):
            adhesives = set(self.model.adhesives)
        # Default to GLUE if none specified
        if not adhesives:
            adhesives = {AdhesiveType.GLUE}

        self.memories = {}
        # Attach adapters for each adhesive
        for adhesive in adhesives:
            if adhesive == AdhesiveType.GLUE:
                # Team-wide persistent memory
                self.memories['glue'] = GLUEPersistentAdapter(team_id=getattr(self, 'team_id', 'default'), memory_dir='memory')
            elif adhesive == AdhesiveType.VELCRO:
                # Session-based, agent-specific memory
                self.memories['velcro'] = VELCROSessionAdapter()
            elif adhesive == AdhesiveType.TAPE:
                # Ephemeral, one-time-use memory
                self.memories['tape'] = TAPEEphemeralAdapter()

        # Set self.memory to the highest-priority adhesive (GLUE > VELCRO > TAPE)
        if 'glue' in self.memories:
            self.memory = self.memories['glue']
        elif 'velcro' in self.memories:
            self.memory = self.memories['velcro']
        elif 'tape' in self.memories:
            self.memory = self.memories['tape']
        else:
            self.memory = None

        # Log memory setup for debugging
        logger = logging.getLogger("glue.smolagent")
        logger.info(f"[GLUE] Agent '{getattr(self, 'name', 'unnamed')}' memory adapters: {list(self.memories.keys())}, default: {type(self.memory).__name__ if self.memory else None}")

    def debug_print_agent_memory(self):
        """
        Print/log the memory adapters and default memory for this agent.
        """
        print(f"[DEBUG] Agent '{getattr(self, 'name', 'unnamed')}' memory adapters:")
        for k, v in getattr(self, 'memories', {}).items():
            print(f"  - {k}: {type(v).__name__}")
        print(f"  Default memory: {type(getattr(self, 'memory', None)).__name__ if getattr(self, 'memory', None) else None}")

    def debug_print_tool_schemas(self):
        """
        Print the schema (name, description, inputs, output_type) for all tools in this agent.
        """
        print(f"[DEBUG] Tool schemas for agent '{getattr(self, 'name', 'unnamed')}':")
        for tool in getattr(self, 'tools', {}).values():
            name = getattr(tool, 'name', None)
            description = getattr(tool, 'description', None)
            inputs = getattr(tool, 'inputs', None)
            output_type = getattr(tool, 'output_type', None)
            print(f"  - Name: {name}")
            print(f"    Description: {description}")
            print(f"    Inputs: {inputs}")
            print(f"    Output type: {output_type}")
        print("")

    def run(self, query: str, **kwargs):
        # Store the current task text for prompt rendering
        self.user_task = query
        # Ensure managed agent callables are injected before every run
        if hasattr(self, 'team') and hasattr(self.team, 'inject_managed_agents'):
            self.team.inject_managed_agents()
        return super().run(query, **kwargs)

    def generate(self, *args, **kwargs):
        # Ensure managed agent callables are injected before every generate
        if hasattr(self, 'team') and hasattr(self.team, 'inject_managed_agents'):
            self.team.inject_managed_agents()
        return super().generate(*args, **kwargs)

    @property
    def interpreter_globals(self):
        """
        Returns the interpreter's globals dict, ensuring the interpreter is initialized.
        If the interpreter does not exist, runs a dummy query to trigger setup.
        Returns None if interpreter is still unavailable.
        """
        if not hasattr(self, "interpreter"):
            try:
                self.run("__interpreter_init__")
            except Exception:
                pass
        return getattr(self, "interpreter", None).globals if hasattr(self, "interpreter") else None

    def force_interpreter(self):
        """
        Force creation of a dummy interpreter with a globals dict and inject all tools.
        This is for testing and debugging only. Does not execute any code or leak references.
        """
        if not hasattr(self, "interpreter"):
            # Create a dummy interpreter with a globals dict
            self.interpreter = types.SimpleNamespace()
            self.interpreter.globals = {}
        # Inject all tools as callables with correct __name__
        if hasattr(self, "tools"):
            for tool_key, tool_callable in self.tools.items():
                # Prefer the .name attribute if present (as in GlueSmolTool)
                tool_name = getattr(tool_callable, "name", tool_key)
                def make_func(tc, name):
                    def f(*args, _tc=tc, **kwargs):
                        return _tc(*args, **kwargs)
                    f.__name__ = name
                    return f
                wrapped = make_func(tool_callable, tool_name)
                self.interpreter.globals[tool_name] = wrapped

    def initialize_system_prompt(self):
        """Render the system prompt with complete context, bypassing default super implementation."""
        from jinja2 import Environment, BaseLoader, Undefined
        env = Environment(loader=BaseLoader(), undefined=Undefined)
        # Choose sub-agent vs lead-agent prompt
        template_str = SUB_AGENT_SYSTEM_PROMPT_TEMPLATE if not getattr(self, 'managed_agents', None) else self.prompt_templates.get("system_prompt", "")
        template = env.from_string(template_str)
        # Build context for template rendering
        context = {
            "team_name": getattr(self, "name", ""),
            "tool_descriptions": "\n".join(f"{t.name}: {t.description}" for t in getattr(self, "tools", {}).values()),
            "authorized_imports": getattr(self, "authorized_imports", ""),
            "managed_agents_description": "\n".join(f"{name}: {getattr(agent, '_subteam_description', '') or agent.description}" for name, agent in getattr(self, "managed_agents", {}).items()),
            "managed_agents": getattr(self, "managed_agents", {}),
            "subteam_descriptions": getattr(self, "_subteam_descriptions", {}),
            "user_task": getattr(self, "user_task", ""),
            "tools": {t.name: t for t in getattr(self, "tools", {}).values()},
        }
        try:
            return template.render(**context)
        except Exception as e:
            logging.getLogger("glue.smolagent").error(f"[GLUE] Error rendering system prompt: {e}. Using raw template.")
            return self.prompt_templates.get("system_prompt", "")

def make_glue_smol_agent(*, model, tools, glue_config, name, description, **kwargs):
    """
    Utility to create a GlueSmolAgent with required metadata.
    Ensures name and description are set, as required by SmolAgents best practices.
    Also validates all tools for required schema.
    """
    logger = logging.getLogger("glue.smolagent")
    if not name or not isinstance(name, str):
        logger.error("All agents must have a non-empty string 'name' attribute.")
        raise ValueError("All agents must have a non-empty string 'name' attribute.")
    if not description or not isinstance(description, str):
        logger.error(f"Agent '{name}' must have a non-empty string 'description' attribute.")
        raise ValueError(f"Agent '{name}' must have a non-empty string 'description' attribute.")
    # Tool schema validation
    for t in tools:
        try:
            if hasattr(t, 'name') and hasattr(t, '__call__') and hasattr(t, 'inputs') and hasattr(t, 'output_type'):
                # Class-based tool: check required attributes
                if not t.name or not t.description or not t.inputs or not t.output_type:
                    logger.error(f"Tool '{getattr(t, 'name', repr(t))}' is missing required schema fields.")
                    raise ValueError(f"Tool '{getattr(t, 'name', repr(t))}' must have name, description, inputs, and output_type.")
            elif hasattr(t, '__name__') and callable(t):
                # Function-based tool: must have _glue_tool_schema
                if not hasattr(t, '_glue_tool_schema'):
                    logger.error(f"Function tool '{t.__name__}' must provide a _glue_tool_schema attribute with 'inputs' and 'output_type'.")
                    raise ValueError(f"Function tool '{t.__name__}' must provide a _glue_tool_schema attribute with 'inputs' and 'output_type'.")
                schema = t._glue_tool_schema
                if not schema.get('inputs') or not schema.get('output_type'):
                    logger.error(f"Function tool '{t.__name__}' _glue_tool_schema must include 'inputs' and 'output_type'.")
                    raise ValueError(f"Function tool '{t.__name__}' _glue_tool_schema must include 'inputs' and 'output_type'.")
            else:
                logger.error(f"Tool '{repr(t)}' is not a valid SmolAgents Tool or function.")
                raise ValueError(f"Tool '{repr(t)}' is not a valid SmolAgents Tool or function.")
        except Exception as e:
            logger.error(f"Error validating tool '{repr(t)}': {e}")
            raise
    # Pass system_prompt from model.smol_config if present
    system_prompt = None
    if hasattr(model, 'smol_config') and isinstance(model.smol_config, dict):
        system_prompt = model.smol_config.get('system_prompt', None)
    try:
        agent = GlueSmolAgent(
            model=model,
            tools=tools,
            glue_config=glue_config,
            name=name,
            description=description,
            system_prompt=system_prompt or "",
            **kwargs
        )
        logger.info(f"Created GlueSmolAgent '{name}' with {len(tools)} tools.")
        return agent
    except Exception as e:
        logger.error(f"Error creating GlueSmolAgent '{name}': {e}")
        raise
