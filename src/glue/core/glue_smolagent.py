from smolagents import CodeAgent, InferenceClientModel
from typing import Optional, Dict, Any

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
        system_prompt: str = "",
        glue_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize base CodeAgent with no tools to bypass type assertions
        super().__init__(
            tools=[],  # we'll register GLUE tools manually below
            model=model,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.glue_config = glue_config or {}

        # Register GLUE tools manually to avoid smolagents type checks
        for t in tools:
            try:
                self.register_tool(t.name, t.__call__)
            except Exception:
                pass

        # Initialize GLUE-specific features (adhesives, memory hooks)
        self._init_glue_features()

        # Inject Jinja macros to define tool_descriptions and managed_agents_description before rendering
        orig = self.prompt_templates.get("system_prompt", "")
        new_prompt = (
            "{% set tool_descriptions -%}"  
            "{%- for tool in tools.values() %}\n"
            "- {{ tool.name }}: {{ tool.description }}\n"
            "  Takes inputs: {{ tool.inputs }}\n"
            "  Returns an output of type: {{ tool.output_type }}\n"
            "{%- endfor %}{% endset %}\n"
            "{% set managed_agents_description -%}"  
            "{%- if managed_agents and managed_agents.values() | list %}\n"
            "You can also give tasks to team members. Calling a team member works the same as calling a tool: the only argument is 'task', a long string explaining your task.\n"
            "Here is a list of the team members that you can call:\n"
            "{%- for agent in managed_agents.values() %}\n"
            "- {{ agent.name }}: {{ agent.description }}\n"
            "{%- endfor %}{% endif %}{% endset %}\n"
            + orig
            + "\n\n{{tool_descriptions}}\n{{managed_agents_description}}\n{{authorized_imports}}"
        )
        self.prompt_templates["system_prompt"] = new_prompt

    def _init_glue_features(self):
        """
        Setup GLUE-specific features: adhesives (GLUE, VELCRO, TAPE) and
        memory persistence hooks.
        """
        # TODO: implement adhesive persistence hooks
        pass

    def run(self, query: str, **kwargs) -> Any:
        """
        Run the agent with GLUE-specific pre- and post-processing.
        """
        # Pre-processing: apply adhesive logic before running
        # TODO: apply pre-run glue hooks

        result = super().run(query, **kwargs)

        # Post-processing: handle result persistence via adhesives
        # TODO: apply post-run glue hooks
        return result