from typing import Dict, Any, Optional
from smolagents import InferenceClientModel
from .glue_smolagent import GlueSmolAgent, make_glue_smol_agent
from .glue_smoltool import GlueSmolTool
from glue.core.model import Model as GlueModel
from glue.core.providers.openrouter import OpenrouterProvider
import inspect
import asyncio
from glue.tools.delegate_task_tool import DelegateTaskTool

class SimpleMessage:
    def __init__(self, content):
        self.content = content

# Add a wrapper so that function-based models have a .generate() method
class FunctionModelWrapper:
    def __init__(self, func):
        self.func = func
    def generate(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class GlueSmolTeam:
    """
    Orchestrates a GLUE Team using Smolagents-managed GlueSmolAgent instances.
    Supports delegation of tasks to member agents and reporting.
    """
    def __init__(
        self,
        team: Any,
        model_clients: Dict[str, InferenceClientModel],
        glue_config: Optional[Dict[str, Any]] = None,
    ):
        self.team = team
        self.model_clients = model_clients
        self.glue_config = glue_config or {}
        self.lead_agent: Optional[GlueSmolAgent] = None
        self.member_agents: Dict[str, GlueSmolAgent] = {}

    def setup(self):
        """
        Create and configure GlueSmolAgent instances for lead and members
        and register member agents as tools to the lead.
        Supports hierarchical teams: if a member is a subteam, use its lead agent as the managed agent.
        """
        # Initialize lead agent: resolve model client to InferenceClientModel
        lead_name = self.team.config.lead
        if lead_name not in self.model_clients:
            raise ValueError(f"No model client for lead {lead_name}")
        raw_client = self.model_clients[lead_name]
        # If it's a GLUE Model, create a new InferenceClientModel with smol_config
        if isinstance(raw_client, GlueModel):
            model_id = raw_client.config.get("model") or raw_client.config.get("provider")
            provider_name = raw_client.config.get("provider")
            api_key = raw_client.config.get("api_key")
            opts = getattr(raw_client, "smol_config", {}) or {}
            if provider_name == "openrouter":
                lead_client = self._make_openrouter_callable(raw_client)
            else:
                lead_client = InferenceClientModel(
                    model_id=model_id,
                    provider=provider_name,
                    api_key=api_key,
                    **{k: v for k, v in opts.items() if v is not None}
                )
            lead_description = raw_client.config.get("description", f"Lead agent for team {lead_name}")
        elif isinstance(raw_client, InferenceClientModel):
            lead_client = raw_client
            lead_description = getattr(raw_client, "description", f"Lead agent for team {lead_name}")
        else:
            raise ValueError(f"Unsupported model client type for lead {lead_name}: {type(raw_client)}")
        self.lead_agent = make_glue_smol_agent(
            model=lead_client,
            tools=[GlueSmolTool(t) for t in self.team.tools],
            glue_config=self.glue_config,
            managed_agents=[],  # Will be filled below
            name=lead_name,
            description=lead_description,
        )
        # Debug: print the memory configuration for the lead agent
        self.lead_agent.debug_print_agent_memory()

        # Initialize member agents and add to managed_agents
        managed_agents = []
        for member_name, model_obj in getattr(self.team, 'models', {}).items():
            if member_name == lead_name:
                continue
            raw_member = self.model_clients.get(member_name)
            if raw_member is None:
                # Check if this member is a subteam
                subteam = getattr(self.team, 'subteams', {}).get(member_name)
                if subteam is not None:
                    # Recursively set up the subteam if not already set up
                    if not hasattr(subteam, 'glue_smolteam'):
                        subteam.glue_smolteam = GlueSmolTeam(subteam, self.model_clients, self.glue_config)
                        subteam.glue_smolteam.setup()
                    # Use the subteam's lead agent as the managed agent
                    agent = subteam.glue_smolteam.lead_agent
                    # Attach subteam description for prompt rendering
                    agent._subteam_description = getattr(subteam, 'config', {}).lead if hasattr(subteam, 'config') else agent.name
                    managed_agents.append(agent)
                    self.member_agents[member_name] = agent
                    continue
                else:
                    raise ValueError(f"No model client or subteam for member {member_name}")
            # Resolve member client similarly
            if isinstance(raw_member, GlueModel):
                model_id = raw_member.config.get("model") or raw_member.config.get("provider")
                provider_name = raw_member.config.get("provider")
                api_key = raw_member.config.get("api_key")
                opts = getattr(raw_member, "smol_config", {}) or {}
                if provider_name == "openrouter":
                    member_client = self._make_openrouter_callable(raw_member)
                else:
                    member_client = InferenceClientModel(
                        model_id=model_id,
                        provider=provider_name,
                        api_key=api_key,
                        **{k: v for k, v in opts.items() if v is not None}
                    )
                member_description = raw_member.config.get("description", f"Team member agent {member_name}")
            elif isinstance(raw_member, InferenceClientModel):
                member_client = raw_member
                member_description = getattr(raw_member, "description", f"Team member agent {member_name}")
            else:
                raise ValueError(f"Unsupported model client type for member {member_name}: {type(raw_member)}")
            agent = make_glue_smol_agent(
                model=member_client,
                tools=[GlueSmolTool(t) for t in self.team.tools],
                glue_config=self.glue_config,
                name=member_name,
                description=member_description,
            )
            # Debug: print the memory configuration for each member agent
            agent.debug_print_agent_memory()
            managed_agents.append(agent)
            self.member_agents[member_name] = agent
        # Assign managed_agents to the lead agent as a dict for Jinja compatibility
        self.lead_agent.managed_agents = {agent.name: agent for agent in managed_agents}
        # Store subteam descriptions for prompt rendering
        self.lead_agent._subteam_descriptions = {
            agent.name: getattr(agent, '_subteam_description', None)
            for agent in managed_agents if hasattr(agent, '_subteam_description')
        }

        # Debug: print the rendered system prompt for the lead agent
        self.debug_print_lead_prompt()
        # Override system prompt template with fully rendered prompt to bypass Jinja template on run
        try:
            prompt_template = self.lead_agent.prompt_templates.get("system_prompt", "")
            # Build context including all variables expected by DEFAULT_SYSTEM_PROMPT_TEMPLATE
            context = {
                "team_name": getattr(self.team.config, 'lead', getattr(self.lead_agent, 'name', '')),
                "tool_descriptions": "\n".join(f"{t.name}: {t.description}" for t in getattr(self.lead_agent, 'tools', {}).values()),
                "authorized_imports": getattr(self.lead_agent, 'authorized_imports', ""),
                "managed_agents_description": "\n".join(f"{name}: {getattr(agent, '_subteam_description', '') or agent.description}" for name, agent in self.lead_agent.managed_agents.items()),
                "managed_agents": self.lead_agent.managed_agents,
                "subteam_descriptions": getattr(self.lead_agent, '_subteam_descriptions', {}),
            }
            from jinja2 import Template
            rendered_prompt = Template(prompt_template).render(**context)
        except Exception:
            # Fallback to raw prompt template if rendering fails
            rendered_prompt = prompt_template
        self.lead_agent.prompt_templates["system_prompt"] = rendered_prompt

    def _make_openrouter_callable(self, glue_model):
        provider = OpenrouterProvider(glue_model)
        def call(messages, stop_sequences=None, **kwargs):
            coro = provider.generate_response(messages)
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    import concurrent.futures
                    from threading import current_thread, main_thread
                    if current_thread() is main_thread():
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(lambda: asyncio.run(coro))
                            result = future.result()
                    else:
                        result = loop.run_until_complete(coro)
                else:
                    result = loop.run_until_complete(coro)
            except RuntimeError:
                result = asyncio.run(coro)
            return SimpleMessage(result)
        return FunctionModelWrapper(call)

    def run(self, initial_input: str) -> Any:
        """
        Execute the team orchestration by running the lead agent.
        """
        if not self.lead_agent:
            raise RuntimeError("Team not set up; call setup() first")
        return self.lead_agent.run(initial_input)

    def debug_print_lead_prompt(self):
        """
        Print the rendered system prompt for the lead agent, including managed agents and tools.
        Includes subteam leads and their descriptions in the managed_agents context.
        """
        if not self.lead_agent:
            print("[DEBUG] No lead agent to print prompt for.")
            return
        try:
            # Render the system prompt template with current context
            prompt_template = self.lead_agent.prompt_templates["system_prompt"]
            # Prepare context for rendering
            context = {
                "tools": {t.name: t for t in self.lead_agent.tools.values()} if hasattr(self.lead_agent, "tools") else {},
                "managed_agents": {name: agent for name, agent in getattr(self.lead_agent, "managed_agents", {}).items()},
                "subteam_descriptions": getattr(self.lead_agent, "_subteam_descriptions", {}),
                "authorized_imports": getattr(self.lead_agent, "authorized_imports", ""),
            }
            # Use Jinja2 for rendering if available, else fallback to str.format
            try:
                from jinja2 import Template
                rendered = Template(prompt_template).render(**context)
            except ImportError:
                rendered = prompt_template.format(**context)
            print("\n[DEBUG] Rendered system prompt for lead agent:\n" + "-"*60)
            print(rendered)
            print("-"*60 + "\n")
        except Exception as e:
            print(f"[DEBUG] Error rendering system prompt: {e}") 