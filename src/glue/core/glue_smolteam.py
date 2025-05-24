from typing import Dict, Any, Optional, List
from smolagents import InferenceClientModel
from .glue_smolagent import GlueSmolAgent, make_glue_smol_agent
from .providers.openrouter import OpenrouterProvider
import asyncio
import logging
from ..core.types import FlowType
logger = logging.getLogger(__name__)

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
        # List of OpenrouterProvider instances to close on shutdown
        self._openrouter_providers: List[Any] = []
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
        # Ensure model client is an InferenceClientModel
        if not isinstance(raw_client, InferenceClientModel):
            raise ValueError(f"Unsupported model client type for lead {lead_name}: {type(raw_client)}")
        lead_client = raw_client
        lead_description = getattr(raw_client, "description", f"Lead agent for team {lead_name}")
        # Filter tools for the lead agent to include all tools
        lead_tools = list(self.team.tools)

        # Initialize lead agent with all tools
        self.lead_agent = make_glue_smol_agent(
            model=lead_client,
            tools=lead_tools,
            glue_config=self.glue_config,
            managed_agents_dict={},  # Placeholder for later population
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
                    if agent is not None:
                        setattr(agent, '_subteam_description', getattr(subteam.config, 'lead', agent.name) if hasattr(subteam, 'config') else agent.name)
                        managed_agents.append(agent)
                        self.member_agents[member_name] = agent # Ensure agent is not None
                    continue
                else:
                    raise ValueError(f"No model client or subteam for member {member_name}")
            # Ensure member client is an InferenceClientModel
            if not isinstance(raw_member, InferenceClientModel):
                raise ValueError(f"Unsupported model client type for member {member_name}: {type(raw_member)}")
            member_client = raw_member
            member_description = getattr(raw_member, "description", f"Team member agent {member_name}")

            # Filter tools for member agents to exclude 'user_input'
            member_tools = [tool for tool in self.team.tools if getattr(tool, 'name', None) != 'user_input']

            agent = make_glue_smol_agent(
                model=member_client,
                tools=member_tools,
                glue_config=self.glue_config,
                name=member_name,
                description=member_description,
            )
            # Debug: print the memory configuration for each member agent
            agent.debug_print_agent_memory()
            managed_agents.append(agent)
            self.member_agents[member_name] = agent
        # Assign managed_agents to the lead agent as a dict for Jinja compatibility
        if self.lead_agent is not None:
            self.lead_agent.managed_agents = {agent.name: agent for agent in managed_agents if agent is not None and hasattr(agent, 'name')}
            # Store subteam descriptions for prompt rendering
            setattr(self.lead_agent, '_subteam_descriptions', {
                agent.name: getattr(agent, '_subteam_description', None)
                for agent in managed_agents if agent is not None and hasattr(agent, 'name') and hasattr(agent, '_subteam_description')
            })

        # Debug: print the rendered system prompt for the lead agent
        self.debug_print_lead_prompt()
        # Initialize magnetic flow functions for lead agent
        try:
            # Ensure interpreter and globals exist
            self.lead_agent.force_interpreter()
            flow_logger = logging.getLogger("glue.smolteam")
            # Iterate relationships if any; skip if not present
            for other_team, rel in getattr(self.team, 'relationships', {}).items():
                # PUSH or BIDIRECTIONAL: push_to_<team>
                if rel in {FlowType.PUSH.value, FlowType.BIDIRECTIONAL.value}:
                    def make_push(t):
                        def push(content):
                            """Push content to target team"""
                            asyncio.create_task(self.team.send_information(t, content))
                        return push
                    self.lead_agent.interpreter_globals[f"push_to_{other_team}"] = make_push(other_team)
                # PULL or BIDIRECTIONAL: pull_from_<team>
                if rel in {FlowType.PULL.value, FlowType.BIDIRECTIONAL.value}:
                    def make_pull(t):
                        def pull(content):
                            """Pull content from source team"""
                            return asyncio.get_event_loop().run_until_complete(self.team.receive_information(t, content))
                        return pull
                    self.lead_agent.interpreter_globals[f"pull_from_{other_team}"] = make_pull(other_team)
                # REPEL: do not register any flow
            flow_logger.debug(f"[GLUE] Registered flow functions for lead agent: {list(self.lead_agent.interpreter_globals.keys())}")
        except Exception as e:
            logging.getLogger("glue.smolteam").error(f"Error initializing flow functions: {e}")

    def _make_openrouter_callable(self, glue_model):
        provider = OpenrouterProvider(glue_model)
        # Register provider for shutdown
        self._openrouter_providers.append(provider)
        logger = logging.getLogger("glue.smolteam.openrouter_callable")

        def call(messages, stop_sequences=None, **kwargs):
            logger.debug(f"CALLABLE INPUT messages: {messages}")
            raw_result = None
            loop = None # Initialize loop to None
            try:
                # Ensure loop is defined before use in finally block
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                raw_result = loop.run_until_complete(
                    provider.generate_response(messages) # Removed tool_kwargs
                )
            except Exception as e:
                logger.error(f"Error in OpenRouter callable event loop: {e}", exc_info=True)
                raw_result = f"ERROR_IN_OPENROUTER_CALLABLE: {type(e).__name__}: {e}"
            finally:
                try:
                    if loop is not None:
                        loop.close()
                except Exception:
                    pass

            logger.debug(f"CALLABLE RAW_RESULT from provider: '{str(raw_result)[:500]}...'" )
            # If this is an error code from the provider, invoke final_answer to stop looping
            if isinstance(raw_result, str) and raw_result.startswith("ERROR"):
                logger.debug("OpenRouter callable saw provider error; delegating to final_answer tool")
                return {"tool_calls": [{"id": "error_call", "name": "final_answer", "arguments": {"answer": raw_result}}]}
            # Otherwise, wrap the provider output as a normal message
            content_for_message = str(raw_result) if raw_result is not None else ""
            final_message_obj = SimpleMessage(content_for_message)
            logger.debug(f"CALLABLE RETURNING SimpleMessage with content: '{final_message_obj.content[:500]}'" )
            return final_message_obj

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
            logger.debug("No lead agent to print prompt for.")
            return
        try:
            # Use the agent's initialize_system_prompt to get a fully rendered prompt
            rendered = self.lead_agent.initialize_system_prompt()
            logger.debug("Rendered system prompt for lead agent:\n" + "-"*60)
            logger.debug(rendered)
            logger.debug("-"*60)
        except Exception as e:
            logger.debug(f"Error rendering system prompt: {e}")

    async def close(self):
        """Close all registered OpenRouter providers to free HTTP resources."""
        for provider in getattr(self, '_openrouter_providers', []):
            try:
                await provider.close()
                logger.info(f"Closed OpenRouter provider for model {getattr(provider.model, 'model_name', getattr(provider.model, 'model', 'unknown'))}")
            except Exception as e:
                logger.error(f"Error closing OpenRouter provider: {e}", exc_info=True)
        # Clear the list after closing
        self._openrouter_providers.clear()