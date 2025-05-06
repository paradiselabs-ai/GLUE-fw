from typing import Dict, Any, Optional
from smolagents import InferenceClientModel
from .glue_smolagent import GlueSmolAgent
from .glue_smoltool import GlueSmolTool

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
        """
        # Initialize lead agent
        lead_name = self.team.config.lead
        if lead_name not in self.model_clients:
            raise ValueError(f"No model client for lead {lead_name}")
        lead_model = self.model_clients[lead_name]
        self.lead_agent = GlueSmolAgent(
            model=lead_model,
            tools=[GlueSmolTool(t) for t in self.team.tools],
            glue_config=self.glue_config,
        )

        # Initialize member agents and register to lead
        for member_name, model_obj in getattr(self.team, 'models', {}).items():
            if member_name == lead_name:
                continue
            if member_name not in self.model_clients:
                raise ValueError(f"No model client for member {member_name}")
            member_client = self.model_clients[member_name]
            agent = GlueSmolAgent(
                model=member_client,
                tools=[GlueSmolTool(t) for t in self.team.tools],
                glue_config=self.glue_config,
            )
            # Register member agent as a tool on the lead
            # Wrap agent.run as a GlueSmolTool
            tool = GlueSmolTool(agent)
            if hasattr(self.lead_agent, 'tools') and isinstance(self.lead_agent.tools, list):
                self.lead_agent.tools.append(tool)
            self.member_agents[member_name] = agent

    def run(self, initial_input: str) -> Any:
        """
        Execute the team orchestration by running the lead agent.
        """
        if not self.lead_agent:
            raise RuntimeError("Team not set up; call setup() first")
        return self.lead_agent.run(initial_input) 