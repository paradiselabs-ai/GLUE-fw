# glue/core/agent.py

class Agent:
    """Represents a GLUE agent, which can be a member of a GlueTeam."""
    # Minimal placeholder implementation for now.
    # Will be fleshed out in a future TDD cycle.
    def __init__(self, name: str, role: str, model=None, tools=None, adhesives=None, **kwargs):
        self.id = kwargs.get("id", f"agent_{name}") # Basic ID generation
        self.name = name
        self.role = role
        self.model = model
        self.tools = tools if tools is not None else []
        self.adhesives = adhesives if adhesives is not None else []
        # Placeholder for Agno config generation
        self.config = {} 

    def get_agno_agent_config(self) -> dict:
        # This will eventually build the config needed for an Agno Agent
        # For now, it matches the mock's expectation
        return self.config
