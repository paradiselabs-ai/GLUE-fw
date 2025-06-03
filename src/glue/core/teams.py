import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, AsyncGenerator
from uuid import uuid4

from agno.agent import Agent as AgnoAgent
from agno.team import Team as AgnoTeam
from agno.models.base import Model as AgnoBaseModel
from agno.models.response import ModelResponse
from agno.models.message import Message
from agno.workflow import Workflow as AgnoWorkflow
from agno.run.team import TeamRunResponse
from agno.tools.function import Function as AgnoTool
from agno.tools.function import Function as AgnoFunction

from ..utils.json_utils import extract_json
from .model import Model
from glue.tools import SimpleBaseTool as GlueTool
from .adhesive import AdhesiveSystem as Adhesive, AdhesiveType

# Configure logging
logger = logging.getLogger(__name__)

# Define a minimal test Agno model
class _MinimalTestAgnoModel(AgnoBaseModel):
    def __init__(self, name: str, **kwargs):
        super().__init__(
            name=name,
            id="minimal-test-model",
            provider="test",
            **kwargs
        )

    def invoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        last_message_content = messages[-1].content if messages and hasattr(messages[-1], 'content') else "No input"
        return ModelResponse(
            role="assistant",
            content=f"Test model received: {last_message_content}"
        )

    async def ainvoke(self, messages: List[Message], **kwargs) -> ModelResponse:
        last_message_content = messages[-1].content if messages and hasattr(messages[-1], 'content') else "No input"
        return ModelResponse(
            role="assistant",
            content=f"Test model (async) received: {last_message_content}"
        )

    def invoke_stream(self, messages: List[Message], **kwargs) -> Iterator[ModelResponse]:
        last_message_content = messages[-1].content if messages and hasattr(messages[-1], 'content') else "No input"
        yield ModelResponse(
            role="assistant",
            content=f"Test model stream received: {last_message_content}"
        )

    async def ainvoke_stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[ModelResponse, None]:
        last_message_content = messages[-1].content if messages and hasattr(messages[-1], 'content') else "No input"
        yield ModelResponse(
            role="assistant",
            content=f"Test model (async stream) received: {last_message_content}"
        )

    def parse_provider_response(self, response: Any) -> ModelResponse:
        if isinstance(response, dict) and "content" in response:
            return ModelResponse(role="assistant", content=response.get("content", "Parsed response"))
        return ModelResponse(role="assistant", content="Parsed provider response")

    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        if isinstance(response, dict) and "content" in response:
            return ModelResponse(role="assistant", content=response.get("content", "Parsed delta"))
        return ModelResponse(role="assistant", content="Parsed provider response delta")


class GlueTeam:
    """
    Represents a team of GLUE models (agents) that can collaborate to perform tasks.
    A GlueTeam can optionally be backed by an AgnoTeam for execution.
    """

    def __init__(self, name: str, team_id: Optional[str] = None, use_agno_team: bool = True):
        self.name = name
        self.team_id = team_id or str(uuid4())
        self.members: Dict[str, Model] = {}  # Stores member models by their ID
        self.lead_model: Optional[Model] = None
        self.tools: Dict[str, GlueTool] = {} # Tools available to the team
        self.adhesives: Dict[AdhesiveType, Adhesive] = {} # Adhesives for memory and output
        self.workflow: Optional[AgnoWorkflow] = None # Agno workflow for complex orchestration

        self.use_agno_team = use_agno_team
        self.agno_team: Optional[AgnoTeam] = None
        self.agno_agents: Dict[str, AgnoAgent] = {} # Stores AgnoAgents created from GlueModels

        self.pending_tasks: List[Tuple[str, Any]] = [] # (task_description, task_context)
        self.completed_tasks: List[Any] = [] # Stores results of completed tasks

        # For inter-team communication (Magnetic Flow)
        self.magnetic_subscriptions: Dict[str, List[GlueTeam]] = {} # topic -> list of subscriber teams
        self.magnetic_publications: Dict[str, Any] = {} # topic -> last published data

        logger.info(f"GlueTeam '{self.name}' (ID: {self.team_id}) initialized. use_agno_team: {self.use_agno_team}")


    def add_member(self, model_instance: Model, role: str = "member") -> None:
        """Adds a model to the team."""
        if not isinstance(model_instance, Model):
            raise TypeError("member must be an instance of a GLUE Model.")

        if role == "lead":
            if self.lead_model:
                logger.warning(f"Team '{self.name}' already has a lead model. Replacing.")
            self.lead_model = model_instance
            logger.info(f"Added model '{model_instance.name}' (ID: {model_instance.model_id}) to team '{self.name}' with role '{role}'.")
        elif role == "member":
            if model_instance.model_id in self.members:
                logger.warning(f"Model with ID '{model_instance.model_id}' already exists in team '{self.name}'. Replacing.")
            self.members[model_instance.model_id] = model_instance
            logger.info(f"Added model '{model_instance.name}' (ID: {model_instance.model_id}) to team '{self.name}' with role '{role}'.")
        else:
            raise ValueError(f"Invalid role '{role}'. Must be 'lead' or 'member'.")

        # If using Agno, and the Agno team is already initialized, add the new member as an AgnoAgent
        if self.use_agno_team and self.agno_team:
            self._add_glue_model_to_agno_team(model_instance)


    async def add_member_async(self, model_instance: Model, role: str = "member") -> None:
        """Asynchronously adds a model to the team."""
        # For now, this is a simple wrapper. Could be extended for async-specific logic.
        self.add_member(model_instance, role)
        logger.info(f"Added model {model_instance.name} to team {self.name} with role {role} (async)")


    def remove_member(self, model_id: str) -> None:
        """Removes a model from the team by its ID."""
        if model_id in self.members:
            removed_model = self.members.pop(model_id)
            logger.info(f"Removed model '{removed_model.name}' (ID: {model_id}) from team '{self.name}'.")
            if self.use_agno_team and self.agno_team and model_id in self.agno_agents:
                # AgnoTeam itself doesn't have a direct 'remove_agent' by ID in the same way.
                # We'd need to re-initialize or manage the agno_agents list.
                # For now, just remove from our tracking.
                del self.agno_agents[model_id]
                logger.info(f"Removed corresponding AgnoAgent for model ID {model_id}.")
        elif self.lead_model and self.lead_model.model_id == model_id:
            logger.info(f"Removed lead model '{self.lead_model.name}' (ID: {model_id}) from team '{self.name}'.")
            if self.use_agno_team and self.agno_team and model_id in self.agno_agents:
                del self.agno_agents[model_id] # Assuming lead was also an AgnoAgent
            self.lead_model = None
        else:
            logger.warning(f"Model with ID '{model_id}' not found in team '{self.name}'.")


    def add_tool(self, tool: GlueTool) -> None:
        """Adds a tool to the team's available tools."""
        if not isinstance(tool, GlueTool):
            raise TypeError("tool must be an instance of GlueTool.")
        if tool.name in self.tools:
            logger.warning(f"Tool with name '{tool.name}' already exists in team '{self.name}'. Replacing.")
        self.tools[tool.name] = tool
        logger.info(f"Added tool '{tool.name}' to team '{self.name}'.")
        # If Agno team is initialized, adapt and add this tool to it
        if self.use_agno_team and self.agno_team:
            self._add_glue_tool_to_agno_team(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Removes a tool from the team by its name."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Removed tool '{tool_name}' from team '{self.name}'.")
            # If Agno team exists, potentially remove/update its tools.
            # AgnoTeam tools are typically set at initialization or via its model.
            # Re-initialization or specific AgnoTeam API might be needed.
            if self.use_agno_team and self.agno_team:
                logger.warning(f"Tool '{tool_name}' removed from GlueTeam. AgnoTeam tools might need manual update/re-initialization.")
        else:
            logger.warning(f"Tool with name '{tool_name}' not found in team '{self.name}'.")


    def add_adhesive(self, adhesive: Adhesive) -> None:
        """Adds an adhesive to the team."""
        if not isinstance(adhesive, Adhesive):
            raise TypeError("adhesive must be an instance of Adhesive.")
        if adhesive.adhesive_type in self.adhesives:
            logger.warning(f"Adhesive of type '{adhesive.adhesive_type}' already exists for team '{self.name}'. Replacing.")
        self.adhesives[adhesive.adhesive_type] = adhesive
        logger.info(f"Added adhesive of type '{adhesive.adhesive_type}' to team '{self.name}'.")
        # Adhesives are primarily a GLUE concept for now.
        # Their interaction with Agno's memory/state would be through how GlueModels (and thus AgnoAgents) use them.

    def get_adhesive(self, adhesive_type: AdhesiveType) -> Optional[Adhesive]:
        """Retrieves an adhesive by its type."""
        return self.adhesives.get(adhesive_type)

    def _get_team_context_for_agno(self) -> str:
        """
        Prepares a string representation of the team's context for Agno.
        This could include summaries of adhesives, team goals, etc.
        """
        context_parts = []
        if self.lead_model:
            context_parts.append(f"Team Lead: {self.lead_model.name}")
        context_parts.append(f"Team Members ({len(self.members)}): {', '.join(m.name for m in self.members.values())}")
        
        glue_adhesive = self.get_adhesive(AdhesiveType.GLUE)
        if glue_adhesive:
            # Potentially summarize or get relevant parts of the GLUE adhesive
            # For now, let's assume it has a method to get a summary or relevant data
            # glue_summary = glue_adhesive.get_summary() 
            # context_parts.append(f"GLUE State: {glue_summary}")
            pass # Placeholder for actual adhesive interaction

        # Add more context as needed (e.g., from other adhesives, team goals)
        return "\n".join(context_parts)

    def _convert_glue_tools_to_agno_tools(self) -> List[AgnoTool]:
        """Converts GlueTools to AgnoTools."""
        agno_tools = []
        for glue_tool_name, glue_tool_instance in self.tools.items():
            try:
                # Assuming GlueTool has a method to convert itself or provide necessary details
                # For simplicity, let's assume GlueTool has name, description, and a callable execute method
                # and AgnoTool can be created from this.
                # This is a placeholder for actual conversion logic.
                # The actual conversion might involve creating AgnoFunction objects.
                
                # Placeholder: Create a basic AgnoFunction
                # The parameters for AgnoFunction would need to be derived from glue_tool_instance.parameters_schema
                # For now, let's assume a simple structure.
                # This needs to be more robust based on actual GlueTool and AgnoFunction structure.
                
                # A more direct approach if GlueTool is designed to be compatible:
                if hasattr(glue_tool_instance, 'to_agno_tool'):
                    agno_tool = glue_tool_instance.to_agno_tool()
                    if agno_tool:
                         agno_tools.append(agno_tool)
                    else:
                        logger.warning(f"Tool '{glue_tool_name}' could not be converted to AgnoTool (to_agno_tool returned None).")

                else: # Fallback to a more manual creation if to_agno_tool is not present
                    # This requires GlueTool to have a 'parameters_schema' and 'execute'
                    if hasattr(glue_tool_instance, 'parameters_schema') and callable(glue_tool_instance.execute):
                        # Simplified AgnoFunction creation
                        # The actual parameters for AgnoFunction might be more complex
                        # (e.g., json_schema for args)
                        agno_function = AgnoFunction(
                            name=glue_tool_instance.name,
                            description=glue_tool_instance.description,
                            fn=glue_tool_instance.execute, # This assumes execute can be directly used
                            # parameters=glue_tool_instance.parameters_schema # This needs to be Agno's format
                        )
                        # AgnoTool often wraps an AgnoFunction
                        # This part is highly dependent on Agno's API for Tool creation
                        # For now, let's assume AgnoTool can take an AgnoFunction
                        # This is a conceptual placeholder:
                        # agno_tools.append(AgnoTool(name=agno_function.name, function=agno_function, description=agno_function.description))
                        logger.warning(f"Manual conversion for tool '{glue_tool_name}' is a placeholder. Review AgnoTool/Function creation.")
                        # For now, let's assume we can directly pass the function if the tool is simple
                        # This is a simplification and likely incorrect for complex tools.
                        # A proper adapter pattern is needed here.
                        # For the purpose of moving forward, if a tool is a simple callable:
                        if callable(glue_tool_instance.execute):
                             # This is a very simplified representation of an AgnoTool
                             # Agno's Tool class might require more specific structure (e.g., from agno.tools.tool import Tool)
                             # and might expect a function that matches a certain signature.
                             # For now, we'll represent it as a dict, which some Agno models might accept.
                            agno_tools.append({
                                "type": "function",
                                "function": {
                                    "name": glue_tool_instance.name,
                                    "description": glue_tool_instance.description,
                                    "parameters": glue_tool_instance.parameters_schema if hasattr(glue_tool_instance, 'parameters_schema') else {"type": "object", "properties": {}}
                                }
                            })


                    else:
                        logger.warning(f"Tool '{glue_tool_name}' does not have to_agno_tool, parameters_schema, or execute method. Cannot convert to AgnoTool.")

            except Exception as e:
                logger.error(f"Failed to convert GlueTool '{glue_tool_name}' to AgnoTool: {e}")
        return agno_tools


    def _add_glue_tool_to_agno_team(self, glue_tool: GlueTool):
        """Adapts and adds a single GlueTool to the initialized AgnoTeam."""
        if not self.agno_team or not self.agno_team.model:
            logger.warning("AgnoTeam or its model not initialized. Cannot add tool.")
            return

        # Convert the single GlueTool to Agno's format
        # This logic should mirror part of _convert_glue_tools_to_agno_tools
        agno_tool_representation = None
        if hasattr(glue_tool, 'to_agno_tool'):
            agno_tool_representation = glue_tool.to_agno_tool() # Assuming this returns the dict format Agno expects
        elif hasattr(glue_tool, 'parameters_schema') and callable(glue_tool.execute):
             agno_tool_representation = {
                "type": "function",
                "function": {
                    "name": glue_tool.name,
                    "description": glue_tool.description,
                    "parameters": glue_tool.parameters_schema
                }
            }
        
        if agno_tool_representation:
            # Add to the AgnoTeam's lead model's tools
            # Agno's API for adding tools dynamically might vary.
            # Often, tools are set on the model.
            if not self.agno_team.model._tools:
                self.agno_team.model._tools = []
            
            # Avoid duplicates by name
            if not any(t.get("function", {}).get("name") == glue_tool.name for t in self.agno_team.model._tools):
                self.agno_team.model._tools.append(agno_tool_representation)
                logger.info(f"Adapted and added GlueTool '{glue_tool.name}' to AgnoTeam's lead model tools.")
                # If Agno also uses a _functions dict on the model for execution:
                if hasattr(glue_tool, 'execute') and callable(glue_tool.execute):
                    if not self.agno_team.model._functions:
                        self.agno_team.model._functions = {}
                    # This assumes AgnoFunction can be created this way for the model's internal use
                    self.agno_team.model._functions[glue_tool.name] = AgnoFunction(
                        name=glue_tool.name,
                        description=glue_tool.description,
                        fn=glue_tool.execute 
                        # parameters might be needed here if AgnoFunction expects it
                    )

            else:
                logger.info(f"Tool '{glue_tool.name}' already present in AgnoTeam's lead model tools.")
        else:
            logger.warning(f"Could not adapt GlueTool '{glue_tool.name}' for AgnoTeam.")


    def _add_glue_model_to_agno_team(self, glue_model_instance: Model):
        """Creates an AgnoAgent from a GlueModel and adds it to the AgnoTeam."""
        if not self.agno_team:
            logger.warning("AgnoTeam not initialized. Cannot add AgnoAgent.")
            return

        agno_model_param = self._create_agno_model_param(glue_model_instance)
        if not agno_model_param:
            logger.error(f"Failed to create Agno model parameter for GLUE model {glue_model_instance.name}. Cannot add to AgnoTeam.")
            return

        # Create AgnoAgent
        # Tools for this agent would typically come from the team or be specific to the agent
        # For now, let's assume agents in AgnoTeam use tools defined at the AgnoTeam.model level
        # or the AgnoTeam orchestrates tool calls.
        agno_agent_tools = self._convert_glue_tools_to_agno_tools() # Or a subset relevant to this agent

        try:
            new_agno_agent = AgnoAgent(
                name=f"{glue_model_instance.name}_agno_agent",
                model=agno_model_param, # This should be an instance of agno.models.base.Model
                # tools=agno_agent_tools, # Pass converted tools if AgnoAgent takes them directly
                                        # Otherwise, tools are managed by the AgnoTeam's lead model.
                instructions=glue_model_instance.instructions # Pass instructions
            )
            # Add tools to the agent's model if Agno works that way
            if hasattr(new_agno_agent.model, 'set_tools') and callable(new_agno_agent.model.set_tools):
                new_agno_agent.model.set_tools(agno_agent_tools)


            self.agno_agents[glue_model_instance.model_id] = new_agno_agent
            
            # Add the new AgnoAgent to the AgnoTeam's members
            # AgnoTeam.members is a list of Agents or Teams.
            if new_agno_agent not in self.agno_team.members: # Avoid duplicates
                self.agno_team.members.append(new_agno_agent)
            logger.info(f"Created and added AgnoAgent for GLUE model '{glue_model_instance.name}' to AgnoTeam '{self.name}'.")

        except Exception as e:
            logger.error(f"Failed to create or add AgnoAgent for GLUE model {glue_model_instance.name}: {e}", exc_info=True)


    def _create_agno_model_param(self, glue_model_instance: Model) -> Optional[AgnoBaseModel]:
        """
        Creates an Agno model parameter (instance of agno.models.base.Model) 
        from a GlueModel instance.
        This is a placeholder and needs to be robustly implemented based on provider mapping.
        """
        provider = glue_model_instance.provider_name 
        model_name = glue_model_instance.model_name
        glue_config = glue_model_instance.config

        logger.info(f"Attempting to create Agno model parameter for GLUE model: {glue_model_instance.name}, Provider: {provider}, Model Name: {model_name}")

        agent_model_param: Optional[AgnoBaseModel] = None
        
        # Simplified provider mapping
        # A more robust provider mapping will be needed later.
        if provider == "test":
            logger.info(f"Using _MinimalTestAgnoModel for GLUE model {glue_model_instance.name} with provider {provider}")
            agent_model_param = _MinimalTestAgnoModel(
                name=f"{glue_model_instance.name}_test_agno_model"
            )
        elif provider == "anthropic":
            from agno.models.anthropic import AnthropicChat # Dynamic import
            temp = glue_config.get("temperature")
            # Ensure temperature is float if not None
            temperature = float(temp) if temp is not None else None 
            agent_model_param = AnthropicChat(
                model_name=model_name or "claude-3-opus-20240229", # Default if not specified
                temperature=temperature,
                # api_key=... # API key handling needed
            )
        elif provider == "openai":
            from agno.models.openai import OpenAIChat # Dynamic import
            temp = glue_config.get("temperature")
            temperature = float(temp) if temp is not None else None
            agent_model_param = OpenAIChat(
                model_name=model_name or "gpt-4-turbo-preview",
                temperature=temperature,
                # api_key=...
            )
        # Add other providers (Google Gemini, etc.) here
        # elif provider == "google":
        #     from agno.models.google import GoogleChat
        #     agent_model_param = GoogleChat(model_name=model_name or "gemini-pro")
        else:
            logger.warning(f"Provider '{provider}' not yet supported for Agno integration. Cannot create Agno model parameter.")
            # Fallback to a generic placeholder if possible, or None
            # For now, returning None, which will prevent AgnoAgent creation.
            return None
        
        # Set common properties if they exist on the AgnoBaseModel instance
        if agent_model_param:
            if hasattr(agent_model_param, 'instructions') and glue_model_instance.instructions:
                agent_model_param.instructions = [glue_model_instance.instructions] if isinstance(glue_model_instance.instructions, str) else glue_model_instance.instructions
            if hasattr(agent_model_param, 'system_prompt') and glue_model_instance.system_prompt: # Assuming system_prompt exists on GlueModel
                 agent_model_param.system_prompt = glue_model_instance.system_prompt
        
        return agent_model_param


    def _initialize_agno_team(self) -> Optional[AgnoTeam]:
        """
        Initializes the AgnoTeam based on the current GlueTeam configuration.
        This should only be called if self.use_agno_team is True.
        """
        if not self.use_agno_team:
            logger.info(f"Agno integration disabled for team '{self.name}'. Skipping AgnoTeam initialization.")
            return None
        
        if self.agno_team:
            logger.info(f"AgnoTeam for '{self.name}' already initialized.")
            return self.agno_team

        logger.info(f"Initializing Agno team for {self.name}...")

        if not self.lead_model:
            logger.error(f"Cannot initialize AgnoTeam for '{self.name}': No lead model set for GlueTeam.")
            return None

        # 1. Create Agno model for the GlueTeam's lead model
        lead_agno_model_param = self._create_agno_model_param(self.lead_model)
        if not lead_agno_model_param:
            logger.error(f"Failed to create Agno model parameter for lead model '{self.lead_model.name}'. Cannot initialize AgnoTeam.")
            return None
        
        # Set tools for the lead Agno model
        agno_native_tools = self._convert_glue_tools_to_agno_tools()
        if hasattr(lead_agno_model_param, 'set_tools') and callable(lead_agno_model_param.set_tools):
            lead_agno_model_param.set_tools(agno_native_tools)
            logger.info(f"Set {len(agno_native_tools)} tools on Agno lead model parameter.")
        else:
            logger.warning("Lead Agno model parameter does not have set_tools method. Tools might not be correctly passed.")


        # 2. Create AgnoAgents for each member model in GlueTeam
        current_agno_agents: List[AgnoAgent] = []
        all_glue_models = [self.lead_model] + list(self.members.values()) # Include lead in the agent list for AgnoTeam members if desired, or handle separately

        for glue_model_instance in all_glue_models: # Iterate over lead + members
            if glue_model_instance.model_id in self.agno_agents: # Reuse if already created
                current_agno_agents.append(self.agno_agents[glue_model_instance.model_id])
                continue

            # Create Agno model parameter for the current Glue model
            # This step is somewhat redundant if lead_agno_model_param is the one for the AgnoTeam.model
            # and other glue_model_instances become AgnoAgent.model
            
            # If glue_model_instance is the lead, its Agno model (lead_agno_model_param) will be the AgnoTeam's model.
            # For other members, we create AgnoAgents whose 'model' attribute is an AgnoBaseModel instance.
            if glue_model_instance == self.lead_model:
                # The lead model's Agno representation is used for the AgnoTeam's own model.
                # It doesn't become a separate AgnoAgent *member* of the AgnoTeam unless specifically designed that way.
                # Typically, the AgnoTeam.model *is* the lead.
                pass
            else: # For non-lead members
                member_agno_model_param = self._create_agno_model_param(glue_model_instance)
                if member_agno_model_param:
                    try:
                        # Instructions for the member agent
                        member_instructions = glue_model_instance.instructions
                        if isinstance(member_instructions, str):
                             member_instructions = [member_instructions]
                        
                        # Tools for this specific agent's model
                        # For simplicity, assume they use team tools for now, or this needs refinement
                        member_agno_tools = self._convert_glue_tools_to_agno_tools() # Or agent-specific tools
                        if hasattr(member_agno_model_param, 'set_tools') and callable(member_agno_model_param.set_tools):
                            member_agno_model_param.set_tools(member_agno_tools)


                        agno_member_agent = AgnoAgent(
                            name=f"{glue_model_instance.name}_agno_agent",
                            model=member_agno_model_param,
                            instructions=member_instructions,
                            # tools=... if AgnoAgent takes tools directly
                        )
                        self.agno_agents[glue_model_instance.model_id] = agno_member_agent
                        current_agno_agents.append(agno_member_agent)
                    except TypeError as te:
                         logger.error(f"Failed to create placeholder AgnoAgent for GLUE model {glue_model_instance.name}: {te}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Unexpected error creating AgnoAgent for {glue_model_instance.name}: {e}", exc_info=True)
                else:
                    logger.error(f"Failed to create Agno model parameter for GLUE model {glue_model_instance.name}. Cannot create AgnoAgent.")
        
        if not current_agno_agents and len(all_glue_models) > 1 : # If only lead, current_agno_agents might be empty if lead isn't an agent-member
            # This condition needs refinement: if there are members but no AgnoAgents could be made for them.
            # If there's only a lead model, current_agno_agents (as members) will be empty.
            # The check should be: if there are GlueTeam members, but no corresponding AgnoAgents were created for them.
            if self.members and not any(mid in self.agno_agents for mid in self.members):
                 logger.error(f"Failed to create any AgnoAgents for team {self.name} members. Cannot initialize AgnoTeam effectively.")
                 # Decide if this is a fatal error for AgnoTeam initialization
                 # return None # Or proceed with a team that only has a lead model and no agent members

        # 3. Create the AgnoTeam instance
        team_instructions = self.lead_model.instructions
        if isinstance(team_instructions, str): # Ensure it's a list
            team_instructions = [team_instructions]

        try:
            self.agno_team = AgnoTeam(
                name=f"{self.name}_agno_team",
                model=lead_agno_model_param, # The lead's Agno model instance
                members=current_agno_agents, # List of AgnoAgent instances (excluding the lead if lead is AgnoTeam.model)
                instructions=team_instructions,
                # workflow=self.workflow, # If a Glue workflow is convertible to AgnoWorkflow
                # mode="collaborate" # Default or configurable
            )
            logger.info(f"Successfully initialized AgnoTeam '{self.agno_team.name}' for GlueTeam '{self.name}' with {len(current_agno_agents)} member agents.")
            
            # Synchronize tools again, ensuring the AgnoTeam's model has them.
            # This might be redundant if set_tools on lead_agno_model_param worked.
            if hasattr(self.agno_team.model, 'set_tools') and callable(self.agno_team.model.set_tools):
                final_agno_tools = self._convert_glue_tools_to_agno_tools()
                self.agno_team.model.set_tools(final_agno_tools)
                # Also ensure _functions are populated if Agno relies on it
                if not self.agno_team.model._functions: self.agno_team.model._functions = {}
                for tool_spec in final_agno_tools:
                    tool_name = tool_spec.get("function", {}).get("name")
                    glue_tool_instance = self.tools.get(tool_name)
                    if tool_name and glue_tool_instance and callable(glue_tool_instance.execute):
                        self.agno_team.model._functions[tool_name] = AgnoFunction(
                            name=tool_name,
                            description=tool_spec.get("function",{}).get("description",""),
                            fn=glue_tool_instance.execute
                        )


            logger.info(f"AgnoTeam '{self.agno_team.name}' initialized for GlueTeam '{self.name}'. Lead model: {self.agno_team.model.name if self.agno_team.model else 'None'}. Member agents: {len(self.agno_team.members)}")

        except Exception as e:
            logger.error(f"Failed to initialize AgnoTeam for '{self.name}': {e}", exc_info=True)
            self.agno_team = None # Ensure it's None if initialization fails
            return None

        return self.agno_team


    def run(self, initial_input: Any, **kwargs) -> Union[Any, TeamRunResponse, None]:
        """
        Runs the team's process with the given input.
        If use_agno_team is True, this delegates to the AgnoTeam.
        Otherwise, a native GLUE execution path would be needed (currently not implemented).
        """
        logger.info(f"GlueTeam '{self.name}' run called with input: {str(initial_input)[:100]}...")

        if self.use_agno_team:
            if not self.agno_team:
                logger.info(f"AgnoTeam for '{self.name}' not yet initialized. Attempting initialization now.")
                self._initialize_agno_team() # Attempt to initialize it
            
            if not self.agno_team: # Check again after attempt
                logger.error(f"AgnoTeam for '{self.name}' could not be initialized. Cannot run with Agno.")
                # According to memory: "If Agno integration is not active or fails for a GlueTeam, it's an error state"
                raise RuntimeError(f"AgnoTeam for '{self.name}' failed to initialize. GLUE cannot proceed without Agno.")

            try:
                # Prepare the input for AgnoTeam.run. It expects a message string or a list of Message objects.
                # For simplicity, let's assume initial_input is a string prompt.
                # A more robust solution would involve converting initial_input to Agno's Message format.
                
                # Convert initial_input to Agno Message format
                # This is a simplified conversion. Real conversion might be more complex.
                if isinstance(initial_input, str):
                    agno_input_message = Message(role="user", content=initial_input)
                elif isinstance(initial_input, dict) and "role" in initial_input and "content" in initial_input:
                    agno_input_message = Message(**initial_input)
                elif isinstance(initial_input, Message): # If it's already an Agno Message
                    agno_input_message = initial_input
                else:
                    # Fallback: try to stringify, but log a warning
                    logger.warning(f"Initial input type {type(initial_input)} not directly supported for Agno. Converting to string.")
                    agno_input_message = Message(role="user", content=str(initial_input))

                logger.info(f"Running AgnoTeam '{self.agno_team.name}' with processed input...")
                
                # AgnoTeam.run is synchronous. If an async version is needed, use arun.
                # The return type of AgnoTeam.run is TeamRunResponse.
                response: TeamRunResponse = self.agno_team.run(message=agno_input_message, **kwargs)
                
                logger.info(f"AgnoTeam '{self.agno_team.name}' finished execution.")
                
                # Process/adapt AgnoTeamRunResponse if needed, or return directly.
                # For now, return it directly.
                return response

            except Exception as e:
                logger.error(f"Error running AgnoTeam '{self.agno_team.name}': {e}", exc_info=True)
                # Handle error appropriately, maybe raise a GLUE-specific exception
                raise RuntimeError(f"Error during AgnoTeam execution for '{self.name}': {e}") from e
        else:
            # Native GLUE execution path (Not implemented as per memory - GLUE relies on Agno)
            logger.error("Native GLUE execution path is not implemented. GLUE requires Agno integration (use_agno_team=True).")
            raise NotImplementedError("Native GLUE execution without Agno is not supported.")

    async def arun(self, initial_input: Any, **kwargs) -> Union[Any, TeamRunResponse, None]:
        """
        Asynchronously runs the team's process with the given input.
        Delegates to AgnoTeam.arun() if use_agno_team is True.
        """
        logger.info(f"GlueTeam '{self.name}' arun called with input: {str(initial_input)[:100]}...")

        if self.use_agno_team:
            if not self.agno_team:
                logger.info(f"AgnoTeam for '{self.name}' not yet initialized for arun. Attempting initialization now.")
                self._initialize_agno_team() # Initialize if not already (it's synchronous)
            
            if not self.agno_team:
                logger.error(f"AgnoTeam for '{self.name}' could not be initialized. Cannot arun with Agno.")
                raise RuntimeError(f"AgnoTeam for '{self.name}' failed to initialize for arun. GLUE cannot proceed.")

            try:
                if isinstance(initial_input, str):
                    agno_input_message = Message(role="user", content=initial_input)
                elif isinstance(initial_input, dict) and "role" in initial_input and "content" in initial_input:
                    agno_input_message = Message(**initial_input)
                elif isinstance(initial_input, Message):
                    agno_input_message = initial_input
                else:
                    logger.warning(f"Initial input type {type(initial_input)} for arun not directly Agno Message. Converting to string.")
                    agno_input_message = Message(role="user", content=str(initial_input))

                logger.info(f"Asynchronously running AgnoTeam '{self.agno_team.name}' with processed input...")
                
                response: TeamRunResponse = await self.agno_team.arun(message=agno_input_message, **kwargs)
                
                logger.info(f"AgnoTeam '{self.agno_team.name}' finished asynchronous execution.")
                return response

            except Exception as e:
                logger.error(f"Error asynchronously running AgnoTeam '{self.agno_team.name}': {e}", exc_info=True)
                raise RuntimeError(f"Error during asynchronous AgnoTeam execution for '{self.name}': {e}") from e
        else:
            logger.error("Native GLUE asynchronous execution path is not implemented. GLUE requires Agno integration.")
            raise NotImplementedError("Native GLUE asynchronous execution without Agno is not supported.")

    # --- Magnetic Flow Methods ---
    def subscribe_to_topic(self, topic: str, subscriber_team: 'GlueTeam'):
        """A team subscribes to a topic from this team."""
        if topic not in self.magnetic_subscriptions:
            self.magnetic_subscriptions[topic] = []
        if subscriber_team not in self.magnetic_subscriptions[topic]:
            self.magnetic_subscriptions[topic].append(subscriber_team)
            logger.info(f"Team '{subscriber_team.name}' subscribed to topic '{topic}' from team '{self.name}'.")

    def unsubscribe_from_topic(self, topic: str, subscriber_team: 'GlueTeam'):
        """A team unsubscribes from a topic from this team."""
        if topic in self.magnetic_subscriptions and subscriber_team in self.magnetic_subscriptions[topic]:
            self.magnetic_subscriptions[topic].remove(subscriber_team)
            logger.info(f"Team '{subscriber_team.name}' unsubscribed from topic '{topic}' from team '{self.name}'.")
            if not self.magnetic_subscriptions[topic]: # Cleanup if no subscribers left
                del self.magnetic_subscriptions[topic]

    def publish_to_topic(self, topic: str, data: Any):
        """This team publishes data to a topic, notifying all subscribers."""
        self.magnetic_publications[topic] = data
        logger.info(f"Team '{self.name}' published data to topic '{topic}'.")
        if topic in self.magnetic_subscriptions:
            for subscriber_team in self.magnetic_subscriptions[topic]:
                try:
                    # This is a conceptual PUSH. The subscriber team needs a method to handle incoming data.
                    # For example, subscriber_team.handle_magnetic_push(topic, data, from_team=self)
                    if hasattr(subscriber_team, 'on_magnetic_data_received'):
                         # Potentially run this asynchronously if it involves significant processing
                        asyncio.create_task(subscriber_team.on_magnetic_data_received(topic, data, source_team_name=self.name))
                        logger.info(f"Notified subscriber team '{subscriber_team.name}' about topic '{topic}'.")
                    else:
                        logger.warning(f"Subscriber team '{subscriber_team.name}' does not have 'on_magnetic_data_received' method.")
                except Exception as e:
                    logger.error(f"Error notifying subscriber team '{subscriber_team.name}' for topic '{topic}': {e}")
    
    async def on_magnetic_data_received(self, topic: str, data: Any, source_team_name: str):
        """
        Handles data received via a magnetic PUSH from another team.
        This method should be overridden or extended by specific team implementations
        to process the data according to the team's logic.
        """
        logger.info(f"Team '{self.name}' received data on topic '{topic}' from team '{source_team_name}': {str(data)[:100]}...")
        # Example: Add to a specific adhesive or trigger a new internal task
        # For instance, if there's an "inbox" adhesive:
        # inbox_adhesive = self.get_adhesive(AdhesiveType.INBOX) # Assuming INBOX type
        # if inbox_adhesive:
        #     inbox_adhesive.store(f"magnetic_{topic}_{source_team_name}", data)
        # else:
        #     logger.warning(f"Team '{self.name}' has no INBOX adhesive to store magnetic data.")
        
        # Or, this could trigger the team's run/arun method with this new data as input
        # await self.arun(initial_input={"source_topic": topic, "data": data, "from_team": source_team_name})
        pass # Placeholder for actual data handling logic


    def request_data_from_team(self, target_team: 'GlueTeam', request_details: Any, topic: Optional[str] = None) -> Any:
        """
        This team requests data from a target_team (Magnetic PULL).
        The target_team needs a method to handle such requests.
        """
        logger.info(f"Team '{self.name}' requesting data from team '{target_team.name}'. Request: {str(request_details)[:100]}")
        if hasattr(target_team, 'handle_magnetic_pull_request'):
            try:
                # This is a synchronous PULL. An async version might be needed.
                return target_team.handle_magnetic_pull_request(request_details, requesting_team_name=self.name, topic=topic)
            except Exception as e:
                logger.error(f"Error requesting data from team '{target_team.name}': {e}")
                return None # Or raise
        else:
            logger.warning(f"Target team '{target_team.name}' does not have 'handle_magnetic_pull_request' method.")
            return None

    def handle_magnetic_pull_request(self, request_details: Any, requesting_team_name: str, topic: Optional[str]=None) -> Any:
        """
        Handles a data request (Magnetic PULL) from another team.
        This method should be implemented by teams that can serve data.
        It should process the request_details and return the appropriate data.
        """
        logger.info(f"Team '{self.name}' received data request on topic '{topic if topic else ''}' from team '{requesting_team_name}'. Request: {str(request_details)[:100]}")
        # Example: Retrieve data from an adhesive or run an internal process
        # output_adhesive = self.get_adhesive(AdhesiveType.OUTPUT) # Assuming OUTPUT type
        # if output_adhesive:
        #     # Logic to find relevant data in output_adhesive based on request_details
        #     # data_key = f"pull_response_for_{requesting_team_name}_{str(request_details)[:20]}" 
        #     # return output_adhesive.retrieve(data_key) 
        #     pass
        logger.warning(f"Team '{self.name}' handle_magnetic_pull_request is not fully implemented. Returning None.")
        return None # Placeholder


    def __repr__(self) -> str:
        return f"<GlueTeam name='{self.name}' id='{self.team_id}' lead='{self.lead_model.name if self.lead_model else 'None'}' members={len(self.members)} use_agno={self.use_agno_team}>"
