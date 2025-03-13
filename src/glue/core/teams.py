# glue/core/team.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List
from datetime import datetime
import asyncio
import logging
from pydantic import BaseModel

from .types import AdhesiveType, TeamConfig, ToolResult, Message
from .model import Model

# ==================== Constants ====================
logger = logging.getLogger("glue.team")

# ==================== Class Definition ====================
class Team:
    """
    Team implementation for GLUE framework.
    Manages model collaboration, tool sharing, and result persistence.
    """
    def __init__(
        self,
        name: str,
        config: Optional[TeamConfig] = None
    ):
        self.name = name
        self.config = config or TeamConfig(name=name, lead="", members=[], tools=[])
        
        # Core components
        self.models: Dict[str, Model] = {}
        self.tools: Dict[str, Any] = {}
        self.tool_bindings: Dict[str, AdhesiveType] = {}
        
        # State management
        self.shared_results: Dict[str, ToolResult] = {}
        self.conversation_history: List[Message] = []
        self.relationships: Dict[str, str] = {}  # Team magnetic relationships
        self.repelled_by: Set[str] = set()      # Teams that repel this one

        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    # ==================== Core Methods ====================
    async def add_member(
        self,
        model: Model,
        role: str = "member",
        tools: Optional[Set[str]] = None
    ) -> None:
        """Add a model to the team"""
        if model.name in self.models:
            raise ValueError(f"Model {model.name} already in team")
            
        # Add model
        self.models[model.name] = model
        model.team = self  # Set team reference
        
        # Set up tools
        if tools:
            for tool_name in tools:
                if tool_name in self.tools:
                    model.add_tool(tool_name, self.tools[tool_name])
                    
        # Update config
        if role == "lead":
            self.config.lead = model.name
        else:
            self.config.members.append(model.name)
            
        self.updated_at = datetime.now()
        logger.info(f"Added model {model.name} to team {self.name}")

    async def add_tool(
        self,
        name: str,
        tool: Any,
        binding: AdhesiveType = AdhesiveType.VELCRO
    ) -> None:
        """Add a tool to the team"""
        self.tools[name] = tool
        self.tool_bindings[name] = binding
        
        # Add to all models
        for model in self.models.values():
            model.add_tool(name, tool)
            
        self.config.tools.append(name)
        self.updated_at = datetime.now()
        logger.info(f"Added tool {name} to team {self.name}")

    async def share_result(
        self,
        tool_name: str,
        result: ToolResult
    ) -> None:
        """Share a tool result with the team"""
        if result.adhesive == AdhesiveType.GLUE:
            self.shared_results[tool_name] = result
            logger.info(f"Shared result from {tool_name} in team {self.name}")
        elif result.adhesive == AdhesiveType.VELCRO:
            # Only store in model's session
            pass
        # TAPE results are not stored

    async def process_message(
        self,
        content: str,
        from_model: Optional[str] = None,
        target_model: Optional[str] = None
    ) -> str:
        """Process a message within the team"""
        # Get source model
        source = None
        if from_model:
            source = self.models.get(from_model)
            if not source:
                raise ValueError(f"Model {from_model} not in team")
                
        # Get target model
        target = None
        if target_model:
            target = self.models.get(target_model)
            if not target:
                raise ValueError(f"Model {target_model} not in team")
                
        # Use lead model if no specific models given
        if not source and self.config.lead:
            source = self.models[self.config.lead]
            
        if not source:
            raise ValueError("No source model available")
            
        # Generate response
        response = await source.generate(content)
        
        # Store in history
        message = Message(
            role="model" if from_model else "system",
            content=content
        )
        self.conversation_history.append(message)
        
        response_message = Message(
            role="model",
            content=response
        )
        self.conversation_history.append(response_message)
        
        return response

    # ==================== Magnetic Field Methods ====================
    def set_relationship(self, team_name: str, relationship: str) -> None:
        """Set magnetic relationship with another team"""
        if team_name in self.repelled_by:
            raise ValueError(f"Cannot set relationship with {team_name} - repelled")
            
        self.relationships[team_name] = relationship
        self.updated_at = datetime.now()
        logger.info(f"Set {relationship} relationship with team {team_name}")

    def break_relationship(self, team_name: str) -> None:
        """Break relationship with another team"""
        if team_name in self.relationships:
            del self.relationships[team_name]
            logger.info(f"Broke relationship with team {team_name}")

    def repel(self, team_name: str) -> None:
        """Set repulsion with another team"""
        self.repelled_by.add(team_name)
        if team_name in self.relationships:
            del self.relationships[team_name]
        logger.info(f"Set repulsion with team {team_name}")

    # ==================== Helper Methods ====================
    def get_model_tools(self, model_name: str) -> Set[str]:
        """Get tools available to a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not in team")
        return set(self.models[model_name]._tools.keys())

    def get_shared_results(self) -> Dict[str, ToolResult]:
        """Get all shared (GLUE) results"""
        return self.shared_results.copy()

    def get_relationships(self) -> Dict[str, str]:
        """Get all team relationships"""
        return self.relationships.copy()

    # ==================== Error Handling ====================
    async def _handle_error(self, error: Exception) -> None:
        """Handle team-level errors"""
        logger.error(f"Team error in {self.name}: {str(error)}")
        raise

    async def cleanup(self) -> None:
        """Clean up team resources"""
        try:
            # Clean up tools
            for tool in self.tools.values():
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
                    
            # Clean up models
            for model in self.models.values():
                if hasattr(model, 'cleanup'):
                    await model.cleanup()
                    
            # Clear state
            self.shared_results.clear()
            self.conversation_history.clear()
            
            logger.info(f"Cleaned up team {self.name}")
            
        except Exception as e:
            logger.error(f"Error during team cleanup: {str(e)}")
            raise
