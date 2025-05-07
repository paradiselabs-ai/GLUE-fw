"""
Translator for converting GLUE DSL AST to Agno configuration.

This module provides the GlueDSLAgnoTranslator class, which translates
GLUE DSL Abstract Syntax Tree (AST) into Agno Workflow/Team/Agent configurations.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("glue.adapters.agno.dsl_translator")

class GlueDSLAgnoTranslator:
    """
    Translator class for converting GLUE DSL AST to Agno configuration.
    
    This class handles the translation of GLUE DSL concepts (teams, models, tools, flows)
    into their Agno equivalents (teams, agents, tools, connections).
    """
    
    def __init__(self):
        """Initialize the translator."""
        self.logger = logger
        self.logger.info("Initialized GlueDSLAgnoTranslator")
    
    def translate(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate GLUE DSL AST into Agno configuration.
        
        Args:
            ast: GLUE DSL Abstract Syntax Tree
            
        Returns:
            Agno configuration dictionary
        """
        self.logger.info(f"Translating GLUE DSL AST to Agno configuration")
        
        # Initialize the Agno configuration structure
        agno_config = {
            "workflow": self._translate_app(ast.get("app", {})),
            "agents": self._translate_models(ast.get("models", {})),
            "teams": self._translate_teams(ast, ast.get("teams", [])),
            "tools": self._translate_tools(ast.get("tools", {})),
            "flows": self._translate_flows(ast.get("flows", [])),
        }
        
        self.logger.info("Translation completed")
        return agno_config
    
    def _translate_app(self, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate GLUE app configuration to Agno workflow configuration.
        
        Args:
            app_config: GLUE app configuration
            
        Returns:
            Agno workflow configuration
        """
        name = app_config.get("app_name", "GLUE App")
        description = app_config.get("description", f"GLUE application: {name}")
        
        return {
            "name": name,
            "description": description,
            "config": app_config.get("config", {})
        }
    
    def _translate_models(self, models_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate GLUE models to Agno agents.
        
        Args:
            models_config: GLUE models configuration
            
        Returns:
            Agno agents configuration
        """
        agents = {}
        
        for model_name, model_config in models_config.items():
            agents[model_name] = {
                "provider": model_config.get("provider", "openai"),
                "model_name": model_config.get("model_name", "gpt-3.5-turbo"),
                "config": model_config.get("config", {}),
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 1000),
            }
        
        return agents
    
    def _translate_teams(self, ast: Dict[str, Any], teams_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Translate GLUE teams to Agno teams.
        
        Args:
            ast: Complete GLUE AST (needed for resolving references)
            teams_list: List of GLUE team configurations
            
        Returns:
            Agno teams configuration
        """
        teams = {}
        
        for team_config in teams_list:
            team_name = team_config.get("name")
            if not team_name:
                continue
                
            # Extract team members and lead
            members = team_config.get("models", [])
            lead = team_config.get("lead")
            
            # Extract team tools
            tool_names = team_config.get("tools", [])
            
            # Extract communication pattern
            communication_pattern = team_config.get("communication_pattern", "hierarchical")
            
            teams[team_name] = {
                "lead": lead,
                "members": members,
                "tools": tool_names,
                "communication_pattern": communication_pattern,
                "config": team_config.get("config", {})
            }
        
        return teams
    
    def _translate_tools(self, tools_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate GLUE tools to Agno tools.
        
        Args:
            tools_config: GLUE tools configuration
            
        Returns:
            Agno tools configuration
        """
        tools = {}
        
        for tool_name, tool_config in tools_config.items():
            tools[tool_name] = {
                "description": tool_config.get("description", ""),
                "params": tool_config.get("params", {}),
                "adhesive": tool_config.get("adhesive", "GLUE"),
                "config": tool_config.get("config", {})
            }
        
        return tools
    
    def _translate_flows(self, flows_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Translate GLUE magnetic flows to Agno team connections.
        
        Args:
            flows_list: List of GLUE flow configurations
            
        Returns:
            Agno flows configuration
        """
        flows = {}
        
        for flow_config in flows_list:
            flow_name = flow_config.get("name")
            if not flow_name:
                continue
                
            flows[flow_name] = {
                "from": flow_config.get("from"),
                "to": flow_config.get("to"),
                "type": flow_config.get("type", "PUSH"),
                "config": flow_config.get("config", {})
            }
        
        return flows
