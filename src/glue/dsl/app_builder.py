"""
GLUE Application Builder

This module provides functionality to build a GLUE application from a runtime configuration.
It handles instantiating models, tools, teams, and setting up flows between teams.
"""

import os
import re
import logging
from typing import Dict, Any, List
import asyncio

from ..core.app import GlueApp, AppConfig
from smolagents import InferenceClientModel
from ..core.teams import Team
from ..core.types import AdhesiveType, FlowType
from ..tools.tool_registry import get_tool_class
from ..tools.tool_base import Tool
from ..core.flow import Flow


class GlueAppBuilder:
    """Builder for GLUE applications from runtime configurations."""

    def __init__(self):
        """Initialize a new GlueAppBuilder."""
        self.logger = logging.getLogger("glue.dsl.app_builder")
        self._models = {}  # Cache of created models
        self._tools = {}  # Cache of created tools
        self._teams = {}  # Cache of created teams

    def build(self, config: Dict[str, Any]) -> GlueApp:
        """Build a GLUE application from a runtime configuration.

        Args:
            config: Runtime configuration dictionary

        Returns:
            Instantiated GlueApp

        Raises:
            ValueError: If the configuration is invalid or references missing components
        """
        self.logger.info("Building GLUE application from configuration")

        # Reset caches
        self._models = {}
        self._tools = {}
        self._teams = {}

        # Extract app configuration
        app_config = config.get("app", {})
        app_name = app_config.get("name", "Unnamed GLUE App")
        app_description = app_config.get("description", "")
        app_version = app_config.get("version", "0.1.0")
        app_development = app_config.get("development", True)

        # Create models
        for model_config in config.get("models", []):
            model = self._create_model(model_config)
            self._models[model.name] = model

        # Create tools
        for tool_config in config.get("tools", []):
            tool = self._create_tool(tool_config)
            self._tools[tool.name] = tool

        # Create teams
        for team_config in config.get("teams", []):
            team = self._create_team(team_config)
            self._teams[team.name] = team

        # Create flows
        flows = self._create_flows(config.get("flows", []))

        # Create an AppConfig object
        app_config_obj = AppConfig(app_name, app_description)

        # Add models, tools, and teams to the AppConfig
        for model in self._models.values():
            app_config_obj.models[model.name] = model

        for tool in self._tools.values():
            app_config_obj.tools[tool.name] = tool

        for team in self._teams.values():
            app_config_obj.teams[team.name] = team

        # Create the app with the AppConfig
        app = GlueApp(config=app_config_obj)

        # Add properties expected by tests
        app.name = app_name
        app.description = app_description
        app.version = app_version
        app.development = app_development

        # Add collections expected by tests
        app.models = self._models
        app.tools = self._tools
        app.teams = self._teams
        app.flows = flows

        self.logger.info(f"Successfully built GLUE application: {app_name}")
        return app

    def _create_model(self, config: Dict[str, Any]) -> Any:
        """Create a model from configuration.

        Args:
            config: Model configuration

        Returns:
            Instantiated Model

        Raises:
            ValueError: If the model provider is unknown
        """
        name = config.get("name")
        provider = config.get("provider")
        config.get("role", "assistant")
        model_config = config.get("config", {})

        # Process adhesives if present
        adhesives = set()
        for adhesive_str in config.get("adhesives", []):
            try:
                # Convert to uppercase for enum matching but preserve the original value
                # for direct string comparison with enum values
                adhesive_upper = adhesive_str.upper()
                if adhesive_upper == "GLUE":
                    adhesives.add(AdhesiveType.GLUE)
                elif adhesive_upper == "VELCRO":
                    adhesives.add(AdhesiveType.VELCRO)
                elif adhesive_upper == "TAPE":
                    adhesives.add(AdhesiveType.TAPE)
                else:
                    self.logger.warning(f"Unknown adhesive type: {adhesive_str}")
            except ValueError:
                self.logger.warning(f"Unknown adhesive type: {adhesive_str}")

        # Substitute environment variables in config
        processed_config = self._substitute_env_vars(model_config)

        # Convert dictionary to Pydantic ModelConfig
        from ..core.schemas import ModelConfig as PydanticModelConfig

        # Create a ModelConfig instance with the required fields
        model_pydantic_config = PydanticModelConfig(
            name=name,
            provider=provider,
            model=processed_config.get("model", provider),
            temperature=processed_config.get("temperature", 0.7),
            max_tokens=processed_config.get("max_tokens", 2048),
            description=processed_config.get("description", ""),
            api_key=processed_config.get("api_key"),
            api_params=processed_config.get("api_params", {}),
            provider_class=processed_config.get("provider_class"),
        )

        self.logger.info(f"Creating model: {name}")
        # Instantiate a SmolAgents InferenceClientModel
        model = InferenceClientModel(
            model_id=model_pydantic_config.model,
            provider=model_pydantic_config.provider,
            api_key=model_pydantic_config.api_key,
            **{k: v for k, v in config.get("config", {}).items() if v is not None}
        )
        # Assign the model name for compatibility
        model.name = name
        # Attach adhesives and SmolAgents-specific config
        setattr(model, "adhesives", adhesives)
        setattr(model, "smol_config", {k:v for k,v in config.get("config",{}).items() if v is not None})
        self._models[model.name] = model
        return model

    def _create_tool(self, config: Dict[str, Any]) -> Tool:
        """Create a tool from configuration.

        Args:
            config: Tool configuration

        Returns:
            Instantiated Tool

        Raises:
            ValueError: If the tool type is unknown
        """
        name = config.get("name")
        tool_type = config.get("type")
        tool_description = config.get("description", f"{name} tool")
        tool_config = config.get("config", {})

        # Process adhesive types if present
        if "adhesive_types" in tool_config:
            adhesive_types = set()
            for adhesive_str in tool_config.get("adhesive_types", []):
                try:
                    # Convert to uppercase for enum matching but preserve the original value
                    # for direct string comparison with enum values
                    adhesive_upper = adhesive_str.upper()
                    if adhesive_upper == "GLUE":
                        adhesive_types.add(AdhesiveType.GLUE)
                    elif adhesive_upper == "VELCRO":
                        adhesive_types.add(AdhesiveType.VELCRO)
                    elif adhesive_upper == "TAPE":
                        adhesive_types.add(AdhesiveType.TAPE)
                    else:
                        self.logger.warning(f"Unknown adhesive type: {adhesive_str}")
                except ValueError:
                    self.logger.warning(f"Unknown adhesive type: {adhesive_str}")

            # Add adhesive_types to the tool configuration
            tool_config["adhesive_types"] = list(adhesive_types)

        # Substitute environment variables in config
        processed_config = self._substitute_env_vars(tool_config)

        # Get the tool class from the registry
        tool_class = get_tool_class(tool_type)
        if not tool_class:
            raise ValueError(f"Unknown tool type: {tool_type}")

        self.logger.info(f"Creating tool: {name} (type: {tool_type})")
        return tool_class(
            name=name, description=tool_description, config=processed_config
        )

    def _create_team(self, config: Dict[str, Any]) -> Team:
        """Create a team from configuration.

        Args:
            config: Team configuration

        Returns:
            Instantiated Team

        Raises:
            ValueError: If referenced model or tools are not found
        """
        name = config.get("name")
        model_name = config.get("model")
        tool_names = config.get("tools", [])
        team_config = config.get("config", {})

        # Get the model
        if model_name not in self._models:
            raise ValueError(f"Referenced model not found: {model_name}")
        model = self._models[model_name]

        # Substitute environment variables in config
        processed_config = self._substitute_env_vars(team_config)

        self.logger.info(f"Creating team: {name}")
        # Create the team with just the lead model
        team = Team(name=name, lead=model, config=processed_config)

        # Add tools to the team after creation
        for tool_name in tool_names:
            if tool_name not in self._tools:
                raise ValueError(f"Referenced tool not found: {tool_name}")
            # Use asyncio.run to call the async add_tool method synchronously
            # This is needed because we're in a synchronous context
            asyncio.run(
                team.add_tool(
                    name=tool_name,
                    tool=self._tools[tool_name],
                    binding=AdhesiveType.VELCRO,  # Default binding
                )
            )

        return team

    def _create_flows(self, flow_configs: List[Dict[str, Any]]) -> List[Flow]:
        """Create flows between teams.

        Args:
            flow_configs: List of flow configurations

        Returns:
            List of instantiated Flows

        Raises:
            ValueError: If referenced teams are not found
        """
        flows = []

        for config in flow_configs:
            source_name = config.get("source")
            target_name = config.get("target")
            flow_type_str = config.get("type", "bidirectional")

            # Validate team references
            if source_name not in self._teams:
                raise ValueError(f"Referenced team not found: {source_name}")
            if target_name not in self._teams:
                raise ValueError(f"Referenced team not found: {target_name}")

            # Get flow type
            try:
                # Convert to uppercase for enum matching but preserve the original value
                # for direct string comparison with enum values
                flow_upper = flow_type_str.upper()
                if flow_upper == "PUSH":
                    flow_type = FlowType.PUSH
                elif flow_upper == "PULL":
                    flow_type = FlowType.PULL
                elif flow_upper == "BIDIRECTIONAL":
                    flow_type = FlowType.BIDIRECTIONAL
                else:
                    self.logger.warning(
                        f"Unknown flow type: {flow_type_str}, defaulting to BIDIRECTIONAL"
                    )
                    flow_type = FlowType.BIDIRECTIONAL
            except ValueError:
                self.logger.warning(
                    f"Unknown flow type: {flow_type_str}, defaulting to BIDIRECTIONAL"
                )
                flow_type = FlowType.BIDIRECTIONAL

            # Create the flow
            source = self._teams[source_name]
            target = self._teams[target_name]

            self.logger.info(
                f"Creating flow: {source_name} -> {target_name} ({flow_type_str})"
            )
            flow = Flow(source=source, target=target, flow_type=flow_type)
            flows.append(flow)

        return flows

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment variables substituted
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self._substitute_env_vars(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._substitute_env_vars(item)
                        if isinstance(item, dict)
                        else (
                            self._substitute_env_var_in_string(item)
                            if isinstance(item, str)
                            else item
                        )
                    )
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = self._substitute_env_var_in_string(value)
            else:
                result[key] = value

        return result

    def _substitute_env_var_in_string(self, value: str) -> str:
        """Substitute environment variables in a string.

        Args:
            value: String that may contain environment variable references

        Returns:
            String with environment variables substituted
        """
        # Match ${VAR_NAME} pattern
        pattern = r"\${([A-Za-z0-9_]+)}"

        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_env_var, value)
