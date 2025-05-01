"""
Utilities for validating GLUE framework configurations.

This module contains utilities for converting ConfigGenerator output to Pydantic models
and validating the configuration.
"""

from typing import Dict, Any, List, TypeVar
from pydantic import ValidationError, BaseModel

from glue.core.schemas import (
    AppConfig,
    ModelConfig,
    ToolConfig,
    TeamConfig,
    MagnetConfig,
)

T = TypeVar("T", bound=BaseModel)


def config_to_pydantic(config: Dict[str, Any]) -> AppConfig:
    """
    Convert a configuration dictionary to Pydantic models.

    Args:
        config: Configuration dictionary from ConfigGenerator

    Returns:
        AppConfig: Validated application configuration

    Raises:
        ValueError: If the configuration is invalid
    """
    try:
        # Create a new config dictionary for the AppConfig
        app_config = {}

        # Handle the nested app configuration
        if "app" in config:
            # Extract app details from the nested structure
            app_details = config["app"]
            if "name" not in app_details:
                raise ValueError(
                    "Configuration validation failed:\napp.name: field required"
                )
            app_config["name"] = app_details.get("name", "")
            app_config["description"] = app_details.get("description", "")
            app_config["version"] = app_details.get("version", "0.1.0")

            # Extract app config if present
            if "config" in app_details:
                app_config["development"] = app_details["config"].get(
                    "development", True
                )
                app_config["log_level"] = app_details["config"].get("log_level", "info")
            else:
                app_config["development"] = True
                app_config["log_level"] = "info"
        else:
            # Handle flat configuration structure (for direct AppConfig creation)
            if "name" not in config:
                raise ValueError(
                    "Configuration validation failed:\nname: field required"
                )
            app_config["name"] = config.get("name", "")
            app_config["description"] = config.get("description", "")
            app_config["version"] = config.get("version", "0.1.0")
            app_config["development"] = config.get("development", True)
            app_config["log_level"] = config.get("log_level", "info")

        # Convert models if present
        if "models" in config and config["models"]:
            # Check that each model has a name field
            for i, model in enumerate(config["models"]):
                if "name" not in model:
                    raise ValueError(
                        f"Configuration validation failed:\nmodels.{i}.name: field required"
                    )
            app_config["models"] = [ModelConfig(**model) for model in config["models"]]
        else:
            app_config["models"] = []

        # Convert tools if present
        if "tools" in config and config["tools"]:
            app_config["tools"] = [ToolConfig(**tool) for tool in config["tools"]]
        else:
            app_config["tools"] = []

        # Convert teams if present
        if "teams" in config and config["teams"]:
            app_config["teams"] = [TeamConfig(**team) for team in config["teams"]]
        else:
            app_config["teams"] = []

        # Convert magnets if present
        if "magnets" in config and config["magnets"]:
            app_config["magnets"] = [
                MagnetConfig(**magnet) for magnet in config["magnets"]
            ]
        else:
            app_config["magnets"] = []

        # Create and validate the full AppConfig
        return AppConfig(**app_config)
    except ValidationError as e:
        # Convert Pydantic validation error to a more user-friendly error message
        error_messages = []
        for error in e.errors():
            location = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_messages.append(f"{location}: {message}")

        error_str = "\n".join(error_messages)
        raise ValueError(f"Configuration validation failed:\n{error_str}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary and return a list of error messages.

    Args:
        config: Configuration dictionary from ConfigGenerator

    Returns:
        List[str]: List of error messages, empty if the configuration is valid
    """
    try:
        config_to_pydantic(config)
        return []  # No errors
    except ValueError as e:
        return str(e).split("\n")[
            1:
        ]  # Skip the first line which is the general error message


def get_config_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for the application configuration.

    Returns:
        Dict[str, Any]: JSON schema
    """
    return AppConfig.model_json_schema()


def get_model_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for model configuration.

    Returns:
        Dict[str, Any]: JSON schema
    """
    return ModelConfig.model_json_schema()


def get_tool_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for tool configuration.

    Returns:
        Dict[str, Any]: JSON schema
    """
    return ToolConfig.model_json_schema()


def get_team_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for team configuration.

    Returns:
        Dict[str, Any]: JSON schema
    """
    return TeamConfig.model_json_schema()


def get_magnet_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for magnet configuration.

    Returns:
        Dict[str, Any]: JSON schema
    """
    return MagnetConfig.model_json_schema()
