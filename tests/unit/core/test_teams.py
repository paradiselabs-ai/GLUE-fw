import pytest
import asyncio
# Removed: from unittest.mock import MagicMock, AsyncMock

from glue.core.teams import GlueTeam
from agno.models.openai import OpenAIChat
from agno.models.anthropic.claude import Claude
from glue.core.model import Model
# Removed: from glue.core.tool import Tool
from agno.team import Team as AgnoTeam
from agno.agent import Agent as AgnoAgent
from agno.run.team import TeamRunResponse # Added for new test
# Removed: from agno.tools.tool import Tool as AgnoTool


@pytest.fixture
def basic_model_instance():
    """Provides a basic, real Model instance using TestProvider."""
    model_config = {
        "name": "test_lead_model",
        "provider": "test", # Use 'test' to match new TestProvider setup
        "model_name": "test_model_variant_1"
    }
    model = Model(config=model_config)
    return model

@pytest.fixture
def basic_member_model_instance(basic_model_instance: Model) -> Model: # Renamed and returns Model
    """Provides a basic, real Model instance for a team member."""
    model_config = {
        "name": "TestAgentMember_model",
        "provider": "test",
        "model_name": "test_model_variant_member",
        "system_prompt": "You are a test member model."
    }
    model = Model(config=model_config)
    return model

@pytest.fixture
def basic_glue_team(basic_model_instance: Model):
    """Provides a basic GlueTeam instance for testing with a real lead model."""
    team = GlueTeam(
        name="TestGlueTeam",
        lead=basic_model_instance,
        # config will default to None, members and tools will be empty lists by default
        use_agno_team=True # Ensure this is true for the test's purpose
    )
    return team

class TestGlueTeam:
    @pytest.mark.asyncio
    async def test_initialize_agno_team_creates_agno_team_instance(self, basic_glue_team: GlueTeam, basic_member_model_instance: Model):
        """
        Tests that _initialize_agno_team correctly creates an instance of agno.team.Team
        and assigns it to the agno_team attribute, using real instances.
        """
        # Add a member to the team using a real agent instance
        await basic_glue_team.add_member(basic_member_model_instance)

        # Call the method to be tested (assuming it's synchronous for now)
        # _initialize_agno_team is called internally by run or when use_agno_team is set
        # For this test, we might need to call it directly if it's intended to be callable,
        # or trigger it via team.run() if that's the designed path.
        # Given it's a "private" method, direct call for unit testing is okay.
        basic_glue_team._initialize_agno_team()

        # Assert that an AgnoTeam instance was created and assigned
        assert basic_glue_team.agno_team is not None, "agno_team attribute was not set"
        assert isinstance(basic_glue_team.agno_team, AgnoTeam), \
            f"Expected agno_team to be AgnoTeam, got {type(basic_glue_team.agno_team)}"

    @pytest.mark.asyncio
    async def test_initialize_agno_team_creates_agno_agents_for_all_glue_models(self, basic_glue_team: GlueTeam, basic_member_model_instance: Model):
        """
        Tests that _initialize_agno_team creates AgnoAgent instances for all
        GLUE Model members in the team.
        """
        # The basic_glue_team fixture already has a lead model.
        # Add another member (basic_agent_instance.model) to the team.
        # The model name for basic_member_model_instance is 'TestAgentMember_model'
        await basic_glue_team.add_member(basic_member_model_instance) 

        # There should now be two models in basic_glue_team.models:
        # 1. 'test_lead_model' (the lead)
        # 2. 'TestAgentMember_model' (the member we just added)
        assert len(basic_glue_team.models) == 2, \
            f"Expected 2 models in GlueTeam, found {len(basic_glue_team.models)}"

        basic_glue_team._initialize_agno_team()

        assert basic_glue_team.agno_team is not None, "agno_team was not set after initialization"
        assert isinstance(basic_glue_team.agno_team, AgnoTeam), "agno_team is not an AgnoTeam instance"

        # Assuming AgnoTeam stores its agents in an attribute called 'members'
        # This might need adjustment based on Agno's actual API
        assert hasattr(basic_glue_team.agno_team, 'members'), "AgnoTeam instance does not have a 'members' attribute"
        
        agno_team_members = basic_glue_team.agno_team.members
        assert isinstance(agno_team_members, list), "AgnoTeam.members is not a list"
        assert len(agno_team_members) == 2, \
            f"Expected 2 AgnoAgents in AgnoTeam, found {len(agno_team_members)}"

        created_agno_agent_names = sorted([agent.name for agent in agno_team_members if hasattr(agent, 'name')])
        
        # Expected names are derived from the GLUE Model names
        expected_glue_model_names = sorted(basic_glue_team.models.keys())
        expected_agno_agent_names = sorted([f"{name}_agno_proxy" for name in expected_glue_model_names])

        assert created_agno_agent_names == expected_agno_agent_names, \
            f"AgnoAgent names do not match expected. Got: {created_agno_agent_names}, Expected: {expected_agno_agent_names}"

    @pytest.mark.asyncio
    async def test_initialize_agno_team_maps_system_prompts(self, basic_glue_team: GlueTeam):
        """
        Tests that _initialize_agno_team correctly maps GLUE Model system prompts
        to Agno Agent system_message parameters.
        """
        # Create a model with a specific system prompt
        test_system_prompt = "You are a specialized test agent with custom instructions."
        model_config = {
            "name": "test_model_with_prompt",
            "provider": "test",
            "model_name": "test_model_variant_1",
            "system_prompt": test_system_prompt
        }
        model_with_prompt = Model(config=model_config)
        
        # Add the model to the team
        await basic_glue_team.add_member(model_with_prompt)
        
        # Initialize the Agno team
        basic_glue_team._initialize_agno_team()
        
        # Verify the Agno team was created
        assert basic_glue_team.agno_team is not None
        assert hasattr(basic_glue_team.agno_team, 'members')
        
        # Find the agent corresponding to our test model
        test_agent = None
        for agent in basic_glue_team.agno_team.members:
            if agent.name == f"{model_with_prompt.name}_agno_proxy":
                test_agent = agent
                break
        
        assert test_agent is not None, f"Agent for {model_with_prompt.name} not found in Agno team"
        # Check for system_message attribute first (this is the main system message in Agno)
        assert hasattr(test_agent, 'system_message'), "Agno Agent does not have system_message attribute"
        assert test_agent.system_message == test_system_prompt, "System prompt was not correctly mapped to system_message"

    @pytest.mark.asyncio
    async def test_initialize_agno_team_maps_llm_configuration(self, basic_glue_team: GlueTeam):
        """
        Tests that _initialize_agno_team correctly maps GLUE Model LLM configuration
        (model_name, temperature, max_tokens) to Agno Agent model parameters.
        """
        # Create a model with specific LLM configuration
        model_config = {
            "name": "test_model_with_llm_config",
            "provider": "test",
            "model_name": "gpt-4",
            "temperature": 0.8,
            "max_tokens": 2048
        }
        model_with_config = Model(config=model_config)
        
        # Add the model to the team
        await basic_glue_team.add_member(model_with_config)
        
        # Initialize the Agno team
        basic_glue_team._initialize_agno_team()
        
        # Verify the Agno team was created
        assert basic_glue_team.agno_team is not None
        assert hasattr(basic_glue_team.agno_team, 'members')
        
        # Find the agent corresponding to our test model
        test_agent = None
        for agent in basic_glue_team.agno_team.members:
            if agent.name == f"{model_with_config.name}_agno_proxy":
                test_agent = agent
                break
        
        assert test_agent is not None, f"Agent for {model_with_config.name} not found in Agno team"
        
        # Check that the agent has a model attribute and it's properly configured
        assert hasattr(test_agent, 'model'), "Agno Agent does not have model attribute"
        assert test_agent.model is not None, "Agno Agent model should not be None - LLM configuration was not mapped"
        
        # Check the model configuration attributes
        assert hasattr(test_agent.model, 'model'), "Agno Agent model does not have 'model' attribute"
        assert test_agent.model.model == "gpt-4", f"Model name was not correctly mapped. Expected 'gpt-4', got {test_agent.model.model}"
        
        assert hasattr(test_agent.model, 'temperature'), "Agno Agent model does not have 'temperature' attribute"
        assert test_agent.model.temperature == 0.8, f"Temperature was not correctly mapped. Expected 0.8, got {test_agent.model.temperature}"
        
        assert hasattr(test_agent.model, 'max_tokens'), "Agno Agent model does not have 'max_tokens' attribute"
        assert test_agent.model.max_tokens == 2048, f"Max tokens was not correctly mapped. Expected 2048, got {test_agent.model.max_tokens}"

    @pytest.mark.asyncio
    async def test_glue_team_run_delegates_to_agno_team(self, basic_glue_team: GlueTeam, basic_member_model_instance: Model):
        """
        Tests that GlueTeam.run() correctly delegates to AgnoTeam.run() when use_agno_team is True,
        and returns an AgnoTeamRunResponse.
        This test uses real Agno components without mocking LLM calls, focusing on the interaction.
        """
        # Ensure the team is configured to use AgnoTeam (already done in basic_glue_team fixture)
        assert basic_glue_team.use_agno_team is True

        # Add a member model to the team. This will also trigger _initialize_agno_team if not already done,
        # or run() will do it.
        # The basic_glue_team fixture already has a lead model.
        await basic_glue_team.add_member(basic_member_model_instance)
        
        # Ensure _initialize_agno_team has been called or will be called by run()
        # For this test, let's explicitly initialize to separate concerns, though run() would do it.
        if not basic_glue_team.agno_team:
            basic_glue_team._initialize_agno_team()

        assert basic_glue_team.agno_team is not None, "AgnoTeam was not initialized before run."
        actual_team_instance = basic_glue_team.agno_team
        expected_team_type = AgnoTeam # This is from 'from agno.team import Team as AgnoTeam' in the test file

        error_msg = (
            f"basic_glue_team.agno_team is not an AgnoTeam instance.\n"
            f"Actual type: {type(actual_team_instance)}\n"
            f"Actual module: {getattr(type(actual_team_instance), '__module__', 'N/A')}.{getattr(type(actual_team_instance), '__name__', 'N/A')}\n"
            f"Expected type: {expected_team_type}\n"
            f"Expected module: {getattr(expected_team_type, '__module__', 'N/A')}.{getattr(expected_team_type, '__name__', 'N/A')}\n"
            f"Are they the same class object? {type(actual_team_instance) is expected_team_type}"
        )
        assert isinstance(actual_team_instance, expected_team_type), error_msg

        initial_input = "Test input for Agno team execution. Please echo this back."
        
        # Run the team
        # GlueTeam.run() is synchronous and calls AgnoTeam.run() (sync wrapper)
        try:
            response = basic_glue_team.run(initial_input=initial_input)
        except Exception as e:
            pytest.fail(f"GlueTeam.run() raised an unexpected exception during AgnoTeam delegation: {e}")

        # Assertions
        assert response is not None, "GlueTeam.run() returned None when expecting AgnoTeamRunResponse."
        assert isinstance(response, TeamRunResponse), \
            f"Expected response to be TeamRunResponse, got {type(response)}"

        # Further assertions based on TeamRunResponse structure (if known and testable without LLM mock)
        # For example, check if an ID or some status is set.
        # These depend on the actual structure of TeamRunResponse and AgnoTeam.run() behavior.
        assert hasattr(response, 'run_id'), "TeamRunResponse does not have 'run_id' attribute."
        assert response.run_id is not None, "TeamRunResponse.run_id was None."
        
        assert hasattr(response, 'output_text'), "TeamRunResponse does not have 'output_text' attribute."
        # We cannot reliably assert the content of output_text without mocking the LLM
        # or having a very deterministic test model in Agno.
        # For now, just checking it exists and is perhaps not None if that's expected.
        # If the 'test' provider in Agno simply echoes, we could assert that.
        # logger.info(f"Agno Team Run Response Output: {response.output_text}") # For debugging
        # For a 'test' provider that might echo, you could try:
        # assert initial_input in response.output_text, "Input not found in Agno team output_text"
        # This is commented out as it depends on specific 'test' provider behavior in Agno.

    @pytest.mark.asyncio
    async def test_initialize_agno_team_maps_anthropic_llm_configuration(self, basic_glue_team: GlueTeam):
        # basic_glue_team already has a lead model.
        anthropic_model_name = "claude-3-opus-20240229"
        anthropic_temp = 0.7
        anthropic_max_tokens = 4000

        model_config_anthropic = {
            "name": "test_model_anthropic",
            "provider": "anthropic",
            "model_name": anthropic_model_name,
            "temperature": anthropic_temp,
            "max_tokens": anthropic_max_tokens,
            "system_prompt": "You are an Anthropic test model."
        }
        anthropic_glue_model = Model(config=model_config_anthropic)
        basic_glue_team.add_member_sync(anthropic_glue_model)

        basic_glue_team._initialize_agno_team() # Method under test

        assert basic_glue_team.agno_team is not None
        # The basic_glue_team fixture adds one lead model. We add one more.
        assert len(basic_glue_team.agno_team.members) == 2, \
            f"Expected 2 Agno agents, got {len(basic_glue_team.agno_team.members)}"

        target_agno_agent_name = f"{anthropic_glue_model.name}_agno_proxy"
        agno_agent_for_anthropic = next(
            (agent for agent in basic_glue_team.agno_team.members if agent.name == target_agno_agent_name),
            None
        )

        assert agno_agent_for_anthropic is not None, f"AgnoAgent for {target_agno_agent_name} not found"
        assert agno_agent_for_anthropic.model is not None, "AgnoAgent should have a .model object for Anthropic config"

        assert isinstance(agno_agent_for_anthropic.model, Claude), \
            f"Model is {type(agno_agent_for_anthropic.model)}, expected AnthropicChat"

        # Check attributes of the Agno AnthropicChat model object
        # These assertions might need adjustment based on AnthropicChat's actual attribute names
        actual_model_name_attr = getattr(agno_agent_for_anthropic.model, 'model', getattr(agno_agent_for_anthropic.model, 'id', None))
        assert actual_model_name_attr == anthropic_model_name, \
            f"Model name is {actual_model_name_attr}, expected {anthropic_model_name}"
        assert getattr(agno_agent_for_anthropic.model, 'temperature', None) == anthropic_temp, \
            f"Temperature is {getattr(agno_agent_for_anthropic.model, 'temperature', None)}, expected {anthropic_temp}"
        assert getattr(agno_agent_for_anthropic.model, 'max_tokens', None) == anthropic_max_tokens, \
            f"Max tokens is {getattr(agno_agent_for_anthropic.model, 'max_tokens', None)}, expected {anthropic_max_tokens}"

    # TODO: Add more tests for GlueTeam functionality, e.g.:
    # - test_add_member
    # - test_run_with_agno_team
    # - test_run_native_glue_loop (if applicable as fallback)
    # - test_tool_usage_and_sharing
    # - test_adhesive_interaction
    # - test_fetch_task_for_member (if still relevant after Agno integration)

import sys
import os
import importlib
import pytest

class TestImportDiagnostics:
    def test_check_sys_path_and_imports(self):
        print("\n--- sys.path ---")
        for p in sys.path:
            print(p)
        print("--- end sys.path ---\n")

        print(f"Current working directory: {os.getcwd()}")

        # Path that conftest.py is expected to add
        # (Calculated from tests/unit/core/test_teams.py to project_root/src)
        expected_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        print(f"Path expected to be added by conftest.py (project_root/src): {expected_src_dir}")

        actual_src_in_path = False
        for p_idx, p_val in enumerate(sys.path):
            if os.path.abspath(p_val) == expected_src_dir:
                print(f"Found '{expected_src_dir}' at sys.path[{p_idx}]")
                actual_src_in_path = True
                break
        
        if actual_src_in_path:
            print(f"The directory '{expected_src_dir}' IS in sys.path.")
            
            glue_package_dir = os.path.join(expected_src_dir, 'glue')
            print(f"Checking for glue package directory: '{glue_package_dir}'")
            print(f"  Is directory? {os.path.isdir(glue_package_dir)}")
            
            glue_init_path = os.path.join(glue_package_dir, '__init__.py')
            print(f"Checking for glue package init file: '{glue_init_path}'")
            print(f"  Exists? {os.path.exists(glue_init_path)}")

            glue_utils_dir = os.path.join(glue_package_dir, 'utils')
            print(f"Checking for glue.utils package directory: '{glue_utils_dir}'")
            print(f"  Is directory? {os.path.isdir(glue_utils_dir)}")

            glue_utils_init_path = os.path.join(glue_utils_dir, '__init__.py')
            print(f"Checking for glue.utils package init file: '{glue_utils_init_path}'")
            print(f"  Exists? {os.path.exists(glue_utils_init_path)}")

            glue_utils_logger_path = os.path.join(glue_utils_dir, 'logger.py')
            print(f"Checking for glue.utils.logger module file: '{glue_utils_logger_path}'")
            print(f"  Exists? {os.path.exists(glue_utils_logger_path)}")

        else:
            print(f"CRITICAL WARNING: Expected 'src' directory ({expected_src_dir}) NOT FOUND in sys.path.")
            pytest.fail(f"CRITICAL: {expected_src_dir} not in sys.path")

        try:
            print("Attempting to import glue.utils.logger...")
            import glue.utils.logger
            print("Successfully imported glue.utils.logger")
            logger_module = glue.utils.logger
            print(f"glue.utils.logger location: {getattr(logger_module, '__file__', 'N/A')}")
        except ImportError as e:
            print(f"Failed to import glue.utils.logger: {e}")
            pytest.fail(f"ImportError for glue.utils.logger: {e}")

        try:
            print("Attempting to import glue.core.providers.test...")
            module = importlib.import_module("glue.core.providers.test")
            print("Successfully imported glue.core.providers.test")
            print(f"glue.core.providers.test location: {getattr(module, '__file__', 'N/A')}")
            if hasattr(module, 'TestProvider'):
                print("TestProvider class found in glue.core.providers.test")
            else:
                print("WARNING: TestProvider class NOT found in glue.core.providers.test")
        except ImportError as e:
            print(f"Failed to import glue.core.providers.test: {e}")
            pytest.fail(f"ImportError for glue.core.providers.test: {e}")
        except Exception as e:
            print(f"Other error importing glue.core.providers.test: {e}")
            pytest.fail(f"General error for glue.core.providers.test: {e}")

        assert True
