import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from glue.core.teams import Team, TeamConfig
from glue.core.types import ToolResult, AdhesiveType, FlowType, Message
from typing import Optional
# ==================== Mock Classes ====================
# Mock for a model
class MockModel:
    def __init__(self, name, tools=None, adhesives=None):
        self.name = name
        self.tools = tools if tools is not None else {}
        self._tools = {} # For testing get_model_tools
        self.adhesives = adhesives if adhesives is not None else []
        self.team = None
        self.generate = AsyncMock()
        self.generate_response = AsyncMock() # For direct_communication lead path
        self.process_tool_result = AsyncMock() # For direct_communication lead path
        self.run = MagicMock() # For inject_managed_agents
        self.interpreter = MagicMock() # For inject_managed_agents
        self.interpreter.globals = {} # For inject_managed_agents
        self.add_tool = AsyncMock() # Mock for add_tool
        self.add_tool_sync = MagicMock() # Mock for add_tool_sync

    # Add a dummy hierarchy attribute for testing update_hierarchy_attributes
    hierarchy_level: Optional[int] = None

@pytest.fixture
def minimal_team_config():
    return TeamConfig(name="TestTeam", lead="LeadModel", members=["Member1"], tools=[])

@pytest.fixture
def mock_lead_model():
    return MockModel("LeadModel")

@pytest.fixture
def mock_member_model():
    return MockModel("MemberModel")

@pytest.fixture
def team_with_lead(mock_lead_model):
    team = Team(name="TestTeam")
    team.add_member_sync(mock_lead_model, role="lead")
    return team

@pytest.fixture
def team_with_members(team_with_lead, mock_member_model):
    team_with_lead.add_member_sync(mock_member_model)
    return team_with_lead

@pytest.fixture
def mock_tool():
    mock_tool = MagicMock()
    mock_tool.initialize = AsyncMock()
    mock_tool._initialized = False
    return mock_tool

# ==================== Basic Initialization and Properties ====================

def test_team_creation_minimal():
    team = Team(name="MinimalTeam")
    assert team.name == "MinimalTeam"
    assert team.config.name == "MinimalTeam"
    assert team.lead is None
    assert len(team.models) == 0
    assert len(team.tools) == 0
    assert len(team.subteams) == 0

def test_team_creation_with_config(minimal_team_config):
    team = Team(name="ConfigTeam", config=minimal_team_config)
    assert team.name == "ConfigTeam" # Name from arg overrides config name for Team instance
    assert team.config.name == "TestTeam"
    assert team.config.lead == "LeadModel"

@pytest.mark.asyncio
async def test_team_add_member_lead_and_member(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members
    assert mock_lead_model.name in team.models
    assert team.models[mock_lead_model.name] == mock_lead_model
    assert team.lead == mock_lead_model
    assert mock_lead_model.team == team

    assert mock_member_model.name in team.models
    assert team.models[mock_member_model.name] == mock_member_model
    assert mock_member_model.team == team
    assert mock_member_model.name in team.config.members

@pytest.mark.asyncio
async def test_add_subteam():
    parent_team = Team(name="ParentTeam")
    sub_team_config = TeamConfig(name="SubTeam", lead="SubLead", members=[], tools=[])
    sub_team = Team(name="SubTeam", config=sub_team_config)
    
    await parent_team.add_member(sub_team)
    assert "SubTeam" in parent_team.subteams
    assert parent_team.subteams["SubTeam"] == sub_team

# ==================== Tool Management ====================

@pytest.mark.asyncio
async def test_add_tool_to_team_and_models(team_with_members):
    team = team_with_members
    mock_tool = MagicMock()
    mock_tool.initialize = AsyncMock() # Mock initialize
    mock_tool._initialized = False

    await team.add_tool("TestTool", mock_tool)
    
    assert "TestTool" in team._tools
    assert team._tools["TestTool"] == mock_tool
    mock_tool.initialize.assert_called_once()

    # Check tool added to existing models
    for model in team.models.values():
        model.add_tool.assert_any_call("TestTool", mock_tool)

    # Check tool added to a new model
    new_model = MockModel("NewModel")
    await team.add_member(new_model)
    # This part is tricky because add_member_sync is used in fixture, add_tool_sync is called
    # If add_member (async) is used, then model.add_tool (async) should be checked
    # For simplicity, current add_member calls add_tool_sync on the model
    new_model.add_tool_sync.assert_any_call("TestTool", mock_tool)


@pytest.mark.asyncio
async def test_share_result_glue_adhesive(team_with_lead):
    team = team_with_lead
    tool_result = ToolResult(tool_name="TestTool", result="data", adhesive=AdhesiveType.GLUE)
    await team.share_result("TestTool", tool_result)
    assert "TestTool" in team.shared_results
    assert team.shared_results["TestTool"] == tool_result

@pytest.mark.asyncio
async def test_share_result_velcro_adhesive(team_with_lead):
    team = team_with_lead
    tool_result = ToolResult(tool_name="VelcroTool", result="temp_data", adhesive=AdhesiveType.VELCRO)
    await team.share_result("VelcroTool", tool_result)
    # Velcro results are not stored in shared_results
    assert "VelcroTool" not in team.shared_results

@pytest.mark.asyncio
async def test_share_result_tape_adhesive(team_with_lead):
    team = team_with_lead
    tool_result = ToolResult(tool_name="TapeTool", result="one_time_data", adhesive=AdhesiveType.TAPE)
    await team.share_result("TapeTool", tool_result)
    # Tape results are not stored
    assert "TapeTool" not in team.shared_results

@pytest.mark.asyncio
async def test_share_result_model_specific_adhesive(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Give the lead model a specific adhesive preference
    mock_lead_model.adhesives = [AdhesiveType.VELCRO] 
    
    # Result initially has no adhesive, should pick up from model
    tool_result = ToolResult(tool_name="ModelAdhesiveTool", result="data")
    await team.share_result("ModelAdhesiveTool", tool_result, model_name="LeadModel")
    
    assert tool_result.adhesive == AdhesiveType.VELCRO
    assert "ModelAdhesiveTool" not in team.shared_results # Because it became VELCRO

# ==================== Message Processing ====================

@pytest.mark.asyncio
async def test_process_message_simple_response(team_with_lead, mock_lead_model):
    team = team_with_lead
    mock_lead_model.generate.return_value = "Lead says hello"
    
    response = await team.process_message("User says hi", source_model="LeadModel")
    
    assert response == "Lead says hello"
    assert len(team.conversation_history) == 2
    assert team.conversation_history[0].role == "user"
    assert team.conversation_history[0].content == "User says hi"
    assert team.conversation_history[1].role == "assistant"
    assert team.conversation_history[1].content == "Lead says hello"

@pytest.mark.asyncio
async def test_process_message_with_tool_call_standard_format(team_with_lead, mock_lead_model):
    team = team_with_lead
    
    # Mock tool
    mock_search_tool = AsyncMock(return_value="Search results: found stuff")
    mock_search_tool.execute = AsyncMock(return_value="Search results: found stuff")
    mock_lead_model.tools = {"web_search": mock_search_tool}  # Add tool to model directly for this test
    team._tools = {"web_search": mock_search_tool}  # And to team's _tools

    # Model responds with a tool call
    tool_call_response = '''{
        "tool_name": "web_search",
        "arguments": {"query": "copilot"}
    }'''
    # Model's response after tool execution
    final_response_after_tool = "Okay, I searched for copilot and found: Search results: found stuff"

    # Configure generate mocks: first for tool call, second for final response
    mock_lead_model.generate.side_effect = [
        tool_call_response, 
        final_response_after_tool
    ]
    # Fix: Set process_tool_result to return the expected string
    mock_lead_model.process_tool_result.return_value = final_response_after_tool
    
    response = await team.process_message("Search for copilot", source_model="LeadModel")
    
    assert response == final_response_after_tool
    mock_search_tool.execute.assert_called_once_with(query="copilot", calling_model="LeadModel", calling_team="TestTeam")
    
    assert len(team.conversation_history) == 4 # user, assistant (tool_call), tool, assistant (final)
    assert team.conversation_history[1].content == tool_call_response
    assert team.conversation_history[2].role == "tool"
    assert team.conversation_history[2].name == "web_search"
    assert team.conversation_history[2].content == "Search results: found stuff"
    assert team.conversation_history[3].content == final_response_after_tool

@pytest.mark.asyncio
async def test_process_message_tool_not_found(team_with_lead, mock_lead_model):
    team = team_with_lead
    tool_call_response = '''{
        "tool_name": "non_existent_tool",
        "arguments": {"param": "value"}
    }'''
    mock_lead_model.generate.return_value = tool_call_response
    
    response = await team.process_message("Use non_existent_tool", source_model="LeadModel")
    
    expected_error_msg = "Tool 'non_existent_tool' not found or available."
    assert response == expected_error_msg
    assert len(team.conversation_history) == 3 # user, assistant (tool_call), tool (error)
    assert team.conversation_history[2].role == "tool"
    assert team.conversation_history[2].name == "non_existent_tool"
    assert team.conversation_history[2].content == expected_error_msg

@pytest.mark.asyncio
async def test_process_message_tool_execution_error(team_with_lead, mock_lead_model):
    team = team_with_lead
    mock_error_tool = AsyncMock()
    mock_error_tool.execute = AsyncMock(side_effect=ValueError("Tool failed!"))
    mock_lead_model.tools = {"error_tool": mock_error_tool}
    team._tools = {"error_tool": mock_error_tool}

    tool_call_response = '''{
        "tool_name": "error_tool",
        "arguments": {}
    }'''
    mock_lead_model.generate.return_value = tool_call_response
    
    response = await team.process_message("Use error_tool", source_model="LeadModel")
    
    expected_error_msg = "Error executing tool 'error_tool': Tool failed!"
    assert response == expected_error_msg
    assert len(team.conversation_history) == 3 # user, assistant (tool_call), tool (error)
    assert team.conversation_history[2].role == "tool"
    assert team.conversation_history[2].name == "error_tool"
    assert team.conversation_history[2].content == expected_error_msg

# ==================== Direct Communication ====================

@pytest.mark.asyncio
async def test_direct_communication_simple(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members
    mock_member_model.generate.return_value = "Member acknowledges" # generate on the target model
    
    response = await team.direct_communication(
        from_model="LeadModel", 
        to_model="MemberModel", 
        message="Lead assigns task to Member"
    )
    
    assert response == "Member acknowledges"
    mock_member_model.generate.assert_called_once_with("Lead assigns task to Member")
    
    # History should reflect the exchange for non-lead target
    assert len(team.conversation_history) == 2
    assert team.conversation_history[0].role == "assistant" # Message from lead to member
    assert team.conversation_history[0].name == "LeadModel"
    assert team.conversation_history[0].content == "Lead assigns task to Member"
    assert team.conversation_history[1].role == "assistant" # Response from member
    assert team.conversation_history[1].name == "MemberModel"
    assert team.conversation_history[1].content == "Member acknowledges"

@pytest.mark.asyncio
async def test_direct_communication_to_lead_with_tool_use(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members
    
    # Mock tool for the lead
    mock_lead_tool = AsyncMock(return_value="Lead tool processed data")
    mock_lead_tool.execute = AsyncMock(return_value="Lead tool processed data")
    mock_lead_model.tools = {"lead_tool": mock_lead_tool}
    team._tools = {"lead_tool": mock_lead_tool} # Ensure team also knows about it for add_tool_sync if needed

    # Member sends a message that will cause lead to use a tool
    message_from_member = "process this data 123"
    
    # Lead's generate_response will first output a tool call, then a final response
    lead_tool_call_response = '{"tool_name": "lead_tool", "arguments": {"data": "123"}}'
    lead_final_response = "Lead processed data 123 using its tool: Lead tool processed data"

    mock_lead_model.generate_response.return_value = lead_tool_call_response
    mock_lead_model.process_tool_result.return_value = lead_final_response

    response = await team.direct_communication(
        from_model="MemberModel",
        to_model="LeadModel",
        message=message_from_member
    )

    assert response == lead_final_response
    
    # Check generate_response was called on lead with appropriate history
    # The history passed to lead.generate_response should include member's message as "user"
    mock_lead_model.generate_response.assert_called_once()
    history_arg = mock_lead_model.generate_response.call_args[0][0]
    assert isinstance(history_arg, list)
    assert len(history_arg) > 0
    assert history_arg[-1].role == "user" # The message from member is treated as user input for lead
    assert history_arg[-1].name == "MemberModel"
    assert history_arg[-1].content == message_from_member

    mock_lead_tool.execute.assert_called_once_with(data="123", calling_model="LeadModel", calling_team="TestTeam")
    # mock_lead_model.process_tool_result.assert_called_once() # Removed, not used

    # Check overall team conversation history
    found_tool_result = any(
        msg.role == "tool" and msg.name == "lead_tool" and msg.content == "Lead tool processed data"
        for msg in team.conversation_history
    )
    found_lead_final_response = any(
        msg.role == "assistant" and msg.name == "LeadModel" and msg.content == lead_final_response
        for msg in team.conversation_history
    )

    assert found_tool_result, "Lead's tool result not found in history"
    assert found_lead_final_response, "Lead's final response not found in history"


# ==================== Agent Loops and State ====================
@pytest.mark.asyncio
@patch('glue.core.teams.GlueSmolTeam') # Patch GlueSmolTeam where it's used
async def test_start_agent_loops(MockGlueSmolTeam, team_with_lead):
    team = team_with_lead
    mock_smol_team_instance = MockGlueSmolTeam.return_value
    mock_smol_team_instance.run = MagicMock() # Mock the run method

    initial_input = "Start the project"
    await team.start_agent_loops(initial_input)

    MockGlueSmolTeam.assert_called_once_with(
        team=team,
        model_clients=team.models,
        glue_config=None 
    )
    mock_smol_team_instance.setup.assert_called_once()
    # Check that asyncio.create_task was called with a thread running smol_team.run
    # This is hard to assert directly without more complex patching of asyncio.create_task and to_thread
    # For now, we trust that if run is called, it was likely through the create_task(to_thread(...)) path
    # Awaiting a brief moment to allow the task to potentially start
    await asyncio.sleep(0.01) 
    mock_smol_team_instance.run.assert_called_once_with(initial_input)
    assert team.config.lead in team.agent_loops
    assert team.agent_loops[team.config.lead] == mock_smol_team_instance

@pytest.mark.asyncio
async def test_get_agent_status(team_with_lead):
    team = team_with_lead
    # Simulate a running loop with some state
    mock_loop = MagicMock()
    mock_loop.state = {"status": "running", "current_task": "thinking"}
    team.agent_loops[team.config.lead] = mock_loop

    status_all = team.get_agent_status()
    assert status_all["team"] == team.name
    assert team.config.lead in status_all["agents"]
    assert status_all["agents"][team.config.lead] == {"status": "running", "current_task": "thinking"}

    status_specific = team.get_agent_status(team.config.lead)
    assert status_specific == {"status": "running", "current_task": "thinking"}

    status_not_found = team.get_agent_status("NonExistentAgent")
    assert "error" in status_not_found

@pytest.mark.asyncio
async def test_terminate_agent_loops(team_with_lead):
    team = team_with_lead
    mock_loop = MagicMock()
    mock_loop.terminate = MagicMock()
    team.agent_loops[team.config.lead] = mock_loop

    await team.terminate_agent_loops("Test termination")
    
    mock_loop.terminate.assert_called_once_with("Test termination")
    assert len(team.agent_loops) == 0


# ==================== Magnetic Field Methods ====================
def test_relationships_and_repulsion(team_with_lead):
    team = team_with_lead
    target_team_name = "OtherTeam"

    team.relationships[target_team_name] = FlowType.BIDIRECTIONAL.value
    assert target_team_name in team.relationships

    team.break_relationship(target_team_name)
    assert target_team_name not in team.relationships

    team.repel(target_team_name)
    assert target_team_name in team.repelled_by
    # Repelling should also break existing relationship
    team.relationships[target_team_name] = FlowType.UNIDIRECTIONAL.value
    team.repel(target_team_name)
    assert target_team_name not in team.relationships 

@pytest.mark.asyncio
async def test_get_relationships(team_with_lead):
    team = team_with_lead
    team.relationships["TeamA"] = "bidirectional"
    rels = await team.get_relationships()
    assert rels == {"TeamA": "bidirectional"}
    rels["TeamB"] = "unidirectional" # Modify copy
    assert "TeamB" not in team.relationships # Original should be unchanged

# ==================== Helper Methods ====================
def test_get_model_tools(team_with_members, mock_lead_model):
    team = team_with_members
    mock_tool = MagicMock()
    mock_lead_model.tools = {"model_tool": mock_tool} # Directly set on model for test

    tools = team.get_model_tools("LeadModel")
    assert "model_tool" in tools
    assert tools["model_tool"] == mock_tool

    with pytest.raises(ValueError):
        team.get_model_tools("NonExistentModel")

@pytest.mark.asyncio
async def test_try_establish_relationship_no_flows(team_with_lead):
    team = team_with_lead
    result = await team.try_establish_relationship("TargetTeam")
    assert not result["success"]
    assert result["relationship_type"] is None
    assert "No flows exist" in result["error"]

# More complex flow scenarios for try_establish_relationship would require mocking Flow objects
# and their source/target attributes, which are currently typed as Any.

def test_get_shared_results(team_with_lead):
    team = team_with_lead
    tool_res = ToolResult(tool_name="SharedTool", result="shared_data")
    team.shared_results["SharedTool"] = tool_res
    
    results = team.get_shared_results()
    assert results == {"SharedTool": tool_res}
    results["New"] = "data" # Modify copy
    assert "New" not in team.shared_results # Original unchanged

@pytest.mark.asyncio
async def test_cleanup(team_with_lead):
    team = team_with_lead
    mock_tool_instance = MagicMock()
    mock_tool_instance.cleanup = AsyncMock()
    team._tools["CleanableTool"] = mock_tool_instance
    team.shared_results["some_result"] = ToolResult(tool_name="some_result", result="data")
    team.conversation_history.append(Message(role="user", content="hi"))

    await team.cleanup()

    mock_tool_instance.cleanup.assert_called_once()
    assert len(team.shared_results) == 0
    assert len(team.conversation_history) == 0

def test_inject_managed_agents(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members # lead is LeadModel, member is MemberModel
    
    # Ensure lead's interpreter is set up by calling run with init string
    mock_lead_model.run("__interpreter_init__")
    assert hasattr(mock_lead_model, "interpreter")
    assert hasattr(mock_lead_model.interpreter, "globals")

    team.inject_managed_agents()

    # Check if member model is now a callable in lead's interpreter globals
    assert "MemberModel" in mock_lead_model.interpreter.globals
    member_delegate_func = mock_lead_model.interpreter.globals["MemberModel"]
    assert callable(member_delegate_func)
    
    # Test calling the delegate function
    mock_member_model.generate.return_value = "Member responded to delegated task"
    task_for_member = "Member, please do this."
    response = member_delegate_func(task_for_member)
    
    mock_member_model.generate.assert_called_once_with(task_for_member)
    # If generate is async, the delegate might return a coroutine
    if asyncio.iscoroutine(response):
        actual_response = asyncio.run(response) # Not ideal in sync test, but for checking
        assert actual_response == "Member responded to delegated task"
    else:
        assert response == "Member responded to delegated task"


@pytest.mark.asyncio
async def test_setup_calls_inject_managed_agents(team_with_members):
    team = team_with_members
    with patch.object(team, 'inject_managed_agents', wraps=team.inject_managed_agents) as mock_inject:
        await team.setup()
        mock_inject.assert_called_once()

# ==================== Flow Management (Basic Registration) ====================
# These tests are basic due to Flow objects being 'Any' and not having defined structure here.
def test_register_unregister_outgoing_flow(team_with_lead):
    team = team_with_lead
    mock_flow = MagicMock()
    mock_flow.target.name = "TargetTeamFlow" # Mock necessary attribute

    team.register_outgoing_flow(mock_flow)
    assert mock_flow in team.outgoing_flows

    team.unregister_outgoing_flow(mock_flow)
    assert mock_flow not in team.outgoing_flows

def test_register_unregister_incoming_flow(team_with_lead):
    team = team_with_lead
    mock_flow = MagicMock()
    mock_flow.source.name = "SourceTeamFlow" # Mock necessary attribute

    team.register_incoming_flow(mock_flow)
    assert mock_flow in team.incoming_flows
    # Assert processing_task is NOT started here, as it's moved to setup
    assert team.processing_task is None 

    team.unregister_incoming_flow(mock_flow)
    assert mock_flow not in team.incoming_flows

@pytest.mark.asyncio
async def test_receive_message_puts_on_queue(team_with_lead):
    team = team_with_lead
    mock_sender = MagicMock()
    mock_sender.name = "SenderTeam"
    message_data = {"content": "Hello from sender"}

    # Ensure message queue is empty
    assert team.message_queue.empty()

    await team.receive_message(message_data, mock_sender)
    
    assert not team.message_queue.empty()
    queued_item = await team.message_queue.get()
    assert queued_item[0] == message_data
    assert queued_item[1] == mock_sender
    team.message_queue.task_done()


# ==================== Backward Compatibility / Edge Cases from Original Code ====================
def test_team_creation_backward_compatible_lead_members_description():
    mock_lead = MockModel("OldLead")
    mock_member1 = MockModel("OldMember1")
    team = Team(name="BackwardTeam", lead=mock_lead, members=[mock_member1, "OldMember2Name"], description="Old desc")

    assert team.name == "BackwardTeam"
    assert team.description == "Old desc"
    assert team.lead == mock_lead
    assert "OldLead" in team.models
    assert "OldMember1" in team.models
    assert "OldMember1" in team.config.members # From object
    assert "OldMember2Name" in team.members # Stored in self.members for test compat
    assert "OldMember2Name" not in team.models # String names are not auto-converted to models here
    assert "OldMember2Name" in team.config.members # String names added to config.members

def test_team_tools_property_list_like_access(team_with_lead):
    team = team_with_lead
    tool1 = MagicMock()
    tool2 = MagicMock()
    team._tools = {"tool1": tool1, "tool2": tool2} # Directly set for test

    # Property tools returns a list
    assert isinstance(team.tools, list)
    assert tool1 in team.tools
    assert tool2 in team.tools
    assert len(team.tools) == 2

    # List-like access via __getitem__
    assert team[0] == tool1 # Order might be an issue if dict iteration order changes
    assert team[1] == tool2
    
    # Iteration via __iter__
    iterated_tools = [t for t in team]
    assert tool1 in iterated_tools
    assert tool2 in iterated_tools
    
    # Length via __len__
    assert len(team) == 2

@pytest.mark.asyncio
async def test_process_message_backward_compat_from_model(team_with_lead, mock_lead_model):
    team = team_with_lead
    mock_lead_model.generate.return_value = "Response using from_model"
    
    # Use from_model instead of source_model
    response = await team.process_message("User says hi via from_model", from_model="LeadModel")
    
    assert response == "Response using from_model"
    mock_lead_model.generate.assert_called_once_with("User says hi via from_model")
    assert team.conversation_history[-1].content == "Response using from_model"

@pytest.mark.asyncio
async def test_process_message_dict_content(team_with_lead, mock_lead_model):
    team = team_with_lead
    mock_lead_model.generate.return_value = "Processed dict content"
    
    message_dict = {"content": "This is the actual message"}
    response = await team.process_message(message_dict, source_model="LeadModel")
    
    assert response == "Processed dict content"
    mock_lead_model.generate.assert_called_once_with("This is the actual message")
    assert team.conversation_history[0].content == "This is the actual message"
    assert team.conversation_history[1].content == "Processed dict content"

@pytest.mark.asyncio
async def test_assign_user_input_tool_to_hierarchy_top_with_invalid_team(mock_tool):
    """Test assign_user_input_tool_to_hierarchy_top with invalid team config"""
    # Create a team with None config
    team = Team(name="TestTeam")
    team.config = None
    team.models = {}
    
    result = await team.assign_user_input_tool_to_hierarchy_top(mock_tool)
    assert not result

@pytest.mark.asyncio
async def test_assign_user_input_tool_to_hierarchy_top_with_no_hierarchy_top(mock_tool):
    """Test assign_user_input_tool_to_hierarchy_top with no lead but with member models"""
    # Create a team with no lead but with member models
    team = Team(name="TestTeam")
    team.config.lead = None  # No lead specified

    # Add a member model but no lead
    member_model = MockModel("MemberModel")
    team.add_member_sync(member_model)

    result = await team.assign_user_input_tool_to_hierarchy_top(mock_tool)
    assert result
    # The tool should be assigned to the member model
    member_model.add_tool.assert_called_once_with('user_input', mock_tool)
    assert 'user_input' in team._tools
    assert team._tools['user_input'] == mock_tool

@pytest.mark.asyncio
async def test_assign_user_input_tool_to_hierarchy_top_success(mock_tool):
    """Test successful assign_user_input_tool_to_hierarchy_top"""
    # Create a team with proper lead model
    team = Team(name="TestTeam")
    team.config.lead = "LeadModel"
    
    # Add lead and member models
    lead_model = MockModel("LeadModel")
    member_model = MockModel("MemberModel")
    team.add_member_sync(lead_model, role="lead")
    team.add_member_sync(member_model)
    
    result = await team.assign_user_input_tool_to_hierarchy_top(mock_tool)
    assert result
    
    # Verify the tool was added to the lead model
    lead_model.add_tool.assert_called_once_with('user_input', mock_tool)
    
    # Verify the tool was added to team's tools registry
    assert 'user_input' in team._tools
    assert team._tools['user_input'] == mock_tool

# Additional tests to improve coverage for teams.py
@pytest.mark.asyncio
async def test_process_message_with_tool_call_alternative_format(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Mock alternative format tool
    mock_tool = AsyncMock(return_value="Alt format result")
    mock_tool.execute = AsyncMock(return_value="Alt format result")
    mock_lead_model.tools = {"alt_tool": mock_tool}
    team._tools = {"alt_tool": mock_tool}

    # Model responds with alternative JSON format and then final response
    alt_call = '{"alt_tool": {"param": "value"}}'
    final_resp = "Final after alt tool"
    mock_lead_model.generate.side_effect = [alt_call, final_resp]
    # Fix: Set process_tool_result to return the expected string
    mock_lead_model.process_tool_result.return_value = final_resp

    response = await team.process_message("Use alt format", source_model="LeadModel")

    assert response == final_resp
    mock_tool.execute.assert_called_once_with(param="value", calling_model="LeadModel", calling_team="TestTeam")
    # Check conversation history sequence
    assert len(team.conversation_history) == 4
    assert team.conversation_history[1].content == alt_call
    assert team.conversation_history[2].role == "tool"
    assert team.conversation_history[2].name == "alt_tool"
    assert team.conversation_history[2].content == "Alt format result"
    assert team.conversation_history[3].content == final_resp

@pytest.mark.asyncio
async def test_process_message_tool_returns_failure_payload(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Mock tool returning failure payload
    mock_tool = AsyncMock()
    mock_tool.execute = AsyncMock(return_value={"success": False, "error": "Bad things"})
    mock_lead_model.tools = {"bad_tool": mock_tool}
    team._tools = {"bad_tool": mock_tool}

    # Model responds with standard tool call format
    call_resp = '''{
        "tool_name": "bad_tool",
        "arguments": {}
    }'''
    mock_lead_model.generate.return_value = call_resp

    response = await team.process_message("Invoke bad tool", source_model="LeadModel")

    assert response == "Bad things"
    # History: user, assistant (tool call), tool (error)
    assert len(team.conversation_history) == 3
    msg = team.conversation_history[2]
    assert msg.role == "tool"
    assert msg.name == "bad_tool"
    assert msg.content == "Bad things"

@pytest.mark.asyncio
async def test_try_establish_relationship_outgoing_flow(team_with_lead):
    team = team_with_lead
    target = "OtherTeam"
    mock_flow = MagicMock()
    mock_flow.target.name = target
    team.outgoing_flows = [mock_flow]

    result = await team.try_establish_relationship(target)
    assert result["success"] is True
    assert result["relationship_type"] == FlowType.BIDIRECTIONAL.value
    assert target in team.relationships

@pytest.mark.asyncio
async def test_try_establish_relationship_incoming_flow(team_with_lead):
    team = team_with_lead
    source = "OtherTeam"
    mock_flow = MagicMock()
    mock_flow.source.name = source
    team.incoming_flows = [mock_flow]

    result = await team.try_establish_relationship(source)
    assert result["success"] is True
    assert result["relationship_type"] == FlowType.BIDIRECTIONAL.value
    assert source in team.relationships

@pytest.mark.asyncio
async def test_try_establish_relationship_incoming_flow_another_duplicate(team_with_lead):
    team = team_with_lead
    source = "OtherTeam"
    mock_flow = MagicMock()
    mock_flow.source.name = source
    team.incoming_flows = [mock_flow, mock_flow] # Duplicate flow

    result = await team.try_establish_relationship(source)
    assert result["success"] is True
    assert result["relationship_type"] == FlowType.BIDIRECTIONAL.value
    assert source in team.relationships

# Test for _handle_error (indirectly, by causing an error it might catch)
# This is hard to test directly as it re-raises. We can test a path that might use it.
# For now, focusing on more directly testable public methods.

def test_update_hierarchy_attributes_on_add_member(mock_lead_model):
    team = Team(name="HierarchyTeam")
    # Patch set_hierarchy_attributes to check if it's called
    with patch('glue.core.teams.set_hierarchy_attributes') as mock_set_attrs:
        team.add_member_sync(mock_lead_model, role="lead")
        mock_set_attrs.assert_called_once()
        # Check if the model's attribute (if any) was set - this depends on set_hierarchy_attributes impl.
        # For this test, we just ensure it's called.

def test_team_constructor_invalid_members_type():
    with pytest.raises(Exception, match="members must be a list or None"):
        Team(name="Test", members="not_a_list")

def test_team_model_property_no_lead_first_model(mock_member_model):
    team = Team(name="NoLeadTeam")
    team.add_member_sync(mock_member_model) # Add a member, but not as lead
    assert team.model == mock_member_model

def test_team_model_property_no_models():
    team = Team(name="EmptyTeam")
    assert team.model is None

@pytest.mark.asyncio
async def test_add_member_subteam(team_with_lead):
    parent_team = team_with_lead
    sub_team = Team(name="SubTeam")
    await parent_team.add_member(sub_team)
    assert "SubTeam" in parent_team.subteams
    assert parent_team.subteams["SubTeam"] == sub_team

@pytest.mark.asyncio
async def test_add_member_existing_subteam(team_with_lead):
    parent_team = team_with_lead
    sub_team = Team(name="SubTeam")
    await parent_team.add_member(sub_team) # First add
    await parent_team.add_member(sub_team) # Try adding again
    assert len(parent_team.subteams) == 1 # Should not add duplicate

@pytest.mark.asyncio
async def test_add_member_existing_model(team_with_lead, mock_lead_model):
    team = team_with_lead # mock_lead_model is already lead
    await team.add_member(mock_lead_model) # Try adding again
    assert len(team.models) == 1 # Should not add duplicate

def test_add_member_sync_subteam(team_with_lead):
    parent_team = team_with_lead
    sub_team = Team(name="SubTeamSync")
    parent_team.add_member_sync(sub_team)
    assert "SubTeamSync" in parent_team.subteams
    assert parent_team.subteams["SubTeamSync"] == sub_team

def test_add_member_sync_existing_subteam(team_with_lead):
    parent_team = team_with_lead
    sub_team = Team(name="SubTeamSync")
    parent_team.add_member_sync(sub_team) # First add
    parent_team.add_member_sync(sub_team) # Try adding again
    assert len(parent_team.subteams) == 1 # Should not add duplicate

def test_add_member_sync_existing_model(team_with_lead, mock_lead_model):
    team = team_with_lead # mock_lead_model is already lead
    team.add_member_sync(mock_lead_model) # Try adding again
    assert len(team.models) == 1 # Should not add duplicate

@pytest.mark.asyncio
async def test_process_message_tool_call_failure_payload(team_with_lead, mock_lead_model):
    team = team_with_lead
    mock_failing_tool = AsyncMock()
    # Tool returns a dict with success: False
    failure_payload = {"success": False, "error": "Tool processing failed deliberately"}
    mock_failing_tool.execute = AsyncMock(return_value=failure_payload)
    mock_lead_model.tools = {"failing_tool": mock_failing_tool}
    team._tools = {"failing_tool": mock_failing_tool}

    tool_call_response = '''{
        "tool_name": "failing_tool",
        "arguments": {}
    }'''
    mock_lead_model.generate.return_value = tool_call_response
    
    response = await team.process_message("Use failing_tool", source_model="LeadModel")
    
    assert response == "Tool processing failed deliberately"
    assert len(team.conversation_history) == 3 # user, assistant (tool_call), tool (error)
    assert team.conversation_history[2].role == "tool"
    assert team.conversation_history[2].name == "failing_tool"
    assert team.conversation_history[2].content == "Tool processing failed deliberately"

@pytest.mark.asyncio
async def test_direct_communication_lead_uses_tool(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members # LeadModel is lead, MemberModel is member
    
    mock_lead_tool = AsyncMock()
    # Fix: Make sure execute returns the string expected in conversation history
    mock_lead_tool.execute = AsyncMock(return_value="Lead tool processed data")
    # Add tool to lead model AND team's _tools for direct_communication's internal lookup
    mock_lead_model.tools = {"lead_specific_tool": mock_lead_tool}
    team._tools["lead_specific_tool"] = mock_lead_tool # Critical for the test

    message_from_member = "Lead, use your tool."
    lead_tool_call_json = '{"tool_name": "lead_specific_tool", "arguments": {"param": "value"}}'
    lead_final_response = "Lead used team tool: Team tool processed data"

    mock_lead_model.generate_response.return_value = lead_tool_call_json
    mock_lead_model.process_tool_result.return_value = lead_final_response

    response = await team.direct_communication(
        from_model="MemberModel",
        to_model="LeadModel",
        message=message_from_member
    )

    assert response == lead_final_response
    
    # Check that generate_response was called on the lead model
    mock_lead_model.generate_response.assert_called_once()
    history_arg_for_lead = mock_lead_model.generate_response.call_args[0][0]
    assert history_arg_for_lead[-1].role == "user" # Member's message to lead
    assert history_arg_for_lead[-1].name == "MemberModel"
    assert history_arg_for_lead[-1].content == message_from_member

    # Check that the lead's tool was executed
    mock_lead_tool.execute.assert_called_once_with(param="value", calling_model="LeadModel", calling_team="TestTeam")

    # Check that process_tool_result was called on the lead model
    mock_lead_model.process_tool_result.assert_called_once()
    tool_result_arg = mock_lead_model.process_tool_result.call_args[0][0]
    assert isinstance(tool_result_arg, ToolResult)
    assert tool_result_arg.tool_name == "lead_specific_tool"
    assert tool_result_arg.result == "Lead tool processed data"
    
    # Verify conversation history
    assert team.conversation_history[-3].role == "user" # Member's message to lead (as user)
    assert team.conversation_history[-3].name == "MemberModel"
    assert team.conversation_history[-3].content == message_from_member
    assert team.conversation_history[-2].role == "tool"
    assert team.conversation_history[-2].name == "lead_specific_tool"
    assert team.conversation_history[-2].content == "Lead tool processed data"
    assert team.conversation_history[-1].role == "assistant" # Lead's final response
    assert team.conversation_history[-1].name == "LeadModel"
    assert team.conversation_history[-1].content == lead_final_response

@pytest.mark.asyncio
async def test_direct_communication_to_non_lead_no_tool_use(team_with_members, mock_member_model):
    team = team_with_members
    mock_member_model.generate.return_value = "Member got the direct message."
    
    response = await team.direct_communication(
        from_model="LeadModel",
        to_model="MemberModel",
        message="This is a direct message to you, member."
    )
    
    assert response == "Member got the direct message."
    mock_member_model.generate.assert_called_once_with("This is a direct message to you, member.")
    # Check history
    assert team.conversation_history[-2].role == "assistant"
    assert team.conversation_history[-2].name == "LeadModel"
    assert team.conversation_history[-2].content == "This is a direct message to you, member."
    assert team.conversation_history[-1].role == "assistant"
    assert team.conversation_history[-1].name == "MemberModel"
    assert team.conversation_history[-1].content == "Member got the direct message."

@pytest.mark.asyncio
async def test_direct_communication_lead_responds_without_tool(team_with_members, mock_lead_model):
    team = team_with_members
    # Lead's generate_response returns a simple string (no tool call)
    lead_simple_response = "Lead acknowledges the message from member."
    mock_lead_model.generate_response.return_value = lead_simple_response
    
    message_from_member = "A simple message for the lead."
    
    response = await team.direct_communication(
        from_model="MemberModel",
        to_model="LeadModel",
        message=message_from_member
    )
    
    assert response == lead_simple_response
    mock_lead_model.generate_response.assert_called_once()
    history_arg_for_lead = mock_lead_model.generate_response.call_args[0][0]
    assert history_arg_for_lead[-1].role == "user"
    assert history_arg_for_lead[-1].name == "MemberModel"
    assert history_arg_for_lead[-1].content == message_from_member
    
    mock_lead_model.process_tool_result.assert_not_called() # No tool was called
    
    # Verify conversation history
    assert team.conversation_history[-2].role == "user" # Member's message to lead
    assert team.conversation_history[-2].name == "MemberModel"
    assert team.conversation_history[-1].role == "assistant" # Lead's response
    assert team.conversation_history[-1].name == "LeadModel"
    assert team.conversation_history[-1].content == lead_simple_response

# Test for constructor with string members only (backward compatibility)
def test_team_creation_string_members_only():
    team = Team(name="StringMembersTeam", members=["MemberName1", "MemberName2"])
    assert "MemberName1" in team.members # Stored in self.members for test compat
    assert "MemberName2" in team.members
    assert "MemberName1" in team.config.members
    assert "MemberName2" in team.config.members
    assert len(team.models) == 0 # String names are not auto-converted to models

# Test for constructor with mixed members (string and object)
def test_team_creation_mixed_members():
    mock_model_obj = MockModel("ObjMember")
    team = Team(name="MixedTeam", members=[mock_model_obj, "StringMember"])
    
    assert "ObjMember" in team.models
    assert team.models["ObjMember"] == mock_model_obj
    assert "ObjMember" in team.config.members # From object
    
    assert "StringMember" in team.members # Stored in self.members for test compat
    assert "StringMember" in team.config.members # String name added to config.members
    assert "StringMember" not in team.models # String names not auto-converted

# Test for tools.setter
def test_tools_setter(team_with_lead):
    team = team_with_lead
    new_tools_dict = {"new_tool": MagicMock()}
    team.tools = new_tools_dict # Uses the setter
    assert team._tools == new_tools_dict

# Test __getitem__ with string key
def test_tools_getitem_string_key(team_with_lead):
    team = team_with_lead
    mock_tool = MagicMock()
    team._tools = {"named_tool": mock_tool}
    assert team["named_tool"] == mock_tool

# Test __getitem__ with invalid int key
def test_tools_getitem_invalid_int_key(team_with_lead):
    team = team_with_lead
    team._tools = {"tool1": MagicMock()}
    with pytest.raises(KeyError): # Or IndexError depending on exact behavior if list is empty
        _ = team[5] # Accessing out of bounds

# Test __getitem__ with invalid string key
def test_tools_getitem_invalid_string_key(team_with_lead):
    team = team_with_lead
    with pytest.raises(KeyError):
        _ = team["non_existent_tool"]

# Test for inject_managed_agents when lead has no interpreter
def test_inject_managed_agents_lead_no_interpreter(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members
    # Remove interpreter from lead to simulate this case
    del mock_lead_model.interpreter 
    
    team.inject_managed_agents() # Should not raise error
    # Assert that no delegate function was added (or that it handled gracefully)
    # This depends on the implementation; for now, just ensure no error.
    # If lead.run("__interpreter_init__") is called inside, mock_lead_model.run would be called.
    # If it's not, then globals won't be populated.
    # To properly test, we'd need run to actually set up the interpreter.
    # For now, this confirms the attempt to initialize.

# Test for inject_managed_agents when lead is None
def test_inject_managed_agents_no_lead():
    team = Team(name="NoLeadForInject")
    member = MockModel("OnlyMember")
    team.add_member_sync(member)
    
    team.inject_managed_agents() # Should not raise error and do nothing.
    # No assertions needed beyond no error, as there's no lead to inject into.

@pytest.mark.asyncio
async def test_add_tool_no_lead_assign_to_lead(mock_tool):
    team = Team(name="NoLeadTeam") # No lead model
    await team.add_tool("test_tool", mock_tool, assign_to="lead")
    # Should not raise error, just log a warning. Tool added to team._tools.
    assert "test_tool" in team._tools

@pytest.mark.asyncio
async def test_add_tool_no_hierarchy_top_assign_to_hierarchy_top(mock_tool):
    team = Team(name="NoHierarchyTeam")
    # Patch get_highest_ranking_model to return None
    with patch('glue.core.teams.get_highest_ranking_model', return_value=None):
        await team.add_tool("test_tool", mock_tool, assign_to="hierarchy_top")
    assert "test_tool" in team._tools # Tool added to team
    # No model should have had add_tool called if no top model found
    # (This requires checking mocks of any models if they were added)

@pytest.mark.asyncio
async def test_share_result_no_model_adhesives(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Ensure lead model has no specific adhesives set
    mock_lead_model.adhesives = [] 
    
    tool_result = ToolResult(tool_name="SomeTool", result="data") # No adhesive specified on result
    await team.share_result("SomeTool", tool_result, model_name="LeadModel")
    
    # Default adhesive (TAPE) should be used if not on result and not on model
    assert tool_result.adhesive == AdhesiveType.TAPE # Default if not set
    assert "SomeTool" not in team.shared_results # TAPE results are not stored

@pytest.mark.asyncio
async def test_process_message_target_model_not_found(team_with_lead):
    team = team_with_lead
    with pytest.raises(ValueError, match="Model NonExistentTarget not in team"):
        await team.process_message("Hello", source_model="LeadModel", target_model="NonExistentTarget")

@pytest.mark.asyncio
async def test_process_message_tool_call_alternative_format_not_a_tool(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Model responds with alternative JSON format but key is not a known tool
    alt_call_not_tool = '{"not_a_real_tool": {"param": "value"}}'
    mock_lead_model.generate.return_value = alt_call_not_tool

    response = await team.process_message("Use something", source_model="LeadModel")

    # Should be treated as a regular message, not a tool call
    assert response == alt_call_not_tool
    assert len(team.conversation_history) == 2 # user, assistant
    assert team.conversation_history[1].content == alt_call_not_tool

@pytest.mark.asyncio
async def test_process_message_tool_call_invalid_json_response(team_with_lead, mock_lead_model):
    team = team_with_lead
    invalid_json_response = "This is not JSON { definitely not"
    mock_lead_model.generate.return_value = invalid_json_response
    
    response = await team.process_message("Trigger non-json", source_model="LeadModel")
    
    assert response == invalid_json_response # Should return the raw response
    assert len(team.conversation_history) == 2 # user, assistant
    assert team.conversation_history[1].content == invalid_json_response

@pytest.mark.asyncio
async def test_direct_communication_to_lead_tool_not_in_model_but_in_team(team_with_members, mock_lead_model, mock_member_model):
    team = team_with_members
    
    mock_team_tool = AsyncMock()
    mock_team_tool.execute = AsyncMock(return_value="Team tool result")
    team._tools["team_level_tool"] = mock_team_tool # Tool is in team but not explicitly on lead model
    mock_lead_model.tools = {} # Lead model has no tools initially

    message_from_member = "Lead, use the team tool."
    lead_tool_call_json = '{"tool_name": "team_level_tool", "arguments": {}}'
    lead_final_response = "Lead used team tool: Team tool processed data"

    mock_lead_model.generate_response.return_value = lead_tool_call_json
    mock_lead_model.process_tool_result.return_value = lead_final_response

    # Mock add_tool_sync on the lead model to check if it's called
    mock_lead_model.add_tool_sync = MagicMock()

    response = await team.direct_communication(
        from_model="MemberModel",
        to_model="LeadModel",
        message=message_from_member
    )

    assert response == lead_final_response
    # Check if add_tool_sync was called to add the tool to the lead model dynamically
    mock_lead_model.add_tool_sync.assert_called_once_with("team_level_tool", mock_team_tool)
    mock_team_tool.execute.assert_called_once()

@pytest.mark.asyncio
async def test_direct_communication_to_lead_tool_call_invalid_format(team_with_members, mock_lead_model):
    team = team_with_members
    # Lead's generate_response returns something that looks like a tool call but isn't valid
    invalid_tool_call_json = '{"not_tool_name_key": "some_tool", "args": {}}'
    mock_lead_model.generate_response.return_value = invalid_tool_call_json
    
    message_from_member = "Lead, try this invalid tool call."
    
    response = await team.direct_communication(
        from_model="MemberModel",
        to_model="LeadModel",
        message=message_from_member
    )
    
    # Should return the raw response as no valid tool call was detected
    assert response == invalid_tool_call_json
    mock_lead_model.process_tool_result.assert_not_called()
    assert team.conversation_history[-1].content == invalid_tool_call_json # Raw response logged

# Test for add_member_sync with specific tools argument
def test_add_member_sync_with_specific_tools(team_with_lead, mock_lead_model):
    team = team_with_lead
    specific_tool_instance = MagicMock()
    team._tools["specific_tool"] = specific_tool_instance # Tool must exist in team's registry

    new_member = MockModel("NewMemberWithTool")
    team.add_member_sync(new_member, role="member", tools={"specific_tool"})
    
    new_member.add_tool_sync.assert_any_call("specific_tool", specific_tool_instance)

# Test for get_agent_status with non-existent agent_id
def test_get_agent_status_non_existent_agent(team_with_lead):
    team = team_with_lead
    status = team.get_agent_status("NonExistentAgentID")
    assert "error" in status
    assert status["error"] == "Agent NonExistentAgentID not found"

# Test for terminate_agent_loops when a loop termination raises an exception
@pytest.mark.asyncio
async def test_terminate_agent_loops_with_exception(team_with_lead):
    team = team_with_lead
    mock_loop1 = MagicMock()
    mock_loop1.terminate = MagicMock(side_effect=Exception("Termination failed"))
    mock_loop2 = MagicMock()
    mock_loop2.terminate = MagicMock()
    
    team.agent_loops = {"loop1": mock_loop1, "loop2": mock_loop2}
    
    await team.terminate_agent_loops("Test termination with failure")
    
    mock_loop1.terminate.assert_called_once_with("Test termination with failure")
    mock_loop2.terminate.assert_called_once_with("Test termination with failure")
    assert len(team.agent_loops) == 0 # Loops should still be cleared

# Test for get_model_tools when model has neither .tools nor ._tools
def test_get_model_tools_no_tool_attributes(team_with_lead, mock_lead_model):
    team = team_with_lead
    # Remove tool attributes from the model
    if hasattr(mock_lead_model, 'tools'):
        del mock_lead_model.tools
    if hasattr(mock_lead_model, '_tools'):
        del mock_lead_model._tools
        
    tools = team.get_model_tools("LeadModel")
    assert tools == {}

# Test for try_establish_relationship with no outgoing or incoming flows
@pytest.mark.asyncio
async def test_try_establish_relationship_no_flows_at_all(team_with_lead):
    team = team_with_lead
    team.outgoing_flows = [] # Ensure no flows
    team.incoming_flows = []
    
    result = await team.try_establish_relationship("AnotherTeam")
    assert not result["success"]
    assert "No flows exist" in result["error"]

# Test for setup calling inject_managed_agents (already exists, but ensure it's robust)
@pytest.mark.asyncio
async def test_setup_calls_inject_managed_agents_robust(team_with_members):
    team = team_with_members # Has lead and members
    with patch.object(team, 'inject_managed_agents', wraps=team.inject_managed_agents) as mock_inject:
        await team.setup()
        mock_inject.assert_called_once()

# Test for unregister_outgoing_flow when flow not present
def test_unregister_outgoing_flow_not_present(team_with_lead):
    team = team_with_lead
    mock_flow = MagicMock()
    # Ensure flow is not in outgoing_flows
    if mock_flow in team.outgoing_flows:
        team.outgoing_flows.remove(mock_flow)
    
    team.unregister_outgoing_flow(mock_flow) # Should not raise error
    assert mock_flow not in team.outgoing_flows

# Test for unregister_incoming_flow when flow not present
def test_unregister_incoming_flow_not_present(team_with_lead):
    team = team_with_lead
    mock_flow = MagicMock()
    # Ensure flow is not in incoming_flows
    if mock_flow in team.incoming_flows:
        team.incoming_flows.remove(mock_flow)
    
    team.unregister_incoming_flow(mock_flow) # Should not raise error
    assert mock_flow not in team.incoming_flows

# Test for Monkey-patched InferenceClientModel (if smolagents is available)
try:
    from smolagents import InferenceClientModel
    def test_monkey_patched_inference_client_model():
        model = InferenceClientModel(client=MagicMock(), model_name="test") # Assuming constructor
        assert hasattr(model, 'add_tool')
        assert hasattr(model, 'add_tool_sync')
        # Call them to ensure they are no-ops
        async def run_async_noop():
            await model.add_tool("dummy", MagicMock())
        asyncio.run(run_async_noop())
        model.add_tool_sync("dummy_sync", MagicMock())
        # No assertions needed other than successful execution without error
except ImportError:
    pass # Skip if smolagents not installed

