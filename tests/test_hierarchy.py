"""
Tests for the hierarchy detection and user input tool assignment system.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from glue.core.teams import Team
from glue.core.hierarchy import (
    HierarchyLevel, 
    get_hierarchy_structure, 
    get_highest_ranking_model, 
    is_top_hierarchy, 
    set_hierarchy_attributes,
    HierarchyDetectionError
)
from glue.core.types import TeamConfig


class MockModel:
    def __init__(self, name):
        self.name = name
        self.team = None
        self.tools = {}
        
    async def add_tool(self, name: str, tool):
        self.tools[name] = tool
        
    def add_tool_sync(self, name: str, tool):
        self.tools[name] = tool


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.initialize = AsyncMock()
    tool._initialized = False
    return tool


@pytest.fixture
def simple_team_config():
    return TeamConfig(
        name="SimpleTeam",
        lead="LeadModel",
        members=["MemberModel1", "MemberModel2"],
        tools=[]
    )


@pytest.fixture
def orchestrator_team_config():
    """Team config with orchestrator structure"""
    return TeamConfig(
        name="OrchestratorTeam",
        lead="OrchestratorModel",
        members=["TeamLead1", "TeamLead2"],
        tools=[]
    )


@pytest.fixture
def team_with_hierarchy(simple_team_config):
    """Team with properly set up hierarchy"""
    team = Team(name="HierarchyTeam", config=simple_team_config)
    
    lead_model = MockModel("LeadModel")
    member1 = MockModel("MemberModel1")
    member2 = MockModel("MemberModel2")
    
    team.add_member_sync(lead_model, role="lead")
    team.add_member_sync(member1, role="member")
    team.add_member_sync(member2, role="member")
    
    return team


# ==================== Hierarchy Structure Tests ====================

def test_get_hierarchy_structure_lead_only():
    """Test hierarchy structure detection with lead-only team"""
    team = Team(name="LeadOnlyTeam")
    lead_model = MockModel("OnlyLead")
    team.add_member_sync(lead_model, role="lead")
    
    hierarchy = get_hierarchy_structure(team)
    
    assert hierarchy["OnlyLead"] == HierarchyLevel.LEAD.value
    assert len(hierarchy) == 1


def test_get_hierarchy_structure_lead_and_members(team_with_hierarchy):
    """Test hierarchy structure detection with lead and members"""
    hierarchy = get_hierarchy_structure(team_with_hierarchy)
    
    assert hierarchy["LeadModel"] == HierarchyLevel.LEAD.value
    assert hierarchy["MemberModel1"] == HierarchyLevel.MEMBER.value
    assert hierarchy["MemberModel2"] == HierarchyLevel.MEMBER.value
    assert len(hierarchy) == 3


def test_get_hierarchy_structure_no_lead():
    """Test hierarchy structure detection with no designated lead"""
    team = Team(name="NoLeadTeam")
    member1 = MockModel("Member1")
    member2 = MockModel("Member2")
    
    team.add_member_sync(member1, role="member")
    team.add_member_sync(member2, role="member")
    
    hierarchy = get_hierarchy_structure(team)
    
    # All members should be at member level
    assert hierarchy["Member1"] == HierarchyLevel.MEMBER.value
    assert hierarchy["Member2"] == HierarchyLevel.MEMBER.value
    assert len(hierarchy) == 2


def test_get_hierarchy_structure_empty_team():
    """Test hierarchy structure detection with empty team"""
    team = Team(name="EmptyTeam")
    
    with pytest.raises(HierarchyDetectionError, match="No models found in team"):
        get_hierarchy_structure(team)


def test_get_hierarchy_structure_with_invalid_team():
    team = None
    with pytest.raises(HierarchyDetectionError, match="Team instance is None"):
        get_hierarchy_structure(team)

    team = MagicMock()
    team.config = None
    team.models = {}
    with pytest.raises(HierarchyDetectionError, match="Team config or models are improperly initialized"):
        get_hierarchy_structure(team)


# ==================== Highest Ranking Model Tests ====================

def test_get_highest_ranking_model_with_lead(team_with_hierarchy):
    """Test getting highest ranking model when lead exists"""
    highest = get_highest_ranking_model(team_with_hierarchy)
    assert highest == "LeadModel"


def test_get_highest_ranking_model_no_lead():
    """Test getting highest ranking model when no lead exists"""
    team = Team(name="NoLeadTeam")
    member1 = MockModel("Member1")
    member2 = MockModel("Member2")
    
    team.add_member_sync(member1, role="member")
    team.add_member_sync(member2, role="member")
    
    # Should return first member alphabetically when no lead
    highest = get_highest_ranking_model(team)
    assert highest == "Member1"


def test_get_highest_ranking_model_empty_team():
    """Test getting highest ranking model from empty team"""
    team = Team(name="EmptyTeam")
    
    with pytest.raises(HierarchyDetectionError, match="No models found in team"):
        get_highest_ranking_model(team)


def test_get_highest_ranking_model_orchestrator_level():
    """Test hierarchy with potential orchestrator level (future extensibility)"""
    team = Team(name="OrchestratorTeam")
    orchestrator = MockModel("OrchestratorModel")
    lead = MockModel("TeamLead")
    
    team.add_member_sync(orchestrator, role="lead")  # Currently treated as lead
    team.add_member_sync(lead, role="member")
    
    highest = get_highest_ranking_model(team)
    assert highest == "OrchestratorModel"


def test_get_highest_ranking_model_with_no_models():
    team = MagicMock()
    team.config = MagicMock(lead=None)
    team.models = {}
    with pytest.raises(HierarchyDetectionError, match="No models found in team"):
        get_highest_ranking_model(team)


def test_get_highest_ranking_model_with_valid_hierarchy():
    team = MagicMock()
    team.config = MagicMock(lead="LeadModel")
    team.models = {"LeadModel": MagicMock(), "MemberModel": MagicMock()}

    result = get_highest_ranking_model(team)
    assert result == "LeadModel"


# ==================== Top Hierarchy Tests ====================

def test_is_top_hierarchy_true(team_with_hierarchy):
    """Test is_top_hierarchy returns True for lead model"""
    assert is_top_hierarchy(team_with_hierarchy, "LeadModel") is True


def test_is_top_hierarchy_false(team_with_hierarchy):
    """Test is_top_hierarchy returns False for member models"""
    assert is_top_hierarchy(team_with_hierarchy, "MemberModel1") is False
    assert is_top_hierarchy(team_with_hierarchy, "MemberModel2") is False


def test_is_top_hierarchy_invalid_model(team_with_hierarchy):
    """Test is_top_hierarchy with invalid model name"""
    with pytest.raises(HierarchyDetectionError, match="Model 'InvalidModel' not found"):
        is_top_hierarchy(team_with_hierarchy, "InvalidModel")


# ==================== Set Hierarchy Attributes Tests ====================

def test_set_hierarchy_attributes(team_with_hierarchy):
    """Test setting hierarchy attributes on models"""
    set_hierarchy_attributes(team_with_hierarchy)
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # Check lead model attributes
    assert lead_model.hierarchy_level == HierarchyLevel.LEAD.value
    assert lead_model.is_hierarchy_top is True
    assert lead_model.has_user_input_access is True
    
    # Check member model attributes
    assert member1.hierarchy_level == HierarchyLevel.MEMBER.value
    assert member1.is_hierarchy_top is False
    assert member1.has_user_input_access is False
    
    assert member2.hierarchy_level == HierarchyLevel.MEMBER.value
    assert member2.is_hierarchy_top is False
    assert member2.has_user_input_access is False


# ==================== Team Method Tests ====================

@pytest.mark.asyncio
async def test_assign_user_input_tool_to_hierarchy_top_success(team_with_hierarchy, mock_tool):
    """Test successful user input tool assignment to hierarchy top"""
    success = await team_with_hierarchy.assign_user_input_tool_to_hierarchy_top(mock_tool)
    
    assert success is True
    
    # Tool should be assigned only to the lead
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    assert "user_input" in lead_model.tools
    assert "user_input" not in member1.tools
    assert "user_input" not in member2.tools


@pytest.mark.asyncio
async def test_assign_user_input_tool_to_hierarchy_top_empty_team(mock_tool):
    """Test user input tool assignment to empty team"""
    team = Team(name="EmptyTeam")
    
    success = await team.assign_user_input_tool_to_hierarchy_top(mock_tool)
    
    assert success is False


@pytest.mark.asyncio
async def test_add_tool_hierarchy_top_assignment(team_with_hierarchy, mock_tool):
    """Test add_tool with hierarchy_top assignment"""
    await team_with_hierarchy.add_tool("test_tool", mock_tool, assign_to="hierarchy_top")
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # Tool should be assigned only to the lead
    assert "test_tool" in lead_model.tools
    assert "test_tool" not in member1.tools
    assert "test_tool" not in member2.tools


@pytest.mark.asyncio
async def test_add_tool_lead_assignment(team_with_hierarchy, mock_tool):
    """Test add_tool with lead assignment"""
    await team_with_hierarchy.add_tool("lead_tool", mock_tool, assign_to="lead")
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # Tool should be assigned only to the lead
    assert "lead_tool" in lead_model.tools
    assert "lead_tool" not in member1.tools
    assert "lead_tool" not in member2.tools


@pytest.mark.asyncio
async def test_add_tool_members_assignment(team_with_hierarchy, mock_tool):
    """Test add_tool with members assignment"""
    await team_with_hierarchy.add_tool("member_tool", mock_tool, assign_to="members")
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # Tool should be assigned only to members
    assert "member_tool" not in lead_model.tools
    assert "member_tool" in member1.tools
    assert "member_tool" in member2.tools


@pytest.mark.asyncio
async def test_add_tool_all_assignment_default(team_with_hierarchy, mock_tool):
    """Test add_tool with default (all) assignment"""
    await team_with_hierarchy.add_tool("all_tool", mock_tool)  # Default is "all"
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # Tool should be assigned to all models
    assert "all_tool" in lead_model.tools
    assert "all_tool" in member1.tools
    assert "all_tool" in member2.tools


def test_update_hierarchy_attributes_called_on_add_member(team_with_hierarchy):
    """Test that hierarchy attributes are updated when adding members"""
    # Add a new member
    new_member = MockModel("NewMember")
    team_with_hierarchy.add_member_sync(new_member, role="member")
    
    # Check that hierarchy attributes were set
    assert hasattr(new_member, 'hierarchy_level')
    assert hasattr(new_member, 'is_hierarchy_top')
    assert hasattr(new_member, 'has_user_input_access')
    
    assert new_member.hierarchy_level == HierarchyLevel.MEMBER.value
    assert new_member.is_hierarchy_top is False
    assert new_member.has_user_input_access is False


@pytest.mark.asyncio
async def test_update_hierarchy_attributes_called_on_async_add_member(team_with_hierarchy):
    """Test that hierarchy attributes are updated when adding members asynchronously"""
    # Add a new member asynchronously
    new_member = MockModel("AsyncMember")
    await team_with_hierarchy.add_member(new_member, role="member")
    
    # Check that hierarchy attributes were set
    assert hasattr(new_member, 'hierarchy_level')
    assert hasattr(new_member, 'is_hierarchy_top')
    assert hasattr(new_member, 'has_user_input_access')
    
    assert new_member.hierarchy_level == HierarchyLevel.MEMBER.value
    assert new_member.is_hierarchy_top is False
    assert new_member.has_user_input_access is False


# ==================== Error Handling Tests ====================

def test_hierarchy_detection_error_handling():
    """Test error handling in hierarchy detection"""
    # Test with invalid team object
    with pytest.raises(HierarchyDetectionError):
        get_hierarchy_structure(None)


@pytest.mark.asyncio
async def test_add_tool_invalid_assign_to(team_with_hierarchy, mock_tool):
    """Test add_tool with invalid assign_to parameter"""
    # Should fallback to "all" assignment with warning
    await team_with_hierarchy.add_tool("invalid_tool", mock_tool, assign_to="invalid")
    
    # Should assign to all models as fallback
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    assert "invalid_tool" in lead_model.tools
    assert "invalid_tool" in member1.tools
    assert "invalid_tool" in member2.tools


# ==================== Backward Compatibility Tests ====================

@pytest.mark.asyncio
async def test_backward_compatibility_default_behavior(team_with_hierarchy, mock_tool):
    """Test that default behavior remains unchanged for backward compatibility"""
    # Default assignment should still be "all"
    await team_with_hierarchy.add_tool("compat_tool", mock_tool)
    
    lead_model = team_with_hierarchy.models["LeadModel"]
    member1 = team_with_hierarchy.models["MemberModel1"]
    member2 = team_with_hierarchy.models["MemberModel2"]
    
    # All models should have the tool
    assert "compat_tool" in lead_model.tools
    assert "compat_tool" in member1.tools
    assert "compat_tool" in member2.tools


# ==================== Edge Cases ====================

def test_team_with_only_members_no_lead():
    """Test hierarchy detection in team with only members, no designated lead"""
    team = Team(name="MembersOnlyTeam")
    member1 = MockModel("Member1")
    member2 = MockModel("Member2")
    
    team.add_member_sync(member1, role="member")
    team.add_member_sync(member2, role="member")
    
    # Should still work, returning first member alphabetically
    highest = get_highest_ranking_model(team)
    assert highest == "Member1"
    
    # Set attributes should work
    set_hierarchy_attributes(team)
    assert member1.is_hierarchy_top is True
    assert member2.is_hierarchy_top is False


def test_single_member_team():
    """Test hierarchy detection in team with single member"""
    team = Team(name="SingleMemberTeam")
    single_member = MockModel("OnlyMember")
    
    team.add_member_sync(single_member, role="member")
    
    highest = get_highest_ranking_model(team)
    assert highest == "OnlyMember"
    
    set_hierarchy_attributes(team)
    assert single_member.is_hierarchy_top is True
    assert single_member.has_user_input_access is True


@pytest.mark.asyncio
async def test_user_input_tool_assignment_team_with_no_lead(mock_tool):
    """Test user input tool assignment in team with no designated lead"""
    team = Team(name="NoLeadTeam")
    member1 = MockModel("Member1")
    member2 = MockModel("Member2")
    
    team.add_member_sync(member1, role="member")
    team.add_member_sync(member2, role="member")
    
    success = await team.assign_user_input_tool_to_hierarchy_top(mock_tool)
    
    assert success is True
    # First member alphabetically should get the tool
    assert "user_input" in member1.tools
    assert "user_input" not in member2.tools
