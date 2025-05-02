import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from glue.magnetic.field import MagneticField, FlowType, FlowState, FlowMetrics
from glue.core.teams import Team


@pytest.fixture
def mock_team():
    team = AsyncMock(spec=Team)
    team.name = "test_team"
    team.set_relationship = AsyncMock()
    team.break_relationship = AsyncMock()
    team.repel = AsyncMock()
    return team


@pytest.fixture
def mock_target_team():
    team = AsyncMock(spec=Team)
    team.name = "target_team"
    team.set_relationship = AsyncMock()
    team.break_relationship = AsyncMock()
    team.repel = AsyncMock()
    return team


@pytest.fixture
def magnetic_field():
    return MagneticField(name="test_field")


class TestMagneticField:
    """Tests for the MagneticField class."""

    @pytest.mark.asyncio
    async def test_add_team(self, magnetic_field, mock_team):
        """Test adding a team to the magnetic field."""
        await magnetic_field.add_team(mock_team)
        
        assert mock_team.name in magnetic_field.teams
        assert magnetic_field.teams[mock_team.name] == mock_team

    @pytest.mark.asyncio
    async def test_set_flow_bidirectional(self, magnetic_field, mock_team, mock_target_team):
        """Test setting a bidirectional flow between teams."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        flow_id = f"{mock_team.name}->{mock_target_team.name}"
        assert flow_id in magnetic_field.flows
        assert magnetic_field.flows[flow_id].flow_type == FlowType.BIDIRECTIONAL
        
        # Check team relationships were set
        mock_team.set_relationship.assert_called_once_with(
            mock_target_team.name, FlowType.BIDIRECTIONAL.value
        )
        mock_target_team.set_relationship.assert_called_once_with(
            mock_team.name, FlowType.BIDIRECTIONAL.value
        )

    @pytest.mark.asyncio
    async def test_set_flow_push(self, magnetic_field, mock_team, mock_target_team):
        """Test setting a push flow between teams."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.PUSH
        )
        
        flow_id = f"{mock_team.name}->{mock_target_team.name}"
        assert flow_id in magnetic_field.flows
        assert magnetic_field.flows[flow_id].flow_type == FlowType.PUSH
        
        # Check only source team relationship was set
        mock_team.set_relationship.assert_called_once_with(
            mock_target_team.name, FlowType.PUSH.value
        )
        mock_target_team.set_relationship.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_flow_pull(self, magnetic_field, mock_team, mock_target_team):
        """Test setting a pull flow between teams."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.PULL
        )
        
        flow_id = f"{mock_team.name}->{mock_target_team.name}"
        assert flow_id in magnetic_field.flows
        assert magnetic_field.flows[flow_id].flow_type == FlowType.PULL
        
        # Check only target team relationship was set
        mock_team.set_relationship.assert_not_called()
        mock_target_team.set_relationship.assert_called_once_with(
            mock_team.name, FlowType.PULL.value
        )

    @pytest.mark.asyncio
    async def test_set_flow_repelled_teams(self, magnetic_field, mock_team, mock_target_team):
        """Test setting a flow between repelled teams raises an error."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        # Set repulsion
        magnetic_field.repulsions.add(f"{mock_team.name}:{mock_target_team.name}")
        
        with pytest.raises(ValueError, match="Teams .* and .* repel"):
            await magnetic_field.set_flow(
                source_team=mock_team.name,
                target_team=mock_target_team.name,
                flow_type=FlowType.BIDIRECTIONAL
            )

    @pytest.mark.asyncio
    async def test_transfer_information_bidirectional(self, magnetic_field, mock_team, mock_target_team):
        """Test transferring information with a bidirectional flow."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        # Mock the bidirectional transfer method
        with patch.object(
            magnetic_field, '_bidirectional_transfer', AsyncMock()
        ) as mock_transfer:
            success = await magnetic_field.transfer_information(
                source_team=mock_team.name,
                target_team=mock_target_team.name,
                content="Test content"
            )
            
            assert success is True
            mock_transfer.assert_called_once_with(
                mock_team.name, mock_target_team.name, "Test content"
            )

    @pytest.mark.asyncio
    async def test_transfer_information_push(self, magnetic_field, mock_team, mock_target_team):
        """Test transferring information with a push flow."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.PUSH
        )
        
        # Mock the push transfer method
        with patch.object(
            magnetic_field, '_push_transfer', AsyncMock()
        ) as mock_transfer:
            success = await magnetic_field.transfer_information(
                source_team=mock_team.name,
                target_team=mock_target_team.name,
                content="Test content"
            )
            
            assert success is True
            mock_transfer.assert_called_once_with(
                mock_team.name, mock_target_team.name, "Test content"
            )

    @pytest.mark.asyncio
    async def test_transfer_information_pull(self, magnetic_field, mock_team, mock_target_team):
        """Test transferring information with a pull flow."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.PULL
        )
        
        # Mock the pull transfer method
        with patch.object(
            magnetic_field, '_pull_transfer', AsyncMock()
        ) as mock_transfer:
            success = await magnetic_field.transfer_information(
                source_team=mock_team.name,
                target_team=mock_target_team.name,
                content="Test content"
            )
            
            assert success is True
            mock_transfer.assert_called_once_with(
                mock_team.name, mock_target_team.name, "Test content"
            )

    @pytest.mark.asyncio
    async def test_transfer_information_inactive_flow(self, magnetic_field, mock_team, mock_target_team):
        """Test transferring information with an inactive flow."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        # Set flow to inactive
        flow_id = f"{mock_team.name}->{mock_target_team.name}"
        magnetic_field.flows[flow_id].active = False
        
        success = await magnetic_field.transfer_information(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            content="Test content"
        )
        
        assert success is False

    @pytest.mark.asyncio
    async def test_reroute_flow(self, magnetic_field, mock_team, mock_target_team):
        """Test rerouting a flow through an alternate path."""
        # Add a third team for the alternate route
        middle_team = AsyncMock(spec=Team)
        middle_team.name = "middle_team"
        
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        await magnetic_field.add_team(middle_team)
        
        # Set up the flow with an alternate route
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        flow_id = f"{mock_team.name}->{mock_target_team.name}"
        flow = magnetic_field.flows[flow_id]
        flow.alternate_routes = [[mock_team.name, middle_team.name, mock_target_team.name]]
        
        # Mock the try_alternate_route method
        with patch.object(
            magnetic_field, '_try_alternate_route', AsyncMock(return_value=True)
        ) as mock_try_route:
            await magnetic_field._reroute_flow(flow, "Test content")
            
            mock_try_route.assert_called_once_with(
                [mock_team.name, middle_team.name, mock_target_team.name],
                "Test content"
            )

    @pytest.mark.asyncio
    async def test_needs_reroute(self, magnetic_field):
        """Test checking if a flow needs rerouting."""
        # Create a flow with high congestion
        flow = FlowState(
            source_team="team1",
            target_team="team2",
            flow_type=FlowType.BIDIRECTIONAL
        )
        flow.metrics.congestion = 0.9  # Above threshold
        
        assert magnetic_field._needs_reroute(flow) is True
        
        # Create a flow with high latency
        flow = FlowState(
            source_team="team1",
            target_team="team2",
            flow_type=FlowType.BIDIRECTIONAL
        )
        flow.metrics.latency = 6.0  # Above threshold
        
        assert magnetic_field._needs_reroute(flow) is True
        
        # Create a flow with good metrics
        flow = FlowState(
            source_team="team1",
            target_team="team2",
            flow_type=FlowType.BIDIRECTIONAL
        )
        flow.metrics.congestion = 0.5  # Below threshold
        flow.metrics.latency = 2.0     # Below threshold
        
        assert magnetic_field._needs_reroute(flow) is False

    @pytest.mark.asyncio
    async def test_update_flow_metrics(self, magnetic_field):
        """Test updating flow metrics."""
        flow = FlowState(
            source_team="team1",
            target_team="team2",
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        start_time = datetime.now() - timedelta(seconds=2)
        await magnetic_field._update_flow_metrics(flow, start_time)
        
        assert flow.metrics.latency == pytest.approx(2.0, abs=0.5)
        assert flow.metrics.throughput > 0
        assert flow.metrics.last_update > start_time

    @pytest.mark.asyncio
    async def test_calculate_congestion(self, magnetic_field):
        """Test calculating flow congestion."""
        flow = FlowState(
            source_team="team1",
            target_team="team2",
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        # Test high latency
        flow.metrics.latency = 10.0  # Above threshold
        congestion = magnetic_field._calculate_congestion(flow)
        assert congestion == 1.0
        
        # Test medium latency
        flow.metrics.latency = 2.5  # Half of threshold
        congestion = magnetic_field._calculate_congestion(flow)
        assert congestion == 0.5
        
        # Test low latency
        flow.metrics.latency = 0.1  # Well below threshold
        congestion = magnetic_field._calculate_congestion(flow)
        assert congestion == 0.02

    @pytest.mark.asyncio
    async def test_find_alternate_routes(self, magnetic_field, mock_team, mock_target_team):
        """Test finding alternate routes between teams."""
        # Add three teams to create potential routes
        team1 = AsyncMock(spec=Team)
        team1.name = "team1"
        
        team2 = AsyncMock(spec=Team)
        team2.name = "team2"
        
        team3 = AsyncMock(spec=Team)
        team3.name = "team3"
        
        await magnetic_field.add_team(team1)
        await magnetic_field.add_team(team2)
        await magnetic_field.add_team(team3)
        
        # Find routes from team1 to team3
        routes = magnetic_field._find_alternate_routes("team1", "team3", max_hops=2)
        
        # Should find at least the direct route and one through team2
        assert len(routes) >= 2
        assert ["team1", "team3"] in routes
        assert ["team1", "team2", "team3"] in routes

    @pytest.mark.asyncio
    async def test_cleanup(self, magnetic_field, mock_team, mock_target_team):
        """Test cleaning up the magnetic field."""
        await magnetic_field.add_team(mock_team)
        await magnetic_field.add_team(mock_target_team)
        
        await magnetic_field.set_flow(
            source_team=mock_team.name,
            target_team=mock_target_team.name,
            flow_type=FlowType.BIDIRECTIONAL
        )
        
        # Add a repulsion
        magnetic_field.repulsions.add(f"team1:team2")
        
        # Patch the break_flow method
        with patch.object(magnetic_field, 'break_flow', AsyncMock()) as mock_break_flow:
            await magnetic_field.cleanup()
            
            # Check that break_flow was called
            mock_break_flow.assert_called_once_with(mock_team.name, mock_target_team.name)
            
            # Check that states were cleared
            assert not magnetic_field.flows
            assert not magnetic_field.repulsions
