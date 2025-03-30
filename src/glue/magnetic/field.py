# glue/magnetic/field.py
# ==================== Imports ====================
from typing import Dict, Set, Any, Optional, List, Tuple
from datetime import datetime
import logging
import asyncio
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from ..core.teams import Team
from ..core.types import FlowType

# ==================== Constants ====================
logger = logging.getLogger("glue.magnetic")

class FlowMetrics(BaseModel):
    """Metrics for flow health monitoring"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    latency: float = 0.0        # Time for information transfer
    congestion: float = 0.0     # Flow congestion level (0-1)
    throughput: float = 0.0     # Successful transfers per minute
    errors: int = 0            # Failed transfers
    last_update: datetime = Field(default_factory=datetime.now)

# ==================== Class Definitions ====================
class FlowState(BaseModel):
    """State of a magnetic flow between teams"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    source_team: str
    target_team: str
    flow_type: FlowType
    active: bool = True
    metrics: FlowMetrics = Field(default_factory=FlowMetrics)
    alternate_routes: List[List[str]] = Field(default_factory=list)  # List of team paths
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class MagneticField:
    """
    Manages team coordination and information flow.
    Provides self-healing and dynamic route adjustment.
    """
    _instances: Dict[str, 'MagneticField'] = {}

    def __init__(self, name: str):
        self.name = name
        self.teams: Dict[str, Team] = {}
        self.flows: Dict[str, FlowState] = {}
        self.repulsions: Set[str] = set()
        
        # Flow control settings
        self.congestion_threshold = 0.8  # 80% congestion triggers reroute
        self.latency_threshold = 5.0    # 5 second latency triggers reroute
        self.healing_interval = 60.0    # Check flows every 60 seconds
        
        # Start flow monitoring
        self._start_monitoring()
        
        MagneticField._instances[name] = self

    # ==================== Core Methods ====================
    async def add_team(self, team: Team) -> None:
        """Register a team with the field"""
        self.teams[team.name] = team
        await self._analyze_team_compatibility(team)
        logger.info(f"Added team {team.name} to field {self.name}")

    async def set_flow(
        self,
        source_team: str,
        target_team: str,
        flow_type: FlowType
    ) -> None:
        """Establish magnetic flow between teams"""
        # Validate teams
        source = self.teams.get(source_team)
        target = self.teams.get(target_team)
        if not source or not target:
            raise ValueError("Teams not found")
            
        # Check repulsion
        if self._are_teams_repelled(source_team, target_team):
            raise ValueError(f"Teams {source_team} and {target_team} repel")
            
        # Create flow with alternate routes
        flow_id = f"{source_team}->{target_team}"
        flow = FlowState(
            source_team=source_team,
            target_team=target_team,
            flow_type=flow_type,
            alternate_routes=self._find_alternate_routes(source_team, target_team)
        )
        self.flows[flow_id] = flow
        
        # Set up team relationships
        await self._establish_team_relationship(source, target, flow_type)
        logger.info(f"Established {flow_type.value} flow between {source_team} and {target_team}")

    async def transfer_information(
        self,
        source_team: str,
        target_team: str,
        content: Any
    ) -> bool:
        """Transfer information between teams with flow control"""
        flow = self._get_flow(source_team, target_team)
        if not flow or not flow.active:
            return False
            
        try:
            start_time = datetime.now()
            
            # Check flow health
            if self._needs_reroute(flow):
                await self._reroute_flow(flow, content)
                return True
                
            # Direct transfer
            if flow.flow_type == FlowType.BIDIRECTIONAL:
                await self._bidirectional_transfer(source_team, target_team, content)
            elif flow.flow_type == FlowType.PUSH:
                await self._push_transfer(source_team, target_team, content)
            elif flow.flow_type == FlowType.PULL:
                await self._pull_transfer(source_team, target_team, content)
                
            # Update metrics
            await self._update_flow_metrics(flow, start_time)
            return True
            
        except Exception as e:
            logger.error(f"Transfer failed: {str(e)}")
            flow.metrics.errors += 1
            return False

    # ==================== Flow Control Methods ====================
    def _needs_reroute(self, flow: FlowState) -> bool:
        """Check if flow needs rerouting"""
        return (
            flow.metrics.congestion > self.congestion_threshold or
            flow.metrics.latency > self.latency_threshold
        )

    async def _reroute_flow(self, flow: FlowState, content: Any) -> None:
        """Reroute flow through alternate path"""
        if not flow.alternate_routes:
            return
            
        # Try alternate routes
        for route in flow.alternate_routes:
            success = await self._try_alternate_route(route, content)
            if success:
                logger.info(f"Successfully rerouted flow through {' -> '.join(route)}")
                return
                
        logger.warning("All alternate routes failed")

    async def _try_alternate_route(self, route: List[str], content: Any) -> bool:
        """Attempt transfer through alternate route"""
        for i in range(len(route) - 1):
            source = route[i]
            target = route[i + 1]
            
            if not await self.transfer_information(source, target, content):
                return False
        return True

    async def _update_flow_metrics(self, flow: FlowState, start_time: datetime) -> None:
        """Update flow health metrics"""
        now = datetime.now()
        duration = (now - start_time).total_seconds()
        
        # Update metrics
        flow.metrics.latency = duration
        flow.metrics.throughput = 1 / duration if duration > 0 else 0
        flow.metrics.last_update = now
        
        # Calculate congestion based on recent metrics
        flow.metrics.congestion = self._calculate_congestion(flow)

    def _calculate_congestion(self, flow: FlowState) -> float:
        """Calculate flow congestion level"""
        if flow.metrics.latency > self.latency_threshold:
            return 1.0
        return min(flow.metrics.latency / self.latency_threshold, 1.0)

    # ==================== Helper Methods ====================
    def _get_flow(self, source_team: str, target_team: str) -> Optional[FlowState]:
        """Get flow between teams"""
        flow_id = f"{source_team}->{target_team}"
        return self.flows.get(flow_id)

    def _are_teams_repelled(self, team1: str, team2: str) -> bool:
        """Check if teams repel each other"""
        return f"{team1}:{team2}" in self.repulsions or f"{team2}:{team1}" in self.repulsions

    def _find_alternate_routes(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """Find alternate routes between teams"""
        routes = []
        visited = {source}
        
        def dfs(current: str, path: List[str], hops: int) -> None:
            if hops > max_hops:
                return
                
            if current == target:
                routes.append(path.copy())
                return
                
            for team in self.teams:
                if team not in visited and not self._are_teams_repelled(current, team):
                    visited.add(team)
                    path.append(team)
                    dfs(team, path, hops + 1)
                    path.pop()
                    visited.remove(team)
                    
        dfs(source, [source], 0)
        return routes

    async def _establish_team_relationship(
        self,
        source: Team,
        target: Team,
        flow_type: FlowType
    ) -> None:
        """Set up team relationships"""
        if flow_type == FlowType.BIDIRECTIONAL:
            await source.set_relationship(target.name, flow_type.value)
            await target.set_relationship(source.name, flow_type.value)
        elif flow_type == FlowType.PUSH:
            await source.set_relationship(target.name, flow_type.value)
        elif flow_type == FlowType.PULL:
            await target.set_relationship(source.name, flow_type.value)

    def _start_monitoring(self) -> None:
        """Start background flow monitoring"""
        async def monitor_flows():
            while True:
                await self._check_flow_health()
                await asyncio.sleep(self.healing_interval)
                
        asyncio.create_task(monitor_flows())

    async def _check_flow_health(self) -> None:
        """Check and heal flow issues"""
        for flow in self.flows.values():
            if self._needs_reroute(flow):
                logger.info(f"Flow health check triggered reroute for {flow.source_team}->{flow.target_team}")
                # Clear metrics to give new route a chance
                flow.metrics = FlowMetrics()

    # ==================== Error Handling ====================
    async def cleanup(self) -> None:
        """Clean up field resources"""
        try:
            # Break all flows
            for flow_id in list(self.flows.keys()):
                flow = self.flows[flow_id]
                await self.break_flow(flow.source_team, flow.target_team)
                
            # Clear states
            self.flows.clear()
            self.repulsions.clear()
            
            logger.info(f"Cleaned up field {self.name}")
            
        except Exception as e:
            logger.error(f"Error during field cleanup: {str(e)}")
            raise
