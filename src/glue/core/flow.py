"""
GLUE Flow System

This module provides the Flow class, which represents a communication channel
between teams in a GLUE application.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, Optional

from .teams import Team
from .types import FlowType


class Flow:
    """Represents a communication flow between teams.
    
    A Flow establishes a channel for messages to pass between teams,
    with a specified direction (push, pull, or bidirectional).
    """
    
    def __init__(
        self, 
        source: Team, 
        target: Team, 
        flow_type: FlowType = FlowType.BIDIRECTIONAL,
        config: Dict[str, Any] = None
    ):
        """Initialize a new Flow.
        
        Args:
            source: Source team
            target: Target team
            flow_type: Type of flow (push, pull, or bidirectional)
            config: Additional configuration for the flow
        """
        self.source = source
        self.target = target
        self.flow_type = flow_type
        self.config = config or {}
        self.logger = logging.getLogger("glue.core.flow")
        self.active = False
        
        # Queues for message passing
        self.source_to_target_queue = asyncio.Queue()
        self.target_to_source_queue = asyncio.Queue()
        
        # Tasks for message processing
        self.source_to_target_task = None
        self.target_to_source_task = None
        
    async def establish(self) -> None:
        """Establish the flow between teams."""
        if self.active:
            self.logger.warning(f"Flow between {self.source.name} and {self.target.name} already established")
            return
            
        self.logger.info(f"Establishing flow between {self.source.name} and {self.target.name} ({self.flow_type.name})")
        
        # Set up message processing tasks based on flow type
        if self.flow_type in [FlowType.PUSH, FlowType.BIDIRECTIONAL]:
            self.source_to_target_task = asyncio.create_task(
                self._process_messages(self.source_to_target_queue, self.source, self.target)
            )
            
        if self.flow_type in [FlowType.PULL, FlowType.BIDIRECTIONAL]:
            self.target_to_source_task = asyncio.create_task(
                self._process_messages(self.target_to_source_queue, self.target, self.source)
            )
            
        # Register flow with teams
        self.source.register_outgoing_flow(self)
        self.target.register_incoming_flow(self)
        
        self.active = True
        
    async def terminate(self) -> None:
        """Terminate the flow between teams."""
        if not self.active:
            self.logger.warning(f"Flow between {self.source.name} and {self.target.name} not active")
            return
            
        self.logger.info(f"Terminating flow between {self.source.name} and {self.target.name}")
        
        # Cancel message processing tasks
        if self.source_to_target_task:
            self.source_to_target_task.cancel()
            try:
                await self.source_to_target_task
            except asyncio.CancelledError:
                pass
                
        if self.target_to_source_task:
            self.target_to_source_task.cancel()
            try:
                await self.target_to_source_task
            except asyncio.CancelledError:
                pass
                
        # Unregister flow with teams
        self.source.unregister_outgoing_flow(self)
        self.target.unregister_incoming_flow(self)
        
        self.active = False
        
    async def send_from_source(self, message: Dict[str, Any]) -> None:
        """Send a message from the source team to the target team.
        
        Args:
            message: Message to send
            
        Raises:
            RuntimeError: If flow is not active or flow type doesn't allow this direction
        """
        if not self.active:
            raise RuntimeError(f"Flow between {self.source.name} and {self.target.name} not active")
            
        if self.flow_type not in [FlowType.PUSH, FlowType.BIDIRECTIONAL]:
            raise RuntimeError(f"Flow type {self.flow_type.name} doesn't allow messages from source to target")
            
        await self.source_to_target_queue.put(message)
        
    async def send_from_target(self, message: Dict[str, Any]) -> None:
        """Send a message from the target team to the source team.
        
        Args:
            message: Message to send
            
        Raises:
            RuntimeError: If flow is not active or flow type doesn't allow this direction
        """
        if not self.active:
            raise RuntimeError(f"Flow between {self.source.name} and {self.target.name} not active")
            
        if self.flow_type not in [FlowType.PULL, FlowType.BIDIRECTIONAL]:
            raise RuntimeError(f"Flow type {self.flow_type.name} doesn't allow messages from target to source")
            
        await self.target_to_source_queue.put(message)
        
    async def _process_messages(self, queue: asyncio.Queue, sender: Team, receiver: Team) -> None:
        """Process messages from one team to another.
        
        Args:
            queue: Queue to process messages from
            sender: Sending team
            receiver: Receiving team
        """
        while True:
            message = await queue.get()
            
            try:
                self.logger.debug(f"Processing message from {sender.name} to {receiver.name}")
                
                # Add metadata to the message
                if "metadata" not in message:
                    message["metadata"] = {}
                    
                message["metadata"]["source_team"] = sender.name
                message["metadata"]["target_team"] = receiver.name
                message["metadata"]["timestamp"] = asyncio.get_event_loop().time()
                
                # Deliver the message to the receiver
                await receiver.receive_message(message, sender)
                
            except Exception as e:
                self.logger.error(f"Error processing message from {sender.name} to {receiver.name}: {e}")
                
            finally:
                queue.task_done()
