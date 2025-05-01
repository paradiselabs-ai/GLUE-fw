"""
GLUE Flow System

This module provides the Flow class, which represents a communication channel
between teams in a GLUE application.
"""

import asyncio
import logging
from typing import Any, Dict

from .teams import Team
from .types import FlowType
from .schemas import Message


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
        config: Dict[str, Any] = None,
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

    async def setup(self) -> None:
        """Set up the flow by establishing connections.

        This is an alias for establish() to maintain compatibility with the app setup process.
        """
        await self.establish()

    async def establish(self) -> None:
        """Establish the flow between teams."""
        if self.active:
            self.logger.warning(
                f"Flow between {self.source.name} and {self.target.name} already established"
            )
            return

        self.logger.info(
            f"Establishing flow between {self.source.name} and {self.target.name} ({self.flow_type.name})"
        )

        # Set up message processing tasks based on flow type
        if self.flow_type in [FlowType.PUSH, FlowType.BIDIRECTIONAL]:
            self.source_to_target_task = asyncio.create_task(
                self._process_messages(
                    self.source_to_target_queue, self.source, self.target
                )
            )

        if self.flow_type in [FlowType.PULL, FlowType.BIDIRECTIONAL]:
            self.target_to_source_task = asyncio.create_task(
                self._process_messages(
                    self.target_to_source_queue, self.target, self.source
                )
            )

        # Register flow with teams
        self.source.register_outgoing_flow(self)
        self.target.register_incoming_flow(self)

        self.active = True

    async def terminate(self) -> None:
        """Terminate the flow between teams."""
        if not self.active:
            self.logger.warning(
                f"Flow between {self.source.name} and {self.target.name} not active"
            )
            return

        self.logger.info(
            f"Terminating flow between {self.source.name} and {self.target.name}"
        )

        # Mark flow as inactive first to prevent new messages from being processed
        self.active = False

        tasks_to_cancel = []

        # Cancel message processing tasks
        if self.source_to_target_task:
            self.source_to_target_task.cancel()
            tasks_to_cancel.append(self.source_to_target_task)

        if self.target_to_source_task:
            self.target_to_source_task.cancel()
            tasks_to_cancel.append(self.target_to_source_task)

        # Wait for all tasks to complete cancellation
        if tasks_to_cancel:
            try:
                # Use wait_for with a timeout to avoid hanging
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Tasks didn't complete in time, log but continue
                self.logger.warning(
                    "Some flow tasks didn't complete cancellation in time"
                )
            except Exception as e:
                self.logger.error(f"Error cancelling flow tasks: {e}")

        # Clear task references
        self.source_to_target_task = None
        self.target_to_source_task = None

        # Unregister flow with teams
        self.source.unregister_outgoing_flow(self)
        self.target.unregister_incoming_flow(self)

    async def send_from_source(self, message: Dict[str, Any]) -> None:
        """Send a message from the source team to the target team.

        Args:
            message: Message to send

        Raises:
            RuntimeError: If flow is not active or flow type doesn't allow this direction
        """
        if not self.active:
            raise RuntimeError(
                f"Flow between {self.source.name} and {self.target.name} not active"
            )

        if self.flow_type not in [FlowType.PUSH, FlowType.BIDIRECTIONAL]:
            raise RuntimeError(
                f"Flow type {self.flow_type.name} doesn't allow messages from source to target"
            )

        await self.source_to_target_queue.put(message)

    async def send_from_target(self, message: Dict[str, Any]) -> None:
        """Send a message from the target team to the source team.

        Args:
            message: Message to send

        Raises:
            RuntimeError: If flow is not active or flow type doesn't allow this direction
        """
        if not self.active:
            raise RuntimeError(
                f"Flow between {self.source.name} and {self.target.name} not active"
            )

        if self.flow_type not in [FlowType.PULL, FlowType.BIDIRECTIONAL]:
            raise RuntimeError(
                f"Flow type {self.flow_type.name} doesn't allow messages from target to source"
            )

        await self.target_to_source_queue.put(message)

    async def _process_messages(
        self, queue: asyncio.Queue, sender: Team, receiver: Team
    ) -> None:
        """Process messages from one team to another.

        Args:
            queue: Queue to process messages from
            sender: Sending team
            receiver: Receiving team
        """
        # Flag to track if we should continue processing
        running = True

        while running:
            try:
                # Check for cancellation before waiting for queue
                if asyncio.current_task().cancelled():
                    break

                # Use wait_for with a timeout to allow periodic cancellation checks
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No message yet, just loop and check for cancellation again
                    continue

                try:
                    self.logger.debug(
                        f"Processing message from {sender.name} to {receiver.name}: {message}"
                    )

                    # Add metadata to the message
                    if "metadata" not in message:
                        message["metadata"] = {}

                    message["metadata"]["source_team"] = sender.name
                    message["metadata"]["target_team"] = receiver.name
                    message["metadata"]["timestamp"] = asyncio.get_event_loop().time()

                    # Check if there's a specific target model
                    target_model = message.get("metadata", {}).get("target_model")

                    # Deliver the message to the receiver
                    await receiver.receive_message(message, sender)

                    # If there's a specific target model, try to deliver directly to that model
                    if target_model and target_model in receiver.models:
                        try:
                            # Get the target model
                            model = receiver.models[target_model]

                            # Get the source model name
                            source_model = message.get("metadata", {}).get(
                                "source_model"
                            )
                            source_team = message.get("metadata", {}).get("source_team")

                            # Extract content
                            content = message.get("content", "")
                            if isinstance(content, dict) and "content" in content:
                                content = content["content"]

                            # Create a message for the model
                            model_message = Message(
                                role="system",  # Always use system role for messages from other models
                                content=f"Message from {source_model or 'unknown'} in team {source_team or 'unknown'}: {content}",
                            )

                            # Generate a response
                            response = await model.generate_response([model_message])

                            # Store in team's conversation history
                            receiver.conversation_history.append(
                                Message(
                                    role="system",
                                    content=f"From {source_model or 'unknown'} in team {source_team or 'unknown'} to {target_model}: {content}",
                                )
                            )

                            receiver.conversation_history.append(
                                Message(
                                    role="assistant",  # Changed from "model" to "assistant" to match allowed roles
                                    content=f"From {target_model} to {source_model or 'unknown'} in team {source_team or 'unknown'}: {response}",
                                )
                            )

                            # Send response back to sender if appropriate
                            if (
                                "is_response" not in message.get("metadata", {})
                                or not message["metadata"]["is_response"]
                            ):
                                response_message = {
                                    "content": response,
                                    "metadata": {
                                        "source_team": receiver.name,
                                        "target_team": sender.name,
                                        "source_model": target_model,
                                        "target_model": source_model,
                                        "is_response": True,
                                        "in_reply_to": message.get("metadata", {}).get(
                                            "timestamp"
                                        ),
                                    },
                                }

                                # Send response back through the appropriate direction
                                if self.flow_type == FlowType.PUSH:
                                    await self.send_from_target(response_message)
                                elif self.flow_type == FlowType.PULL:
                                    await self.send_from_source(response_message)
                                elif self.flow_type == FlowType.BIDIRECTIONAL:
                                    # In bidirectional flow, send from the receiver's side
                                    if sender == self.source:
                                        await self.send_from_target(response_message)
                                    else:
                                        await self.send_from_source(response_message)
                        except Exception as e:
                            self.logger.error(
                                f"Error delivering message to model {target_model}: {e}"
                            )

                except Exception as e:
                    self.logger.error(
                        f"Error processing message from {sender.name} to {receiver.name}: {e}"
                    )

                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                # Handle task cancellation gracefully
                self.logger.debug(
                    f"Message processing task cancelled for {sender.name} to {receiver.name}"
                )
                # Make sure to break out of the loop when cancelled
                break
            except RuntimeError as e:
                # Handle event loop closed or other runtime errors
                self.logger.debug(
                    f"Runtime error in message processing for {sender.name} to {receiver.name}: {e}"
                )
                running = False  # Exit the loop
            except Exception as e:
                # Log any errors but keep processing
                self.logger.error(
                    f"Error processing message from {sender.name} to {receiver.name}: {e}"
                )
