"""
Example script demonstrating agent-to-agent communication in GLUE.

This script creates a simple application with two teams, each with two models,
and demonstrates how models can communicate with each other.
"""

import asyncio
import logging
import os
import sys
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.glue.core.app import GlueApp
from src.glue.core.model import Model
from src.glue.core.teams import Team
from src.glue.core.types import FlowType, TeamConfig
from src.glue.core.schemas import Message

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent_communication")

# Uncomment to enable debug logging
logging.getLogger("agent_communication").setLevel(logging.DEBUG)

async def wait_for_response(team, source_team_name, timeout=15):
    """Wait for a response from another team.
    
    Args:
        team: The team to check for responses
        source_team_name: The name of the team that sent the original message
        timeout: Maximum time to wait for a response in seconds
        
    Returns:
        The response message if found, None otherwise
    """
    start_time = time.time()
    message_time = start_time
    initial_history_length = len(team.conversation_history)
    
    logger.debug(f"Waiting for response from {team.name} to {source_team_name}")
    logger.debug(f"Initial conversation history length: {initial_history_length}")
    
    # Check the conversation history for responses
    while time.time() - start_time < timeout:
        current_history_length = len(team.conversation_history)
        
        # If new messages have been added to the conversation history
        if current_history_length > initial_history_length:
            logger.debug(f"New messages detected in {team.name}'s conversation history")
            
            # Look for messages in the conversation history that came from the source team
            for i in range(initial_history_length, current_history_length):
                message = team.conversation_history[i]
                
                # Check if this is a response message (role is assistant or model)
                if message.role in ["assistant", "model"]:
                    logger.debug(f"Found response in {team.name}'s conversation history: {message.content[:100]}...")
                    return message.content
            
            # If we've checked all new messages and didn't find a response, update the initial length
            initial_history_length = current_history_length
        
        # Also check for system messages that might contain responses
        for i in range(len(team.conversation_history) - 1, -1, -1):
            message = team.conversation_history[i]
            
            # Check if this is a system message from the source team
            if (message.role == "system" and 
                isinstance(message.content, str) and 
                (f"From {source_team_name}:" in message.content or 
                 f"Message from team {source_team_name}" in message.content)):
                
                # Look for the response to this message (should be the next message)
                if i + 1 < len(team.conversation_history):
                    response = team.conversation_history[i + 1]
                    if response.role in ["assistant", "model"]:
                        logger.debug(f"Found response after system message in {team.name}'s conversation history")
                        return response.content
        
        # Check for metadata in messages that might indicate a response
        for i in range(len(team.conversation_history) - 1, -1, -1):
            message = team.conversation_history[i]
            
            # Check if this message has metadata indicating it's a response to the source team
            if hasattr(message, 'metadata') and message.metadata:
                metadata = message.metadata
                if (metadata.get('source_team') == source_team_name or 
                    metadata.get('target_team') == source_team_name):
                    logger.debug(f"Found message with metadata linking to {source_team_name}")
                    return message.content
        
        # Wait a bit before checking again
        await asyncio.sleep(0.5)
    
    # Log the conversation history if no response was found
    logger.debug(f"No response found in {team.name}'s conversation history after {timeout} seconds")
    logger.debug(f"Conversation history length: {len(team.conversation_history)}")
    for i, msg in enumerate(team.conversation_history[-5:]):
        logger.debug(f"Message {i}: role={msg.role}, content={msg.content[:50]}...")
        if hasattr(msg, 'metadata') and msg.metadata:
            logger.debug(f"Message {i} metadata: {msg.metadata}")
    
    return None

async def main():
    """Run the agent communication example."""
    logger.info("Starting agent communication example")
    
    # Set debug level for relevant loggers to help troubleshoot
    logging.getLogger("glue.team").setLevel(logging.DEBUG)
    logging.getLogger("glue.model").setLevel(logging.DEBUG)
    
    # Create app configuration
    app_config = {
        "app": {
            "name": "Agent Communication Example",
            "description": "Demonstrates agent-to-agent communication in GLUE",
            "version": "0.1.0",
            "development": True
        },
        "models": {
            "researcher": {
                "name": "researcher",
                "provider": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "adhesives": ["glue", "velcro"],
                "role": "Researcher who finds information"
            },
            "assistant": {
                "name": "assistant",
                "provider": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "adhesives": ["glue", "velcro"],
                "role": "Assistant who helps organize information"
            },
            "writer": {
                "name": "writer",
                "provider": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "adhesives": ["glue", "velcro"],  # Changed from "tape" to "glue" and "velcro" for better inter-team communication
                "role": "Writer who creates content"
            },
            "editor": {
                "name": "editor",
                "provider": "openrouter",
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "adhesives": ["glue"],
                "role": "Editor who reviews and improves content"
            }
        },
        "tools": {
            "web_search": {
                "name": "web_search",
                "description": "Search the web for information"
            },
            "file_handler": {
                "name": "file_handler",
                "description": "Read and write files"
            },
            "code_interpreter": {
                "name": "code_interpreter",
                "description": "Execute Python code"
            }
        },
        "magnetize": {
            "research": {
                "lead": "researcher",
                "tools": ["web_search", "code_interpreter"]
            },
            "docs": {
                "lead": "writer",
                "tools": ["file_handler"]
            }
        },
        "flows": [
            {
                "source": "research",
                "target": "docs",
                "type": "BIDIRECTIONAL"
            }
        ]
    }
    
    # Create the app
    app = GlueApp(config=app_config)
    
    # Set up the app
    await app.setup()
    
    # Add the assistant model to the research team
    research_team = app.teams["research"]
    assistant_model = app.models["assistant"]
    await research_team.add_member(assistant_model)
    
    # Add the editor model to the docs team
    docs_team = app.teams["docs"]
    editor_model = app.models["editor"]
    await docs_team.add_member(editor_model)
    
    # Print team information
    logger.info(f"Research team members: {list(research_team.models.keys())}")
    logger.info(f"Docs team members: {list(docs_team.models.keys())}")
    
    # Demonstrate intra-team communication
    logger.info("Demonstrating intra-team communication:")
    
    try:
        # Researcher to Assistant
        logger.info("Researcher to Assistant:")
        researcher = app.models["researcher"]
        response = await researcher.communicate_with_model("assistant", 
            "Hello Assistant, I need help organizing research on quantum computing.")
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error in researcher to assistant communication: {e}")
    
    try:
        # Writer to Editor
        logger.info("Writer to Editor:")
        writer = app.models["writer"]
        response = await writer.communicate_with_model("editor", 
            "Hello Editor, can you review this article on quantum computing?")
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error in writer to editor communication: {e}")
    
    # Demonstrate inter-team communication
    logger.info("Demonstrating inter-team communication:")
    
    try:
        # Researcher to Writer
        logger.info("Researcher to Writer:")
        response = await researcher.communicate_with_model("writer", 
            "Hello Writer, I've completed my research on quantum computing. Here are the key points: 1) Quantum bits can exist in multiple states simultaneously. 2) Quantum entanglement allows for instant communication. 3) Quantum computers excel at certain types of calculations.")
        logger.info(f"Initial response: {response}")
        
        # Add a small delay to ensure the message is processed
        await asyncio.sleep(2)
        
        # Log the initial conversation history length
        logger.info(f"Docs team conversation history length before waiting: {len(docs_team.conversation_history)}")
        
        # Wait for and get the actual response from the writer
        writer_response = await wait_for_response(docs_team, "research")
        if writer_response:
            logger.info(f"Writer's response: {writer_response}")
        else:
            logger.warning("No response received from writer")
            
            # Log the conversation history to help debug
            logger.info(f"Docs team conversation history length after waiting: {len(docs_team.conversation_history)}")
            for i, msg in enumerate(docs_team.conversation_history[-3:]):
                logger.info(f"Recent message {i}: role={msg.role}, content={msg.content[:100]}...")
    except Exception as e:
        logger.error(f"Error in researcher to writer communication: {e}")
    
    try:
        # Assistant to Editor
        logger.info("Assistant to Editor:")
        assistant = app.models["assistant"]
        response = await assistant.communicate_with_model("editor", 
            "Hello Editor, I've helped organize the research on quantum computing. Can you ensure the final article is clear and accessible to a general audience?")
        logger.info(f"Initial response: {response}")
        
        # Add a small delay to ensure the message is processed
        await asyncio.sleep(2)
        
        # Log the initial conversation history length
        logger.info(f"Docs team conversation history length before waiting: {len(docs_team.conversation_history)}")
        
        # Wait for and get the actual response from the editor
        editor_response = await wait_for_response(docs_team, "research")
        if editor_response:
            logger.info(f"Editor's response: {editor_response}")
        else:
            logger.warning("No response received from editor")
            
            # Log the conversation history to help debug
            logger.info(f"Docs team conversation history length after waiting: {len(docs_team.conversation_history)}")
            for i, msg in enumerate(docs_team.conversation_history[-3:]):
                logger.info(f"Recent message {i}: role={msg.role}, content={msg.content[:100]}...")
    except Exception as e:
        logger.error(f"Error in assistant to editor communication: {e}")
    
    # Demonstrate team-to-team communication
    logger.info("Demonstrating team-to-team communication:")
    
    try:
        # Research team to Docs team
        logger.info("Research team to Docs team:")
        response = await researcher.communicate_with_team("docs", 
            "The research team has completed our investigation into quantum computing. We're ready to hand off our findings for documentation.")
        logger.info(f"Initial response: {response}")
        
        # Add a small delay to ensure the message is processed
        await asyncio.sleep(2)
        
        # Log the initial conversation history length
        logger.info(f"Docs team conversation history length before waiting: {len(docs_team.conversation_history)}")
        
        # Wait for and get the actual response from the docs team
        docs_response = await wait_for_response(docs_team, "research")
        if docs_response:
            logger.info(f"Docs team response: {docs_response}")
        else:
            logger.warning("No response received from docs team")
            
            # Log the conversation history to help debug
            logger.info(f"Docs team conversation history length after waiting: {len(docs_team.conversation_history)}")
            for i, msg in enumerate(docs_team.conversation_history[-3:]):
                logger.info(f"Recent message {i}: role={msg.role}, content={msg.content[:100]}...")
    except Exception as e:
        logger.error(f"Error in research team to docs team communication: {e}")
    
    # Clean up
    await app.cleanup()
    logger.info("Agent communication example completed")

if __name__ == "__main__":
    asyncio.run(main())