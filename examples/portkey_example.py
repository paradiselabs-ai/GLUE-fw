#!/usr/bin/env python3
"""
Example application demonstrating Portkey.ai integration with GLUE framework.

This example shows how to use the GLUE framework with Portkey.ai for API key
management, usage tracking, and cost optimization.

To run this example:
1. Set PORTKEY_ENABLED=true
2. Set PORTKEY_API_KEY=your_portkey_api_key
3. Set OPENAI_API_KEY=your_openai_api_key
4. Run: python portkey_example.py
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta

from glue import GlueApp
from glue.core.team import Team
from glue.utils.portkey_client import get_portkey_client
from glue.cli import setup_logging

# Initialize central logging configuration
setup_logging()

# Set up logging
logger = logging.getLogger(__name__)


async def main():
    """Run the example application."""
    # Check if Portkey is enabled
    portkey_enabled = os.environ.get("PORTKEY_ENABLED", "").lower() in ("true", "1", "yes")
    if not portkey_enabled:
        logger.warning(
            "Portkey integration is not enabled. "
            "Set PORTKEY_ENABLED=true to enable Portkey integration."
        )
    
    # Check for required API keys
    if not os.environ.get("PORTKEY_API_KEY") and portkey_enabled:
        logger.error("PORTKEY_API_KEY environment variable not set.")
        return
    
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        return
    
    # Create a GLUE app with two models
    app = GlueApp(
        models=[
            {
                "name": "researcher",
                "provider": "openai",
                "model": "gpt-4",
                "role": "researcher",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 0.7,
                "max_tokens": 1000,
                "description": "A research assistant that helps find information."
            },
            {
                "name": "writer",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "role": "writer",
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": 0.5,
                "max_tokens": 1000,
                "description": "A writer that helps create content."
            }
        ],
        teams=[
            {
                "name": "research_team",
                "models": ["researcher", "writer"],
                "tools": []
            }
        ]
    )
    
    # Get the research team
    research_team = app.get_team("research_team")
    
    # Add a simple task
    await research_team.add_task(
        "Research the benefits of AI frameworks and write a short summary."
    )
    
    # Run the app
    await app.run()
    
    # Get the results
    results = research_team.get_results()
    print("\n=== Research Results ===")
    for result in results:
        print(f"\n{result['role'].upper()}: {result['content']}")
    
    # If Portkey is enabled, get usage statistics
    if portkey_enabled:
        try:
            # Get dates for the last day
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Get usage statistics
            client = get_portkey_client()
            usage = await client.get_usage(yesterday, today)
            
            print("\n=== Portkey Usage Statistics ===")
            print(f"Total requests: {usage.get('total_requests', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
            print(f"Total cost: ${usage.get('total_cost', 'N/A')}")
            
            # Print model-specific usage if available
            if 'models' in usage:
                print("\nUsage by model:")
                for model, model_usage in usage['models'].items():
                    print(f"  {model}: {model_usage.get('tokens', 'N/A')} tokens, ${model_usage.get('cost', 'N/A')}")
        
        except Exception as e:
            logger.error(f"Error getting Portkey usage statistics: {str(e)}")
    
    # Close the app
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
