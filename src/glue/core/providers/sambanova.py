# src/glue/core/providers/sambanova.py
import os
import aiohttp
import logging
from typing import List, Optional, Dict, Any
from ..schemas import Message
from .provider_base import ProviderBase

logger = logging.getLogger(__name__)

class SambanovaProvider(ProviderBase):
    """Native Sambanova.ai LLM provider for GLUE."""

    def _initialize_client(self):
        # Load API key from config or environment
        key = getattr(self, 'api_key', None) or os.getenv('SAMBANOVA_API_KEY')
        if not key:
            raise ValueError('Sambanova API key not found. Set SAMBANOVA_API_KEY or provide api_key in config.')
        self.api_key = key
        # Base URL for Sambanova inference
        self.base_url = os.getenv('SAMBANOVA_BASE_URL', 'https://api.sambanova.ai/v1')
        # Log initialization success
        logger.info(f"Initialized Sambanova client for model {self.model}")

    async def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        # Build chat messages payload
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get('role')
                content = m.get('content')
            else:
                role = getattr(m, 'role', None)
                content = getattr(m, 'content', None)
            msg_list.append({'role': role, 'content': content})
        payload = {
            'model': self.model,
            'messages': msg_list,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            **kwargs,
        }
        # Log input messages and payload
        logger.debug(f"SambanovaProvider INPUT messages: {msg_list}")
        logger.debug(f"SambanovaProvider API payload: {payload}")
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # Log raw API response
                logger.debug(f"SambanovaProvider RAW API RESPONSE: {data}")
        # Extract assistant message.content and log
        if isinstance(data, dict) and 'choices' in data and data['choices']:
            choice = data['choices'][0]
            msg = choice.get('message', {})
            content = msg.get('content', '')
            logger.debug(f"SambanovaProvider CHOICE: {choice}")
            logger.debug(f"SambanovaProvider MSG: {msg}")
            logger.debug(f"SambanovaProvider RETURNING content: {content}")
            return content
        logger.error("SambanovaProvider no choices in response; returning empty string")
        return "" 