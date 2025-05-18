# src/glue/core/providers/nebius.py
import os
import aiohttp
import logging
import asyncio
from typing import List, Optional, Dict, Any
from ..schemas import Message
from .provider_base import ProviderBase

logger = logging.getLogger(__name__)

class NebiusProvider(ProviderBase):
    """Native Nebius.ai LLM provider for GLUE."""

    def _initialize_client(self):
        # Load API key from config or environment
        key = getattr(self, 'api_key', None) or os.getenv('NEBIUS_API_KEY')
        if not key:
            raise ValueError('Nebius API key not found. Set NEBIUS_API_KEY or provide api_key in config.')
        self.api_key = key
        # Default to Nebius chat completions v1 endpoint
        self.base_url = os.getenv('NEBIUS_BASE_URL', 'https://api.studio.nebius.com/v1')
        # Remove trailing slash to prevent double-slash in endpoints
        self.base_url = self.base_url.rstrip('/')
        logger.info(f"Initialized Nebius client for model {self.model}")

    async def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        # Build prompt string from conversation history
        prompt_parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get('role')
                content = m.get('content')
            else:
                role = getattr(m, 'role', None)
                content = getattr(m, 'content', None)
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts)
        # Prepare payload for /completions endpoint
        payload = {
            'model': self.model,
            'prompt': prompt,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'n': 1,
            'stream': False,
            **kwargs,
        }
        # Log inputs
        logger.debug(f"NebiusProvider PROMPT: {prompt}")
        logger.debug(f"NebiusProvider API payload: {payload}")
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        # Call the completions endpoint
        url = f"{self.base_url}/completions"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    logger.debug(f"NebiusProvider RAW API RESPONSE: {data}")
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    logger.error(f"NebiusProvider model not found or invalid endpoint: '{self.model}'")
                    raise RuntimeError(f"ERROR_NEBIUS_MODEL_NOT_FOUND: '{self.model}'")
                logger.error(f"NebiusProvider HTTP error: {e.status} {e.message}")
                raise
        # Extract completion text
        if isinstance(data, dict) and 'choices' in data and data['choices']:
            choice = data['choices'][0]
            text = choice.get('text') or choice.get('message', {}).get('content') or ''
            logger.debug(f"NebiusProvider CHOICE: {choice}")
            logger.debug(f"NebiusProvider RETURNING text: {text}")
            return text
        logger.error("NebiusProvider no choices in response; returning empty string")
        return '' 