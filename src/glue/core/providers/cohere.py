import os
import aiohttp
import logging
import asyncio
from typing import List, Optional, Dict, Any
from ..schemas import Message
from .provider_base import ProviderBase
from enum import Enum

logger = logging.getLogger(__name__)

class CohereProvider(ProviderBase):
    """Native Cohere LLM provider for GLUE."""

    def _initialize_client(self):
        # Load API key from config or environment
        key = getattr(self, 'api_key', None) or os.getenv('CO_API_KEY')
        if not key:
            raise ValueError('Cohere API key not found. Set CO_API_KEY or provide api_key in config.')
        self.api_key = key
        # Default to Cohere chat endpoint
        self.base_url = os.getenv('COHERE_BASE_URL', 'https://api.cohere.com/v2/chat')
        self.base_url = self.base_url.rstrip('/')
        logger.info(f"Initialized Cohere client for model {self.model}")

    async def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        # Build messages payload with plain-text content
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                role_raw = m.get('role')
                content_raw = m.get('content')
            else:
                role_raw = getattr(m, 'role', None)
                content_raw = getattr(m, 'content', None)
            # Convert role enum to string
            if isinstance(role_raw, Enum):
                role = role_raw.value
            else:
                role = str(role_raw) if role_raw is not None else ''
            # Normalize content: join list of blocks or cast to string
            if isinstance(content_raw, list):
                content = ''.join(
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in content_raw
                )
            else:
                content = str(content_raw)
            msg_list.append({'role': role, 'content': content})
        # Prepare payload for chat endpoint with only supported parameters
        payload = {
            'model': self.model,
            'messages': msg_list,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }
        # Optionally include stop_sequences if provided
        stop_seqs = kwargs.get('stop_sequences') or kwargs.get('stop_sequences')
        if stop_seqs is not None:
            payload['stop_sequences'] = stop_seqs
        # Log inputs
        logger.debug(f"CohereProvider API payload: {payload}")
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        url = self.base_url
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    logger.debug(f"CohereProvider RAW API RESPONSE: {data}")
            except aiohttp.ClientResponseError as e:
                # Fallback for non-chat models: use generate endpoint on 422
                if e.status == 422:
                    logger.warning(f"CohereProvider chat failed (422); falling back to generate endpoint for model '{self.model}'")
                    # Use only the last user message as prompt to avoid oversized payloads
                    user_contents = [m['content'] for m in msg_list if m['role'] == 'user']
                    if user_contents:
                        prompt = user_contents[-1]
                    else:
                        prompt = msg_list[-1]['content'] if msg_list else ''
                    gen_url = os.getenv('COHERE_GENERATE_URL', 'https://api.cohere.com/v1/generate')
                    gen_payload = {
                        'model': self.model,
                        'prompt': prompt,
                        'max_tokens': self.max_tokens,
                        'temperature': self.temperature,
                    }
                    # include stop_sequences if defined
                    if 'stop_sequences' in payload:
                        gen_payload['stop_sequences'] = payload['stop_sequences']
                    logger.debug(f"CohereProvider fallback generate payload: {gen_payload}")
                    async with session.post(gen_url, json=gen_payload, headers=headers) as gen_resp:
                        gen_resp.raise_for_status()
                        gen_data = await gen_resp.json()
                        logger.debug(f"CohereProvider fallback generate response: {gen_data}")
                    # Extract text from generations
                    gens = gen_data.get('generations') or []
                    if isinstance(gens, list) and gens:
                        return gens[0].get('text', '')
                    return gen_data.get('text', '')
                # Handle model not found
                if e.status == 404:
                    logger.error(f"CohereProvider model not found or invalid endpoint: '{self.model}'")
                    raise RuntimeError(f"ERROR_COHERE_MODEL_NOT_FOUND: '{self.model}'")
                logger.error(f"CohereProvider HTTP error: {e.status} {e.message}")
                raise
        # Parse response content
        msg = data.get('message', {})
        content_blocks = msg.get('content', [])
        if isinstance(content_blocks, list):
            text = ''.join([block.get('text', '') for block in content_blocks])
        else:
            text = content_blocks if isinstance(content_blocks, str) else ''
        return text 