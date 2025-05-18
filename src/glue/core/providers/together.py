# src/glue/core/providers/together.py
import os
import aiohttp
import logging
import asyncio
from typing import List, Optional, Dict, Any
from ..schemas import Message
from .provider_base import ProviderBase

logger = logging.getLogger(__name__)

class TogetherProvider(ProviderBase):
    """Native Together.ai LLM provider for GLUE."""

    def _initialize_client(self):
        # Load API key from model config or environment
        if hasattr(self, 'api_key') and self.api_key:
            key = self.api_key
        else:
            key = os.getenv('TOGETHER_API_KEY')
        if not key:
            raise ValueError('Together API key not found. Set TOGETHER_API_KEY or provide api_key in config.')
        self.api_key = key
        # Base URL for Together inference (v1 endpoint)
        # Default to Together API host; can be overridden via env var
        self.base_url = os.getenv('TOGETHER_BASE_URL', 'https://api.together.xyz/v1')

    async def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        # Retry parameters
        max_retries = int(os.getenv('TOGETHER_MAX_RETRIES', 3))
        backoff_base = float(os.getenv('TOGETHER_BACKOFF_BASE', 1.0))
        # Build chat messages payload from conversation history
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
            else:
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)
            msg_list.append({"role": role, "content": content})
        payload = {
            "model": self.model,
            "messages": msg_list,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        # Call Together's chat completions endpoint with retry on 429
        data = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(1, max_retries + 1):
                async with session.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as resp:
                    if resp.status == 429 and attempt < max_retries:
                        # Rate limited: inspect Retry-After or use exponential backoff
                        retry_after = resp.headers.get('Retry-After')
                        delay = float(retry_after) if retry_after else backoff_base * attempt
                        logger.warning(
                            f"Rate limited by Together.ai (429). Retrying in {delay} seconds (attempt {attempt}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    break
        if data is None:
            raise RuntimeError('Failed to get response from Together.ai after retries')
        # Extract assistant message.content from 'choices'
        if isinstance(data, dict) and 'choices' in data and data['choices']:
            first = data['choices'][0]
            # new Together payload: {choices:[{message:{content:...}}]}
            msg = first.get('message', {})
            content = msg.get('content')
            if content is not None:
                return content
        # Fallback for other response formats
        return data.get('text') or data.get('generated_text', '')
