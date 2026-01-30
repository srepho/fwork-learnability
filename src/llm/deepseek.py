"""DeepSeek V3 client implementation."""

import time

import httpx
import tiktoken

from .base import LLMClient, LLMResponse


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_MULTIPLIER = 2  # exponential backoff


class DeepSeekClient(LLMClient):
    """Client for DeepSeek V3 API.

    Uses temperature=0 for deterministic output as required by the methodology.
    """

    API_BASE = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"  # DeepSeek V3

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        super().__init__(api_key, model)
        self._client = httpx.Client(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        # Use cl100k_base as approximation for token counting
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a completion from DeepSeek V3.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,  # Deterministic output
            "max_tokens": max_tokens,
        }

        # Retry logic with exponential backoff
        last_error = None
        delay = RETRY_DELAY

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= RETRY_MULTIPLIER
                else:
                    raise
            except httpx.HTTPStatusError as e:
                # Retry on 5xx errors
                if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                    last_error = e
                    time.sleep(delay)
                    delay *= RETRY_MULTIPLIER
                else:
                    raise

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw_response=data,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken approximation."""
        return len(self._tokenizer.encode(text))

    @property
    def name(self) -> str:
        return f"DeepSeek ({self.model})"

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
