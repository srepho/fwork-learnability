"""DeepSeek V3 client implementation."""

import httpx
import tiktoken

from .base import LLMClient, LLMResponse


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

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

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
