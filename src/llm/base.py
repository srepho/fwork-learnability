"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM generation request."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    raw_response: dict = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        """Check if generation completed successfully."""
        return self.finish_reason in ("stop", "end_turn")


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    All implementations must use temperature=0 for deterministic output.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with the generated content and metadata.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this client."""
        pass
