"""LLM provider abstraction — wraps `agno.Agent` for Claude and Gemini."""
from __future__ import annotations

from typing import Any, Literal

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.google import Gemini

Provider = Literal["claude", "gemini"]

_MODEL_REGISTRY: dict[Provider, type] = {
    "claude": Claude,
    "gemini": Gemini,
}


class AgnoAgent:
    """Factory wrapper around `agno.agent.Agent` supporting multiple providers."""

    def __init__(
        self,
        name: str,
        provider: Provider,
        model: str,
        *,
        model_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        if provider not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported provider {provider!r}. "
                f"Expected one of {sorted(_MODEL_REGISTRY)}."
            )

        self.name = name
        self.provider: Provider = provider
        self.model_id = model

        model_cls = _MODEL_REGISTRY[provider]
        self._model = model_cls(id=model, **(model_kwargs or {}))
        self.agent = Agent(name=name, model=self._model, **agent_kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent.run(*args, **kwargs)

    def print_response(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent.print_response(*args, **kwargs)
