"""A scripted stand-in for the provider model behind `llm.ModelAgent`.

`script(monkeypatch, replies)` makes every agent built afterwards answer
from `replies` — (text, finish_reason) pairs, one per model request — and
returns the list of requests it saw (each one's model settings and
messages), so a test can assert what was sent and how many calls it took.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from stock_analyzer import llm


@dataclass
class Seen:
    requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.requests)


def script(
    monkeypatch,
    replies: list[tuple[str, str]],
    *,
    usage: RequestUsage | None = None,
) -> Seen:
    seen = Seen()
    pending = list(replies)

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.requests.append({"settings": info.model_settings, "messages": messages})
        if not pending:
            raise AssertionError("the model was called more often than scripted")
        text, finish_reason = pending.pop(0)
        return ModelResponse(
            parts=[TextPart(text)],
            usage=usage or RequestUsage(input_tokens=100, output_tokens=10),
            finish_reason=finish_reason,  # ty: ignore[invalid-argument-type]
        )

    model = FunctionModel(answer, model_name="scripted")

    @asynccontextmanager
    async def open_model(*_a, **_k):
        yield model

    monkeypatch.setattr(llm, "open_model", open_model)
    return seen
