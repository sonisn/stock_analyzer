"""The LLM provider names, apart from the SDKs that serve them.

`config` needs the type to validate settings; importing it from `llm`
pulled Pydantic AI and all three provider SDKs into every command,
including ones that never call a model.
"""

from __future__ import annotations

from typing import Literal

Provider = Literal["claude", "gemini", "openai"]
# The helper roles (news rerank, insider synthesis) may also run on an open
# model over OpenRouter (openrouter.OpenRouterAgent); the deciding roles may not.
HelperProvider = Literal["claude", "gemini", "openai", "openrouter"]
