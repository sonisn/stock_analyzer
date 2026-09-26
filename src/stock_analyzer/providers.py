"""The LLM provider names, apart from the SDKs that serve them.

`config` needs the type to validate settings; importing it from `llm`
pulled agno and all three provider SDKs (~0.75 s) into every command,
including ones that never call a model.
"""

from __future__ import annotations

from typing import Literal

Provider = Literal["claude", "gemini", "openai"]
