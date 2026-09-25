"""Startup health checks — fail loud and fast before burning API tokens.

Each pipeline depends on a small set of external services (Anthropic for
most LLM calls — plus Gemini/OpenAI too, when the discover pipeline's
ranker/red-team/fallback settings reference them — SnapTrade for
brokerage data, SMTP for delivery). When any of these is mis-configured
or down, we want to know in seconds, not after step 10 of an LLM-heavy
run.

`preflight(settings, ...)` issues one cheap, auth-validating call per
required service and raises `PreflightError` with a bundled list of
problems if anything is broken.
"""

from __future__ import annotations

import finnhub
from anthropic import Anthropic
from snaptrade_client import SnapTrade
from snaptrade_client.auth import SnapTradeAuth

from .config import Settings
from .logging import get_logger

# `google-genai` and `openai` are only imported lazily inside the checks
# below — most CLI entry points never touch Gemini/OpenAI, and importing
# either unconditionally would add startup cost (and a hard dependency on
# credentials being configured) to runs that don't need them.

logger = get_logger(__name__)

# Short, deliberate: a slow/hanging external service should not waste the
# user's time at startup — better to surface the latency as a failure
# than block on a full default timeout.
_PING_TIMEOUT = 5.0


class PreflightError(RuntimeError):
    """Raised by `preflight(...)` when one or more services are unreachable."""


def _check_anthropic(settings: Settings) -> str | None:
    if not settings.anthropic_api_key:
        return "ANTHROPIC_API_KEY is empty"
    try:
        client = Anthropic(api_key=settings.anthropic_api_key, timeout=_PING_TIMEOUT)
        # /v1/models is auth-validated, doesn't consume tokens, fast.
        client.models.list(limit=1)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "401" in msg or "auth" in lower or "invalid api key" in lower:
            return "ANTHROPIC_API_KEY rejected by api.anthropic.com (401)"
        return f"could not reach api.anthropic.com: {msg}"
    return None


def _check_gemini(settings: Settings) -> str | None:
    if not settings.google_api_key:
        return "GOOGLE_API_KEY is empty"
    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        # Listing models is auth-validated, doesn't consume generation
        # tokens, and is fast.
        next(iter(client.models.list(config={"page_size": 1})), None)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "401" in msg or "403" in msg or "api key" in lower or "permission" in lower:
            return "GOOGLE_API_KEY rejected by Gemini API (401/403)"
        return f"could not reach Gemini API: {msg}"
    return None


def _check_openai(settings: Settings) -> str | None:
    if not settings.openai_api_key:
        return "OPENAI_API_KEY is empty"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=_PING_TIMEOUT)
        # /v1/models is auth-validated, doesn't consume tokens, fast.
        client.models.list()
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "401" in msg or "auth" in lower or "invalid api key" in lower:
            return "OPENAI_API_KEY rejected by api.openai.com (401)"
        return f"could not reach api.openai.com: {msg}"
    return None


def _check_snaptrade(settings: Settings) -> str | None:
    missing = [
        name
        for name, val in (
            ("SNAPTRADE_CLIENT_ID", settings.snaptrade_client_id),
            ("SNAPTRADE_CONSUMER_KEY", settings.snaptrade_consumer_key),
            ("SNAPTRADE_USER_ID", settings.snaptrade_user_id),
            ("SNAPTRADE_USER_SECRET", settings.snaptrade_user_secret),
        )
        if not val
    ]
    user_id, user_secret = settings.snaptrade_user_id, settings.snaptrade_user_secret
    if missing or not user_id or not user_secret:
        return f"SnapTrade env vars missing: {', '.join(missing)}"
    try:
        client = SnapTrade(
            auth=SnapTradeAuth.commercial_api_key(
                client_id=settings.snaptrade_client_id,
                consumer_key=settings.snaptrade_consumer_key,
            )
        )
        # list_user_accounts is the cheapest auth-validating call.
        resp = client.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=user_secret,
        )
        # Touch the body so any deserialization error surfaces here, not later.
        _ = resp.body if hasattr(resp, "body") else resp
    except Exception as e:
        return f"SnapTrade ping failed: {e}"
    return None


def _check_finnhub(settings: Settings) -> str | None:
    if not settings.finnhub_api_key:
        return "FINNHUB_API_KEY is empty"
    try:
        client = finnhub.Client(api_key=settings.finnhub_api_key)
        # company_profile2 on a known ticker is the cheapest auth check.
        client.company_profile2(symbol="AAPL")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "auth" in msg.lower() or "api_key" in msg.lower():
            return "FINNHUB_API_KEY rejected by finnhub.io (401)"
        return f"could not reach Finnhub: {msg}"
    return None


def _check_email(settings: Settings) -> str | None:
    # Only a coherence check — we don't open an SMTP session at startup
    # because that's slow and most providers throttle connection attempts.
    if settings.email_to and not settings.smtp_host:
        return (
            "EMAIL_TO is set but SMTP_HOST is missing — email delivery "
            "will be skipped. Set SMTP_* env vars or unset EMAIL_TO."
        )
    return None


def preflight(
    settings: Settings,
    *,
    needs_llm: bool = True,
    needs_brokerage: bool = False,
    needs_finnhub: bool = False,
    needs_email: bool = False,
    needs_discover_providers: bool = False,
) -> None:
    """Verify required external services are reachable and creds work.

    Raises `PreflightError` if anything is broken; all failures are
    collected so one run surfaces every problem (not one-at-a-time).

    `needs_discover_providers` additionally checks whichever of
    Gemini/OpenAI the discover pipeline's ranker/red-team/fallback settings
    actually reference — set it from `cli/discover.py` only, since other
    entry points (rebalance, portfolio, insider) are still Claude-only.
    """
    errors: list[str] = []
    if needs_llm and (err := _check_anthropic(settings)):
        errors.append(err)
    if needs_discover_providers:
        providers = {p for p, _ in settings.resolve_ranker_rounds()}
        providers.add(settings.discover_redteam_provider)
        providers.add(settings.discover_fallback_provider)
        if "gemini" in providers and (err := _check_gemini(settings)):
            errors.append(err)
        if "openai" in providers and (err := _check_openai(settings)):
            errors.append(err)
    if needs_brokerage and (err := _check_snaptrade(settings)):
        errors.append(err)
    if needs_finnhub and (err := _check_finnhub(settings)):
        errors.append(err)
    if needs_email and (err := _check_email(settings)):
        errors.append(err)

    if errors:
        bullets = "\n  - ".join(errors)
        raise PreflightError(f"Preflight failed:\n  - {bullets}")

    logger.info(
        "Preflight OK (llm=%s, discover_providers=%s, brokerage=%s, finnhub=%s, email=%s)",
        needs_llm,
        needs_discover_providers,
        needs_brokerage,
        needs_finnhub,
        needs_email,
    )
