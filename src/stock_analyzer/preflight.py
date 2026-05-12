"""Startup health checks — fail loud and fast before burning API tokens.

Each pipeline depends on a small set of external services (Anthropic for
all LLM calls, SnapTrade for brokerage data, SMTP for delivery). When
any of these is mis-configured or down, we want to know in seconds, not
after step 10 of an LLM-heavy run.

`preflight(settings, ...)` issues one cheap, auth-validating call per
required service and raises `PreflightError` with a bundled list of
problems if anything is broken.
"""
from __future__ import annotations

from anthropic import Anthropic
from snaptrade_client import SnapTrade

from .config import Settings
from .logging import get_logger

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
        client = Anthropic(
            api_key=settings.anthropic_api_key, timeout=_PING_TIMEOUT
        )
        # /v1/models is auth-validated, doesn't consume tokens, fast.
        client.models.list(limit=1)
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "401" in msg or "auth" in lower or "invalid api key" in lower:
            return "ANTHROPIC_API_KEY rejected by api.anthropic.com (401)"
        return f"could not reach api.anthropic.com: {msg}"
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
    if missing:
        return f"SnapTrade env vars missing: {', '.join(missing)}"
    try:
        client = SnapTrade(
            client_id=settings.snaptrade_client_id,
            consumer_key=settings.snaptrade_consumer_key,
        )
        # list_user_accounts is the cheapest auth-validating call.
        resp = client.account_information.list_user_accounts(
            user_id=settings.snaptrade_user_id,
            user_secret=settings.snaptrade_user_secret,
        )
        # Touch the body so any deserialization error surfaces here, not later.
        _ = resp.body if hasattr(resp, "body") else resp
    except Exception as e:
        return f"SnapTrade ping failed: {e}"
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
    needs_email: bool = False,
) -> None:
    """Verify required external services are reachable and creds work.

    Raises `PreflightError` if anything is broken; all failures are
    collected so one run surfaces every problem (not one-at-a-time).
    """
    errors: list[str] = []
    if needs_llm:
        if err := _check_anthropic(settings):
            errors.append(err)
    if needs_brokerage:
        if err := _check_snaptrade(settings):
            errors.append(err)
    if needs_email:
        if err := _check_email(settings):
            errors.append(err)

    if errors:
        bullets = "\n  - ".join(errors)
        raise PreflightError(f"Preflight failed:\n  - {bullets}")

    logger.info(
        "Preflight OK (llm=%s, brokerage=%s, email=%s)",
        needs_llm,
        needs_brokerage,
        needs_email,
    )
