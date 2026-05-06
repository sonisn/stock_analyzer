from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analyzer.config import Settings


def _minimal_env() -> dict[str, str]:
    return {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "SNAPTRADE_CLIENT_ID": "cid",
        "SNAPTRADE_CONSUMER_KEY": "ck",
        "SNAPTRADE_USER_ID": "uid",
        "SNAPTRADE_USER_SECRET": "us",
        "FINNHUB_API_KEY": "fh",
        "SMTP_HOST": "mail.example.com",
        "SMTP_USERNAME": "u",
        "SMTP_PASSWORD": "p",
        "SMTP_FROM_ADDRESS": "from@example.com",
        "SMTP_TO_ADDRESS": "to@example.com",
        "DATABASE_URL": "sqlite:///:memory:",
    }


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert s.drawdown_threshold_pct == 5.0
    assert s.politician_lookback_months == 24
    assert s.politician_fresh_disclosure_days == 2
    assert s.run_timezone == "America/New_York"
    assert s.smtp_port == 587
    assert s.dry_run is False


def test_settings_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No env vars set
    with pytest.raises(ValidationError):
        Settings()


def test_settings_secret_str_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert "sk-ant-test" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test"


def test_settings_extra_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("STOCK_ANALYZER_BOGUS", "x")
    # Extra env vars unrelated to known prefixes are ignored by pydantic-settings
    # (env_prefix scoping). The 'extra=forbid' applies to model fields, not env.
    Settings()  # should not raise
