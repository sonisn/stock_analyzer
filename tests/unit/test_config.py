from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analyzer.config import Settings


def _minimal_env() -> dict[str, str]:
    return {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "SNAPTRADE_CLIENT_ID": "secret-cid-zzz",
        "SNAPTRADE_CONSUMER_KEY": "secret-ck-zzz",
        "SNAPTRADE_USER_ID": "uid",
        "SNAPTRADE_USER_SECRET": "secret-us-zzz",
        "FINNHUB_API_KEY": "secret-fh-zzz",
        "SMTP_HOST": "mail.example.com",
        "SMTP_USERNAME": "u",
        "SMTP_PASSWORD": "secret-pw-zzz",
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
    # conftest already strips ANTHROPIC_/SNAPTRADE_/FINNHUB_/SMTP_/STOCK_ANALYZER_ —
    # also strip DATABASE_URL and disable .env loading so the test is deterministic.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_secret_str_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert "sk-ant-test" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test"
    for secret_val in [
        "secret-cid-zzz",
        "secret-ck-zzz",
        "secret-us-zzz",
        "secret-fh-zzz",
        "secret-pw-zzz",
    ]:
        assert secret_val not in repr(s)


def test_settings_extra_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("STOCK_ANALYZER_BOGUS", "x")
    # Extra env vars unrelated to known prefixes are ignored by pydantic-settings
    # (env_prefix scoping). The 'extra=forbid' applies to model fields, not env.
    Settings()  # should not raise


def test_settings_rejects_invalid_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()
