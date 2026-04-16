import os
import json
import pytest
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "managed_tool_gateway.py"
MODULE_SPEC = spec_from_file_location("managed_tool_gateway_test_module", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
managed_tool_gateway = module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = managed_tool_gateway
MODULE_SPEC.loader.exec_module(managed_tool_gateway)
resolve_managed_tool_gateway = managed_tool_gateway.resolve_managed_tool_gateway


def test_resolve_managed_tool_gateway_derives_vendor_origin_from_shared_domain():
    with patch.dict(
        os.environ,
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: "nous-token",
        )

    assert result is not None
    assert result.gateway_origin == "https://firecrawl-gateway.nousresearch.com"
    assert result.nous_user_token == "nous-token"
    assert result.managed_mode is True


def test_resolve_managed_tool_gateway_uses_vendor_specific_override():
    with patch.dict(
        os.environ,
        {
            "BROWSER_USE_GATEWAY_URL": "http://browser-use-gateway.localhost:3009/",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "browser-use",
            token_reader=lambda: "nous-token",
        )

    assert result is not None
    assert result.gateway_origin == "http://browser-use-gateway.localhost:3009"


def test_resolve_managed_tool_gateway_is_inactive_without_nous_token():
    with patch.dict(
        os.environ,
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: None,
        )

    assert result is None


def test_resolve_managed_tool_gateway_is_disabled_without_subscription():
    with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}, clear=False), \
         patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=False):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: "nous-token",
        )

    assert result is None


def test_read_nous_access_token_refreshes_expiring_cached_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "nous": {
                "access_token": "stale-token",
                "refresh_token": "refresh-token",
                "expires_at": expires_at,
            }
        }
    }))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        lambda refresh_skew_seconds=120: "fresh-token",
    )

    assert managed_tool_gateway.read_nous_access_token() == "fresh-token"


# ── _parse_timestamp ──────────────────────────────────────────────────────────

def test_parse_timestamp_z_suffix():
    result = managed_tool_gateway._parse_timestamp("2026-04-17T12:00:00Z")
    assert result is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_timestamp_naive_string():
    result = managed_tool_gateway._parse_timestamp("2026-04-17T08:30:00")
    assert result is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2026, 4, 17, 8, 30, 0, tzinfo=timezone.utc)


def test_parse_timestamp_empty_string():
    assert managed_tool_gateway._parse_timestamp("") is None


def test_parse_timestamp_invalid_string():
    assert managed_tool_gateway._parse_timestamp("not-a-date") is None


def test_parse_timestamp_none():
    assert managed_tool_gateway._parse_timestamp(None) is None


# ── _access_token_is_expiring ─────────────────────────────────────────────────

def test_access_token_is_expiring_fresh_token():
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert managed_tool_gateway._access_token_is_expiring(expires_at, skew_seconds=120) is False


def test_access_token_is_expiring_expired_token():
    expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert managed_tool_gateway._access_token_is_expiring(expires_at, skew_seconds=120) is True


def test_access_token_is_expiring_none():
    assert managed_tool_gateway._access_token_is_expiring(None, skew_seconds=120) is True


# ── get_tool_gateway_scheme ───────────────────────────────────────────────────

def test_get_tool_gateway_scheme_default(monkeypatch):
    monkeypatch.delenv("TOOL_GATEWAY_SCHEME", raising=False)
    assert managed_tool_gateway.get_tool_gateway_scheme() == "https"


def test_get_tool_gateway_scheme_http(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "http")
    assert managed_tool_gateway.get_tool_gateway_scheme() == "http"


def test_get_tool_gateway_scheme_https(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "https")
    assert managed_tool_gateway.get_tool_gateway_scheme() == "https"


def test_get_tool_gateway_scheme_invalid_raises(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "ftp")
    with pytest.raises(ValueError):
        managed_tool_gateway.get_tool_gateway_scheme()


# ── build_vendor_gateway_url ──────────────────────────────────────────────────

def test_build_vendor_gateway_url_vendor_specific_override(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", "http://firecrawl.localhost:4001/")
    monkeypatch.delenv("TOOL_GATEWAY_DOMAIN", raising=False)
    monkeypatch.delenv("TOOL_GATEWAY_SCHEME", raising=False)
    result = managed_tool_gateway.build_vendor_gateway_url("firecrawl")
    assert result == "http://firecrawl.localhost:4001"


def test_build_vendor_gateway_url_shared_domain(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_GATEWAY_URL", raising=False)
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "example.com")
    monkeypatch.delenv("TOOL_GATEWAY_SCHEME", raising=False)
    result = managed_tool_gateway.build_vendor_gateway_url("firecrawl")
    assert result == "https://firecrawl-gateway.example.com"


def test_build_vendor_gateway_url_default_domain(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_GATEWAY_URL", raising=False)
    monkeypatch.delenv("TOOL_GATEWAY_DOMAIN", raising=False)
    monkeypatch.delenv("TOOL_GATEWAY_SCHEME", raising=False)
    result = managed_tool_gateway.build_vendor_gateway_url("firecrawl")
    assert result == "https://firecrawl-gateway.nousresearch.com"
