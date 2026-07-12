"""Cross-turn stream-stale circuit breaker (issue #58962).

A session wedged against an unresponsive provider can hit the stale-stream
detector on every turn and loop forever, burning the full 180s×retries each
turn with no response (observed: 494 consecutive failures over 3+ days).

These tests cover the guard added to ``interruptible_streaming_api_call``:

- a session that has already tripped the consecutive-stale threshold short
  circuits immediately (no network attempt, no 180s wait) with a clear error;
- a successful stream resets the consecutive-stale streak;
- a stale-stream kill increments the consecutive-stale streak.

The harness mirrors tests/run_agent/test_28161_anthropic_stream_pool_cleanup.py.
"""

import threading

import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    return agent


def _good_stream_cm():
    """Context manager whose stream yields no events and returns a valid message."""
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter([]))
    msg = MagicMock()
    msg.content = []
    msg.stop_reason = "end_turn"
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    stream.get_final_message = MagicMock(return_value=msg)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


class TestStreamStaleCircuitBreaker:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_short_circuits_when_streak_at_threshold(self, monkeypatch):
        """A session already past the consecutive-stale threshold must abort
        immediately without opening a stream or waiting out the stale timeout."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 3  # simulate prior wedged turns

        # The stream must never be opened on the short-circuit path.
        # _check_stale_giveup now raises TimeoutError (with _stale_giveup=True)
        # so the conversation loop can bypass the retry_count >= 2 gate.
        with pytest.raises(TimeoutError, match="unresponsive"):
            agent._interruptible_streaming_api_call({})

        agent._anthropic_client.messages.stream.assert_not_called()
        # The streak is NOT reset on the short-circuit so subsequent turns
        # keep failing fast instead of re-attempting forever.
        assert agent._consecutive_stale_streams == 3

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_success_resets_streak(self, monkeypatch):
        """A stream that completes successfully clears the consecutive-stale
        streak so a recovered provider resumes normally."""
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 2  # below the giveup=3 threshold
        agent._anthropic_client.messages.stream.return_value = _good_stream_cm()

        resp = agent._interruptible_streaming_api_call({})
        assert resp is not None
        assert agent._consecutive_stale_streams == 0

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_stale_kill_increments_streak(self, monkeypatch):
        """Each stale-stream kill increments the consecutive-stale streak so a
        wedged session eventually trips the breaker."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

        agent = _make_anthropic_agent()
        agent._consecutive_stale_streams = 0
        unblock = threading.Event()

        def _blocking_gen():
            unblock.wait(timeout=5.0)
            raise httpx.ConnectError("connection dropped after close()")
            yield  # make this a generator so next() triggers the wait

        def _stream_side_effect(*args, **kwargs):
            cm = MagicMock()
            stream = MagicMock()
            stream.__iter__ = MagicMock(return_value=_blocking_gen())
            cm.__enter__ = MagicMock(return_value=stream)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        # Every attempt blocks, trips the stale detector, and fails.
        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        agent._anthropic_client.close.side_effect = unblock.set

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        # At least one stale kill happened; the streak must have advanced.
        assert agent._consecutive_stale_streams >= 1


# ---------------------------------------------------------------------------
# Conversation-loop recovery path: stale-giveup vs. normal RuntimeError
# ---------------------------------------------------------------------------


def _make_agent_with_fallback():
    """Build a minimal AIAgent with one configured fallback."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="gpt-4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openai", "model": "gpt-3.5-turbo"}],
        )
        agent.client = MagicMock()
        return agent


def _good_response(text="ok"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="gpt-3.5-turbo", usage=None)


class TestStaleGiveupFallbackPath:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_runtime_error_retries_normally(self):
        """RuntimeError from the API is not treated as stale-giveup: it does
        not carry _stale_giveup so the bypass in _should_fallback never fires.
        The error must exhaust normal retries before the fallback chain is consulted,
        meaning multiple primary calls happen before any fallback switch."""
        agent = _make_agent_with_fallback()
        # Three retries: primary called 3× before retry exhaustion → fallback
        agent._api_max_retries = 3
        calls = []

        def fake_api(kwargs):
            calls.append(agent.provider)
            # Fail on primary (provider not set == ''), succeed on fallback
            if agent.provider != "openai":
                raise RuntimeError("transient internal failure")
            return _good_response()

        mock_fb = MagicMock()
        mock_fb.api_key = "fb-key"
        mock_fb.base_url = "https://example.com/v1"
        mock_fb._custom_headers = None
        mock_fb.default_headers = None

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb, "gpt-3.5-turbo"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        primary_calls = [c for c in calls if c != "openai"]
        # RuntimeError must exhaust all retries on primary before fallback fires.
        # With max_retries=3 the primary must be called at least twice.
        assert len(primary_calls) >= 2, (
            f"RuntimeError should retry on the primary provider, got primary_calls={primary_calls}"
        )
        assert agent._fallback_activated

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_stale_giveup_timeout_goes_directly_to_fallback(self):
        """TimeoutError with _stale_giveup=True must activate the fallback
        immediately on the first attempt — the _should_fallback bypass means
        the loop never waits for retry_count >= 2."""
        from agent.chat_completion_helpers import _check_stale_giveup

        agent = _make_agent_with_fallback()
        agent._consecutive_stale_streams = 99
        calls = []

        def fake_api(kwargs):
            calls.append(agent.provider)
            if len(calls) == 1:
                _check_stale_giveup(agent)
            return _good_response("fallback answer")

        mock_fb = MagicMock()
        mock_fb.api_key = "fb-key"
        mock_fb.base_url = "https://example.com/v1"
        mock_fb._custom_headers = None
        mock_fb.default_headers = None

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb, "gpt-3.5-turbo"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "fallback answer"
        # First call was on the primary; second on the fallback (no intermediate retry)
        assert len(calls) == 2
        assert calls[0] != calls[1], "stale-giveup must bypass to fallback without retrying same provider"
        assert agent._fallback_activated is True
