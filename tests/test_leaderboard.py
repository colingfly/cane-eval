"""Tests for the leaderboard module -- all offline, no API calls."""

import json

import pytest

from cane_eval.engine import ReliabilitySummary, LatencyStats
from cane_eval.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    Competitor,
    score_to_grade,
    demo_leaderboard,
)


def _summary(name, overall, passed, total, p95=1000, schema=None):
    """Build a ReliabilitySummary for tests without running anything."""
    schema_pass = schema_fail = 0
    if schema is not None:
        schema_pass = round(total * schema / 100)
        schema_fail = total - schema_pass
    return ReliabilitySummary(
        suite_name=name,
        total=total,
        passed=passed,
        warned=0,
        failed=total - passed,
        overall_score=overall,
        latency=LatencyStats(p95_ms=p95),
        schema_pass=schema_pass,
        schema_fail=schema_fail,
        reliability_score=overall,
        reliability_grade=score_to_grade(overall),
    )


def test_score_to_grade_boundaries():
    assert score_to_grade(90) == "A"
    assert score_to_grade(89.9) == "B"
    assert score_to_grade(75) == "B"
    assert score_to_grade(60) == "C"
    assert score_to_grade(40) == "D"
    assert score_to_grade(39.9) == "F"
    assert score_to_grade(0) == "F"


def test_ranking_orders_by_reliability_desc():
    board = Leaderboard.from_summaries(
        "suite",
        {
            "low": _summary("s", 60, 6, 10),
            "high": _summary("s", 95, 9, 10),
            "mid": _summary("s", 80, 8, 10),
        },
    )
    assert [e.name for e in board.entries] == ["high", "mid", "low"]
    assert [e.rank for e in board.entries] == [1, 2, 3]
    assert board.winner.name == "high"


def test_tie_break_by_pass_rate_then_latency():
    # Same reliability score; higher pass rate should win.
    board = Leaderboard.from_summaries(
        "suite",
        {
            "slow_low_pass": _summary("s", 80, 5, 10, p95=5000),
            "fast_high_pass": _summary("s", 80, 9, 10, p95=1000),
        },
    )
    assert board.entries[0].name == "fast_high_pass"


def test_latency_tie_break_when_pass_rate_equal():
    board = Leaderboard.from_summaries(
        "suite",
        {
            "slow": _summary("s", 80, 8, 10, p95=8000),
            "fast": _summary("s", 80, 8, 10, p95=1200),
        },
    )
    assert board.entries[0].name == "fast"


def test_errored_entry_sorts_last_and_marked():
    board = Leaderboard(suite_name="suite")
    board.entries = [
        LeaderboardEntry.errored("broken", "openai", "gpt-x", "boom"),
        LeaderboardEntry.from_summary("good", _summary("s", 70, 7, 10)),
    ]
    board.rank()
    assert board.entries[0].name == "good"
    assert board.entries[-1].name == "broken"
    assert board.entries[-1].ok is False
    assert board.winner.name == "good"


def test_from_summary_flattens_fields():
    s = _summary("s", 88, 8, 10, p95=2500, schema=90)
    e = LeaderboardEntry.from_summary("Agent X", s, provider="openai", model="gpt-4o")
    assert e.correctness == 88
    assert e.pass_rate == 80.0
    assert e.p95_ms == 2500
    assert e.schema_pct == 90.0
    assert e.reliability_grade == "B"
    assert e.provider == "openai"


def test_from_summary_falls_back_to_overall_when_no_reliability():
    s = ReliabilitySummary(suite_name="s", total=4, passed=4, overall_score=82.0)
    s.reliability_score = None  # no reliability pillar configured
    e = LeaderboardEntry.from_summary("A", s)
    assert e.reliability_score == 82.0
    assert e.reliability_grade == "B"


def test_schema_column_hidden_when_no_schema():
    board = Leaderboard.from_summaries(
        "suite", {"a": _summary("s", 90, 9, 10), "b": _summary("s", 80, 8, 10)}
    )
    md = board.to_markdown()
    assert "Schema" not in md


def test_schema_column_shown_when_present():
    board = Leaderboard.from_summaries(
        "suite", {"a": _summary("s", 90, 9, 10, schema=100)}
    )
    md = board.to_markdown()
    assert "Schema" in md


def test_markdown_contains_ranked_names_and_footer():
    board = Leaderboard.from_summaries(
        "My Suite",
        {"Alpha": _summary("s", 92, 9, 10), "Beta": _summary("s", 71, 7, 10)},
        judge_model="claude-sonnet-4-5-20250929",
    )
    md = board.to_markdown()
    assert "# Agent Reliability Leaderboard" in md
    assert "My Suite" in md
    assert "Alpha" in md and "Beta" in md
    assert "cane-eval" in md
    # Winner appears before runner-up in the table.
    assert md.index("Alpha") < md.index("Beta")


def test_markdown_escapes_pipe_in_name():
    board = Leaderboard(suite_name="s")
    board.entries = [LeaderboardEntry.from_summary("a|b", _summary("s", 90, 9, 10))]
    board.rank()
    assert "a\\|b" in board.to_markdown()


def test_to_dict_roundtrips_json():
    board = Leaderboard.from_summaries(
        "suite", {"a": _summary("s", 90, 9, 10, schema=100)}
    )
    d = board.to_dict()
    s = json.dumps(d)  # must be JSON-serializable
    assert json.loads(s)["entries"][0]["name"] == "a"
    assert d["entries"][0]["rank"] == 1


def test_html_is_self_contained_and_filled():
    board = demo_leaderboard(generated_at="2026-07-17")
    html = board.to_html()
    assert html.startswith("<!doctype html>")
    assert "{rows}" not in html and "{suite}" not in html
    # Self-contained: styles inline, no external stylesheet/script fetches.
    assert "<style>" in html
    assert "<link " not in html and "<script" not in html
    # One header row + one row per entry.
    assert html.count("<tr>") == 1 + len(board.entries)


def test_demo_leaderboard_is_ranked_and_deterministic():
    a = demo_leaderboard()
    b = demo_leaderboard()
    assert [e.name for e in a.entries] == [e.name for e in b.entries]
    scores = [e.reliability_score for e in a.entries]
    assert scores == sorted(scores, reverse=True)
    assert a.entries[0].rank == 1


def test_competitor_from_dict_defaults():
    c = Competitor.from_dict({"model": "gpt-4o", "provider": "openai"})
    assert c.name == "gpt-4o"  # falls back to model when name missing
    assert c.provider == "openai"
    c2 = Competitor.from_dict({"name": "Custom", "provider": "anthropic"})
    assert c2.name == "Custom"


def test_competitor_resolves_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    c = Competitor.from_dict({
        "name": "Kimi K2",
        "provider": "openai-compatible",
        "model": "moonshotai/kimi-k2",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    })
    assert c.api_key_env == "OPENROUTER_API_KEY"
    assert c.resolve_api_key() == "sk-or-test"


def test_competitor_explicit_key_beats_env(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "from-env")
    c = Competitor(name="x", api_key="explicit", api_key_env="SOME_KEY")
    assert c.resolve_api_key() == "explicit"


def test_competitor_missing_env_key_is_none(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    c = Competitor(name="x", api_key_env="MISSING_KEY")
    assert c.resolve_api_key() is None


def test_empty_leaderboard_renders():
    board = Leaderboard(suite_name="empty")
    md = board.to_markdown()
    assert "Agent Reliability Leaderboard" in md
    assert board.winner is None
