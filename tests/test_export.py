"""Tests for the export module."""

import json
import tempfile
import os

from cane_eval.engine import EvalResult, RunSummary
from cane_eval.judge import JudgeResult, CriteriaScore
from cane_eval.export import Exporter


def _make_result(question, score, status, expected="expected", agent="agent"):
    return EvalResult(
        question=question,
        expected_answer=expected,
        agent_answer=agent,
        judge_result=JudgeResult(
            criteria_scores=[CriteriaScore(key="accuracy", score=score)],
            overall_score=score,
            overall_reasoning="test reasoning",
            status=status,
        ),
        tags=["test"],
    )


def _make_summary():
    return RunSummary(
        suite_name="Test",
        total=3,
        passed=1,
        warned=1,
        failed=1,
        overall_score=60,
        results=[
            _make_result("Q1", 90, "pass"),
            _make_result("Q2", 65, "warn"),
            _make_result("Q3", 30, "fail"),
        ],
    )


def test_dpo_export():
    summary = _make_summary()
    exporter = Exporter(summary)

    pairs = exporter.as_dpo()
    assert len(pairs) == 3
    assert pairs[0]["prompt"] == "Q1"
    assert pairs[0]["chosen"] == "expected"
    assert pairs[0]["rejected"] == "agent"


def test_dpo_filter_by_score():
    summary = _make_summary()
    exporter = Exporter(summary)

    pairs = exporter.as_dpo(max_score=60)
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "Q3"


def test_sft_export():
    summary = _make_summary()
    exporter = Exporter(summary)

    examples = exporter.as_sft(min_score=80)
    assert len(examples) == 1
    assert examples[0]["prompt"] == "Q1"
    assert examples[0]["completion"] == "expected"
    assert examples[0]["metadata"]["source"] == "cane-eval"


def test_openai_export():
    summary = _make_summary()
    exporter = Exporter(summary)

    examples = exporter.as_openai(min_score=80, system_prompt="You are helpful.")
    assert len(examples) == 1
    msgs = examples[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_raw_export():
    summary = _make_summary()
    exporter = Exporter(summary)

    raw = exporter.as_raw()
    assert len(raw) == 3
    assert raw[0]["question"] == "Q1"
    assert raw[0]["overall_score"] == 90


def test_jsonl_string():
    summary = _make_summary()
    exporter = Exporter(summary)

    jsonl = exporter.as_dpo_string()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 3

    # Each line should be valid JSON
    for line in lines:
        parsed = json.loads(line)
        assert "prompt" in parsed


def test_to_file():
    summary = _make_summary()
    exporter = Exporter(summary)

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        exporter.to_file(path, format="dpo")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3
    finally:
        os.unlink(path)


def test_filter_by_status():
    summary = _make_summary()
    exporter = Exporter(summary)

    pairs = exporter.as_dpo(status="fail")
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == "Q3"


def test_filter_by_tags():
    summary = _make_summary()
    exporter = Exporter(summary)

    pairs = exporter.as_dpo(tags=["test"])
    assert len(pairs) == 3

    pairs = exporter.as_dpo(tags=["nonexistent"])
    assert len(pairs) == 0


def test_skip_no_expected_answer():
    summary = RunSummary(
        suite_name="Test",
        total=1,
        results=[_make_result("Q1", 50, "fail", expected="", agent="bad answer")],
    )
    exporter = Exporter(summary)
    pairs = exporter.as_dpo()
    assert len(pairs) == 0  # skipped because no expected answer
