"""Tests for the judge module (unit tests that don't call the API)."""

import pytest

from cane_eval.judge import Judge, JudgeResult, CriteriaScore


def test_classify_pass():
    assert Judge.classify(90) == "pass"
    assert Judge.classify(80) == "pass"
    assert Judge.classify(100) == "pass"


def test_classify_warn():
    assert Judge.classify(79) == "warn"
    assert Judge.classify(60) == "warn"
    assert Judge.classify(70) == "warn"


def test_classify_fail():
    assert Judge.classify(59) == "fail"
    assert Judge.classify(0) == "fail"
    assert Judge.classify(30) == "fail"


def test_judge_result_score_dict():
    result = JudgeResult(
        criteria_scores=[
            CriteriaScore(key="accuracy", score=85, reasoning="Good"),
            CriteriaScore(key="tone", score=70, reasoning="Ok"),
        ],
        overall_score=77.5,
        status="warn",
    )
    scores = result.score_dict()
    assert scores == {"accuracy": 85, "tone": 70}


def test_judge_result_full_dict():
    result = JudgeResult(
        criteria_scores=[
            CriteriaScore(key="accuracy", score=85, reasoning="Good"),
        ],
    )
    full = result.full_dict()
    assert full["accuracy"]["score"] == 85
    assert full["accuracy"]["reasoning"] == "Good"


def test_compute_overall():
    judge = Judge.__new__(Judge)  # skip __init__ (no API key needed)

    criteria = [
        {"key": "a", "weight": 60, "is_enabled": True},
        {"key": "b", "weight": 40, "is_enabled": True},
    ]
    scores = {"a": {"score": 80}, "b": {"score": 60}}

    result = judge._compute_overall(scores, criteria)
    expected = 80 * 0.6 + 60 * 0.4  # 48 + 24 = 72
    assert result == 72.0


def test_compute_overall_disabled_criterion():
    judge = Judge.__new__(Judge)

    criteria = [
        {"key": "a", "weight": 60, "is_enabled": True},
        {"key": "b", "weight": 40, "is_enabled": False},
    ]
    scores = {"a": {"score": 80}, "b": {"score": 20}}

    result = judge._compute_overall(scores, criteria)
    assert result == 80.0  # only "a" counts


def test_compute_overall_zero_weight():
    judge = Judge.__new__(Judge)

    criteria = [
        {"key": "a", "weight": 0, "is_enabled": True},
    ]
    scores = {"a": {"score": 80}}

    result = judge._compute_overall(scores, criteria)
    assert result == 0.0


def test_parse_response_valid_json():
    judge = Judge.__new__(Judge)

    criteria = [{"key": "accuracy"}]
    raw = '{"criteria_scores": {"accuracy": {"score": 90, "reasoning": "Great"}}, "overall_reasoning": "Well done"}'

    parsed = judge._parse_response(raw, criteria)
    assert parsed["criteria_scores"]["accuracy"]["score"] == 90


def test_parse_response_with_markdown_fences():
    judge = Judge.__new__(Judge)

    criteria = [{"key": "accuracy"}]
    raw = '```json\n{"criteria_scores": {"accuracy": {"score": 75}}, "overall_reasoning": "Ok"}\n```'

    parsed = judge._parse_response(raw, criteria)
    assert parsed["criteria_scores"]["accuracy"]["score"] == 75


def test_parse_response_invalid_json():
    judge = Judge.__new__(Judge)

    criteria = [{"key": "accuracy"}, {"key": "tone"}]
    raw = "this is not json at all"

    parsed = judge._parse_response(raw, criteria)
    assert parsed["criteria_scores"]["accuracy"]["score"] == 50
    assert parsed["criteria_scores"]["tone"]["score"] == 50


def test_build_prompt_with_custom_rules():
    judge = Judge.__new__(Judge)

    criteria = [{"key": "accuracy", "label": "Accuracy", "description": "Correctness"}]
    custom_rules = ["Be polite", "Never guess"]

    prompt = judge._build_prompt(
        question="What is 2+2?",
        expected_answer="4",
        agent_answer="22",
        criteria=criteria,
        custom_rules=custom_rules,
    )

    assert "What is 2+2?" in prompt
    assert "22" in prompt
    assert "Be polite" in prompt
    assert "Never guess" in prompt


def test_build_prompt_without_expected():
    judge = Judge.__new__(Judge)

    criteria = [{"key": "accuracy", "label": "Accuracy", "description": "Correctness"}]

    prompt = judge._build_prompt(
        question="Hello?",
        expected_answer="",
        agent_answer="Hi",
        criteria=criteria,
    )

    assert "Expected Answer" not in prompt
