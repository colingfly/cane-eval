"""Tests for the engine module (unit tests that don't call the API)."""

from cane_eval.engine import _extract_by_path, EvalResult, RunSummary
from cane_eval.judge import JudgeResult, CriteriaScore


def test_extract_by_path_simple():
    data = {"response": "hello"}
    assert _extract_by_path(data, "response") == "hello"


def test_extract_by_path_nested():
    data = {"data": {"answer": {"text": "hello"}}}
    assert _extract_by_path(data, "data.answer.text") == "hello"


def test_extract_by_path_empty():
    data = {"response": "hello"}
    assert _extract_by_path(data, "") == str(data)


def test_extract_by_path_missing():
    data = {"response": "hello"}
    result = _extract_by_path(data, "nonexistent")
    assert result == ""


def test_eval_result_properties():
    result = EvalResult(
        question="Q",
        expected_answer="A",
        agent_answer="B",
        judge_result=JudgeResult(
            overall_score=85,
            status="pass",
        ),
    )
    assert result.score == 85
    assert result.status == "pass"


def test_eval_result_to_dict():
    result = EvalResult(
        question="Q",
        expected_answer="A",
        agent_answer="B",
        judge_result=JudgeResult(
            criteria_scores=[CriteriaScore(key="accuracy", score=85)],
            overall_score=85,
            overall_reasoning="Good",
            status="pass",
        ),
        tags=["test"],
    )
    d = result.to_dict()
    assert d["question"] == "Q"
    assert d["overall_score"] == 85
    assert d["status"] == "pass"
    assert d["criteria_scores"] == {"accuracy": 85}


def test_run_summary_pass_rate():
    summary = RunSummary(
        suite_name="Test",
        total=4,
        passed=3,
        warned=0,
        failed=1,
    )
    assert summary.pass_rate == 75.0


def test_run_summary_pass_rate_zero():
    summary = RunSummary(suite_name="Test", total=0)
    assert summary.pass_rate == 0.0


def test_run_summary_failures():
    r1 = EvalResult(
        question="Q1", expected_answer="", agent_answer="",
        judge_result=JudgeResult(overall_score=90, status="pass"),
    )
    r2 = EvalResult(
        question="Q2", expected_answer="", agent_answer="",
        judge_result=JudgeResult(overall_score=30, status="fail"),
    )
    summary = RunSummary(suite_name="Test", total=2, results=[r1, r2])

    failures = summary.failures()
    assert len(failures) == 1
    assert failures[0].question == "Q2"


def test_run_summary_by_tag():
    r1 = EvalResult(
        question="Q1", expected_answer="", agent_answer="",
        judge_result=JudgeResult(overall_score=90, status="pass"),
        tags=["math"],
    )
    r2 = EvalResult(
        question="Q2", expected_answer="", agent_answer="",
        judge_result=JudgeResult(overall_score=30, status="fail"),
        tags=["science"],
    )
    summary = RunSummary(suite_name="Test", total=2, results=[r1, r2])

    math = summary.by_tag("math")
    assert len(math) == 1
    assert math[0].question == "Q1"
