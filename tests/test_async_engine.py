"""Tests for async/parallel execution in the engine."""

import asyncio
import time
from unittest.mock import MagicMock, patch

from cane_eval.engine import ReliabilityRunner, ReliabilityResult, ReliabilitySummary
from cane_eval.suite import ReliabilitySuite, ReliabilityCase
from cane_eval.judge import JudgeResult, CriteriaScore


def _make_suite(n_tests=5):
    """Create a test suite with n mock test cases."""
    tests = [
        ReliabilityCase(
            question=f"Question {i}",
            expected_answer=f"Answer {i}",
            tags=["test"],
        )
        for i in range(n_tests)
    ]
    return ReliabilitySuite(
        name="Async Test Suite",
        tests=tests,
    )


def _mock_judge_score(**kwargs):
    """Return a mock JudgeResult."""
    return JudgeResult(
        criteria_scores=[
            CriteriaScore(key="accuracy", score=80.0),
            CriteriaScore(key="completeness", score=75.0),
        ],
        overall_score=78.0,
        overall_reasoning="Good answer",
        status="pass",
    )


def _slow_agent(question: str) -> str:
    """Agent that sleeps briefly to simulate latency."""
    time.sleep(0.05)
    return f"Answer to: {question}"


def _fast_agent(question: str) -> str:
    """Agent that responds immediately."""
    return f"Answer to: {question}"


# ---- Tests ----

def test_run_async_basic():
    """run_async() returns a valid summary."""
    suite = _make_suite(3)

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        runner = ReliabilityRunner(api_key="test-key", verbose=False)
        summary = asyncio.run(runner.run_async(suite, agent=_fast_agent))

    assert isinstance(summary, ReliabilitySummary)
    assert summary.total == 3
    assert summary.passed == 3
    assert len(summary.results) == 3


def test_run_async_preserves_order():
    """Results maintain the same order as test cases."""
    suite = _make_suite(5)

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        runner = ReliabilityRunner(api_key="test-key", verbose=False)
        summary = asyncio.run(runner.run_async(suite, agent=_fast_agent))

    for i, result in enumerate(summary.results):
        assert result.question == f"Question {i}"


def test_run_async_concurrency_faster():
    """Concurrent execution is faster than sequential for slow agents."""
    suite = _make_suite(5)

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        # Sequential
        runner_seq = ReliabilityRunner(api_key="test-key", verbose=False, concurrency=1)
        start = time.time()
        asyncio.run(runner_seq.run_async(suite, agent=_slow_agent))
        seq_time = time.time() - start

        # Concurrent
        runner_par = ReliabilityRunner(api_key="test-key", verbose=False, concurrency=5)
        start = time.time()
        asyncio.run(runner_par.run_async(suite, agent=_slow_agent))
        par_time = time.time() - start

    # Parallel should be meaningfully faster
    assert par_time < seq_time * 0.8


def test_run_with_concurrency():
    """run() with concurrency > 1 still works (calls run_async internally)."""
    suite = _make_suite(3)

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        runner = ReliabilityRunner(api_key="test-key", verbose=False, concurrency=3)
        summary = runner.run(suite, agent=_fast_agent)

    assert summary.total == 3


def test_run_sequential_default():
    """run() with default concurrency=1 runs sequentially."""
    suite = _make_suite(3)

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        runner = ReliabilityRunner(api_key="test-key", verbose=False)
        assert runner.concurrency == 1
        summary = runner.run(suite, agent=_fast_agent)

    assert summary.total == 3


def test_run_async_handles_agent_error():
    """Agent errors are caught and reported, not raised."""
    suite = _make_suite(2)

    def failing_agent(q):
        raise RuntimeError("Agent crashed")

    with patch("cane_eval.engine.Judge") as MockJudge:
        MockJudge.return_value.score = _mock_judge_score

        runner = ReliabilityRunner(api_key="test-key", verbose=False, concurrency=2)
        summary = asyncio.run(runner.run_async(suite, agent=failing_agent))

    assert summary.total == 2
    for r in summary.results:
        assert "Agent error" in r.agent_answer


def test_backward_compat_aliases():
    """Old names EvalRunner, TestSuite, TestCase still work."""
    from cane_eval.engine import EvalRunner
    from cane_eval.suite import TestSuite, TestCase

    assert EvalRunner is ReliabilityRunner
    assert TestSuite is ReliabilitySuite
    assert TestCase is ReliabilityCase
