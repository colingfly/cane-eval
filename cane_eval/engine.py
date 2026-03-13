"""
engine.py -- Eval runner.

Orchestrates running a test suite against an agent target,
collecting judge scores, and producing a run summary.
"""

import json
import time
import subprocess
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from cane_eval.suite import TestSuite, TestCase, AgentTarget
from cane_eval.judge import Judge, JudgeResult


@dataclass
class EvalResult:
    """Result for a single test case."""
    question: str
    expected_answer: str
    agent_answer: str
    judge_result: JudgeResult
    sources: list[str] = field(default_factory=list)
    response_time_ms: int = 0
    tags: list[str] = field(default_factory=list)
    context: Optional[str] = None

    @property
    def score(self) -> float:
        return self.judge_result.overall_score

    @property
    def status(self) -> str:
        return self.judge_result.status

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "agent_answer": self.agent_answer,
            "overall_score": self.score,
            "status": self.status,
            "criteria_scores": self.judge_result.score_dict(),
            "judge_reasoning": self.judge_result.overall_reasoning,
            "response_time_ms": self.response_time_ms,
            "tags": self.tags,
        }


@dataclass
class RunSummary:
    """Summary of a complete eval run."""
    suite_name: str
    total: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    overall_score: float = 0.0
    results: list[EvalResult] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0

    def failures(self) -> list[EvalResult]:
        """Return only failing results."""
        return [r for r in self.results if r.status == "fail"]

    def warnings(self) -> list[EvalResult]:
        """Return only warning results."""
        return [r for r in self.results if r.status == "warn"]

    def by_tag(self, tag: str) -> list[EvalResult]:
        """Return results matching a specific tag."""
        return [r for r in self.results if tag in r.tags]

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "overall_score": self.overall_score,
            "pass_rate": round(self.pass_rate, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }


# ---- Agent callers ----

def _extract_by_path(data, path: str) -> str:
    """Extract a value from nested dict using dot notation: 'data.response.text'"""
    if not path:
        return str(data) if data else ""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return str(current)
    return str(current) if current else ""


def _call_http_agent(question: str, target: AgentTarget) -> dict:
    """Call an HTTP agent endpoint."""
    start = time.time()

    try:
        payload_str = target.payload_template.replace("{{question}}", question)
        payload_bytes = payload_str.encode("utf-8")

        headers = dict(target.headers)
        headers.setdefault("Content-Type", "application/json")

        method = target.method.upper()

        req = urllib.request.Request(
            target.url,
            data=payload_bytes if method == "POST" else None,
            headers=headers,
            method=method,
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")

        try:
            resp_data = json.loads(body)
            answer = _extract_by_path(resp_data, target.response_path)
        except json.JSONDecodeError:
            answer = body.strip()

        if not answer:
            answer = "(Agent returned an empty response)"

        elapsed = int((time.time() - start) * 1000)
        return {"answer": answer, "sources": [], "response_time_ms": elapsed}

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {
            "answer": f"Error calling agent: {str(e)}",
            "sources": [],
            "response_time_ms": elapsed,
        }


def _call_command_agent(question: str, target: AgentTarget) -> dict:
    """Call a CLI agent via subprocess."""
    start = time.time()

    try:
        cmd = target.command.replace("{{question}}", question)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        answer = result.stdout.strip()
        if not answer:
            answer = result.stderr.strip() or "(Command produced no output)"

        elapsed = int((time.time() - start) * 1000)
        return {"answer": answer, "sources": [], "response_time_ms": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = int((time.time() - start) * 1000)
        return {"answer": "Error: Command timed out after 60s", "sources": [], "response_time_ms": elapsed}
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {"answer": f"Error: {str(e)}", "sources": [], "response_time_ms": elapsed}


# ---- Runner ----

class EvalRunner:
    """
    Run a test suite against an agent and collect judge scores.

    Usage:
        suite = TestSuite.from_yaml("tests.yaml")
        runner = EvalRunner(api_key="sk-ant-...")

        # Option 1: Agent callable
        summary = runner.run(suite, agent=lambda q: my_agent(q))

        # Option 2: HTTP target defined in YAML
        summary = runner.run(suite)

        # Option 3: CLI command target
        summary = runner.run(suite)

        print(f"Score: {summary.overall_score} | {summary.passed}P {summary.warned}W {summary.failed}F")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = True,
        on_result: Optional[Callable[[EvalResult, int, int], None]] = None,
    ):
        """
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Override judge model (default: suite's model)
            verbose: Print progress to stdout
            on_result: Callback after each result: fn(result, index, total)
        """
        self.api_key = api_key
        self.model_override = model
        self.verbose = verbose
        self.on_result = on_result

    def run(
        self,
        suite: TestSuite,
        agent: Optional[Callable[[str], str]] = None,
        tags: Optional[list[str]] = None,
    ) -> RunSummary:
        """
        Execute eval run.

        Args:
            suite: Test suite to run
            agent: Optional callable that takes a question and returns an answer string.
                   If not provided, uses the suite's target config (HTTP or command).
            tags: Optional tag filter -- only run tests matching these tags

        Returns:
            RunSummary with all results
        """
        judge = Judge(
            api_key=self.api_key,
            model=self.model_override or suite.model,
        )

        # Filter tests by tags if specified
        tests = suite.filter_by_tags(tags) if tags else suite.tests
        if not tests:
            return RunSummary(suite_name=suite.name)

        criteria = suite.criteria_dicts()
        custom_rules = suite.custom_rules

        start_time = time.time()
        started_at = datetime.utcnow().isoformat()

        results = []

        for i, tc in enumerate(tests):
            if self.verbose:
                q_preview = tc.question[:60] + "..." if len(tc.question) > 60 else tc.question
                print(f"  [{i+1}/{len(tests)}] {q_preview}")

            # Step 1: Get agent answer
            agent_result = self._get_answer(tc, suite.target, agent)

            # Step 2: Judge the response
            try:
                judge_result = judge.score(
                    question=tc.question,
                    agent_answer=agent_result["answer"],
                    expected_answer=tc.expected_answer or "",
                    criteria=criteria,
                    custom_rules=custom_rules,
                    context=tc.context,
                )
            except Exception as e:
                if self.verbose:
                    print(f"    Judge error: {e}")
                judge_result = JudgeResult(
                    overall_score=50,
                    overall_reasoning=f"Judge failed: {str(e)}",
                    status="warn",
                )

            # Build result
            eval_result = EvalResult(
                question=tc.question,
                expected_answer=tc.expected_answer or "",
                agent_answer=agent_result["answer"],
                judge_result=judge_result,
                sources=agent_result.get("sources", []),
                response_time_ms=agent_result.get("response_time_ms", 0),
                tags=tc.tags,
                context=tc.context,
            )
            results.append(eval_result)

            if self.verbose:
                status_icon = {"pass": "P", "warn": "W", "fail": "F"}.get(eval_result.status, "?")
                print(f"    Score: {eval_result.score} ({status_icon})")

            if self.on_result:
                self.on_result(eval_result, i + 1, len(tests))

        # Compute summary
        duration = time.time() - start_time
        overall = round(sum(r.score for r in results) / len(results), 1) if results else 0.0

        summary = RunSummary(
            suite_name=suite.name,
            total=len(results),
            passed=sum(1 for r in results if r.status == "pass"),
            warned=sum(1 for r in results if r.status == "warn"),
            failed=sum(1 for r in results if r.status == "fail"),
            overall_score=overall,
            results=results,
            started_at=started_at,
            completed_at=datetime.utcnow().isoformat(),
            duration_seconds=duration,
        )

        if self.verbose:
            print(f"\n  Done: {summary.overall_score} ({summary.passed}P/{summary.warned}W/{summary.failed}F) in {duration:.1f}s")

        return summary

    def _get_answer(
        self,
        tc: TestCase,
        target: AgentTarget,
        agent: Optional[Callable],
    ) -> dict:
        """Get agent answer using callable, HTTP, or command."""
        start = time.time()

        if agent is not None:
            try:
                answer = agent(tc.question)
                elapsed = int((time.time() - start) * 1000)
                return {"answer": str(answer), "sources": [], "response_time_ms": elapsed}
            except Exception as e:
                elapsed = int((time.time() - start) * 1000)
                return {"answer": f"Agent error: {str(e)}", "sources": [], "response_time_ms": elapsed}

        if target.type == "http" and target.url:
            return _call_http_agent(tc.question, target)

        if target.type == "command" and target.command:
            return _call_command_agent(tc.question, target)

        return {
            "answer": "(No agent configured. Pass an agent callable or set a target in your YAML suite.)",
            "sources": [],
            "response_time_ms": 0,
        }
