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
from typing import Optional, Callable, Generator

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
    schema_result: Optional[SchemaResult] = None

    @property
    def score(self) -> float:
        return self.judge_result.overall_score

    @property
    def status(self) -> str:
        return self.judge_result.status

    def to_dict(self) -> dict:
        d = {
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
        if self.schema_result is not None:
            d["schema"] = self.schema_result.to_dict()
        return d


@dataclass
class LatencyStats:
    """Latency statistics for an eval run."""
    p50_ms: int = 0
    p95_ms: int = 0
    p99_ms: int = 0
    max_ms: int = 0
    min_ms: int = 0
    mean_ms: int = 0

    @classmethod
    def from_results(cls, results: list) -> "LatencyStats":
        """Compute latency stats from a list of EvalResults."""
        times = sorted(r.response_time_ms for r in results if r.response_time_ms > 0)
        if not times:
            return cls()

        def percentile(sorted_vals, pct):
            idx = int(len(sorted_vals) * pct / 100)
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

        return cls(
            p50_ms=percentile(times, 50),
            p95_ms=percentile(times, 95),
            p99_ms=percentile(times, 99),
            max_ms=times[-1],
            min_ms=times[0],
            mean_ms=int(sum(times) / len(times)),
        )

    def to_dict(self) -> dict:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
        }


@dataclass
class SchemaResult:
    """Schema validation result for a single test case."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"valid": self.valid, "errors": self.errors}


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
    latency: Optional[LatencyStats] = None
    schema_pass: int = 0
    schema_fail: int = 0
    reliability_score: Optional[float] = None
    reliability_grade: str = ""

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
        d = {
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
        if self.latency:
            d["latency"] = self.latency.to_dict()
        if self.schema_pass or self.schema_fail:
            d["schema_pass"] = self.schema_pass
            d["schema_fail"] = self.schema_fail
        if self.reliability_score is not None:
            d["reliability_score"] = self.reliability_score
            d["reliability_grade"] = self.reliability_grade
        return d


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
        provider: str = "anthropic",
        base_url: Optional[str] = None,
        schema: Optional[dict] = None,
        latency_p95: Optional[int] = None,
        latency_target: int = 5000,
    ):
        """
        Args:
            api_key: API key for the judge provider (or set env var)
            model: Override judge model (default: suite's model)
            verbose: Print progress to stdout
            on_result: Callback after each result: fn(result, index, total)
            provider: Judge provider ("anthropic", "openai", "gemini", "openai-compatible")
            base_url: Base URL for OpenAI-compatible endpoints
            schema: JSON Schema dict to validate agent responses against
            latency_p95: p95 latency threshold in ms (fail run if exceeded)
            latency_target: Target latency in ms for reliability scoring (default: 5000)
        """
        self.api_key = api_key
        self.model_override = model
        self.verbose = verbose
        self.on_result = on_result
        self.provider = provider
        self.base_url = base_url
        self.schema = schema
        self.latency_p95 = latency_p95
        self.latency_target = latency_target

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
            provider=self.provider,
            base_url=self.base_url,
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

            # Schema validation
            schema_result = None
            if self.schema:
                schema_result = self._validate_schema(agent_result["answer"])

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
                schema_result=schema_result,
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

        # Latency stats
        latency_stats = LatencyStats.from_results(results)

        # Schema stats
        schema_pass = sum(1 for r in results if r.schema_result and r.schema_result.valid)
        schema_fail = sum(1 for r in results if r.schema_result and not r.schema_result.valid)

        # Reliability score
        reliability_score, reliability_grade = self._compute_reliability(
            results, latency_stats, overall, schema_pass, schema_fail,
        )

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
            latency=latency_stats,
            schema_pass=schema_pass,
            schema_fail=schema_fail,
            reliability_score=reliability_score,
            reliability_grade=reliability_grade,
        )

        if self.verbose:
            print(f"\n  Done: {summary.overall_score} ({summary.passed}P/{summary.warned}W/{summary.failed}F) in {duration:.1f}s")

        return summary

    def run_stream(
        self,
        suite: TestSuite,
        agent: Optional[Callable[[str], str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Generator[EvalResult, None, RunSummary]:
        """
        Execute eval run, yielding each result as it completes.

        Usage:
            gen = runner.run_stream(suite, agent=my_agent)
            for result in gen:
                print(f"{result.status}: {result.score}")
            summary = gen.value  # available after iteration

        Or capture the summary via wrapper:
            results = []
            summary = None
            gen = runner.run_stream(suite)
            try:
                while True:
                    result = next(gen)
                    results.append(result)
            except StopIteration as e:
                summary = e.value

        Args:
            suite: Test suite to run
            agent: Optional callable that takes a question and returns an answer
            tags: Optional tag filter

        Yields:
            EvalResult for each completed test case

        Returns:
            RunSummary (accessible via StopIteration.value)
        """
        judge = Judge(
            api_key=self.api_key,
            model=self.model_override or suite.model,
            provider=self.provider,
            base_url=self.base_url,
        )

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

            agent_result = self._get_answer(tc, suite.target, agent)

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

            schema_result = None
            if self.schema:
                schema_result = self._validate_schema(agent_result["answer"])

            eval_result = EvalResult(
                question=tc.question,
                expected_answer=tc.expected_answer or "",
                agent_answer=agent_result["answer"],
                judge_result=judge_result,
                sources=agent_result.get("sources", []),
                response_time_ms=agent_result.get("response_time_ms", 0),
                tags=tc.tags,
                context=tc.context,
                schema_result=schema_result,
            )
            results.append(eval_result)

            if self.on_result:
                self.on_result(eval_result, i + 1, len(tests))

            yield eval_result

        duration = time.time() - start_time
        overall = round(sum(r.score for r in results) / len(results), 1) if results else 0.0

        latency_stats = LatencyStats.from_results(results)
        schema_pass = sum(1 for r in results if r.schema_result and r.schema_result.valid)
        schema_fail = sum(1 for r in results if r.schema_result and not r.schema_result.valid)
        reliability_score, reliability_grade = self._compute_reliability(
            results, latency_stats, overall, schema_pass, schema_fail,
        )

        return RunSummary(
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
            latency=latency_stats,
            schema_pass=schema_pass,
            schema_fail=schema_fail,
            reliability_score=reliability_score,
            reliability_grade=reliability_grade,
        )

    @staticmethod
    def preflight(
        suite: TestSuite,
        timeout: int = 5,
        verbose: bool = True,
        provider: str = "anthropic",
    ) -> dict:
        """
        Pre-flight health check: verify agent endpoints are reachable
        before burning API credits on a full eval run.

        Checks:
        - HTTP targets: sends a HEAD/GET request to verify connectivity
        - Command targets: verifies the command binary exists
        - Callable targets: always pass (no remote dependency)

        Args:
            suite: Test suite with target config
            timeout: Max seconds to wait per check
            verbose: Print results to stdout

        Returns:
            dict with 'ok' (bool), 'checks' (list of check results),
            and 'errors' (list of error messages)
        """
        checks = []
        errors = []
        target = suite.target

        if target.type == "http" and target.url:
            url = target.url
            check = {"type": "http", "url": url, "status": "unknown"}

            try:
                req = urllib.request.Request(url, method="HEAD")
                for k, v in target.headers.items():
                    req.add_header(k, v)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    check["status"] = "ok"
                    check["http_status"] = resp.status
                    if verbose:
                        print(f"  [OK] {url} (HTTP {resp.status})")
            except urllib.error.HTTPError as e:
                # 4xx/5xx but the server is reachable
                if e.code in (400, 404, 405, 422):
                    check["status"] = "ok"
                    check["http_status"] = e.code
                    check["note"] = "HEAD not supported, but endpoint is reachable"
                    if verbose:
                        print(f"  [OK] {url} (reachable, HTTP {e.code})")
                else:
                    check["status"] = "error"
                    check["error"] = f"HTTP {e.code}: {e.reason}"
                    errors.append(f"HTTP target returned {e.code}: {e.reason}")
                    if verbose:
                        print(f"  [FAIL] {url} (HTTP {e.code}: {e.reason})")
            except urllib.error.URLError as e:
                check["status"] = "error"
                check["error"] = str(e.reason)
                errors.append(f"Cannot reach {url}: {e.reason}")
                if verbose:
                    print(f"  [FAIL] {url} ({e.reason})")
            except Exception as e:
                check["status"] = "error"
                check["error"] = str(e)
                errors.append(f"Cannot reach {url}: {e}")
                if verbose:
                    print(f"  [FAIL] {url} ({e})")

            checks.append(check)

        elif target.type == "command" and target.command:
            binary = target.command.split()[0]
            check = {"type": "command", "binary": binary, "status": "unknown"}

            import shutil
            if shutil.which(binary):
                check["status"] = "ok"
                if verbose:
                    print(f"  [OK] command '{binary}' found")
            else:
                check["status"] = "error"
                check["error"] = f"Command '{binary}' not found in PATH"
                errors.append(f"Command '{binary}' not found in PATH")
                if verbose:
                    print(f"  [FAIL] command '{binary}' not found")

            checks.append(check)

        elif target.type == "callable":
            check = {"type": "callable", "status": "ok"}
            if verbose:
                print(f"  [OK] callable target (no remote dependency)")
            checks.append(check)

        # Check API key availability for the selected provider
        import os
        from cane_eval.providers import PROVIDERS, PROVIDER_ALIASES

        resolved_provider = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
        provider_cls = PROVIDERS.get(resolved_provider)
        env_key_name = provider_cls.env_key() if provider_cls else "ANTHROPIC_API_KEY"

        # OpenAI-compatible with local endpoints may not need a key
        skip_key_check = (resolved_provider == "openai-compatible")

        api_key = os.environ.get(env_key_name, "")
        key_check = {"type": "api_key", "status": "unknown"}
        if api_key or skip_key_check:
            key_check["status"] = "ok"
            if verbose:
                if skip_key_check and not api_key:
                    print(f"  [OK] {env_key_name} not required for local endpoints")
                else:
                    print(f"  [OK] {env_key_name} is set")
        else:
            key_check["status"] = "error"
            key_check["error"] = f"{env_key_name} not set"
            errors.append(f"{env_key_name} environment variable not set")
            if verbose:
                print(f"  [FAIL] {env_key_name} not set")
        checks.append(key_check)

        ok = len(errors) == 0
        if verbose:
            if ok:
                print(f"\n  Pre-flight: all checks passed")
            else:
                print(f"\n  Pre-flight: {len(errors)} issue(s) found")

        return {"ok": ok, "checks": checks, "errors": errors}

    def _validate_schema(self, agent_answer: str) -> SchemaResult:
        """Validate agent answer against JSON schema."""
        try:
            import jsonschema
        except ImportError:
            return SchemaResult(valid=False, errors=["jsonschema package not installed. pip install jsonschema"])

        # Try to parse as JSON
        try:
            data = json.loads(agent_answer)
        except (json.JSONDecodeError, TypeError):
            return SchemaResult(valid=False, errors=["Response is not valid JSON"])

        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return SchemaResult(valid=True)
        except jsonschema.ValidationError as e:
            return SchemaResult(valid=False, errors=[e.message])
        except jsonschema.SchemaError as e:
            return SchemaResult(valid=False, errors=[f"Invalid schema: {e.message}"])

    def _compute_reliability(
        self,
        results: list,
        latency: LatencyStats,
        accuracy_score: float,
        schema_pass: int,
        schema_fail: int,
    ) -> tuple:
        """
        Compute Agent Reliability Score.

        Three pillars:
        - Correctness (accuracy_score from judge, 0-100)
        - Structural (schema adherence, 0 or 100 per test)
        - Performance (latency vs target)

        Returns (score, grade) tuple.
        """
        has_schema = (schema_pass + schema_fail) > 0
        has_latency = latency.p95_ms > 0

        # Correctness: just the judge score
        correctness = accuracy_score

        # Schema: percentage valid
        if has_schema:
            schema_score = (schema_pass / (schema_pass + schema_fail)) * 100
        else:
            schema_score = None

        # Latency: score based on p95 vs target
        if has_latency:
            target = self.latency_target
            if latency.p95_ms <= target:
                latency_score = 100.0
            else:
                # Linearly degrade: at 2x target = 0
                latency_score = max(0, 100 - ((latency.p95_ms - target) / target) * 100)
        else:
            latency_score = None

        # Weighted composite
        if has_schema and has_latency:
            # All three pillars
            reliability = (correctness * 0.5) + (schema_score * 0.25) + (latency_score * 0.25)
        elif has_schema:
            reliability = (correctness * 0.6) + (schema_score * 0.4)
        elif has_latency:
            reliability = (correctness * 0.7) + (latency_score * 0.3)
        else:
            # Only correctness available
            reliability = correctness

        reliability = round(reliability, 1)

        # Grade
        if reliability >= 90:
            grade = "A"
        elif reliability >= 75:
            grade = "B"
        elif reliability >= 60:
            grade = "C"
        elif reliability >= 40:
            grade = "D"
        else:
            grade = "F"

        return reliability, grade

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
