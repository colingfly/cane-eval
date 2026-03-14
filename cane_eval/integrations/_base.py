"""
_base.py -- Shared helpers for integrations.

All integrations follow the same pattern:
1. Adapt the framework's agent into a callable(str) -> str
2. Load the test suite from YAML or dict
3. Run eval via EvalRunner
4. Optionally run mining, RCA, or push to Cane Cloud
5. Return RunSummary
"""

import os
from pathlib import Path
from typing import Optional, Callable, Union

from cane_eval.suite import TestSuite
from cane_eval.engine import EvalRunner, RunSummary


def _load_suite(suite: Union[str, Path, dict, TestSuite]) -> TestSuite:
    """Load suite from YAML path, dict, or pass through TestSuite."""
    if isinstance(suite, TestSuite):
        return suite
    if isinstance(suite, dict):
        return TestSuite.from_dict(suite)
    return TestSuite.from_yaml(str(suite))


def _run_eval(
    agent_fn: Callable[[str], str],
    suite: Union[str, Path, dict, TestSuite],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    tags: Optional[list[str]] = None,
    verbose: bool = True,
    mine: bool = False,
    mine_threshold: float = 60,
    rca: bool = False,
    rca_threshold: float = 60,
    cloud: Optional[str] = None,
    cloud_api_key: Optional[str] = None,
    environment_id: Optional[str] = None,
) -> RunSummary:
    """
    Core eval runner used by all integrations.

    Args:
        agent_fn: Callable that takes a question string and returns an answer string.
        suite: YAML path, dict config, or TestSuite object.
        api_key: Anthropic API key for judging.
        model: Override judge model.
        tags: Only run tests matching these tags.
        verbose: Print progress to stdout.
        mine: Run failure mining after eval.
        mine_threshold: Score threshold for mining failures.
        rca: Run root cause analysis after eval.
        rca_threshold: Score threshold for RCA.
        cloud: Cane Cloud URL to push results to (e.g. "https://app.cane.dev").
        cloud_api_key: API key for Cane Cloud.
        environment_id: Environment ID for Cane Cloud.

    Returns:
        RunSummary with all results, plus .mining_result and .rca_result
        attributes if those were requested.
    """
    loaded_suite = _load_suite(suite)
    anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    runner = EvalRunner(
        api_key=anthropic_key,
        model=model,
        verbose=verbose,
    )

    summary = runner.run(loaded_suite, agent=agent_fn, tags=tags)

    # Optional: failure mining
    if mine and summary.failed > 0:
        from cane_eval.mining import FailureMiner
        miner = FailureMiner(
            api_key=anthropic_key,
            model=model or loaded_suite.model,
            verbose=verbose,
        )
        mining_result = miner.mine(summary, max_score=mine_threshold)
        summary.mining_result = mining_result  # type: ignore[attr-defined]

    # Optional: root cause analysis
    if rca and summary.failed > 0:
        from cane_eval.rca import RootCauseAnalyzer
        analyzer = RootCauseAnalyzer(
            api_key=anthropic_key,
            model=model or loaded_suite.model,
            verbose=verbose,
        )
        rca_result = analyzer.analyze(summary, max_score=rca_threshold)
        summary.rca_result = rca_result  # type: ignore[attr-defined]

    # Optional: push to Cane Cloud
    if cloud and cloud_api_key and environment_id:
        _push_to_cloud(summary, cloud, cloud_api_key, environment_id, verbose)

    return summary


def _push_to_cloud(
    summary: RunSummary,
    cloud_url: str,
    api_key: str,
    environment_id: str,
    verbose: bool = True,
) -> None:
    """Push eval results to Cane Cloud platform."""
    import json
    import urllib.request

    url = f"{cloud_url.rstrip('/')}/v1/eval/results/{environment_id}"

    payload = json.dumps({
        "results": [r.to_dict() for r in summary.results],
        "suite_name": summary.suite_name,
        "overall_score": summary.overall_score,
        "total": summary.total,
        "passed": summary.passed,
        "warned": summary.warned,
        "failed": summary.failed,
        "duration_seconds": summary.duration_seconds,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if verbose:
                print(f"  Results pushed to Cane Cloud ({resp.status})")
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to push to Cane Cloud: {e}")
