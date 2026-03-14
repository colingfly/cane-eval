"""
fastapi_agent.py -- FastAPI agent integration for cane-eval.

Evaluate any FastAPI-based agent by pointing at its endpoint, or test it
in-process using the TestClient (no server needed).

Usage:
    # Option 1: Running server
    from cane_eval.integrations import evaluate_fastapi
    results = evaluate_fastapi(
        "http://localhost:8000/ask",
        suite="qa.yaml",
    )

    # Option 2: In-process via TestClient (no server needed)
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/ask")
    def ask(request: dict):
        return {"response": my_agent(request["message"])}

    results = evaluate_fastapi(app, suite="qa.yaml", endpoint="/ask")

    # Option 3: Custom payload shape
    results = evaluate_fastapi(
        "http://localhost:8000/chat",
        suite="qa.yaml",
        payload_template='{"query": "{{question}}"}',
        response_path="data.answer",
    )
"""

import json
import urllib.request
from typing import Any, Optional, Union, Callable
from pathlib import Path

from cane_eval.engine import RunSummary
from cane_eval.integrations._base import _run_eval


def _extract_by_path(data: Any, path: str) -> str:
    """Extract a value from nested dict using dot notation."""
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


def _adapt_fastapi_url(
    url: str,
    payload_template: str,
    response_path: str,
    headers: Optional[dict] = None,
) -> Callable[[str], str]:
    """Create a caller for a running FastAPI server."""
    def call_agent(question: str) -> str:
        payload_str = payload_template.replace("{{question}}", question)
        payload_bytes = payload_str.encode("utf-8")

        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(
            url, data=payload_bytes, headers=req_headers, method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")

        try:
            resp_data = json.loads(body)
            answer = _extract_by_path(resp_data, response_path)
        except json.JSONDecodeError:
            answer = body.strip()

        return answer or "(Agent returned an empty response)"

    return call_agent


def _adapt_fastapi_app(
    app: Any,
    endpoint: str,
    payload_template: str,
    response_path: str,
) -> Callable[[str], str]:
    """Create a caller using FastAPI TestClient (in-process, no server)."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise ImportError(
                "FastAPI TestClient requires 'httpx' to be installed. "
                "Run: pip install httpx"
            )

    client = TestClient(app)

    def call_agent(question: str) -> str:
        payload_str = payload_template.replace("{{question}}", question)
        payload = json.loads(payload_str)

        resp = client.post(endpoint, json=payload)
        resp.raise_for_status()

        resp_data = resp.json()
        answer = _extract_by_path(resp_data, response_path)
        return answer or "(Agent returned an empty response)"

    return call_agent


def evaluate_fastapi(
    target: Any,
    suite: Union[str, Path, dict] = "tests.yaml",
    endpoint: str = "/ask",
    payload_template: str = '{"message": "{{question}}"}',
    response_path: str = "response",
    headers: Optional[dict] = None,
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
    Evaluate a FastAPI agent against a test suite.

    Can target a running server (URL) or a FastAPI app instance (in-process).

    Args:
        target: URL string (e.g. "http://localhost:8000/ask") or a FastAPI
            app instance for in-process testing via TestClient.
        suite: Path to YAML test suite, dict config, or TestSuite object.
        endpoint: Endpoint path when using app instance (default: "/ask").
            Ignored when target is a URL.
        payload_template: JSON template with {{question}} placeholder.
            Default: '{"message": "{{question}}"}'
        response_path: Dot-notation path to extract answer from response JSON.
            Default: "response". Examples: "data.answer", "choices.0.text"
        headers: Extra HTTP headers for URL targets.
        api_key: Anthropic API key for judging (or set ANTHROPIC_API_KEY env var).
        model: Override judge model.
        tags: Only run tests matching these tags.
        verbose: Print progress to stdout.
        mine: Run failure mining after eval.
        mine_threshold: Score threshold for mining failures (default: 60).
        rca: Run root cause analysis after eval.
        rca_threshold: Score threshold for RCA (default: 60).
        cloud: Cane Cloud URL to push results (e.g. "https://app.cane.dev").
        cloud_api_key: API key for Cane Cloud.
        environment_id: Environment ID on Cane Cloud.

    Returns:
        RunSummary with all eval results.

    Example:
        >>> # Running server
        >>> results = evaluate_fastapi(
        ...     "http://localhost:8000/ask",
        ...     suite="qa.yaml",
        ... )
        >>> print(f"Score: {results.overall_score}")

        >>> # In-process
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> results = evaluate_fastapi(app, suite="qa.yaml", endpoint="/ask")
    """
    if isinstance(target, str) and target.startswith("http"):
        agent_fn = _adapt_fastapi_url(
            url=target,
            payload_template=payload_template,
            response_path=response_path,
            headers=headers,
        )
    else:
        agent_fn = _adapt_fastapi_app(
            app=target,
            endpoint=endpoint,
            payload_template=payload_template,
            response_path=response_path,
        )

    return _run_eval(
        agent_fn=agent_fn,
        suite=suite,
        api_key=api_key,
        model=model,
        tags=tags,
        verbose=verbose,
        mine=mine,
        mine_threshold=mine_threshold,
        rca=rca,
        rca_threshold=rca_threshold,
        cloud=cloud,
        cloud_api_key=cloud_api_key,
        environment_id=environment_id,
    )
