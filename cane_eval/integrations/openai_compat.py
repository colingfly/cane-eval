"""
openai_compat.py -- OpenAI-compatible endpoint integration for cane-eval.

Evaluate any OpenAI-compatible API endpoint (OpenAI, vLLM, Ollama, LiteLLM,
LocalAI, text-generation-inference, etc.) with a single function call.

Usage:
    from cane_eval.integrations import evaluate_openai

    # OpenAI API
    results = evaluate_openai(
        "https://api.openai.com/v1/chat/completions",
        suite="qa.yaml",
        openai_api_key="sk-...",
        openai_model="gpt-4o",
    )

    # Local vLLM / Ollama / LiteLLM
    results = evaluate_openai(
        "http://localhost:8000/v1/chat/completions",
        suite="qa.yaml",
        openai_model="llama3",
    )

    # Using the openai SDK client directly
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    results = evaluate_openai(client, suite="qa.yaml", openai_model="llama3")
"""

import json
import urllib.request
from typing import Any, Optional, Union, Callable
from pathlib import Path

from cane_eval.engine import RunSummary
from cane_eval.integrations._base import _run_eval


def _adapt_openai_endpoint(
    endpoint: Any,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Callable[[str], str]:
    """
    Adapt an OpenAI-compatible endpoint or client into a callable.

    Supports:
    - URL string (uses urllib, no openai dependency required)
    - openai.OpenAI client instance
    - Any object with .chat.completions.create()
    """
    # If it's a string URL, use raw HTTP
    if isinstance(endpoint, str) and endpoint.startswith("http"):
        return _make_http_caller(
            url=endpoint,
            api_key=openai_api_key,
            model=openai_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # If it's an OpenAI client object
    if hasattr(endpoint, "chat") and hasattr(endpoint.chat, "completions"):
        return _make_sdk_caller(
            client=endpoint,
            model=openai_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise TypeError(
        f"Expected an OpenAI-compatible URL string or openai.OpenAI client, "
        f"got {type(endpoint).__name__}. "
        "Pass a URL like 'http://localhost:8000/v1/chat/completions' "
        "or an openai.OpenAI() instance."
    )


def _make_http_caller(
    url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Callable[[str], str]:
    """Create a caller using raw HTTP (no openai dependency)."""
    def call_agent(question: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        # Standard OpenAI response format
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")

        return str(body)

    return call_agent


def _make_sdk_caller(
    client: Any,
    model: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Callable[[str], str]:
    """Create a caller using the openai SDK client."""
    def call_agent(question: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    return call_agent


def evaluate_openai(
    endpoint: Any,
    suite: Union[str, Path, dict] = "tests.yaml",
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
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
    Evaluate an OpenAI-compatible endpoint against a test suite.

    Works with any OpenAI-compatible API: OpenAI, vLLM, Ollama, LiteLLM,
    LocalAI, text-generation-inference, and more.

    Args:
        endpoint: URL string (e.g. "http://localhost:8000/v1/chat/completions")
            or an openai.OpenAI client instance.
        suite: Path to YAML test suite, dict config, or TestSuite object.
        openai_api_key: API key for the OpenAI-compatible endpoint.
        openai_model: Model name to use (default: "gpt-4o").
        system_prompt: Optional system message prepended to each call.
        temperature: Sampling temperature (default: 0.0).
        max_tokens: Max tokens per response (default: 1024).
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
        >>> results = evaluate_openai(
        ...     "http://localhost:11434/v1/chat/completions",
        ...     suite="qa.yaml",
        ...     openai_model="llama3",
        ... )
        >>> print(f"Score: {results.overall_score}")
    """
    agent_fn = _adapt_openai_endpoint(
        endpoint=endpoint,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
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
