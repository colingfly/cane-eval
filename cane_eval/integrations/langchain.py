"""
langchain.py -- LangChain integration for cane-eval.

Evaluate any LangChain chain, agent, or runnable with a single function call.

Usage:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_template("{input}") | ChatOpenAI()

    from cane_eval.integrations import evaluate_langchain
    results = evaluate_langchain(chain, suite="qa.yaml")

    # With mining and RCA
    results = evaluate_langchain(
        chain,
        suite="qa.yaml",
        mine=True,
        rca=True,
    )

    # Push to Cane Cloud
    results = evaluate_langchain(
        chain,
        suite="qa.yaml",
        cloud="https://app.cane.dev",
        cloud_api_key="sk-...",
        environment_id="env_abc123",
    )
"""

from typing import Any, Optional, Union, Callable
from pathlib import Path

from cane_eval.engine import RunSummary
from cane_eval.integrations._base import _run_eval


def _adapt_langchain_runnable(runnable: Any, input_key: str = "input") -> Callable[[str], str]:
    """
    Adapt a LangChain Runnable/Chain/Agent into a simple callable.

    Supports:
    - Runnable (invoke with dict or string)
    - Chain (run or invoke)
    - AgentExecutor (invoke)
    - Any object with .invoke() or .run() or __call__
    """
    def call_agent(question: str) -> str:
        # Try invoke with dict first (standard Runnable interface)
        if hasattr(runnable, "invoke"):
            try:
                result = runnable.invoke({input_key: question})
            except (TypeError, KeyError):
                # Fall back to string input
                try:
                    result = runnable.invoke(question)
                except Exception:
                    result = runnable.invoke({"query": question})
        elif hasattr(runnable, "run"):
            result = runnable.run(question)
        elif callable(runnable):
            result = runnable(question)
        else:
            raise TypeError(
                f"Cannot call LangChain object of type {type(runnable).__name__}. "
                "Expected a Runnable, Chain, AgentExecutor, or callable."
            )

        # Extract string from various result types
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # AgentExecutor returns {"output": "..."}
            for key in ("output", "result", "answer", "response", "text"):
                if key in result:
                    return str(result[key])
            return str(result)
        if hasattr(result, "content"):
            # AIMessage
            return str(result.content)
        return str(result)

    return call_agent


def evaluate_langchain(
    runnable: Any,
    suite: Union[str, Path, dict] = "tests.yaml",
    input_key: str = "input",
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
    Evaluate a LangChain runnable against a test suite.

    Args:
        runnable: Any LangChain Runnable, Chain, AgentExecutor, or callable.
            Supports LCEL chains, legacy chains, and agent executors.
        suite: Path to YAML test suite, dict config, or TestSuite object.
        input_key: Key name for the input dict (default: "input").
            Set to match your chain's expected input key.
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
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> chain = ChatPromptTemplate.from_template("{input}") | ChatOpenAI()
        >>> results = evaluate_langchain(chain, suite="qa.yaml")
        >>> print(f"Score: {results.overall_score}")
    """
    agent_fn = _adapt_langchain_runnable(runnable, input_key=input_key)

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
