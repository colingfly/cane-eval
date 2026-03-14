"""
llamaindex.py -- LlamaIndex integration for cane-eval.

Evaluate any LlamaIndex query engine, chat engine, or agent with a single call.

Usage:
    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    from cane_eval.integrations import evaluate_llamaindex
    results = evaluate_llamaindex(query_engine, suite="qa.yaml")

    # Chat engine
    chat_engine = index.as_chat_engine()
    results = evaluate_llamaindex(chat_engine, suite="qa.yaml")
"""

from typing import Any, Optional, Union, Callable
from pathlib import Path

from cane_eval.engine import RunSummary
from cane_eval.integrations._base import _run_eval


def _adapt_llamaindex_engine(engine: Any) -> Callable[[str], str]:
    """
    Adapt a LlamaIndex query engine, chat engine, or agent into a callable.

    Supports:
    - QueryEngine (query)
    - ChatEngine (chat)
    - BaseAgent (chat)
    - Any object with .query() or .chat()
    """
    def call_agent(question: str) -> str:
        # Try query engine first
        if hasattr(engine, "query"):
            result = engine.query(question)
        elif hasattr(engine, "chat"):
            result = engine.chat(question)
        elif callable(engine):
            result = engine(question)
        else:
            raise TypeError(
                f"Cannot call LlamaIndex object of type {type(engine).__name__}. "
                "Expected a QueryEngine, ChatEngine, Agent, or callable."
            )

        # Extract string from Response objects
        if isinstance(result, str):
            return result
        if hasattr(result, "response"):
            return str(result.response)
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    return call_agent


def evaluate_llamaindex(
    engine: Any,
    suite: Union[str, Path, dict] = "tests.yaml",
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
    Evaluate a LlamaIndex engine against a test suite.

    Args:
        engine: Any LlamaIndex QueryEngine, ChatEngine, or Agent.
        suite: Path to YAML test suite, dict config, or TestSuite object.
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
        >>> from llama_index.core import VectorStoreIndex
        >>> index = VectorStoreIndex.from_documents(docs)
        >>> engine = index.as_query_engine()
        >>> results = evaluate_llamaindex(engine, suite="qa.yaml")
        >>> print(f"Score: {results.overall_score}")
    """
    agent_fn = _adapt_llamaindex_engine(engine)

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
