"""
integrations -- One-liner eval helpers for popular frameworks.

Wrap your existing agent in a single function call:

    from cane_eval.integrations import evaluate_langchain
    results = evaluate_langchain(chain, suite="qa.yaml")

    from cane_eval.integrations import evaluate_openai
    results = evaluate_openai("http://localhost:8000/v1/chat/completions", suite="qa.yaml")

    from cane_eval.integrations import evaluate_phlote
    results = evaluate_phlote(music_agent, suite="music_eval.yaml")

Each integration adapts the framework's agent interface into cane-eval's
callable format, runs the eval, and optionally pushes results to Cane Cloud.
"""

from cane_eval.integrations.langchain import evaluate_langchain
from cane_eval.integrations.llamaindex import evaluate_llamaindex
from cane_eval.integrations.openai_compat import evaluate_openai
from cane_eval.integrations.fastapi_agent import evaluate_fastapi
from cane_eval.integrations.phlote import evaluate_phlote

__all__ = [
    "evaluate_langchain",
    "evaluate_llamaindex",
    "evaluate_openai",
    "evaluate_fastapi",
    "evaluate_phlote",
]
