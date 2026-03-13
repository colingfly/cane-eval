"""
export.py -- Training data export.

Converts eval results and mined examples into training-ready formats:
DPO, SFT, OpenAI fine-tune, and raw JSON.
"""

import json
from typing import Optional

from cane_eval.engine import RunSummary, EvalResult


class Exporter:
    """
    Export eval results as training data in various formats.

    Supported formats:
    - dpo: Direct Preference Optimization pairs (prompt + chosen + rejected)
    - sft: Supervised Fine-Tuning examples (prompt + completion)
    - openai: OpenAI fine-tuning format (messages array)
    - raw: Full eval results as JSON

    Usage:
        summary = runner.run(suite, agent=my_agent)
        exporter = Exporter(summary)

        # Export failures as DPO pairs
        exporter.to_dpo("training_dpo.jsonl")

        # Export passing results as SFT data
        exporter.to_sft("training_sft.jsonl", min_score=80)

        # Get as string
        jsonl = exporter.as_dpo_string()
    """

    def __init__(self, summary: Optional[RunSummary] = None):
        self.summary = summary

    def _filter_results(
        self,
        results: Optional[list[EvalResult]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Filter results by score range, status, or tags."""
        items = results or (self.summary.results if self.summary else [])

        if min_score is not None:
            items = [r for r in items if r.score >= min_score]
        if max_score is not None:
            items = [r for r in items if r.score <= max_score]
        if status:
            items = [r for r in items if r.status == status]
        if tags:
            tag_set = set(tags)
            items = [r for r in items if tag_set & set(r.tags)]

        return items

    # ---- DPO format ----

    def as_dpo(
        self,
        results: Optional[list[EvalResult]] = None,
        improved_answers: Optional[dict[str, str]] = None,
        **filter_kwargs,
    ) -> list[dict]:
        """
        Generate DPO training pairs.

        For each result, creates a pair where:
        - chosen = expected_answer (or improved_answer if provided)
        - rejected = agent_answer (the bad response)

        Args:
            results: Override results list
            improved_answers: Optional dict mapping question to improved answer
                (from failure mining). If provided, uses these as "chosen" instead
                of expected_answer.
            **filter_kwargs: Passed to _filter_results (min_score, max_score, status, tags)
        """
        items = self._filter_results(results, **filter_kwargs)
        pairs = []

        for r in items:
            chosen = r.expected_answer
            if improved_answers and r.question in improved_answers:
                chosen = improved_answers[r.question]

            if not chosen:
                continue  # skip if no expected/improved answer

            pairs.append({
                "prompt": r.question,
                "chosen": chosen,
                "rejected": r.agent_answer,
                "chosen_score": None,
                "rejected_score": r.score,
                "failure_type": None,
                "judge_reasoning": r.judge_result.overall_reasoning,
            })

        return pairs

    def as_dpo_string(self, **kwargs) -> str:
        """Return DPO pairs as JSONL string."""
        return "\n".join(json.dumps(p) for p in self.as_dpo(**kwargs))

    def to_dpo(self, path: str, **kwargs):
        """Write DPO pairs to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_dpo_string(**kwargs))

    # ---- SFT format ----

    def as_sft(
        self,
        results: Optional[list[EvalResult]] = None,
        use_expected: bool = True,
        **filter_kwargs,
    ) -> list[dict]:
        """
        Generate SFT training examples.

        Args:
            results: Override results list
            use_expected: If True, use expected_answer as completion.
                If False, use agent_answer (for passing results).
            **filter_kwargs: Passed to _filter_results
        """
        items = self._filter_results(results, **filter_kwargs)
        examples = []

        for r in items:
            completion = r.expected_answer if use_expected else r.agent_answer
            if not completion:
                continue

            examples.append({
                "prompt": r.question,
                "completion": completion,
                "metadata": {
                    "score": r.score,
                    "status": r.status,
                    "source": "cane-eval",
                    "judge_reasoning": r.judge_result.overall_reasoning,
                },
            })

        return examples

    def as_sft_string(self, **kwargs) -> str:
        """Return SFT examples as JSONL string."""
        return "\n".join(json.dumps(e) for e in self.as_sft(**kwargs))

    def to_sft(self, path: str, **kwargs):
        """Write SFT examples to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_sft_string(**kwargs))

    # ---- OpenAI fine-tune format ----

    def as_openai(
        self,
        results: Optional[list[EvalResult]] = None,
        system_prompt: str = "You are a helpful assistant.",
        use_expected: bool = True,
        **filter_kwargs,
    ) -> list[dict]:
        """
        Generate OpenAI fine-tuning format (messages array).

        Args:
            results: Override results list
            system_prompt: System message for the conversation
            use_expected: Use expected_answer as assistant response
            **filter_kwargs: Passed to _filter_results
        """
        items = self._filter_results(results, **filter_kwargs)
        examples = []

        for r in items:
            completion = r.expected_answer if use_expected else r.agent_answer
            if not completion:
                continue

            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": r.question},
                    {"role": "assistant", "content": completion},
                ],
            })

        return examples

    def as_openai_string(self, **kwargs) -> str:
        """Return OpenAI examples as JSONL string."""
        return "\n".join(json.dumps(e) for e in self.as_openai(**kwargs))

    def to_openai(self, path: str, **kwargs):
        """Write OpenAI examples to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_openai_string(**kwargs))

    # ---- Raw format ----

    def as_raw(
        self,
        results: Optional[list[EvalResult]] = None,
        **filter_kwargs,
    ) -> list[dict]:
        """Return full eval results as dicts."""
        items = self._filter_results(results, **filter_kwargs)
        return [r.to_dict() for r in items]

    def as_raw_string(self, **kwargs) -> str:
        """Return raw results as JSONL string."""
        return "\n".join(json.dumps(r) for r in self.as_raw(**kwargs))

    def to_raw(self, path: str, **kwargs):
        """Write raw results to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_raw_string(**kwargs))

    # ---- Convenience ----

    def to_file(self, path: str, format: str = "dpo", **kwargs):
        """
        Write training data to file in specified format.

        Args:
            path: Output file path
            format: One of "dpo", "sft", "openai", "raw"
            **kwargs: Passed to the format-specific method
        """
        writers = {
            "dpo": self.to_dpo,
            "sft": self.to_sft,
            "openai": self.to_openai,
            "raw": self.to_raw,
        }
        writer = writers.get(format)
        if not writer:
            raise ValueError(f"Unknown format: {format}. Use one of: {list(writers.keys())}")
        writer(path, **kwargs)
