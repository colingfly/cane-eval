"""
mining.py -- Automated failure mining.

Takes low-scoring eval results, classifies failure types via LLM,
generates improved answers via LLM rewrite, and produces DPO training pairs.

This closes the feedback loop:
  Agents run > Eval scores responses > Failures are mined >
  LLM rewrites bad answers > Training data is generated > Models improve
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from cane_eval.engine import RunSummary, EvalResult


# ---- Failure types ----

FAILURE_TYPES = [
    "hallucination",
    "incomplete",
    "off_topic",
    "wrong_format",
    "factual_error",
    "other",
]


# ---- Data classes ----

@dataclass
class MinedExample:
    """A single mined training example with improved answer."""
    question: str
    original_answer: str
    improved_answer: str
    expected_answer: str = ""
    failure_type: str = "other"
    improvement_reasoning: str = ""
    original_score: float = 0.0
    estimated_improved_score: Optional[float] = None
    context: Optional[str] = None

    def to_dpo(self) -> dict:
        """Export as DPO training pair."""
        return {
            "prompt": self.question,
            "chosen": self.improved_answer,
            "rejected": self.original_answer,
            "chosen_score": self.estimated_improved_score,
            "rejected_score": self.original_score,
            "failure_type": self.failure_type,
            "improvement_reasoning": self.improvement_reasoning,
        }

    def to_sft(self) -> dict:
        """Export as SFT training example."""
        return {
            "prompt": self.question,
            "completion": self.improved_answer,
            "metadata": {
                "failure_type": self.failure_type,
                "original_score": self.original_score,
                "source": "failure_mining",
                "improvement_reasoning": self.improvement_reasoning,
            },
        }


@dataclass
class MiningResult:
    """Result of a failure mining run."""
    total_failures: int = 0
    total_mined: int = 0
    examples: list[MinedExample] = field(default_factory=list)
    failure_distribution: dict = field(default_factory=dict)

    def to_dpo_string(self) -> str:
        """Export all examples as DPO JSONL."""
        return "\n".join(json.dumps(ex.to_dpo()) for ex in self.examples)

    def to_sft_string(self) -> str:
        """Export all examples as SFT JSONL."""
        return "\n".join(json.dumps(ex.to_sft()) for ex in self.examples)

    def to_file(self, path: str, format: str = "dpo"):
        """Write mined examples to JSONL file."""
        content = self.to_dpo_string() if format == "dpo" else self.to_sft_string()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# ---- System prompts ----

CLASSIFY_SYSTEM = """You are a failure classifier for AI agent responses.
Given judge reasoning about why an agent response scored poorly, classify the
primary failure type into exactly ONE category. Respond with ONLY the category
name, nothing else.

Categories:
- hallucination: The agent fabricated information not supported by sources
- incomplete: The answer is partially correct but missing key information
- off_topic: The answer does not address the question asked
- wrong_format: The answer has correct info but wrong structure or format
- factual_error: The answer contains incorrect facts that contradict sources
- other: Does not clearly fit the above categories"""


REWRITE_SYSTEM = """You are an expert AI response improver. You will receive a question
that an AI agent answered poorly, along with the judge's criticism and optionally
the expected answer and source documents.

Your job is to write an improved answer that addresses ALL of the judge's criticisms.

Rules:
- If source documents are provided, ground your answer ONLY in those sources
- If an expected answer is provided, ensure your answer covers the same key points
- Address every specific criticism from the judge
- Be accurate, complete, and well-structured
- Do not fabricate information beyond what the sources support
- Match the appropriate tone and format for the question

Respond with valid JSON only:
{"improved_answer": "<your improved answer>", "reasoning": "<brief explanation of what you fixed>", "confidence": <0-100 integer>}"""


# ---- Miner class ----

class FailureMiner:
    """
    Automated failure mining: classify failures and generate improved answers.

    Usage:
        summary = runner.run(suite, agent=my_agent)
        miner = FailureMiner(api_key="sk-ant-...")
        result = miner.mine(summary, max_score=60)

        print(f"Mined {result.total_mined} examples from {result.total_failures} failures")
        result.to_file("mined_dpo.jsonl", format="dpo")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _call(self, prompt: str, system: str, max_tokens: int = 1024) -> str:
        """Call Claude and return text."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def classify_failure(self, judge_reasoning: str) -> str:
        """Classify a failure into one of the standard failure types."""
        if not judge_reasoning:
            return "other"

        try:
            raw = self._call(
                prompt=f"Judge reasoning:\n{judge_reasoning}\n\nCategory:",
                system=CLASSIFY_SYSTEM,
                max_tokens=20,
            )
            category = raw.strip().lower().replace("-", "_").replace(" ", "_")
            if category in FAILURE_TYPES:
                return category
        except Exception:
            pass

        return "other"

    def generate_improved_answer(
        self,
        question: str,
        bad_answer: str,
        expected_answer: str = "",
        judge_reasoning: str = "",
        context: Optional[str] = None,
    ) -> dict:
        """
        Generate an improved answer using LLM rewrite.

        Returns dict with improved_answer, reasoning, confidence.
        """
        parts = [
            f"Question: {question}",
            f"\nOriginal (Bad) Answer: {bad_answer}",
        ]

        if judge_reasoning:
            parts.append(f"\nJudge's Criticism: {judge_reasoning}")

        if expected_answer:
            parts.append(f"\nExpected Answer: {expected_answer}")

        if context:
            truncated = context[:4000]
            parts.append(f"\nSource Documents:\n{truncated}")

        parts.append("\nWrite an improved answer (JSON only):")
        prompt = "\n".join(parts)

        try:
            raw = self._call(prompt=prompt, system=REWRITE_SYSTEM, max_tokens=1024)

            # Parse JSON, strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
            return {
                "improved_answer": result.get("improved_answer", cleaned),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", None),
            }
        except (json.JSONDecodeError, Exception):
            return {
                "improved_answer": raw.strip() if 'raw' in dir() else "",
                "reasoning": "",
                "confidence": None,
            }

    def mine(
        self,
        summary: RunSummary,
        max_score: float = 60,
        min_score: float = 0,
        max_examples: int = 100,
    ) -> MiningResult:
        """
        Mine failures from an eval run.

        Args:
            summary: Eval run summary to mine
            max_score: Only mine results scoring at or below this threshold
            min_score: Only mine results scoring at or above this threshold
            max_examples: Maximum number of examples to mine (cost cap)

        Returns:
            MiningResult with classified failures and improved answers
        """
        # Filter to failing results within score range
        failures = [
            r for r in summary.results
            if r.score <= max_score and r.score >= min_score
        ]
        failures.sort(key=lambda r: r.score)

        # Cap the number of examples
        failures = failures[:max_examples]

        result = MiningResult(total_failures=len(failures))

        if not failures:
            if self.verbose:
                print("  No failures found matching criteria")
            return result

        if self.verbose:
            print(f"  Mining {len(failures)} failures (score range: {min_score}-{max_score})")

        from collections import Counter
        type_counts = Counter()

        for i, r in enumerate(failures):
            try:
                # 1. Classify failure type
                failure_type = self.classify_failure(r.judge_result.overall_reasoning)
                type_counts[failure_type] += 1

                # 2. Generate improved answer
                rewrite = self.generate_improved_answer(
                    question=r.question,
                    bad_answer=r.agent_answer,
                    expected_answer=r.expected_answer,
                    judge_reasoning=r.judge_result.overall_reasoning,
                    context=r.context,
                )

                # 3. Build mined example
                confidence = rewrite.get("confidence")
                example = MinedExample(
                    question=r.question,
                    original_answer=r.agent_answer,
                    improved_answer=rewrite["improved_answer"],
                    expected_answer=r.expected_answer,
                    failure_type=failure_type,
                    improvement_reasoning=rewrite["reasoning"],
                    original_score=r.score,
                    estimated_improved_score=float(confidence) if confidence is not None else None,
                    context=r.context,
                )
                result.examples.append(example)
                result.total_mined += 1

                if self.verbose:
                    print(f"    [{i+1}/{len(failures)}] {failure_type} (score: {r.score})")

            except Exception as e:
                if self.verbose:
                    print(f"    [{i+1}/{len(failures)}] Error: {e}")
                continue

        result.failure_distribution = dict(type_counts)

        if self.verbose:
            print(f"  Done: mined {result.total_mined}/{result.total_failures} examples")
            for ftype, count in type_counts.most_common():
                print(f"    {ftype}: {count}")

        return result

    def mine_results(
        self,
        results: list[EvalResult],
        max_score: float = 60,
        min_score: float = 0,
        max_examples: int = 100,
    ) -> MiningResult:
        """
        Mine failures from a list of EvalResult objects directly.

        Same as mine() but takes raw results instead of a RunSummary.
        """
        dummy_summary = RunSummary(
            suite_name="direct",
            total=len(results),
            results=results,
        )
        return self.mine(dummy_summary, max_score=max_score, min_score=min_score, max_examples=max_examples)
