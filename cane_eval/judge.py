"""
judge.py -- LLM-as-Judge scoring engine.

Uses any supported LLM provider (Anthropic, OpenAI, Gemini, or OpenAI-compatible)
as an impartial judge to score agent responses against expected answers
across weighted criteria.
"""

import json
from dataclasses import dataclass, field
from typing import Optional


# ---- Data classes ----

@dataclass
class CriteriaScore:
    """Score for a single evaluation criterion."""
    key: str
    score: float
    reasoning: str = ""


@dataclass
class JudgeResult:
    """Complete judge verdict for one test case."""
    criteria_scores: list[CriteriaScore] = field(default_factory=list)
    overall_score: float = 0.0
    overall_reasoning: str = ""
    status: str = "pending"  # pass | warn | fail

    def score_dict(self) -> dict:
        """Return scores as {key: score} dict."""
        return {cs.key: cs.score for cs in self.criteria_scores}

    def full_dict(self) -> dict:
        """Return full scores as {key: {score, reasoning}} dict."""
        return {
            cs.key: {"score": cs.score, "reasoning": cs.reasoning}
            for cs in self.criteria_scores
        }


# ---- System prompt ----

JUDGE_SYSTEM = """You are an expert evaluator assessing an AI agent's response quality.
You will score the response on specific criteria, each from 0 to 100.
Be strict but fair. Base your evaluation on the expected answer and the actual response.

Scoring guidelines:
- 90-100: Excellent. Accurate, complete, well-cited.
- 70-89: Good. Mostly correct, minor gaps.
- 50-69: Partial. Some correct info but significant gaps or minor errors.
- 30-49: Poor. Major inaccuracies or mostly incomplete.
- 0-29: Failing. Wrong, hallucinated, or completely missed.

IMPORTANT: If the agent fabricated information not supported by the sources, any hallucination-related score must be below 30.
If the agent directly contradicts the expected answer on key facts, accuracy must be below 40.

Respond ONLY with valid JSON, no markdown, no backticks."""


# ---- Judge class ----

class Judge:
    """
    LLM-as-Judge evaluator.

    Uses any supported LLM provider to score agent responses against expected answers.

    Usage:
        # Anthropic (default)
        judge = Judge(api_key="sk-ant-...")

        # OpenAI
        judge = Judge(provider="openai", model="gpt-4o")

        # Gemini
        judge = Judge(provider="gemini", model="gemini-2.0-flash")

        # Ollama (local, free)
        judge = Judge(provider="openai-compatible", model="llama3",
                      base_url="http://localhost:11434/v1")

        result = judge.score(
            question="What is the return policy?",
            expected_answer="30-day returns for unused items",
            agent_answer="We offer 30-day returns.",
            criteria=[{"key": "accuracy", "label": "Accuracy", ...}],
        )
        print(result.overall_score, result.status)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        provider: str = "anthropic",
        base_url: Optional[str] = None,
    ):
        from cane_eval.providers import get_provider, detect_provider_from_model

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Auto-detect provider from model name if not explicitly set
        if provider == "anthropic" and not model.startswith("claude"):
            provider = detect_provider_from_model(model)

        self._provider = get_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self._provider.model

    def _call(self, prompt: str, system: str = "", max_tokens: int = None) -> str:
        """Call the LLM provider and return text response."""
        return self._provider.call(
            prompt=prompt,
            system=system or JUDGE_SYSTEM,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
        )

    def _build_prompt(
        self,
        question: str,
        expected_answer: str,
        agent_answer: str,
        criteria: list[dict],
        custom_rules: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build the judge evaluation prompt."""
        criteria_desc = "\n".join(
            f"- {c['key']}: {c.get('label', c['key'])} -- {c.get('description', '')}"
            for c in criteria
        )

        rules_text = ""
        if custom_rules:
            rules_text = "\n\nCustom Rules (the judge MUST consider these):\n" + "\n".join(
                f"- {r}" for r in custom_rules
            )

        expected_section = f"\n\nExpected Answer:\n{expected_answer}" if expected_answer else ""

        context_section = ""
        if context:
            truncated = context[:4000]
            context_section = f"\n\nSource Documents:\n{truncated}"

        criteria_json = ", ".join(
            f'"{c["key"]}": {{"score": <0-100>, "reasoning": "<1-2 sentences>"}}'
            for c in criteria
        )

        return f"""Evaluate this AI agent's response.

Question:
{question}
{expected_section}
{context_section}

Agent's Response:
{agent_answer}

Evaluation Criteria:
{criteria_desc}
{rules_text}

Return this exact JSON structure:
{{
  "criteria_scores": {{
    {criteria_json}
  }},
  "overall_reasoning": "<Brief 1-2 sentence summary>"
}}"""

    def _parse_response(self, raw: str, criteria: list[dict]) -> dict:
        """Parse JSON response from judge. Handles markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: return neutral scores
            return {
                "criteria_scores": {
                    c["key"]: {"score": 50, "reasoning": "Judge response could not be parsed"}
                    for c in criteria
                },
                "overall_reasoning": "Evaluation parsing failed",
            }

    def _compute_overall(self, criteria_scores: dict, criteria: list[dict]) -> float:
        """Compute weighted overall score."""
        total_weight = sum(c.get("weight", 50) for c in criteria if c.get("is_enabled", True))
        if total_weight == 0:
            return 0.0

        weighted = 0.0
        for c in criteria:
            if not c.get("is_enabled", True):
                continue
            key = c["key"]
            score_data = criteria_scores.get(key, {})
            score = score_data.get("score", 50) if isinstance(score_data, dict) else score_data
            weighted += score * (c.get("weight", 50) / total_weight)

        return round(weighted, 1)

    @staticmethod
    def classify(score: float) -> str:
        """Classify score into pass/warn/fail."""
        if score >= 80:
            return "pass"
        if score >= 60:
            return "warn"
        return "fail"

    def score(
        self,
        question: str,
        agent_answer: str,
        expected_answer: str = "",
        criteria: Optional[list[dict]] = None,
        custom_rules: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> JudgeResult:
        """
        Score a single agent response.

        Args:
            question: The test question
            agent_answer: The agent's response to evaluate
            expected_answer: The expected/ideal answer
            criteria: List of criteria dicts with key, label, description, weight, is_enabled
            custom_rules: Optional list of custom evaluation rules
            context: Optional source documents for grounded evaluation

        Returns:
            JudgeResult with criteria scores, overall score, and pass/warn/fail status
        """
        if criteria is None:
            from cane_eval.suite import DEFAULT_CRITERIA
            criteria = [c.to_dict() for c in DEFAULT_CRITERIA]

        prompt = self._build_prompt(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
            criteria=criteria,
            custom_rules=custom_rules,
            context=context,
        )

        raw = self._call(prompt)
        parsed = self._parse_response(raw, criteria)

        criteria_scores_raw = parsed.get("criteria_scores", {})
        overall = self._compute_overall(criteria_scores_raw, criteria)
        status = self.classify(overall)

        # Build CriteriaScore list
        criteria_scores = []
        for key, val in criteria_scores_raw.items():
            if isinstance(val, dict):
                criteria_scores.append(CriteriaScore(
                    key=key,
                    score=val.get("score", 50),
                    reasoning=val.get("reasoning", ""),
                ))
            else:
                criteria_scores.append(CriteriaScore(key=key, score=float(val)))

        return JudgeResult(
            criteria_scores=criteria_scores,
            overall_score=overall,
            overall_reasoning=parsed.get("overall_reasoning", ""),
            status=status,
        )
