"""
rca.py -- Root cause analysis for eval failures.

Goes beyond failure classification to identify the underlying reasons
agents fail and generate actionable recommendations.

Example insights:
- "Agent lacks documentation about refund policies (3 of 5 failures reference refunds)"
- "System prompt does not instruct the agent to cite sources (4 hallucination failures)"
- "Agent response quality degrades for multi-part questions (avg score 42 vs 78 for simple)"
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from cane_eval.engine import RunSummary, EvalResult


# ---- Data classes ----

@dataclass
class RootCause:
    """A single identified root cause."""
    id: str = ""
    title: str = ""
    severity: str = "medium"
    category: str = "behavior_pattern"
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class RCAResult:
    """Result of a batch root cause analysis."""
    root_causes: list[RootCause] = field(default_factory=list)
    summary: str = ""
    top_recommendation: str = ""
    total_analyzed: int = 0
    avg_failure_score: float = 0.0
    score_range: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def to_dict(self) -> dict:
        return {
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "summary": self.summary,
            "top_recommendation": self.top_recommendation,
            "total_analyzed": self.total_analyzed,
            "avg_failure_score": self.avg_failure_score,
            "score_range": self.score_range,
        }


@dataclass
class FixAction:
    """A single recommended fix action."""
    action: str = ""
    priority: str = "medium"
    effort: str = "moderate"

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "priority": self.priority,
            "effort": self.effort,
        }


@dataclass
class TargetedRCAResult:
    """Result of a targeted (single result) root cause analysis."""
    question: str = ""
    score: float = 0.0
    diagnosis: str = ""
    likely_cause: str = "unknown"
    contributing_factors: list[str] = field(default_factory=list)
    fix_actions: list[FixAction] = field(default_factory=list)
    confidence: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "score": self.score,
            "diagnosis": self.diagnosis,
            "likely_cause": self.likely_cause,
            "contributing_factors": self.contributing_factors,
            "fix_actions": [fa.to_dict() for fa in self.fix_actions],
            "confidence": self.confidence,
        }


# ---- System prompts ----

RCA_BATCH_SYSTEM = """You are an expert AI systems debugger. You analyze patterns across
multiple failed eval results to identify the ROOT CAUSES of why an AI agent is
performing poorly.

You receive a batch of failing test cases with:
- The question asked
- The agent's answer
- The expected answer (if available)
- The judge's reasoning and score

Your job is to find PATTERNS across failures and identify actionable root causes.
Do not just restate that the agent failed. Dig deeper:
- Is the agent missing specific knowledge domains?
- Is the system prompt missing instructions?
- Are there patterns in question types that fail?
- Is the agent fabricating information consistently?
- Are source documents incomplete or outdated?

Respond with valid JSON only, no markdown fences:
{
  "root_causes": [
    {
      "id": "<short-kebab-case-id>",
      "title": "<concise title, max 80 chars>",
      "severity": "<critical|high|medium|low>",
      "category": "<knowledge_gap|prompt_issue|source_gap|behavior_pattern|data_quality>",
      "description": "<2-3 sentence explanation of the root cause>",
      "evidence": ["<specific question or pattern that supports this>"],
      "recommendation": "<specific actionable fix>"
    }
  ],
  "summary": "<1-2 sentence executive summary of findings>",
  "top_recommendation": "<single most impactful action to take>"
}"""


RCA_TARGETED_SYSTEM = """You are an expert AI systems debugger. You analyze a single
failing eval result in depth to determine exactly why the agent produced a bad answer.

You receive:
- The question asked
- The agent's answer
- The expected answer
- The judge's reasoning and score

Perform a deep analysis:
1. What specific information is wrong or missing?
2. Why might the agent have produced this response?
3. What would need to change to fix this?

Respond with valid JSON only, no markdown fences:
{
  "diagnosis": "<detailed explanation of what went wrong>",
  "likely_cause": "<knowledge_gap|prompt_issue|source_gap|hallucination|reasoning_error|context_overflow>",
  "contributing_factors": ["<factor1>", "<factor2>"],
  "fix_actions": [
    {"action": "<what to do>", "priority": "<high|medium|low>", "effort": "<quick|moderate|significant>"}
  ],
  "confidence": <0-100>
}"""


# ---- JSON parsing ----

def _parse_json_response(raw: str) -> dict:
    """Parse a JSON response, stripping markdown fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


# ---- Analyzer class ----

class RootCauseAnalyzer:
    """
    AI-powered root cause analysis for eval failures.

    Analyzes patterns across failing eval results to identify actionable
    root causes and recommendations.

    Usage:
        summary = runner.run(suite, agent=my_agent)
        analyzer = RootCauseAnalyzer(api_key="sk-ant-...")

        # Batch analysis across all failures
        rca = analyzer.analyze(summary, max_score=60)
        print(rca.summary)
        for rc in rca.root_causes:
            print(f"  [{rc.severity}] {rc.title}")
            print(f"    {rc.recommendation}")

        # Deep dive on a single result
        targeted = analyzer.analyze_result(summary.results[0])
        print(targeted.diagnosis)
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

    def _call(self, prompt: str, system: str, max_tokens: int = 2048) -> str:
        """Call Claude and return text."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def analyze(
        self,
        summary: RunSummary,
        max_score: float = 60,
        min_score: float = 0,
        max_failures: int = 30,
    ) -> RCAResult:
        """
        Run batch root cause analysis across failing eval results.

        Args:
            summary: Eval run summary to analyze.
            max_score: Analyze results scoring at or below this threshold.
            min_score: Analyze results scoring at or above this threshold.
            max_failures: Max failures to include in analysis (context cap).

        Returns:
            RCAResult with root causes, summary, and recommendations.
        """
        # Filter to failing results
        failures = [
            r for r in summary.results
            if r.score <= max_score and r.score >= min_score
        ]
        failures.sort(key=lambda r: r.score)
        failures = failures[:max_failures]

        if not failures:
            if self.verbose:
                print("  No failures found matching criteria")
            return RCAResult(
                summary="No failures found below the score threshold.",
                top_recommendation="All results are scoring above the threshold. Consider lowering it.",
            )

        if self.verbose:
            print(f"  Analyzing {len(failures)} failures (score range: {min_score}-{max_score})")

        # Build prompt
        prompt = self._build_batch_prompt(failures, summary.suite_name)

        try:
            raw = self._call(prompt=prompt, system=RCA_BATCH_SYSTEM, max_tokens=2048)
            analysis = _parse_json_response(raw)
        except (json.JSONDecodeError, Exception) as e:
            if self.verbose:
                print(f"  Analysis failed: {e}")
            return RCAResult(
                summary=f"Analysis failed: {str(e)}",
                top_recommendation="Try again or reduce the number of failures to analyze.",
                total_analyzed=len(failures),
            )

        # Build root causes
        root_causes = []
        for rc_data in analysis.get("root_causes", []):
            root_causes.append(RootCause(
                id=rc_data.get("id", ""),
                title=rc_data.get("title", ""),
                severity=rc_data.get("severity", "medium"),
                category=rc_data.get("category", "behavior_pattern"),
                description=rc_data.get("description", ""),
                evidence=rc_data.get("evidence", []),
                recommendation=rc_data.get("recommendation", ""),
            ))

        # Compute score stats
        scores = [f.score for f in failures]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0

        result = RCAResult(
            root_causes=root_causes,
            summary=analysis.get("summary", ""),
            top_recommendation=analysis.get("top_recommendation", ""),
            total_analyzed=len(failures),
            avg_failure_score=avg_score,
            score_range=[min(scores), max(scores)] if scores else [0, 0],
        )

        if self.verbose:
            print(f"  Found {len(root_causes)} root causes across {len(failures)} failures")

        return result

    def analyze_result(self, result: EvalResult) -> TargetedRCAResult:
        """
        Run deep root cause analysis on a single eval result.

        Args:
            result: The specific EvalResult to analyze.

        Returns:
            TargetedRCAResult with diagnosis, likely cause, and fix actions.
        """
        if self.verbose:
            q_preview = result.question[:60] + "..." if len(result.question) > 60 else result.question
            print(f"  Analyzing: {q_preview}")

        prompt = self._build_targeted_prompt(result)

        try:
            raw = self._call(prompt=prompt, system=RCA_TARGETED_SYSTEM, max_tokens=1024)
            analysis = _parse_json_response(raw)
        except (json.JSONDecodeError, Exception) as e:
            if self.verbose:
                print(f"  Analysis failed: {e}")
            return TargetedRCAResult(
                question=result.question,
                score=result.score,
                diagnosis=f"Analysis failed: {str(e)}",
                likely_cause="unknown",
            )

        # Build fix actions
        fix_actions = []
        for fa_data in analysis.get("fix_actions", []):
            fix_actions.append(FixAction(
                action=fa_data.get("action", ""),
                priority=fa_data.get("priority", "medium"),
                effort=fa_data.get("effort", "moderate"),
            ))

        targeted = TargetedRCAResult(
            question=result.question,
            score=result.score,
            diagnosis=analysis.get("diagnosis", ""),
            likely_cause=analysis.get("likely_cause", "unknown"),
            contributing_factors=analysis.get("contributing_factors", []),
            fix_actions=fix_actions,
            confidence=analysis.get("confidence", 0),
        )

        if self.verbose:
            print(f"  Cause: {targeted.likely_cause} (confidence: {targeted.confidence}%)")

        return targeted

    def analyze_results(
        self,
        results: list[EvalResult],
        max_score: float = 60,
        min_score: float = 0,
        max_failures: int = 30,
    ) -> RCAResult:
        """
        Run batch RCA on a list of EvalResult objects directly.

        Same as analyze() but takes raw results instead of a RunSummary.
        """
        dummy_summary = RunSummary(
            suite_name="direct",
            total=len(results),
            results=results,
        )
        return self.analyze(dummy_summary, max_score=max_score, min_score=min_score, max_failures=max_failures)

    # ---- Prompt builders ----

    def _build_batch_prompt(self, failures: list[EvalResult], suite_name: str) -> str:
        """Build the prompt for batch root cause analysis."""
        parts = [
            f"Suite: {suite_name}",
            f"Total failing results analyzed: {len(failures)}",
            "",
        ]

        for i, r in enumerate(failures[:30]):
            parts.append(f"--- Failure {i+1} (score: {r.score}) ---")
            parts.append(f"Question: {r.question}")
            if r.expected_answer:
                parts.append(f"Expected: {r.expected_answer[:500]}")
            parts.append(f"Agent answer: {r.agent_answer[:500]}")
            parts.append(f"Judge reasoning: {r.judge_result.overall_reasoning}")
            if r.judge_result.criteria_scores:
                score_strs = [f"{cs.name}: {cs.score}" for cs in r.judge_result.criteria_scores]
                parts.append(f"Criteria scores: {', '.join(score_strs)}")
            parts.append("")

        parts.append("Analyze these failures. Find root causes and patterns. Return JSON only.")
        return "\n".join(parts)

    def _build_targeted_prompt(self, result: EvalResult) -> str:
        """Build the prompt for single-result deep analysis."""
        parts = [
            f"Question: {result.question}",
        ]
        if result.expected_answer:
            parts.append(f"Expected answer: {result.expected_answer}")
        parts.append(f"Agent answer: {result.agent_answer}")
        parts.append(f"Judge reasoning: {result.judge_result.overall_reasoning}")
        parts.append(f"Overall score: {result.score}")
        if result.judge_result.criteria_scores:
            score_strs = [f"{cs.name}: {cs.score}" for cs in result.judge_result.criteria_scores]
            parts.append(f"Criteria scores: {', '.join(score_strs)}")
        parts.append("")
        parts.append("Perform deep root cause analysis. Return JSON only.")
        return "\n".join(parts)
