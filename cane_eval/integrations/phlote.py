"""
phlote.py -- Phlote.xyz integration for cane-eval.

Evaluate AI music agents, metadata/rights APIs, and adaptive music systems
for reliability. Provides domain-specific criteria for music infrastructure:
rights clearance, metadata completeness, licensing validity, and contextual
relevance.

Usage:
    from cane_eval.integrations.phlote import evaluate_phlote

    results = evaluate_phlote(
        agent=my_music_agent,
        suite="music_agent_eval.yaml",
    )

    # With domain-specific criteria auto-registered
    results = evaluate_phlote(
        agent=my_music_agent,
        suite="music_agent_eval.yaml",
        register_criteria=True,
    )
"""

import json
from typing import Any, Optional, Union, Callable
from pathlib import Path

from cane_eval.engine import ReliabilityRunner, ReliabilitySummary
from cane_eval.criteria import CriterionPlugin, CriterionType, EvalContext, CriteriaRegistry
from cane_eval.integrations._base import _load_suite, _run_eval


# ---- Phlote-specific failure types ----

PHLOTE_FAILURE_TYPES = [
    "rights_violation",
    "metadata_missing",
    "licensing_error",
]


def register_phlote_failure_types():
    """Register Phlote-specific failure types with the mining module."""
    from cane_eval.mining import register_failure_type
    for ft in PHLOTE_FAILURE_TYPES:
        register_failure_type(ft)


# ---- Domain-specific criteria ----

class RightsClearanceCriterion(CriterionPlugin):
    """
    Checks that agent responses include proper rights/licensing metadata.

    Looks for rights-related fields in JSON responses or keywords in text
    responses that indicate rights clearance status.
    """

    name = "rights_clearance"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Response includes proper rights/licensing clearance information"

    REQUIRED_FIELDS = {"rights", "license", "rights_holder", "clearance_status"}
    RIGHTS_KEYWORDS = {"rights", "licensed", "clearance", "copyright", "permission", "royalty"}

    def score(self, ctx: EvalContext) -> float:
        answer = ctx.agent_answer

        # Try JSON parsing first
        try:
            data = json.loads(answer)
            if isinstance(data, dict):
                found = sum(1 for f in self.REQUIRED_FIELDS if f in data)
                if found > 0:
                    return min(100.0, (found / 2) * 100)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to keyword matching in text
        answer_lower = answer.lower()
        found = sum(1 for kw in self.RIGHTS_KEYWORDS if kw in answer_lower)
        if found >= 3:
            return 100.0
        elif found >= 1:
            return 50.0
        return 0.0


class MetadataCompletenessCriterion(CriterionPlugin):
    """
    Validates that required music metadata fields are present in the response.

    Expected fields for music metadata: title, artist, duration, genre, isrc,
    release_date, album.
    """

    name = "metadata_completeness"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Required music metadata fields are present and populated"

    REQUIRED_FIELDS = {"title", "artist"}
    OPTIONAL_FIELDS = {"duration", "genre", "isrc", "release_date", "album", "bpm", "key"}

    def score(self, ctx: EvalContext) -> float:
        answer = ctx.agent_answer

        try:
            data = json.loads(answer)
            if not isinstance(data, dict):
                return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0

        # Check required fields
        required_found = sum(1 for f in self.REQUIRED_FIELDS if data.get(f))
        if required_found < len(self.REQUIRED_FIELDS):
            return (required_found / len(self.REQUIRED_FIELDS)) * 50.0

        # Required fields present, check optional for bonus
        optional_found = sum(1 for f in self.OPTIONAL_FIELDS if data.get(f))
        optional_score = (optional_found / len(self.OPTIONAL_FIELDS)) * 50.0

        return 50.0 + optional_score


class LicensingValidityCriterion(CriterionPlugin):
    """
    Validates that licensing information in the response is structurally valid.

    Checks for valid license types, non-empty license holders, and proper
    licensing status values.
    """

    name = "licensing_validity"
    criterion_type = CriterionType.DETERMINISTIC
    weight = 1.0
    description = "Licensing information is valid and properly structured"

    VALID_LICENSE_TYPES = {
        "creative_commons", "cc", "cc0", "cc-by", "cc-by-sa", "cc-by-nc",
        "commercial", "exclusive", "non-exclusive", "sync", "mechanical",
        "master", "public_domain", "royalty_free", "rights_managed",
    }

    VALID_STATUSES = {"active", "expired", "pending", "revoked", "cleared", "restricted"}

    def score(self, ctx: EvalContext) -> float:
        answer = ctx.agent_answer

        try:
            data = json.loads(answer)
            if not isinstance(data, dict):
                return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0

        score = 0.0
        checks = 0

        # Check license type
        license_type = str(data.get("license_type", data.get("license", ""))).lower().replace(" ", "_")
        if license_type:
            checks += 1
            if license_type in self.VALID_LICENSE_TYPES:
                score += 100.0

        # Check license holder
        holder = data.get("license_holder", data.get("rights_holder", ""))
        if holder:
            checks += 1
            score += 100.0  # Non-empty holder is valid

        # Check licensing status
        status = str(data.get("licensing_status", data.get("status", ""))).lower()
        if status:
            checks += 1
            if status in self.VALID_STATUSES:
                score += 100.0

        if checks == 0:
            return 0.0
        return score / checks


class ContextualRelevanceCriterion(CriterionPlugin):
    """
    LLM-based criterion that judges whether the response is contextually
    relevant to the musical query. Delegates to the existing judge score
    if available, or returns a neutral score.
    """

    name = "contextual_relevance"
    criterion_type = CriterionType.LLM
    weight = 1.0
    description = "Response is contextually relevant to the musical query"

    def score(self, ctx: EvalContext) -> float:
        if ctx.judge_result is None:
            return 50.0
        # Use overall judge score as proxy for contextual relevance
        return ctx.judge_result.overall_score


# ---- Registry helper ----

def phlote_registry() -> CriteriaRegistry:
    """Create a registry with Phlote-specific criteria."""
    registry = CriteriaRegistry()
    registry.register(RightsClearanceCriterion())
    registry.register(MetadataCompletenessCriterion())
    registry.register(LicensingValidityCriterion())
    registry.register(ContextualRelevanceCriterion())
    return registry


# ---- Main integration function ----

def evaluate_phlote(
    agent: Any,
    suite: Union[str, Path, dict] = "tests.yaml",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    tags: Optional[list[str]] = None,
    verbose: bool = True,
    mine: bool = False,
    mine_threshold: float = 60,
    rca: bool = False,
    rca_threshold: float = 60,
    register_criteria: bool = True,
    cloud: Optional[str] = None,
    cloud_api_key: Optional[str] = None,
    environment_id: Optional[str] = None,
) -> ReliabilitySummary:
    """
    Evaluate a Phlote music agent or API for reliability.

    Automatically registers Phlote-specific failure types and criteria
    for domain-aware evaluation of music infrastructure.

    Args:
        agent: Callable that takes a question string and returns an answer.
            Can also be any object with .invoke(), .run(), or __call__().
        suite: Path to YAML test suite, dict config, or TestSuite/ReliabilitySuite.
        api_key: API key for judging (or set env var).
        model: Override judge model.
        tags: Only run tests matching these tags.
        verbose: Print progress to stdout.
        mine: Run failure mining after eval.
        mine_threshold: Score threshold for mining failures.
        rca: Run root cause analysis after eval.
        rca_threshold: Score threshold for RCA.
        register_criteria: Register Phlote domain criteria (default: True).
        cloud: Cane Cloud URL.
        cloud_api_key: API key for Cane Cloud.
        environment_id: Environment ID on Cane Cloud.

    Returns:
        ReliabilitySummary with all eval results.
    """
    # Register Phlote failure types
    register_phlote_failure_types()

    # Adapt agent if it's not already callable
    if callable(agent):
        agent_fn = agent
    elif hasattr(agent, "invoke"):
        def agent_fn(q):
            result = agent.invoke(q)
            return str(result) if not isinstance(result, str) else result
    elif hasattr(agent, "run"):
        agent_fn = agent.run
    else:
        raise TypeError(
            f"Cannot call Phlote agent of type {type(agent).__name__}. "
            "Expected a callable, or object with .invoke() or .run()."
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
