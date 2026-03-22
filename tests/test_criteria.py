"""Tests for the criteria plugin system."""

from cane_eval.criteria import (
    CriterionType,
    EvalContext,
    CriterionPlugin,
    CriteriaRegistry,
    SchemaValidityCriterion,
    LatencyPerformanceCriterion,
    AccuracyCriterion,
    CompletenessCriterion,
    HallucinationCriterion,
    default_registry,
)
from cane_eval.engine import SchemaResult
from cane_eval.judge import JudgeResult, CriteriaScore


# ---- CriterionType ----

def test_criterion_types():
    assert CriterionType.LLM.value == "llm"
    assert CriterionType.DETERMINISTIC.value == "deterministic"
    assert CriterionType.HYBRID.value == "hybrid"


# ---- EvalContext ----

def test_eval_context_defaults():
    ctx = EvalContext()
    assert ctx.question == ""
    assert ctx.agent_answer == ""
    assert ctx.response_time_ms == 0
    assert ctx.metadata == {}


def test_eval_context_with_data():
    ctx = EvalContext(
        question="What is 2+2?",
        agent_answer="4",
        response_time_ms=100,
        metadata={"source": "test"},
    )
    assert ctx.question == "What is 2+2?"
    assert ctx.response_time_ms == 100


# ---- SchemaValidityCriterion ----

def test_schema_validity_no_schema():
    c = SchemaValidityCriterion()
    ctx = EvalContext(schema_result=None)
    assert c.score(ctx) == 100.0


def test_schema_validity_pass():
    c = SchemaValidityCriterion()
    ctx = EvalContext(schema_result=SchemaResult(valid=True))
    assert c.score(ctx) == 100.0


def test_schema_validity_fail():
    c = SchemaValidityCriterion()
    ctx = EvalContext(schema_result=SchemaResult(valid=False, errors=["bad"]))
    assert c.score(ctx) == 0.0


# ---- LatencyPerformanceCriterion ----

def test_latency_no_data():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=0)
    assert c.score(ctx) == 100.0


def test_latency_under_target():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=3000)
    assert c.score(ctx) == 100.0


def test_latency_at_target():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=5000)
    assert c.score(ctx) == 100.0


def test_latency_over_target():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=7500)
    assert c.score(ctx) == 50.0


def test_latency_at_2x_target():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=10000)
    assert c.score(ctx) == 0.0


def test_latency_beyond_2x():
    c = LatencyPerformanceCriterion(target_ms=5000)
    ctx = EvalContext(response_time_ms=15000)
    assert c.score(ctx) == 0.0


# ---- LLM criteria (extract from JudgeResult) ----

def test_accuracy_criterion_with_judge():
    jr = JudgeResult(
        criteria_scores=[CriteriaScore(key="accuracy", score=85.0)],
        overall_score=85.0,
        status="pass",
    )
    c = AccuracyCriterion()
    ctx = EvalContext(judge_result=jr)
    assert c.score(ctx) == 85.0


def test_completeness_criterion_with_judge():
    jr = JudgeResult(
        criteria_scores=[CriteriaScore(key="completeness", score=70.0)],
        overall_score=70.0,
        status="warn",
    )
    c = CompletenessCriterion()
    ctx = EvalContext(judge_result=jr)
    assert c.score(ctx) == 70.0


def test_hallucination_criterion_with_judge():
    jr = JudgeResult(
        criteria_scores=[CriteriaScore(key="hallucination", score=95.0)],
        overall_score=95.0,
        status="pass",
    )
    c = HallucinationCriterion()
    ctx = EvalContext(judge_result=jr)
    assert c.score(ctx) == 95.0


def test_llm_criterion_no_judge():
    c = AccuracyCriterion()
    ctx = EvalContext(judge_result=None)
    assert c.score(ctx) == 50.0  # Default fallback


def test_llm_criterion_missing_key():
    jr = JudgeResult(
        criteria_scores=[CriteriaScore(key="other", score=80.0)],
        overall_score=80.0,
        status="pass",
    )
    c = AccuracyCriterion()
    ctx = EvalContext(judge_result=jr)
    assert c.score(ctx) == 50.0  # Default for missing key


# ---- CriteriaRegistry ----

def test_registry_register_and_get():
    reg = CriteriaRegistry()
    c = SchemaValidityCriterion()
    reg.register(c)
    assert reg.get("schema_validity") is c


def test_registry_unregister():
    reg = CriteriaRegistry()
    reg.register(SchemaValidityCriterion())
    assert "schema_validity" in reg
    reg.unregister("schema_validity")
    assert "schema_validity" not in reg


def test_registry_all():
    reg = CriteriaRegistry()
    reg.register(SchemaValidityCriterion())
    reg.register(LatencyPerformanceCriterion())
    assert len(reg.all()) == 2


def test_registry_by_type():
    reg = CriteriaRegistry()
    reg.register(SchemaValidityCriterion())
    reg.register(LatencyPerformanceCriterion())
    reg.register(AccuracyCriterion())

    det = reg.by_type(CriterionType.DETERMINISTIC)
    assert len(det) == 2

    llm = reg.by_type(CriterionType.LLM)
    assert len(llm) == 1


def test_registry_names():
    reg = CriteriaRegistry()
    reg.register(SchemaValidityCriterion())
    reg.register(AccuracyCriterion())
    names = reg.names()
    assert "schema_validity" in names
    assert "accuracy" in names


def test_registry_len():
    reg = CriteriaRegistry()
    assert len(reg) == 0
    reg.register(SchemaValidityCriterion())
    assert len(reg) == 1


def test_registry_contains():
    reg = CriteriaRegistry()
    reg.register(SchemaValidityCriterion())
    assert "schema_validity" in reg
    assert "nonexistent" not in reg


def test_registry_empty_name_raises():
    reg = CriteriaRegistry()

    class BadCriterion(CriterionPlugin):
        name = ""
        def score(self, ctx):
            return 0.0

    try:
        reg.register(BadCriterion())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---- Custom criterion ----

def test_custom_criterion():
    class JsonResponseCriterion(CriterionPlugin):
        name = "json_response"
        criterion_type = CriterionType.DETERMINISTIC
        weight = 1.0
        description = "Response is valid JSON"

        def score(self, ctx: EvalContext) -> float:
            import json
            try:
                json.loads(ctx.agent_answer)
                return 100.0
            except (json.JSONDecodeError, TypeError):
                return 0.0

    c = JsonResponseCriterion()
    assert c.score(EvalContext(agent_answer='{"key": "value"}')) == 100.0
    assert c.score(EvalContext(agent_answer="not json")) == 0.0


# ---- Default registry ----

def test_default_registry():
    reg = default_registry()
    assert len(reg) == 5
    assert "schema_validity" in reg
    assert "latency_performance" in reg
    assert "accuracy" in reg
    assert "completeness" in reg
    assert "hallucination" in reg
