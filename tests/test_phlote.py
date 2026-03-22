"""Tests for the Phlote integration (unit tests, no API calls)."""

import json

from cane_eval.integrations.phlote import (
    RightsClearanceCriterion,
    MetadataCompletenessCriterion,
    LicensingValidityCriterion,
    ContextualRelevanceCriterion,
    phlote_registry,
    register_phlote_failure_types,
    PHLOTE_FAILURE_TYPES,
)
from cane_eval.criteria import EvalContext, CriterionType
from cane_eval.judge import JudgeResult


# ---- Phlote failure types ----

def test_phlote_failure_types():
    assert "rights_violation" in PHLOTE_FAILURE_TYPES
    assert "metadata_missing" in PHLOTE_FAILURE_TYPES
    assert "licensing_error" in PHLOTE_FAILURE_TYPES


def test_register_phlote_failure_types():
    register_phlote_failure_types()
    from cane_eval.mining import FAILURE_TYPES
    assert "rights_violation" in FAILURE_TYPES
    assert "metadata_missing" in FAILURE_TYPES
    assert "licensing_error" in FAILURE_TYPES


# ---- RightsClearanceCriterion ----

def test_rights_clearance_json_with_fields():
    c = RightsClearanceCriterion()
    data = json.dumps({"rights": "cleared", "license": "CC-BY", "track": "test"})
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 100.0


def test_rights_clearance_json_partial():
    c = RightsClearanceCriterion()
    data = json.dumps({"rights": "pending", "track": "test"})
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 50.0


def test_rights_clearance_text_keywords():
    c = RightsClearanceCriterion()
    ctx = EvalContext(agent_answer="This track is licensed under copyright with full rights clearance")
    score = c.score(ctx)
    assert score == 100.0


def test_rights_clearance_text_partial():
    c = RightsClearanceCriterion()
    ctx = EvalContext(agent_answer="This track has rights information")
    score = c.score(ctx)
    assert score == 50.0


def test_rights_clearance_no_info():
    c = RightsClearanceCriterion()
    ctx = EvalContext(agent_answer="Here is a track for you")
    score = c.score(ctx)
    assert score == 0.0


# ---- MetadataCompletenessCriterion ----

def test_metadata_complete():
    c = MetadataCompletenessCriterion()
    data = json.dumps({
        "title": "Test Song",
        "artist": "Test Artist",
        "duration": "3:30",
        "genre": "Pop",
        "isrc": "USRC12345678",
        "release_date": "2024-01-01",
        "album": "Test Album",
        "bpm": 120,
        "key": "C Major",
    })
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 100.0


def test_metadata_required_only():
    c = MetadataCompletenessCriterion()
    data = json.dumps({"title": "Test Song", "artist": "Test Artist"})
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 50.0


def test_metadata_missing_required():
    c = MetadataCompletenessCriterion()
    data = json.dumps({"title": "Test Song", "genre": "Pop"})
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 25.0  # 1/2 required * 50


def test_metadata_not_json():
    c = MetadataCompletenessCriterion()
    ctx = EvalContext(agent_answer="Not JSON")
    assert c.score(ctx) == 0.0


def test_metadata_json_array():
    c = MetadataCompletenessCriterion()
    ctx = EvalContext(agent_answer="[1, 2, 3]")
    assert c.score(ctx) == 0.0


# ---- LicensingValidityCriterion ----

def test_licensing_valid_complete():
    c = LicensingValidityCriterion()
    data = json.dumps({
        "license_type": "creative_commons",
        "license_holder": "Test Publisher",
        "licensing_status": "active",
    })
    ctx = EvalContext(agent_answer=data)
    assert c.score(ctx) == 100.0


def test_licensing_partial():
    c = LicensingValidityCriterion()
    data = json.dumps({
        "license_type": "commercial",
        "licensing_status": "invalid_status",
    })
    ctx = EvalContext(agent_answer=data)
    score = c.score(ctx)
    assert score == 50.0  # 1 valid out of 2 checks


def test_licensing_no_fields():
    c = LicensingValidityCriterion()
    data = json.dumps({"title": "Test"})
    ctx = EvalContext(agent_answer=data)
    assert c.score(ctx) == 0.0


def test_licensing_not_json():
    c = LicensingValidityCriterion()
    ctx = EvalContext(agent_answer="Not JSON")
    assert c.score(ctx) == 0.0


# ---- ContextualRelevanceCriterion ----

def test_contextual_relevance_with_judge():
    c = ContextualRelevanceCriterion()
    jr = JudgeResult(overall_score=85.0, status="pass")
    ctx = EvalContext(judge_result=jr)
    assert c.score(ctx) == 85.0


def test_contextual_relevance_no_judge():
    c = ContextualRelevanceCriterion()
    ctx = EvalContext(judge_result=None)
    assert c.score(ctx) == 50.0


# ---- Phlote registry ----

def test_phlote_registry():
    reg = phlote_registry()
    assert len(reg) == 4
    assert "rights_clearance" in reg
    assert "metadata_completeness" in reg
    assert "licensing_validity" in reg
    assert "contextual_relevance" in reg


def test_phlote_criterion_types():
    reg = phlote_registry()
    det = reg.by_type(CriterionType.DETERMINISTIC)
    llm = reg.by_type(CriterionType.LLM)
    assert len(det) == 3
    assert len(llm) == 1
