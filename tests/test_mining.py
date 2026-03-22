"""Tests for the mining module (unit tests that don't call the API)."""

import json

from cane_eval.mining import MinedExample, MiningResult, FAILURE_TYPES


def test_failure_types():
    assert "hallucination" in FAILURE_TYPES
    assert "incomplete" in FAILURE_TYPES
    assert "off_topic" in FAILURE_TYPES
    assert "wrong_format" in FAILURE_TYPES
    assert "factual_error" in FAILURE_TYPES
    assert "other" in FAILURE_TYPES
    assert len(FAILURE_TYPES) >= 6


def test_mined_example_to_dpo():
    ex = MinedExample(
        question="What is 2+2?",
        original_answer="22",
        improved_answer="4",
        failure_type="factual_error",
        original_score=20,
        estimated_improved_score=95,
    )
    dpo = ex.to_dpo()
    assert dpo["prompt"] == "What is 2+2?"
    assert dpo["chosen"] == "4"
    assert dpo["rejected"] == "22"
    assert dpo["failure_type"] == "factual_error"
    assert dpo["rejected_score"] == 20
    assert dpo["chosen_score"] == 95


def test_mined_example_to_sft():
    ex = MinedExample(
        question="What is 2+2?",
        original_answer="22",
        improved_answer="4",
        failure_type="factual_error",
        original_score=20,
        improvement_reasoning="Fixed arithmetic",
    )
    sft = ex.to_sft()
    assert sft["prompt"] == "What is 2+2?"
    assert sft["completion"] == "4"
    assert sft["metadata"]["failure_type"] == "factual_error"
    assert sft["metadata"]["source"] == "failure_mining"


def test_mining_result_to_dpo_string():
    result = MiningResult(
        total_failures=2,
        total_mined=2,
        examples=[
            MinedExample(question="Q1", original_answer="bad1", improved_answer="good1"),
            MinedExample(question="Q2", original_answer="bad2", improved_answer="good2"),
        ],
    )
    jsonl = result.to_dpo_string()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2

    for line in lines:
        parsed = json.loads(line)
        assert "prompt" in parsed
        assert "chosen" in parsed
        assert "rejected" in parsed


def test_mining_result_to_sft_string():
    result = MiningResult(
        total_failures=1,
        total_mined=1,
        examples=[
            MinedExample(question="Q1", original_answer="bad", improved_answer="good"),
        ],
    )
    jsonl = result.to_sft_string()
    parsed = json.loads(jsonl)
    assert parsed["prompt"] == "Q1"
    assert parsed["completion"] == "good"


def test_mining_result_to_file(tmp_path):
    result = MiningResult(
        total_failures=1,
        total_mined=1,
        examples=[
            MinedExample(question="Q1", original_answer="bad", improved_answer="good"),
        ],
    )
    path = tmp_path / "mined.jsonl"
    result.to_file(str(path), format="dpo")

    with open(path) as f:
        content = f.read()
    parsed = json.loads(content)
    assert parsed["prompt"] == "Q1"
