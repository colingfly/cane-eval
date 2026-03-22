"""Tests for the reliability scoring module."""

import pytest

from cane_eval.reliability import ReliabilityConfig, compute_reliability


# ---- ReliabilityConfig ----

def test_default_config():
    config = ReliabilityConfig()
    assert config.correctness_weight == 0.50
    assert config.structural_weight == 0.25
    assert config.performance_weight == 0.25


def test_custom_config():
    config = ReliabilityConfig(
        correctness_weight=0.60,
        structural_weight=0.20,
        performance_weight=0.20,
    )
    assert config.correctness_weight == 0.60


def test_config_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ReliabilityConfig(
            correctness_weight=0.50,
            structural_weight=0.50,
            performance_weight=0.50,
        )


# ---- compute_reliability ----

def test_all_three_pillars():
    score, grade = compute_reliability(
        accuracy_score=80.0,
        schema_score=100.0,
        latency_score=100.0,
    )
    # 80*0.5 + 100*0.25 + 100*0.25 = 40 + 25 + 25 = 90
    assert score == 90.0
    assert grade == "A"


def test_correctness_only():
    score, grade = compute_reliability(
        accuracy_score=75.0,
        schema_score=None,
        latency_score=None,
    )
    assert score == 75.0
    assert grade == "B"


def test_correctness_and_schema():
    score, grade = compute_reliability(
        accuracy_score=80.0,
        schema_score=100.0,
        latency_score=None,
    )
    # Redistributed: c_w = 0.5/0.75 = 0.667, s_w = 0.25/0.75 = 0.333
    # 80*0.667 + 100*0.333 = 53.33 + 33.33 = 86.7
    assert score == 86.7
    assert grade == "B"


def test_correctness_and_latency():
    score, grade = compute_reliability(
        accuracy_score=80.0,
        schema_score=None,
        latency_score=100.0,
    )
    # Redistributed: c_w = 0.5/0.75 = 0.667, p_w = 0.25/0.75 = 0.333
    # 80*0.667 + 100*0.333 = 53.33 + 33.33 = 86.7
    assert score == 86.7
    assert grade == "B"


def test_custom_weights():
    config = ReliabilityConfig(
        correctness_weight=0.60,
        structural_weight=0.20,
        performance_weight=0.20,
    )
    score, grade = compute_reliability(
        accuracy_score=80.0,
        schema_score=100.0,
        latency_score=100.0,
        config=config,
    )
    # 80*0.6 + 100*0.2 + 100*0.2 = 48 + 20 + 20 = 88
    assert score == 88.0
    assert grade == "B"


def test_grade_a():
    score, grade = compute_reliability(95.0, None, None)
    assert grade == "A"


def test_grade_b():
    score, grade = compute_reliability(80.0, None, None)
    assert grade == "B"


def test_grade_c():
    score, grade = compute_reliability(65.0, None, None)
    assert grade == "C"


def test_grade_d():
    score, grade = compute_reliability(45.0, None, None)
    assert grade == "D"


def test_grade_f():
    score, grade = compute_reliability(30.0, None, None)
    assert grade == "F"


def test_zero_score():
    score, grade = compute_reliability(0.0, 0.0, 0.0)
    assert score == 0.0
    assert grade == "F"


def test_perfect_score():
    score, grade = compute_reliability(100.0, 100.0, 100.0)
    assert score == 100.0
    assert grade == "A"
