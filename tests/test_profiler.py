"""Tests for profiler.py -- Agent personality profiling."""

import json
import os
import tempfile
import pytest
import numpy as np

from cane_eval.profiler import (
    Profiler,
    ProfileResult,
    PersonalityProfile,
    SteeringVector,
    ContrastivePair,
    EmbeddedResult,
    _score_hedging,
    _score_verbosity,
    _compute_traits,
    _aggregate_personality,
    _extract_contrastive_pairs,
    _compute_steering_vectors,
    _cluster_kmeans,
    _project_pca,
    _label_clusters,
    PERSONALITY_TRAITS,
    HEDGE_MARKERS,
)
from cane_eval.engine import ReliabilitySummary, ReliabilityResult
from cane_eval.judge import JudgeResult, CriteriaScore


# ---- Helpers ----

def _make_judge_result(accuracy=80, hallucination=90, completeness=70, overall=None, status=None):
    """Create a JudgeResult with specified criterion scores."""
    if overall is None:
        overall = (accuracy + hallucination + completeness) / 3
    if status is None:
        status = "pass" if overall >= 70 else "warn" if overall >= 50 else "fail"
    return JudgeResult(
        overall_score=overall,
        overall_reasoning="Test reasoning",
        status=status,
        criteria_scores=[
            CriteriaScore(key="accuracy", score=accuracy, reasoning=""),
            CriteriaScore(key="hallucination", score=hallucination, reasoning=""),
            CriteriaScore(key="completeness", score=completeness, reasoning=""),
        ],
    )


def _make_result(question="What is X?", agent_answer="X is Y.", expected_answer="X is Y.",
                 accuracy=80, hallucination=90, completeness=70, overall=None, status=None):
    """Create a ReliabilityResult."""
    jr = _make_judge_result(accuracy, hallucination, completeness, overall, status)
    return ReliabilityResult(
        question=question,
        expected_answer=expected_answer,
        agent_answer=agent_answer,
        judge_result=jr,
    )


def _make_summary(n=10, name="test-suite"):
    """Create a summary with n diverse results."""
    results = []
    for i in range(n):
        # Alternate between good and bad results
        if i % 3 == 0:
            # Good: high accuracy, low hallucination risk
            r = _make_result(
                question=f"Question {i}: What is fact {i}?",
                agent_answer=f"Fact {i} is a well-established truth supported by evidence.",
                expected_answer=f"Fact {i} is true.",
                accuracy=85 + (i % 10),
                hallucination=90 + (i % 5),
                completeness=80,
                overall=85,
                status="pass",
            )
        elif i % 3 == 1:
            # Bad: overconfident and wrong
            r = _make_result(
                question=f"Question {i}: What is fact {i}?",
                agent_answer=f"Fact {i} is absolutely and definitively X, with no room for doubt.",
                expected_answer=f"Fact {i} is Y.",
                accuracy=20,
                hallucination=15,
                completeness=30,
                overall=25,
                status="fail",
            )
        else:
            # Hedgy: uncertain language
            r = _make_result(
                question=f"Question {i}: What is fact {i}?",
                agent_answer=f"I think perhaps fact {i} might possibly be something, but I'm not sure and it seems like it could be approximately correct, maybe.",
                expected_answer=f"Fact {i} is Z.",
                accuracy=55,
                hallucination=70,
                completeness=40,
                overall=55,
                status="warn",
            )
        results.append(r)

    return ReliabilitySummary(
        suite_name=name,
        total=len(results),
        passed=sum(1 for r in results if r.status == "pass"),
        warned=sum(1 for r in results if r.status == "warn"),
        failed=sum(1 for r in results if r.status == "fail"),
        overall_score=sum(r.score for r in results) / len(results),
        results=results,
    )


# ---- Trait scoring tests ----

class TestHedgingScore:
    def test_no_hedging(self):
        score = _score_hedging("The answer is definitively X.")
        assert score < 20

    def test_moderate_hedging(self):
        score = _score_hedging("I think the answer is probably X, but perhaps it could be Y.")
        assert score > 20

    def test_extreme_hedging(self):
        score = _score_hedging(
            "I believe it's possible that maybe perhaps the answer might possibly be X, "
            "though I'm not sure and it seems like it could be approximately Y, roughly speaking."
        )
        assert score > 40

    def test_empty_string(self):
        score = _score_hedging("")
        assert score == 0.0


class TestVerbosityScore:
    def test_matched_length(self):
        score = _score_verbosity("one two three", "one two three")
        assert 40 <= score <= 60

    def test_verbose(self):
        score = _score_verbosity("a " * 100, "a " * 10)
        assert score > 70

    def test_terse(self):
        score = _score_verbosity("yes", "The answer is a detailed explanation of the topic.")
        assert score < 30


class TestComputeTraits:
    def test_returns_all_traits(self):
        r = _make_result()
        traits = _compute_traits(r)
        for trait in PERSONALITY_TRAITS:
            assert trait in traits
            assert 0 <= traits[trait] <= 100

    def test_high_accuracy_low_overconfidence(self):
        r = _make_result(accuracy=95, hallucination=95)
        traits = _compute_traits(r)
        assert traits["overconfidence"] < 20
        assert traits["calibration"] > 80

    def test_low_accuracy_high_overconfidence(self):
        r = _make_result(accuracy=10, hallucination=10)
        traits = _compute_traits(r)
        assert traits["overconfidence"] > 50


# ---- Personality aggregation tests ----

class TestAggregatePersonality:
    def test_empty(self):
        profile = _aggregate_personality([])
        assert isinstance(profile, PersonalityProfile)
        assert profile.trait_scores == {}

    def test_produces_all_traits(self):
        results = []
        for i in range(5):
            er = EmbeddedResult(
                index=i, question="q", agent_answer="a", expected_answer="e",
                score=70, status="pass",
                traits={t: 50 + i * 5 for t in PERSONALITY_TRAITS},
            )
            results.append(er)

        profile = _aggregate_personality(results)
        assert len(profile.trait_scores) == len(PERSONALITY_TRAITS)
        assert len(profile.dominant_traits) <= 3

    def test_risk_detection(self):
        results = [
            EmbeddedResult(
                index=0, question="q", agent_answer="a", expected_answer="e",
                score=30, status="fail",
                traits={"overconfidence": 90, "calibration": 20, "verbosity": 50,
                         "hedging": 10, "groundedness": 30, "completeness": 40},
            )
        ]
        profile = _aggregate_personality(results)
        assert "overconfidence" in profile.risk_traits


# ---- Contrastive pair tests ----

class TestContrastivePairs:
    def test_extracts_pairs(self):
        high = EmbeddedResult(
            index=0, question="q1", agent_answer="correct answer", expected_answer="correct",
            score=90, status="pass", embedding=[1.0, 0.0, 0.0],
            traits={"hedging": 10},
        )
        low = EmbeddedResult(
            index=1, question="q2", agent_answer="wrong answer", expected_answer="right",
            score=20, status="fail", embedding=[0.0, 1.0, 0.0],
            traits={"hedging": 10},
        )
        pairs = _extract_contrastive_pairs([high, low], high_threshold=80, low_threshold=40)
        assert len(pairs) == 1
        assert pairs[0].wrong_score == 20
        assert pairs[0].right_score == 90

    def test_excludes_hedgy_results(self):
        high = EmbeddedResult(
            index=0, question="q1", agent_answer="correct", expected_answer="correct",
            score=90, status="pass", embedding=[1.0, 0.0],
            traits={"hedging": 80},  # too hedgy
        )
        low = EmbeddedResult(
            index=1, question="q2", agent_answer="wrong", expected_answer="right",
            score=20, status="fail", embedding=[0.0, 1.0],
            traits={"hedging": 10},
        )
        pairs = _extract_contrastive_pairs([high, low])
        assert len(pairs) == 0  # no confident right to pair with


# ---- Steering vector tests ----

class TestSteeringVectors:
    def test_computes_overconfidence_vector(self):
        right = EmbeddedResult(
            index=0, question="q", agent_answer="a", expected_answer="a",
            score=90, status="pass", embedding=[1.0, 0.0, 0.0],
        )
        wrong = EmbeddedResult(
            index=1, question="q", agent_answer="b", expected_answer="a",
            score=20, status="fail", embedding=[0.0, 1.0, 0.0],
        )
        pair = ContrastivePair(
            question="q", confident_right="a", confident_wrong="b",
            right_score=90, wrong_score=20,
            right_embedding=[1.0, 0.0, 0.0], wrong_embedding=[0.0, 1.0, 0.0],
        )
        vectors = _compute_steering_vectors([right, wrong], [pair])
        overconfidence_vec = [v for v in vectors if v.name == "overconfidence"]
        assert len(overconfidence_vec) == 1
        assert overconfidence_vec[0].magnitude > 0

    def test_computes_quality_vector(self):
        results = [
            EmbeddedResult(index=0, question="q", agent_answer="a", expected_answer="a",
                          score=90, status="pass", embedding=[1.0, 0.0]),
            EmbeddedResult(index=1, question="q", agent_answer="b", expected_answer="a",
                          score=20, status="fail", embedding=[0.0, 1.0]),
        ]
        vectors = _compute_steering_vectors(results, [])
        quality_vec = [v for v in vectors if v.name == "quality"]
        assert len(quality_vec) == 1


# ---- Clustering tests ----

class TestClustering:
    def test_kmeans_basic(self):
        data = np.array([
            [0, 0], [0.1, 0.1], [0.05, 0.05],
            [10, 10], [10.1, 10.1], [10.05, 10.05],
        ])
        labels = _cluster_kmeans(data, n_clusters=2)
        assert len(labels) == 6
        # Points near each other should be in the same cluster
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_kmeans_single_point(self):
        data = np.array([[1.0, 2.0]])
        labels = _cluster_kmeans(data, n_clusters=3)
        assert len(labels) == 1


class TestPCA:
    def test_projects_to_2d(self):
        data = np.random.randn(20, 50)
        projected = _project_pca(data, 2)
        assert projected.shape == (20, 2)

    def test_projects_to_3d(self):
        data = np.random.randn(20, 50)
        projected = _project_pca(data, 3)
        assert projected.shape == (20, 3)


# ---- Cluster labeling tests ----

class TestClusterLabeling:
    def test_labels_clusters(self):
        results = []
        for i in range(6):
            er = EmbeddedResult(
                index=i, question=f"q{i}", agent_answer=f"a{i}", expected_answer=f"e{i}",
                score=90 if i < 3 else 20, status="pass" if i < 3 else "fail",
                cluster_id=0 if i < 3 else 1,
                traits={"overconfidence": 10 if i < 3 else 80, "calibration": 90 if i < 3 else 20,
                         "verbosity": 50, "hedging": 50, "groundedness": 50, "completeness": 50},
            )
            results.append(er)

        labels = _label_clusters(results, 2)
        assert len(labels) == 2
        assert "passing" in labels[0]
        assert "failing" in labels[1]


# ---- Data class tests ----

class TestDataClasses:
    def test_embedded_result_to_dict(self):
        er = EmbeddedResult(
            index=0, question="q", agent_answer="a", expected_answer="e",
            score=80, status="pass", projection_2d=[1.0, 2.0],
            traits={"overconfidence": 30}, cluster_id=1,
        )
        d = er.to_dict()
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["score"] == 80
        assert d["cluster_id"] == 1

    def test_personality_profile_to_dict(self):
        pp = PersonalityProfile(
            trait_scores={"overconfidence": 30, "calibration": 80},
            dominant_traits=["calibration"],
            risk_traits=[],
        )
        d = pp.to_dict()
        assert d["trait_scores"]["calibration"] == 80

    def test_steering_vector_to_dict(self):
        sv = SteeringVector(
            name="test", description="desc", direction=[1.0, 0.0],
            magnitude=0.5, positive_label="pos", negative_label="neg",
        )
        d = sv.to_dict()
        assert d["magnitude"] == 0.5
        assert d["name"] == "test"

    def test_contrastive_pair_to_dict(self):
        cp = ContrastivePair(
            question="q", confident_right="right", confident_wrong="wrong",
            right_score=90, wrong_score=20,
        )
        d = cp.to_dict()
        assert d["right_score"] == 90

    def test_profile_result_to_dict(self):
        pr = ProfileResult(
            suite_name="test", total_results=5,
            personality=PersonalityProfile(trait_scores={"overconfidence": 50}),
        )
        d = pr.to_dict()
        assert d["suite_name"] == "test"
        assert d["personality"]["trait_scores"]["overconfidence"] == 50

    def test_profile_result_to_json(self):
        pr = ProfileResult(suite_name="test", total_results=0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            pr.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["suite_name"] == "test"
        finally:
            os.unlink(path)

    def test_profile_result_to_html(self):
        pr = ProfileResult(
            suite_name="test", total_results=1,
            embedded_results=[EmbeddedResult(
                index=0, question="q", agent_answer="a", expected_answer="e",
                score=80, status="pass", projection_2d=[1, 2], traits={"overconfidence": 30},
            )],
            personality=PersonalityProfile(trait_scores={"overconfidence": 30}),
            embedding_model="test-model", projection_method="pca",
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            path = f.name
        try:
            pr.to_html(path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "plotly" in html.lower()
            assert "Agent Personality Profile" in html
        finally:
            os.unlink(path)


# ---- Integration test (no external model needed) ----

class TestProfilerIntegration:
    """Integration tests using mocked embeddings (no sentence-transformers needed)."""

    def test_profile_empty_summary(self):
        summary = ReliabilitySummary(suite_name="empty", total=0, results=[])
        profiler = Profiler(verbose=False)
        # This should return empty result without trying to embed
        profile = profiler.profile(summary)
        assert profile.total_results == 0

    def test_trait_computation_integration(self):
        """Test that trait computation works end-to-end on a summary."""
        summary = _make_summary(n=6)
        for r in summary.results:
            traits = _compute_traits(r)
            assert all(t in traits for t in PERSONALITY_TRAITS)
            assert all(0 <= v <= 100 for v in traits.values())
