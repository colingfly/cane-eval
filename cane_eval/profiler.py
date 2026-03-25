"""
profiler.py -- Agent personality profiling via embedding analysis.

Embeds model outputs from eval runs, projects them into low-dimensional space,
clusters behavioral patterns, and extracts contrastive steering vectors.

This answers the question: "What kind of thinker is this model, and where does it break?"

Pipeline:
  Eval results → Embed outputs → UMAP/PCA projection → Behavioral clustering
  → Personality trait scoring → Contrastive pair extraction → Steering vectors
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from cane_eval.engine import ReliabilitySummary, ReliabilityResult


# ---- Personality trait definitions ----

PERSONALITY_TRAITS = {
    "overconfidence": {
        "description": "Confidently wrong — high fluency, low accuracy",
        "positive_signals": ["hallucination", "factual_error"],
        "negative_signals": ["accuracy"],
    },
    "calibration": {
        "description": "Expressed certainty matches actual correctness",
        "positive_signals": ["accuracy"],
        "negative_signals": ["hallucination"],
    },
    "verbosity": {
        "description": "Response length relative to expected answer length",
        "positive_signals": [],
        "negative_signals": [],
    },
    "hedging": {
        "description": "Excessive qualification and uncertainty language",
        "positive_signals": [],
        "negative_signals": [],
    },
    "groundedness": {
        "description": "Answers grounded in provided context/sources",
        "positive_signals": ["accuracy", "completeness"],
        "negative_signals": ["hallucination", "off_topic"],
    },
    "completeness": {
        "description": "Covers all key points without omission",
        "positive_signals": ["completeness"],
        "negative_signals": ["incomplete"],
    },
}

# Hedging markers — words/phrases that signal uncertainty
HEDGE_MARKERS = [
    "i think", "i believe", "it seems", "perhaps", "maybe", "possibly",
    "it's possible", "it might", "could be", "i'm not sure", "approximately",
    "roughly", "generally", "typically", "often", "usually", "likely",
    "probably", "it appears", "as far as i know", "in my opinion",
    "it depends", "not entirely", "somewhat", "arguably",
]


# ---- Data classes ----

@dataclass
class EmbeddedResult:
    """A single eval result with its embedding and metadata."""
    index: int
    question: str
    agent_answer: str
    expected_answer: str
    score: float
    status: str  # pass/warn/fail
    embedding: Optional[list[float]] = None
    projection_2d: Optional[list[float]] = None
    projection_3d: Optional[list[float]] = None
    criteria_scores: dict = field(default_factory=dict)
    failure_type: Optional[str] = None
    traits: dict = field(default_factory=dict)
    cluster_id: int = -1

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "question": self.question,
            "agent_answer": self.agent_answer[:200],
            "expected_answer": self.expected_answer[:200],
            "score": self.score,
            "status": self.status,
            "criteria_scores": self.criteria_scores,
            "traits": self.traits,
            "cluster_id": self.cluster_id,
        }
        if self.projection_2d:
            d["x"] = self.projection_2d[0]
            d["y"] = self.projection_2d[1]
        if self.projection_3d:
            d["x3d"] = self.projection_3d[0]
            d["y3d"] = self.projection_3d[1]
            d["z3d"] = self.projection_3d[2]
        if self.failure_type:
            d["failure_type"] = self.failure_type
        return d


@dataclass
class PersonalityProfile:
    """Aggregate personality profile for a model/agent."""
    trait_scores: dict = field(default_factory=dict)  # trait -> 0-100
    trait_descriptions: dict = field(default_factory=dict)
    dominant_traits: list = field(default_factory=list)  # top 3
    risk_traits: list = field(default_factory=list)  # concerning traits

    def to_dict(self) -> dict:
        return {
            "trait_scores": self.trait_scores,
            "trait_descriptions": self.trait_descriptions,
            "dominant_traits": self.dominant_traits,
            "risk_traits": self.risk_traits,
        }


@dataclass
class SteeringVector:
    """A direction in embedding space between behavioral clusters."""
    name: str
    description: str
    direction: list[float]  # unit vector
    magnitude: float  # separation strength
    positive_label: str  # what + direction means
    negative_label: str  # what - direction means
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "direction": self.direction,
            "magnitude": round(self.magnitude, 4),
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


@dataclass
class ContrastivePair:
    """A pair of responses: one confidently right, one confidently wrong."""
    question: str
    confident_right: str
    confident_wrong: str
    right_score: float
    wrong_score: float
    right_embedding: Optional[list[float]] = None
    wrong_embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "confident_right": self.confident_right,
            "confident_wrong": self.confident_wrong,
            "right_score": self.right_score,
            "wrong_score": self.wrong_score,
        }


@dataclass
class ProfileResult:
    """Complete profiling result for an eval run."""
    suite_name: str
    total_results: int = 0
    embedded_results: list[EmbeddedResult] = field(default_factory=list)
    personality: Optional[PersonalityProfile] = None
    clusters: dict = field(default_factory=dict)  # cluster_id -> label
    steering_vectors: list[SteeringVector] = field(default_factory=list)
    contrastive_pairs: list[ContrastivePair] = field(default_factory=list)
    embedding_model: str = ""
    projection_method: str = ""

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total_results": self.total_results,
            "embedding_model": self.embedding_model,
            "projection_method": self.projection_method,
            "personality": self.personality.to_dict() if self.personality else None,
            "clusters": self.clusters,
            "steering_vectors": [sv.to_dict() for sv in self.steering_vectors],
            "contrastive_pairs": [cp.to_dict() for cp in self.contrastive_pairs],
            "results": [r.to_dict() for r in self.embedded_results],
        }

    def to_json(self, path: str):
        """Write full profile to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_html(self, path: str):
        """Generate self-contained HTML report with interactive Plotly charts."""
        html = _generate_html_report(self)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


# ---- Embedding helpers ----

def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence-transformers model for embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for profiling. "
            "Install it: pip install sentence-transformers"
        )


def _embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of texts, returning (n, dim) array."""
    model = _get_embedder(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def _project_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to low-dimensional space via UMAP."""
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        # Fallback to PCA
        return _project_pca(embeddings, n_components)


def _project_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings via PCA (no external dep beyond numpy)."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top n_components (eigenvalues are sorted ascending)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]
    projected = centered @ components
    return projected


def _cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 4, max_iter: int = 100) -> np.ndarray:
    """Simple k-means clustering (no sklearn dependency)."""
    n = len(embeddings)
    if n <= n_clusters:
        return np.arange(n)

    # Initialize centroids via k-means++
    rng = np.random.RandomState(42)
    centroids = [embeddings[rng.randint(n)]]

    for _ in range(1, n_clusters):
        dists = np.array([
            min(np.sum((e - c) ** 2) for c in centroids)
            for e in embeddings
        ])
        probs = dists / dists.sum()
        centroids.append(embeddings[rng.choice(n, p=probs)])

    centroids = np.array(centroids)

    # Iterate
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        for i, e in enumerate(embeddings):
            dists = np.sum((centroids - e) ** 2, axis=1)
            labels[i] = np.argmin(dists)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            members = embeddings[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


# ---- Trait scoring ----

def _score_hedging(text: str) -> float:
    """Score hedging level (0=none, 100=extremely hedgy)."""
    text_lower = text.lower()
    count = sum(1 for marker in HEDGE_MARKERS if marker in text_lower)
    words = len(text.split())
    if words == 0:
        return 0.0
    # Normalize: >5 hedge markers per 100 words = very hedgy
    rate = (count / max(words, 1)) * 100
    return min(100.0, rate * 20)


def _score_verbosity(agent_answer: str, expected_answer: str) -> float:
    """Score verbosity (0=terse, 50=matched, 100=very verbose)."""
    agent_words = len(agent_answer.split())
    expected_words = max(len(expected_answer.split()), 1)
    ratio = agent_words / expected_words
    # 1.0 = perfectly matched = 50, >2.0 = very verbose = 100, <0.5 = terse = 0
    return min(100.0, max(0.0, ratio * 50))


def _compute_traits(result: ReliabilityResult) -> dict:
    """Compute personality trait scores for a single result."""
    scores = result.judge_result.score_dict() if result.judge_result else {}
    traits = {}

    # Overconfidence: high fluency (long, confident answer) but low accuracy
    accuracy = float(scores.get("accuracy", 50))
    hallucination = float(scores.get("hallucination", 100))
    # Hallucination score is inverted: 100 = no hallucination, 0 = lots
    overconfidence = max(0.0, (100 - accuracy) * (100 - hallucination) / 100)
    traits["overconfidence"] = round(overconfidence, 1)

    # Calibration: accuracy and hallucination alignment
    traits["calibration"] = round((accuracy + hallucination) / 2, 1)

    # Verbosity
    traits["verbosity"] = round(
        _score_verbosity(result.agent_answer, result.expected_answer), 1
    )

    # Hedging
    traits["hedging"] = round(_score_hedging(result.agent_answer), 1)

    # Groundedness
    completeness = float(scores.get("completeness", 50))
    traits["groundedness"] = round((accuracy + hallucination + completeness) / 3, 1)

    # Completeness
    traits["completeness"] = round(completeness, 1)

    return traits


def _aggregate_personality(embedded_results: list[EmbeddedResult]) -> PersonalityProfile:
    """Aggregate individual trait scores into a personality profile."""
    if not embedded_results:
        return PersonalityProfile()

    all_traits = {}
    for trait_name in PERSONALITY_TRAITS:
        values = [r.traits.get(trait_name, 50) for r in embedded_results]
        all_traits[trait_name] = round(sum(values) / len(values), 1)

    # Sort by deviation from 50 (neutral) — most extreme traits first
    sorted_traits = sorted(
        all_traits.items(),
        key=lambda x: abs(x[1] - 50),
        reverse=True,
    )

    dominant = [t[0] for t in sorted_traits[:3]]
    risk = [
        t[0] for t in sorted_traits
        if t[0] in ("overconfidence", "hedging") and t[1] > 60
    ]

    return PersonalityProfile(
        trait_scores=all_traits,
        trait_descriptions={k: v["description"] for k, v in PERSONALITY_TRAITS.items()},
        dominant_traits=dominant,
        risk_traits=risk,
    )


# ---- Contrastive extraction ----

def _extract_contrastive_pairs(
    embedded_results: list[EmbeddedResult],
    high_threshold: float = 80,
    low_threshold: float = 40,
) -> list[ContrastivePair]:
    """
    Extract contrastive pairs: confidently right vs. confidently wrong.

    High-scoring results with low hedging = "confidently right"
    Low-scoring results with low hedging = "confidently wrong" (overconfident)
    """
    confident_right = [
        r for r in embedded_results
        if r.score >= high_threshold and r.traits.get("hedging", 50) < 40
    ]
    confident_wrong = [
        r for r in embedded_results
        if r.score <= low_threshold and r.traits.get("hedging", 50) < 40
    ]

    pairs = []
    # Match by closest question similarity (or just pair by index)
    for wrong in confident_wrong:
        # Find best matching "right" result (prefer same question if exists)
        best = None
        for right in confident_right:
            if right.question == wrong.question:
                best = right
                break
        if best is None and confident_right:
            best = confident_right[0]

        if best:
            pairs.append(ContrastivePair(
                question=wrong.question,
                confident_right=best.agent_answer,
                confident_wrong=wrong.agent_answer,
                right_score=best.score,
                wrong_score=wrong.score,
                right_embedding=best.embedding,
                wrong_embedding=wrong.embedding,
            ))

    return pairs


def _compute_steering_vectors(
    embedded_results: list[EmbeddedResult],
    contrastive_pairs: list[ContrastivePair],
) -> list[SteeringVector]:
    """
    Compute steering vectors from contrastive pairs and cluster analysis.

    Primary vector: overconfidence direction (confidently wrong - confidently right)
    """
    vectors = []

    # 1. Overconfidence vector from contrastive pairs
    if contrastive_pairs:
        right_embeds = [
            np.array(p.right_embedding) for p in contrastive_pairs
            if p.right_embedding is not None
        ]
        wrong_embeds = [
            np.array(p.wrong_embedding) for p in contrastive_pairs
            if p.wrong_embedding is not None
        ]

        if right_embeds and wrong_embeds:
            right_mean = np.mean(right_embeds, axis=0)
            wrong_mean = np.mean(wrong_embeds, axis=0)
            direction = wrong_mean - right_mean
            magnitude = float(np.linalg.norm(direction))

            if magnitude > 0:
                unit_direction = (direction / magnitude).tolist()
                vectors.append(SteeringVector(
                    name="overconfidence",
                    description="Direction from calibrated confidence to overconfidence in embedding space",
                    direction=unit_direction,
                    magnitude=magnitude,
                    positive_label="overconfident",
                    negative_label="calibrated",
                    n_positive=len(wrong_embeds),
                    n_negative=len(right_embeds),
                ))

    # 2. Quality vector from pass/fail clusters
    pass_embeds = [
        np.array(r.embedding) for r in embedded_results
        if r.status == "pass" and r.embedding is not None
    ]
    fail_embeds = [
        np.array(r.embedding) for r in embedded_results
        if r.status == "fail" and r.embedding is not None
    ]

    if pass_embeds and fail_embeds:
        pass_mean = np.mean(pass_embeds, axis=0)
        fail_mean = np.mean(fail_embeds, axis=0)
        direction = fail_mean - pass_mean
        magnitude = float(np.linalg.norm(direction))

        if magnitude > 0:
            unit_direction = (direction / magnitude).tolist()
            vectors.append(SteeringVector(
                name="quality",
                description="Direction from high-quality to low-quality responses",
                direction=unit_direction,
                magnitude=magnitude,
                positive_label="low_quality",
                negative_label="high_quality",
                n_positive=len(fail_embeds),
                n_negative=len(pass_embeds),
            ))

    return vectors


# ---- Cluster labeling ----

def _label_clusters(
    embedded_results: list[EmbeddedResult],
    n_clusters: int,
) -> dict:
    """Label clusters based on dominant traits and pass/fail composition."""
    labels = {}
    for k in range(n_clusters):
        members = [r for r in embedded_results if r.cluster_id == k]
        if not members:
            labels[k] = "empty"
            continue

        # Compute cluster stats
        avg_score = sum(r.score for r in members) / len(members)
        fail_rate = sum(1 for r in members if r.status == "fail") / len(members)

        # Find dominant trait
        trait_avgs = {}
        for trait in PERSONALITY_TRAITS:
            vals = [r.traits.get(trait, 50) for r in members]
            trait_avgs[trait] = sum(vals) / len(vals)

        dominant_trait = max(trait_avgs, key=lambda t: abs(trait_avgs[t] - 50))
        dominant_val = trait_avgs[dominant_trait]

        # Build label
        if fail_rate > 0.6:
            quality = "failing"
        elif fail_rate > 0.3:
            quality = "mixed"
        else:
            quality = "passing"

        trait_level = "high" if dominant_val > 60 else "low" if dominant_val < 40 else "moderate"
        labels[k] = f"{quality} | {trait_level} {dominant_trait} (n={len(members)}, avg={avg_score:.0f})"

    return labels


# ---- HTML report generation ----

def _generate_html_report(profile: ProfileResult) -> str:
    """Generate self-contained HTML report with Plotly scatter + radar charts."""
    results_json = json.dumps([r.to_dict() for r in profile.embedded_results])
    personality_json = json.dumps(profile.personality.to_dict() if profile.personality else {})
    steering_json = json.dumps([sv.to_dict() for sv in profile.steering_vectors])
    clusters_json = json.dumps(profile.clusters)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cane-Eval Personality Profile: {profile.suite_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; font-weight: 600; margin-bottom: 8px; color: #fff; }}
  h2 {{ font-size: 18px; font-weight: 500; margin: 24px 0 12px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .card {{ background: #141414; border: 1px solid #222; border-radius: 12px; padding: 20px; }}
  .card-full {{ grid-column: 1 / -1; }}
  .trait-bar {{ display: flex; align-items: center; margin: 8px 0; gap: 12px; }}
  .trait-name {{ width: 140px; font-size: 13px; color: #aaa; text-align: right; }}
  .trait-track {{ flex: 1; height: 8px; background: #222; border-radius: 4px; overflow: hidden; }}
  .trait-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
  .trait-value {{ width: 40px; font-size: 13px; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge-risk {{ background: #3d1111; color: #ff6b6b; }}
  .badge-good {{ background: #0d3311; color: #51cf66; }}
  .badge-neutral {{ background: #1a1a2e; color: #748ffc; }}
  .steering {{ margin: 8px 0; padding: 12px; background: #1a1a1a; border-radius: 8px; }}
  .steering-name {{ font-weight: 600; font-size: 14px; color: #fff; }}
  .steering-desc {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .steering-stat {{ font-size: 13px; color: #aaa; margin-top: 4px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: 700; color: #fff; }}
  .stat-label {{ font-size: 12px; color: #888; margin-top: 4px; }}
  #scatter {{ width: 100%; height: 500px; }}
  #radar {{ width: 100%; height: 400px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Agent Personality Profile</h1>
  <div class="subtitle">{profile.suite_name} | {profile.total_results} responses | {profile.embedding_model} | {profile.projection_method}</div>

  <div class="stat-grid">
    <div class="stat"><div class="stat-value" id="stat-total">{profile.total_results}</div><div class="stat-label">Responses</div></div>
    <div class="stat"><div class="stat-value" id="stat-clusters">{len(profile.clusters)}</div><div class="stat-label">Clusters</div></div>
    <div class="stat"><div class="stat-value" id="stat-vectors">{len(profile.steering_vectors)}</div><div class="stat-label">Steering Vectors</div></div>
    <div class="stat"><div class="stat-value" id="stat-pairs">{len(profile.contrastive_pairs)}</div><div class="stat-label">Contrastive Pairs</div></div>
  </div>

  <div class="grid">
    <div class="card card-full">
      <h2>Embedding Space</h2>
      <div id="scatter"></div>
    </div>

    <div class="card">
      <h2>Personality Radar</h2>
      <div id="radar"></div>
    </div>

    <div class="card">
      <h2>Trait Scores</h2>
      <div id="traits"></div>
    </div>

    <div class="card">
      <h2>Steering Vectors</h2>
      <div id="vectors"></div>
    </div>

    <div class="card">
      <h2>Cluster Labels</h2>
      <div id="cluster-labels"></div>
    </div>
  </div>
</div>

<script>
const results = {results_json};
const personality = {personality_json};
const steeringVectors = {steering_json};
const clusters = {clusters_json};

// Color map for status
const statusColors = {{ pass: '#51cf66', warn: '#ffd43b', fail: '#ff6b6b' }};

// Scatter plot
if (results.length > 0 && results[0].x !== undefined) {{
  const traces = ['pass', 'warn', 'fail'].map(status => {{
    const filtered = results.filter(r => r.status === status);
    return {{
      x: filtered.map(r => r.x),
      y: filtered.map(r => r.y),
      mode: 'markers',
      type: 'scatter',
      name: status,
      marker: {{
        color: statusColors[status],
        size: filtered.map(r => 6 + (r.score / 20)),
        opacity: 0.7,
      }},
      text: filtered.map(r =>
        `Score: ${{r.score}}<br>` +
        `Q: ${{r.question.substring(0, 60)}}...<br>` +
        `Cluster: ${{r.cluster_id}}<br>` +
        `Overconfidence: ${{r.traits.overconfidence || 'N/A'}}`
      ),
      hoverinfo: 'text',
    }};
  }});

  Plotly.newPlot('scatter', traces, {{
    paper_bgcolor: '#141414',
    plot_bgcolor: '#141414',
    font: {{ color: '#e0e0e0' }},
    xaxis: {{ showgrid: false, zeroline: false, title: 'Dimension 1' }},
    yaxis: {{ showgrid: false, zeroline: false, title: 'Dimension 2' }},
    legend: {{ x: 0, y: 1 }},
    margin: {{ l: 40, r: 20, t: 20, b: 40 }},
  }});
}}

// Radar chart
if (personality && personality.trait_scores) {{
  const traits = Object.keys(personality.trait_scores);
  const values = traits.map(t => personality.trait_scores[t]);

  Plotly.newPlot('radar', [{{
    type: 'scatterpolar',
    r: [...values, values[0]],
    theta: [...traits, traits[0]],
    fill: 'toself',
    fillcolor: 'rgba(116, 143, 252, 0.2)',
    line: {{ color: '#748ffc' }},
    marker: {{ color: '#748ffc', size: 6 }},
  }}], {{
    polar: {{
      bgcolor: '#141414',
      radialaxis: {{ visible: true, range: [0, 100], color: '#444', gridcolor: '#222' }},
      angularaxis: {{ color: '#aaa', gridcolor: '#222' }},
    }},
    paper_bgcolor: '#141414',
    font: {{ color: '#e0e0e0', size: 12 }},
    margin: {{ l: 60, r: 60, t: 40, b: 40 }},
    showlegend: false,
  }});
}}

// Trait bars
if (personality && personality.trait_scores) {{
  const container = document.getElementById('traits');
  const traitColors = {{
    overconfidence: '#ff6b6b',
    calibration: '#51cf66',
    verbosity: '#ffd43b',
    hedging: '#ff922b',
    groundedness: '#748ffc',
    completeness: '#20c997',
  }};
  for (const [trait, score] of Object.entries(personality.trait_scores)) {{
    const color = traitColors[trait] || '#748ffc';
    const badge = personality.risk_traits.includes(trait)
      ? '<span class="badge badge-risk">RISK</span>'
      : score > 70 ? '<span class="badge badge-good">STRONG</span>'
      : '<span class="badge badge-neutral">OK</span>';
    container.innerHTML += `
      <div class="trait-bar">
        <div class="trait-name">${{trait}} ${{badge}}</div>
        <div class="trait-track"><div class="trait-fill" style="width:${{score}}%;background:${{color}}"></div></div>
        <div class="trait-value" style="color:${{color}}">${{score}}</div>
      </div>
    `;
  }}
}}

// Steering vectors
const vecContainer = document.getElementById('vectors');
for (const sv of steeringVectors) {{
  vecContainer.innerHTML += `
    <div class="steering">
      <div class="steering-name">${{sv.name}}</div>
      <div class="steering-desc">${{sv.description}}</div>
      <div class="steering-stat">
        Magnitude: ${{sv.magnitude}} | ${{sv.negative_label}} (${{sv.n_negative}}) ←→ ${{sv.positive_label}} (${{sv.n_positive}})
      </div>
    </div>
  `;
}}
if (steeringVectors.length === 0) {{
  vecContainer.innerHTML = '<div class="steering-desc">Not enough contrastive data to compute steering vectors.</div>';
}}

// Cluster labels
const clusterContainer = document.getElementById('cluster-labels');
for (const [id, label] of Object.entries(clusters)) {{
  clusterContainer.innerHTML += `
    <div class="steering">
      <div class="steering-name">Cluster ${{id}}</div>
      <div class="steering-desc">${{label}}</div>
    </div>
  `;
}}
</script>
</body>
</html>"""


# ---- Main profiler class ----

class Profiler:
    """
    Agent personality profiler.

    Embeds eval outputs, projects to 2D/3D, clusters behavioral patterns,
    extracts personality traits, and computes contrastive steering vectors.

    Usage:
        summary = runner.run(suite, agent=my_agent)
        profiler = Profiler()
        profile = profiler.profile(summary)

        # Interactive HTML report
        profile.to_html("personality.html")

        # JSON for API consumption
        profile.to_json("personality.json")

        # Access steering vectors
        for sv in profile.steering_vectors:
            print(f"{sv.name}: magnitude={sv.magnitude:.3f}")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_clusters: int = 4,
        projection: str = "auto",
        verbose: bool = True,
    ):
        """
        Args:
            embedding_model: Sentence-transformers model for embedding outputs.
            n_clusters: Number of behavioral clusters to find.
            projection: Dimensionality reduction method: "umap", "pca", or "auto" (try UMAP, fall back to PCA).
            verbose: Print progress.
        """
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.projection = projection
        self.verbose = verbose

    def profile(
        self,
        summary: ReliabilitySummary,
        high_threshold: float = 80,
        low_threshold: float = 40,
    ) -> ProfileResult:
        """
        Profile an eval run's outputs.

        Args:
            summary: Completed eval run summary.
            high_threshold: Score threshold for "confidently right" (contrastive pairs).
            low_threshold: Score threshold for "confidently wrong" (contrastive pairs).

        Returns:
            ProfileResult with embeddings, projections, personality, and steering vectors.
        """
        if not summary.results:
            return ProfileResult(suite_name=summary.suite_name)

        if self.verbose:
            print(f"  Profiling {len(summary.results)} results...")

        # Step 1: Compute traits for each result
        if self.verbose:
            print(f"  Computing personality traits...")
        embedded_results = []
        for i, r in enumerate(summary.results):
            traits = _compute_traits(r)
            er = EmbeddedResult(
                index=i,
                question=r.question,
                agent_answer=r.agent_answer,
                expected_answer=r.expected_answer,
                score=r.score,
                status=r.status,
                criteria_scores=r.judge_result.score_dict() if r.judge_result else {},
                traits=traits,
            )
            embedded_results.append(er)

        # Step 2: Embed agent answers
        if self.verbose:
            print(f"  Embedding {len(embedded_results)} responses with {self.embedding_model}...")
        texts = [r.agent_answer for r in embedded_results]
        embeddings = _embed_texts(texts, self.embedding_model)

        for i, er in enumerate(embedded_results):
            er.embedding = embeddings[i].tolist()

        # Step 3: Project to 2D and 3D
        projection_method = self.projection
        if self.verbose:
            print(f"  Projecting to 2D/3D ({projection_method})...")

        if len(embeddings) >= 4:
            if projection_method == "auto":
                try:
                    proj_2d = _project_umap(embeddings, 2)
                    proj_3d = _project_umap(embeddings, 3)
                    projection_method = "umap"
                except ImportError:
                    proj_2d = _project_pca(embeddings, 2)
                    proj_3d = _project_pca(embeddings, 3)
                    projection_method = "pca"
            elif projection_method == "umap":
                proj_2d = _project_umap(embeddings, 2)
                proj_3d = _project_umap(embeddings, 3)
            else:
                proj_2d = _project_pca(embeddings, 2)
                proj_3d = _project_pca(embeddings, 3)

            for i, er in enumerate(embedded_results):
                er.projection_2d = proj_2d[i].tolist()
                er.projection_3d = proj_3d[i].tolist()
        else:
            projection_method = "none (too few results)"

        # Step 4: Cluster
        n_clusters = min(self.n_clusters, len(embeddings))
        if self.verbose:
            print(f"  Clustering into {n_clusters} groups...")

        if len(embeddings) >= 2:
            labels = _cluster_kmeans(embeddings, n_clusters)
            for i, er in enumerate(embedded_results):
                er.cluster_id = int(labels[i])
        cluster_labels = _label_clusters(embedded_results, n_clusters)

        # Step 5: Aggregate personality
        if self.verbose:
            print(f"  Computing personality profile...")
        personality = _aggregate_personality(embedded_results)

        # Step 6: Extract contrastive pairs
        if self.verbose:
            print(f"  Extracting contrastive pairs...")
        contrastive_pairs = _extract_contrastive_pairs(
            embedded_results, high_threshold, low_threshold
        )

        # Step 7: Compute steering vectors
        if self.verbose:
            print(f"  Computing steering vectors...")
        steering_vectors = _compute_steering_vectors(embedded_results, contrastive_pairs)

        if self.verbose:
            print(f"  Done: {len(contrastive_pairs)} contrastive pairs, {len(steering_vectors)} steering vectors")

        return ProfileResult(
            suite_name=summary.suite_name,
            total_results=len(embedded_results),
            embedded_results=embedded_results,
            personality=personality,
            clusters={str(k): v for k, v in cluster_labels.items()},
            steering_vectors=steering_vectors,
            contrastive_pairs=contrastive_pairs,
            embedding_model=self.embedding_model,
            projection_method=projection_method,
        )

    def profile_from_json(self, path: str) -> ProfileResult:
        """Profile from a saved eval results JSON file."""
        from cane_eval.engine import ReliabilitySummary, ReliabilityResult
        from cane_eval.judge import JudgeResult, CriteriaScore

        with open(path, "r") as f:
            data = json.load(f)

        results_list = data.get("results", data if isinstance(data, list) else [])
        eval_results = []
        for r in results_list:
            criteria_scores = []
            for name, score_val in (r.get("criteria_scores") or {}).items():
                criteria_scores.append(CriteriaScore(key=name, score=float(score_val), reasoning=""))
            jr = JudgeResult(
                overall_score=r.get("overall_score", 0),
                overall_reasoning=r.get("judge_reasoning", ""),
                status=r.get("status", "fail"),
                criteria_scores=criteria_scores,
            )
            eval_results.append(ReliabilityResult(
                question=r.get("question", ""),
                expected_answer=r.get("expected_answer", ""),
                agent_answer=r.get("agent_answer", ""),
                judge_result=jr,
                tags=r.get("tags", []),
            ))

        summary = ReliabilitySummary(
            suite_name=data.get("suite_name", "loaded"),
            total=len(eval_results),
            results=eval_results,
        )
        return self.profile(summary)
