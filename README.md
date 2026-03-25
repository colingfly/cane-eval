# cane-eval

AI system reliability infrastructure. Evaluate any AI system's reliability across correctness, structure, and performance.

[![PyPI](https://img.shields.io/pypi/v/cane-eval)](https://pypi.org/project/cane-eval/)

```
pip install cane-eval
```

## What it does

Extensible reliability evaluation for AI systems — not just LLMs, but any AI agent, API, or pipeline. One tool, one score, one answer: **would this break in production?**

```
  Support Agent                              28.4s

  Overall: [=========----------] 47

  1 passed  1 warned  3 failed  (5 total)
  Pass rate: 20%

  Latency:  p50: 1.2s  p95: 8.4s  max: 12.1s
  Schema:   3/5 valid (60%)

  Reliability: [=======-----------] 52 (D)
```

## 30-Second Demo

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cane-eval demo
```

## Quick Start

**1. Define tests** (`tests.yaml`):

```yaml
name: Support Agent

criteria:
  - key: accuracy
    weight: 40
  - key: completeness
    weight: 30
  - key: hallucination
    weight: 30

# Optional: validate response structure
schema:
  type: object
  required: [answer, sources]
  properties:
    answer: { type: string }
    sources: { type: array }

# Optional: latency target for reliability scoring
latency_target_ms: 5000

# Optional: configure reliability weights
reliability:
  correctness_weight: 0.60
  structural_weight: 0.20
  performance_weight: 0.20

# Optional: parallel execution
concurrency: 5

tests:
  - question: What is the return policy?
    expected_answer: 30-day return policy for unused items with receipt
  - question: How do I reset my password?
    expected_answer: Go to Settings > Security > Reset Password
```

**2. Run**:

```bash
cane-eval run tests.yaml
```

**3. Production checks**:

```bash
# Parallel execution
cane-eval run tests.yaml -j 5

# Custom reliability weights (correctness:structural:performance)
cane-eval run tests.yaml --reliability-weights 60:20:20

# Validate responses against JSON schema
cane-eval run tests.yaml --schema schema.json --fail-on-schema

# Fail if p95 latency exceeds 10 seconds
cane-eval run tests.yaml --latency-p95 10000

# All together + mine failures into training data
cane-eval run tests.yaml -j 5 --schema schema.json --latency-p95 10000 --mine --export dpo
```

## Reliability Score

Every eval run produces an Agent Reliability Score (0-100) across three pillars:

| Pillar | What it measures | How |
|--------|-----------------|-----|
| **Correctness** | Does the answer look good? | LLM judge (accuracy, completeness, hallucination) |
| **Structural** | Does the response match expected format? | JSON schema validation |
| **Performance** | Is it fast enough for production? | p95 latency vs target |

Grades: **A** (90+) production-ready, **B** (75+) mostly reliable, **C** (60+) needs work, **D** (40+) significant gaps, **F** (<40) not ready.

Weights are configurable per-suite or via CLI:

```yaml
reliability:
  correctness_weight: 0.60
  structural_weight: 0.20
  performance_weight: 0.20
```

## Extensible Criteria

Build custom evaluation criteria for any domain:

```python
from cane_eval import CriterionPlugin, CriterionType, EvalContext, CriteriaRegistry

class ResponseTimeCriterion(CriterionPlugin):
    name = "response_time"
    criterion_type = CriterionType.DETERMINISTIC
    description = "Response under 2 seconds"

    def score(self, ctx: EvalContext) -> float:
        if ctx.response_time_ms <= 2000:
            return 100.0
        return max(0, 100 - (ctx.response_time_ms - 2000) / 20)

registry = CriteriaRegistry()
registry.register(ResponseTimeCriterion())
```

Built-in criteria: `SchemaValidityCriterion`, `LatencyPerformanceCriterion`, `AccuracyCriterion`, `CompletenessCriterion`, `HallucinationCriterion`.

## Multi-Model Judging

Any LLM as judge. Auto-detects provider from model name.

```bash
cane-eval run tests.yaml                                                       # Claude (default)
cane-eval run tests.yaml --provider openai --model gpt-4o                      # OpenAI
cane-eval run tests.yaml --provider gemini --model gemini-2.0-flash            # Gemini
cane-eval run tests.yaml --provider ollama --model llama3 --base-url http://localhost:11434/v1  # Local
```

```bash
pip install cane-eval[openai]          # OpenAI
pip install cane-eval[gemini]          # Google Gemini
pip install cane-eval[all-providers]   # everything
```

## CLI

```bash
cane-eval run tests.yaml                          # run eval
cane-eval run tests.yaml -j 5                     # parallel (5 concurrent)
cane-eval run tests.yaml --reliability-weights 60:20:20  # custom weights
cane-eval run tests.yaml --schema schema.json     # + schema validation
cane-eval run tests.yaml --latency-p95 10000      # + latency threshold
cane-eval run tests.yaml --mine --export dpo      # + failure mining
cane-eval rca tests.yaml --targeted               # root cause analysis
cane-eval diff old.json new.json                  # regression diff
cane-eval demo                                    # try it in 30 seconds
```

## Python API

```python
from cane_eval import ReliabilitySuite, ReliabilityRunner, ReliabilityConfig

suite = ReliabilitySuite.from_yaml("tests.yaml")
runner = ReliabilityRunner(
    schema={"type": "object", "required": ["answer"]},
    latency_p95=10000,
    concurrency=5,
    reliability_config=ReliabilityConfig(
        correctness_weight=0.60,
        structural_weight=0.20,
        performance_weight=0.20,
    ),
)
summary = runner.run(suite, agent=lambda q: my_agent.ask(q))

print(f"Score: {summary.overall_score}")
print(f"Reliability: {summary.reliability_score} ({summary.reliability_grade})")
print(f"Latency p95: {summary.latency.p95_ms}ms")
print(f"Schema: {summary.schema_pass}/{summary.schema_pass + summary.schema_fail} valid")
```

Old names (`EvalRunner`, `TestSuite`, `TestCase`, `EvalResult`, `RunSummary`) still work as aliases.

## Framework Integrations

```python
from cane_eval import evaluate_langchain, evaluate_llamaindex, evaluate_openai, evaluate_fastapi

results = evaluate_langchain(chain, suite="qa.yaml")
results = evaluate_llamaindex(query_engine, suite="qa.yaml")
results = evaluate_openai("http://localhost:11434/v1/chat/completions", suite="qa.yaml")
results = evaluate_fastapi("http://localhost:8000/ask", suite="qa.yaml")
```

## Agent Personality Profiler

Embed model outputs, cluster by behavior, and extract steering vectors — visualize *how* your agent thinks, not just whether it's correct.

```bash
# Profile from a test suite
cane-eval profile tests.yaml --html report.html

# Profile from existing results
cane-eval profile --results eval_results.json --html report.html

# Export steering vectors for post-training
cane-eval profile tests.yaml --export-vectors vectors.json
```

Profiles 6 behavioral traits:

| Trait | What it captures |
|-------|-----------------|
| **Overconfidence** | Confidently wrong — high fluency, low accuracy |
| **Calibration** | Expressed certainty matches actual correctness |
| **Verbosity** | Response length relative to expected |
| **Hedging** | Excessive qualification and uncertainty language |
| **Groundedness** | Answers grounded in retrieved context |
| **Completeness** | Covers all key points |

Generates interactive HTML reports with embedding space scatter plots, personality radar charts, behavioral clusters, contrastive pairs, and steering vectors with magnitudes.

```python
from cane_eval import AgentProfiler

profiler = AgentProfiler(embedding_model="all-MiniLM-L6-v2")
profile = profiler.profile(results, suite_name="my-agent")

# Steering vectors — directions in embedding space
for sv in profile.steering_vectors:
    print(f"{sv.name}: magnitude {sv.magnitude:.3f}")

# Export HTML report
profile.to_html("report.html")
```

```bash
pip install cane-eval[profile]       # sentence-transformers + numpy
pip install cane-eval[profile-umap]  # + UMAP for better projections
```

## Eval Targets

```yaml
# HTTP endpoint
target:
  type: http
  url: https://my-agent.com/api/ask
  payload_template: '{"query": "{{question}}"}'
  response_path: data.answer

# CLI tool
target:
  type: command
  command: python my_agent.py --query "{{question}}"
```

## CI

```yaml
# .github/workflows/eval.yml
- run: pip install cane-eval
- run: cane-eval run tests.yaml -j 5 --schema schema.json --latency-p95 10000 --quiet
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Exit code 1 on failures. Add `--fail-on-warn` or `--fail-on-schema` for stricter checks.

## How It Works

```
YAML Suite --> Agent --> LLM Judge -----> Reliability Score (A-F)
                  |          |                    |
                  |          v                    |
                  |   Schema Check                |
                  |   Latency Stats               |
                  |          |                    v
                  v          v              Training Data
            Root Cause    Failure           (DPO/SFT/OpenAI)
            Analysis      Mining
                             |
                             v
                      Personality Profiler
                    (embeddings, clusters,
                     steering vectors)
```

## License

MIT
