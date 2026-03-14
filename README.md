# cane-eval

LLM-as-Judge evaluation for AI agents. Define test suites in YAML, score responses with Claude, analyze failure root causes, and mine failures into training data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/colingfly/cane-eval/blob/main/examples/quickstart.ipynb)

```
pip install cane-eval
```

## Quick Start

**1. Define a test suite** (`tests.yaml`):

```yaml
name: Support Agent
model: claude-sonnet-4-5-20250929

criteria:
  - key: accuracy
    label: Accuracy
    weight: 40
  - key: completeness
    label: Completeness
    weight: 30
  - key: hallucination
    label: Hallucination Check
    weight: 30

tests:
  - question: What is the return policy?
    expected_answer: 30-day return policy for unused items with receipt
    tags: [policy]

  - question: How do I reset my password?
    expected_answer: Go to Settings > Security > Reset Password
    tags: [account]
```

**2. Run it**:

```bash
cane-eval run tests.yaml
```

```
  cane-eval Support Agent
  2 test cases | model: claude-sonnet-4-5-20250929

   PASS  1/2 [================----] 82  What is the return policy?
   FAIL  2/2 [=========-----------] 45  How do I reset my password?
         Agent fabricated a non-existent Settings page

  ========================================

  Support Agent  4.2s

  Overall: [==============----------------] 63.5

  2 passed  0 warned  1 failed  (2 total)
  Pass rate: 50%
```

**3. Mine failures into training data**:

```bash
cane-eval run tests.yaml --mine --export dpo
```

## Usage

### CLI

```bash
# Run eval suite
cane-eval run tests.yaml

# Filter by tags
cane-eval run tests.yaml --tags policy,account

# Export training data (dpo, sft, openai, raw)
cane-eval run tests.yaml --export dpo --output training.jsonl

# Mine failures and generate improved answers
cane-eval run tests.yaml --mine --mine-threshold 60

# Root cause analysis on failures
cane-eval rca tests.yaml --threshold 60

# RCA from existing results (skip re-running eval)
cane-eval rca tests.yaml --results results.json

# RCA with targeted deep dives on worst failures
cane-eval rca tests.yaml --targeted --targeted-max 5

# Compare two runs (regression diff)
cane-eval diff results_v1.json results_v2.json

# Validate suite YAML
cane-eval validate tests.yaml

# CI mode: exit 1 on any failure
cane-eval run tests.yaml --quiet
```

### Python

```python
from cane_eval import TestSuite, EvalRunner, Exporter, FailureMiner, RootCauseAnalyzer

# Load suite
suite = TestSuite.from_yaml("tests.yaml")

# Run eval with your agent
runner = EvalRunner()
summary = runner.run(suite, agent=lambda q: my_agent.ask(q))

print(f"Score: {summary.overall_score}")
print(f"Pass rate: {summary.pass_rate:.0f}%")

# Root cause analysis on failures
analyzer = RootCauseAnalyzer()
rca = analyzer.analyze(summary, max_score=60)
print(rca.summary)
for rc in rca.root_causes:
    print(f"  [{rc.severity}] {rc.title} -- {rc.recommendation}")

# Export failures as DPO training pairs
exporter = Exporter(summary)
exporter.to_dpo("training_dpo.jsonl", max_score=60)

# Or mine failures with LLM-generated improvements
miner = FailureMiner()
mined = miner.mine(summary, max_score=60)
mined.to_file("mined_dpo.jsonl")
```

### Framework Integrations

One-liner eval for popular frameworks. No boilerplate needed.

**LangChain:**

```python
from cane_eval import evaluate_langchain

chain = prompt | llm | parser  # any LCEL chain
results = evaluate_langchain(chain, suite="qa.yaml")
```

**LlamaIndex:**

```python
from cane_eval import evaluate_llamaindex

query_engine = index.as_query_engine()
results = evaluate_llamaindex(query_engine, suite="qa.yaml")
```

**OpenAI-compatible endpoints** (OpenAI, vLLM, Ollama, LiteLLM):

```python
from cane_eval import evaluate_openai

# Any OpenAI-compatible endpoint
results = evaluate_openai(
    "http://localhost:11434/v1/chat/completions",
    suite="qa.yaml",
    openai_model="llama3",
)

# Or with the openai SDK client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
results = evaluate_openai(client, suite="qa.yaml", openai_model="llama3")
```

**FastAPI agents:**

```python
from cane_eval import evaluate_fastapi

# Running server
results = evaluate_fastapi("http://localhost:8000/ask", suite="qa.yaml")

# Or test in-process (no server needed)
from fastapi import FastAPI
app = FastAPI()
results = evaluate_fastapi(app, suite="qa.yaml", endpoint="/ask")
```

All integrations support mining, RCA, and Cane Cloud push:

```python
results = evaluate_langchain(
    chain,
    suite="qa.yaml",
    mine=True,                             # mine failures into training data
    rca=True,                              # root cause analysis
    cloud="https://app.cane.dev",          # push results to Cane Cloud
    cloud_api_key="sk-...",
    environment_id="env_abc123",
)
```

Install integration dependencies:

```bash
pip install cane-eval[langchain]      # LangChain
pip install cane-eval[llamaindex]     # LlamaIndex
pip install cane-eval[openai]         # OpenAI SDK
pip install cane-eval[fastapi]        # FastAPI TestClient
pip install cane-eval[integrations]   # all of the above
```

### HTTP Agent Target

Point eval at any HTTP endpoint:

```yaml
name: Production Agent Eval

target:
  type: http
  url: https://my-agent.com/api/ask
  method: POST
  payload_template: '{"query": "{{question}}"}'
  response_path: data.answer
  headers:
    Authorization: Bearer ${AGENT_API_KEY}

tests:
  - question: What are your business hours?
    expected_answer: Monday through Friday, 9am to 5pm EST
```

### CLI Agent Target

Eval any command-line tool:

```yaml
name: CLI Agent Eval

target:
  type: command
  command: python my_agent.py --query "{{question}}"

tests:
  - question: Summarize the Q4 report
    expected_answer: Revenue grew 15% year-over-year
```

### Regression Diff

Compare runs to catch regressions:

```bash
# Save results from each run
cane-eval run tests.yaml --output-json results_v1.json
# ... make changes ...
cane-eval run tests.yaml --output-json results_v2.json

# Diff
cane-eval diff results_v1.json results_v2.json
```

```
  Regression Diff
  ------------------------------------------------------------

  2 Regressions
    -25  85 -> 60  How do I cancel my subscription?
    -12  72 -> 60  What payment methods do you accept?

  1 Improvements
    +30  45 -> 75  What is the return policy?
```

### Failure Mining

Automatically classify failures and generate improved training data:

```python
from cane_eval import EvalRunner, TestSuite, FailureMiner

suite = TestSuite.from_yaml("tests.yaml")
runner = EvalRunner()
summary = runner.run(suite, agent=my_agent)

# Mine all failures scoring below 60
miner = FailureMiner()
result = miner.mine(summary, max_score=60)

print(result.failure_distribution)
# {"hallucination": 3, "incomplete": 5, "factual_error": 2}

# Export as DPO training pairs
result.to_file("mined_dpo.jsonl", format="dpo")
```

### Root Cause Analysis

Go beyond "what failed" to understand "why it failed" with AI-powered root cause analysis:

```python
from cane_eval import EvalRunner, TestSuite, RootCauseAnalyzer

suite = TestSuite.from_yaml("tests.yaml")
runner = EvalRunner()
summary = runner.run(suite, agent=my_agent)

# Batch analysis: find patterns across all failures
analyzer = RootCauseAnalyzer()
rca = analyzer.analyze(summary, max_score=60)

print(rca.summary)
# "Agent consistently fails on refund-related questions due to missing policy documentation"

print(rca.top_recommendation)
# "Add refund policy documents to the agent's knowledge base"

for rc in rca.root_causes:
    print(f"[{rc.severity}] {rc.category}: {rc.title}")
    print(f"  {rc.recommendation}")
# [critical] knowledge_gap: Missing refund policy documentation
#   Add refund policy documents to the agent's knowledge base
# [high] prompt_issue: No instruction to cite sources
#   Update system prompt to require source citations

# Deep dive on a single failure
targeted = analyzer.analyze_result(summary.results[0])
print(targeted.diagnosis)
print(targeted.likely_cause)  # "knowledge_gap", "hallucination", etc.
for fix in targeted.fix_actions:
    print(f"  [{fix.priority}] {fix.action} ({fix.effort})")
```

RCA categories: `knowledge_gap`, `prompt_issue`, `source_gap`, `behavior_pattern`, `data_quality`

Severity levels: `critical`, `high`, `medium`, `low`

### Custom Criteria

```yaml
criteria:
  - key: accuracy
    label: Factual Accuracy
    description: Response matches expected answer on key facts
    weight: 40

  - key: tone
    label: Professional Tone
    description: Appropriate, helpful, non-condescending language
    weight: 20

  - key: citations
    label: Source Citations
    description: Claims are backed by referenced documents
    weight: 25

  - key: hallucination
    label: Hallucination Check
    description: No fabricated or unsupported information
    weight: 15

custom_rules:
  - Never recommend competitor products
  - Always include a link to the help center when relevant
  - Responses must be under 200 words
```

### Export Formats

| Format | Use Case | Structure |
|--------|----------|-----------|
| `dpo` | Direct Preference Optimization | `{prompt, chosen, rejected}` |
| `sft` | Supervised Fine-Tuning | `{prompt, completion, metadata}` |
| `openai` | OpenAI fine-tuning API | `{messages: [{role, content}]}` |
| `raw` | Analysis and debugging | Full eval result with all scores |

## CI Integration

```yaml
# .github/workflows/eval.yml
name: Agent Eval
on: [push]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install cane-eval
      - run: cane-eval run tests/eval_suite.yaml --quiet
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## How It Works

```
YAML Suite          Your Agent         Claude Judge        Output
-----------         ----------         ------------        ------
questions    --->   get answers  --->  score 0-100   --->  DPO / SFT / OpenAI
expected             per test          per criteria
criteria                               pass/warn/fail
custom rules                                |
                                            v
                               Root Cause Analysis (optional)
                               find patterns across failures
                               identify knowledge gaps, prompt issues
                               generate actionable recommendations
                                            |
                                            v
                               Failure Mining (optional)
                               classify failure type
                               LLM rewrite bad answers
                               generate improved training pairs
```

## License

MIT
