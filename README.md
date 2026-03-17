# cane-eval

Eval toolkit for AI agents. YAML test suites, multi-model judging, failure mining, root cause analysis, and training data export.

[![PyPI](https://img.shields.io/pypi/v/cane-eval)](https://pypi.org/project/cane-eval/)

```
pip install cane-eval
```

## 30-Second Demo

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cane-eval demo
```

```
  Running a deliberately flawed support agent against 5 test cases...

   FAIL  52  What is your return policy?
         Missing all specific policy details customers need.
   FAIL   0  How do I reset my password?
         Entirely fabricated -- contradicts the expected process.
   WARN  66  Do you offer international shipping?
   FAIL  16  What payment methods do you accept?
         Fabricated a competitor recommendation with false claims.
   PASS 100  How do I contact customer support?

  Overall: 47/100  (28.4s)
  1 passed  1 warned  3 failed
```

Three failures in 28 seconds. That's what your agents are doing without evals.

## Quick Start

**1. Define tests** (`tests.yaml`):

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
  - question: How do I reset my password?
    expected_answer: Go to Settings > Security > Reset Password
```

**2. Run**:

```bash
cane-eval run tests.yaml
```

**3. Mine failures into training data**:

```bash
cane-eval run tests.yaml --mine --export dpo
```

## Multi-Model Judging

Use any LLM as judge. Auto-detects provider from model name.

```bash
# Anthropic (default)
cane-eval run tests.yaml

# OpenAI
cane-eval run tests.yaml --provider openai --model gpt-4o

# Gemini
cane-eval run tests.yaml --provider gemini --model gemini-2.0-flash

# Ollama / vLLM / any OpenAI-compatible endpoint
cane-eval run tests.yaml --provider ollama --model llama3 --base-url http://localhost:11434/v1
```

```python
from cane_eval import Judge

# Auto-detects OpenAI from model name
judge = Judge(model="gpt-4o", api_key="sk-...")

# Gemini
judge = Judge(model="gemini-2.0-flash", api_key="...")

# Local Ollama
judge = Judge(provider="ollama", model="llama3", base_url="http://localhost:11434/v1")
```

Install provider dependencies:

```bash
pip install cane-eval[openai]          # OpenAI
pip install cane-eval[gemini]          # Google Gemini
pip install cane-eval[all-providers]   # everything
```

## CLI

```bash
cane-eval run tests.yaml                          # run eval suite
cane-eval run tests.yaml --tags policy,account    # filter by tags
cane-eval run tests.yaml --export dpo             # export training data
cane-eval run tests.yaml --mine                   # mine failures + rewrite
cane-eval rca tests.yaml --threshold 60           # root cause analysis
cane-eval rca tests.yaml --targeted               # deep dive on worst failures
cane-eval diff results_v1.json results_v2.json    # regression diff
cane-eval validate tests.yaml                     # validate YAML
cane-eval run tests.yaml --quiet                  # CI mode (exit 1 on fail)
```

## Python API

```python
from cane_eval import TestSuite, EvalRunner, FailureMiner, RootCauseAnalyzer

suite = TestSuite.from_yaml("tests.yaml")
runner = EvalRunner()
summary = runner.run(suite, agent=lambda q: my_agent.ask(q))

print(f"Score: {summary.overall_score}")

# Root cause analysis
analyzer = RootCauseAnalyzer()
rca = analyzer.analyze(summary, max_score=60)
for rc in rca.root_causes:
    print(f"  [{rc.severity}] {rc.title} -- {rc.recommendation}")

# Mine failures into DPO training pairs
miner = FailureMiner()
mined = miner.mine(summary, max_score=60)
mined.to_file("training.jsonl", format="dpo")
```

## Framework Integrations

One-liner eval for LangChain, LlamaIndex, OpenAI endpoints, and FastAPI agents.

```python
from cane_eval import evaluate_langchain, evaluate_llamaindex, evaluate_openai, evaluate_fastapi

# LangChain
results = evaluate_langchain(chain, suite="qa.yaml")

# LlamaIndex
results = evaluate_llamaindex(query_engine, suite="qa.yaml")

# OpenAI-compatible (OpenAI, vLLM, Ollama, LiteLLM)
results = evaluate_openai("http://localhost:11434/v1/chat/completions", suite="qa.yaml", openai_model="llama3")

# FastAPI
results = evaluate_fastapi("http://localhost:8000/ask", suite="qa.yaml")
```

```bash
pip install cane-eval[langchain]      # LangChain
pip install cane-eval[llamaindex]     # LlamaIndex
pip install cane-eval[fastapi]        # FastAPI
pip install cane-eval[integrations]   # all of the above
```

## Eval Targets

Point eval at any HTTP endpoint or CLI tool:

```yaml
# HTTP
target:
  type: http
  url: https://my-agent.com/api/ask
  method: POST
  payload_template: '{"query": "{{question}}"}'
  response_path: data.answer
  headers:
    Authorization: Bearer ${AGENT_API_KEY}

# CLI
target:
  type: command
  command: python my_agent.py --query "{{question}}"
```

## Export Formats

| Format | Use Case | Structure |
|--------|----------|-----------|
| `dpo` | Direct Preference Optimization | `{prompt, chosen, rejected}` |
| `sft` | Supervised Fine-Tuning | `{prompt, completion, metadata}` |
| `openai` | OpenAI fine-tuning API | `{messages: [{role, content}]}` |
| `raw` | Debugging | Full eval result with all scores |

## CI

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
YAML Suite --> Your Agent --> LLM Judge --> Training Data (DPO/SFT/OpenAI)
                                |
                                v
                         Root Cause Analysis --> fix recommendations
                                |
                                v
                         Failure Mining --> improved answer rewrites
```

## License

MIT
