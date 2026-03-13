# cane-eval

LLM-as-Judge evaluation for AI agents. Define test suites in YAML, score responses with Claude, mine failures into training data.

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

# Compare two runs (regression diff)
cane-eval diff results_v1.json results_v2.json

# Validate suite YAML
cane-eval validate tests.yaml

# CI mode: exit 1 on any failure
cane-eval run tests.yaml --quiet
```

### Python

```python
from cane_eval import TestSuite, EvalRunner, Exporter, FailureMiner

# Load suite
suite = TestSuite.from_yaml("tests.yaml")

# Run eval with your agent
runner = EvalRunner()
summary = runner.run(suite, agent=lambda q: my_agent.ask(q))

print(f"Score: {summary.overall_score}")
print(f"Pass rate: {summary.pass_rate:.0f}%")

# Export failures as DPO training pairs
exporter = Exporter(summary)
exporter.to_dpo("training_dpo.jsonl", max_score=60)

# Or mine failures with LLM-generated improvements
miner = FailureMiner()
mined = miner.mine(summary, max_score=60)
mined.to_file("mined_dpo.jsonl")
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
YAML Suite          Your Agent         Claude Judge        Training Data
-----------         ----------         ------------        -------------
questions    --->   get answers  --->  score 0-100   --->  DPO pairs
expected             per test          per criteria        SFT examples
criteria                               pass/warn/fail      OpenAI format
custom rules                                               |
                                                           v
                                              Failure Mining (optional)
                                              classify failure type
                                              LLM rewrite bad answers
                                              generate improved pairs
```

## License

MIT
