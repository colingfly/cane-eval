# cane-eval Roadmap: Agent Reliability Layer

## The Shift
From: "does the answer look good?"
To: "would this break in production?"

correct != safe. An agent can score 95 on accuracy and still break your app
with a 30s p95 latency or a missing JSON field.

---

## Phase 1: Latency Intelligence (easy -- data already exists)

Every test case already records `response_time_ms`. We just need to surface it.

### 1a. Latency stats in CLI output
After the score summary, print:

```
Latency: p50: 1.2s  p95: 8.4s  max: 12.1s
```

Compute p50, p95, p99, max from all `response_time_ms` values in the run.

### 1b. Latency threshold flag
```bash
cane-eval run tests.yaml --latency-p95 10000
```
If p95 exceeds threshold (in ms), the run fails (exit code 1).
Print: `FAIL: p95 latency 12100ms exceeds threshold 10000ms`

### 1c. Per-test latency in verbose output
```
[1/5] What is your return policy?
  Score: 92 (P)  Latency: 1.2s
```

**Files touched:** engine.py (compute stats), cli.py (flag + output)
**Effort:** Small. 1-2 hours.

---

## Phase 2: Schema Validation (new capability)

### 2a. --schema flag
```bash
cane-eval run tests.yaml --schema schema.json
```

Load a JSON Schema file. After getting each agent response, attempt `json.loads()`
then validate against schema using `jsonschema.validate()`.

Result per test case:
- `schema_valid: bool`
- `schema_errors: list[str]` (human-readable validation errors)

### 2b. Schema results in output
```
[1/5] What is your return policy?
  Score: 92 (P)  Latency: 1.2s  Schema: PASS
[2/5] How do I reset my password?
  Score: 88 (P)  Latency: 2.3s  Schema: FAIL
    Expected: { "answer": string, "sources": list }
    Got: { "response": "...", "refs": null }
```

### 2c. --fail-on-schema flag
Fail the run (exit 1) if any test case fails schema validation.
Separate from score -- schema is binary pass/fail, not a 0-100 score.

### 2d. Schema in YAML suite (optional inline)
```yaml
schema:
  type: object
  required: [answer, sources]
  properties:
    answer: { type: string }
    sources: { type: array, items: { type: string } }
```

So users can define schema per suite without a separate file, or use --schema
for a shared schema across suites.

**Files touched:** engine.py (validation logic), cli.py (flags + output),
suite.py (inline schema support), new: schema.py (validation helpers)
**Dependency:** jsonschema (add to pyproject.toml)
**Effort:** Medium. 3-4 hours.

---

## Phase 3: Reliability Criteria (the big positioning move)

Make schema adherence and latency first-class eval criteria with weights.

### 3a. Built-in reliability criteria
```yaml
criteria:
  - key: accuracy
    weight: 40
  - key: schema
    weight: 30
  - key: latency
    weight: 30
```

`schema` and `latency` are special -- they don't go through the LLM judge.
They're computed deterministically:

- **schema**: 100 if valid, 0 if invalid. Binary.
- **latency**: 100 if under threshold, scaled down as it exceeds.
  Formula: `max(0, 100 - ((response_time_ms - target) / target) * 100)`
  With configurable target (default: 5000ms).

These mix into the weighted overall score alongside LLM-judged criteria.
A perfect accuracy score with broken schema still tanks the overall.

### 3b. Latency target in suite
```yaml
latency_target_ms: 5000
```

Or via CLI: `--latency-target 5000`

### 3c. Updated summary output
```
Support Agent                              28.4s

Overall: [=======-----------] 71

  accuracy:     92  (weight: 40)
  schema:       60  (weight: 30)  3/5 valid
  latency:      62  (weight: 30)  p95: 8.4s

1 passed  2 warned  2 failed  (5 total)
Latency: p50: 1.2s  p95: 8.4s  max: 12.1s
```

**Files touched:** judge.py (skip LLM for deterministic criteria),
engine.py (compute deterministic scores, merge with LLM scores),
cli.py (updated output), suite.py (new fields)
**Effort:** Medium-large. 4-6 hours.

---

## Phase 4: Agent Reliability Score (the product story)

### 4a. Composite reliability score
Three pillars:
- **Correctness** (LLM-judged: accuracy, completeness, hallucination)
- **Structural** (deterministic: schema adherence)
- **Performance** (deterministic: latency stability)

Single number: Agent Reliability Score (0-100).
This is the number that goes on a dashboard, in a CI badge, in a Slack alert.

### 4b. Reliability grade
```
Agent Reliability Score: 71 (B)

  A (90-100): Production-ready
  B (75-89):  Mostly reliable, fix edge cases
  C (60-74):  Needs work before production
  D (40-59):  Significant reliability gaps
  F (0-39):   Not production-ready
```

### 4c. CI badge output
```bash
cane-eval run tests.yaml --badge reliability-badge.svg
```

Generate an SVG badge: "Agent Reliability: B | 71"
Teams embed this in their README or PR checks.

**Effort:** Medium. Mostly presentation layer on top of Phase 3 data.

---

## Build Order

1. **Phase 1** (latency stats) -- ship today. Data exists, just surface it.
2. **Phase 2** (schema validation) -- ship this week. New dep, new logic, but scoped.
3. **Phase 3** (reliability criteria) -- ship next. Architectural change to scoring.
4. **Phase 4** (reliability score) -- the product wrapper. Marketing + UX.

Phase 1+2 = "production reliability checks" (the feedback says ship this)
Phase 3+4 = "agent reliability layer" (the Sequoia story)

---

## What we're NOT building (yet)

- Fancy dashboards (CLI first, UI later)
- Perfect schema engine (jsonschema is good enough)
- Latency profiling/tracing (just stats for now)
- Reliability certification API (Phase 4b, later)

Ship ugly. Iterate.
