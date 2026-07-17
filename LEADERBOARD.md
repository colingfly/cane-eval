# Agent Reliability Leaderboard

Suite: **Support Agent Benchmark v1 (illustrative sample)** &middot; **12** test cases &middot; judged by `claude-sonnet-4-5-20250929` &middot; generated 2026-07-17

| Rank | System | Reliability | Grade | Correctness | Pass rate | p95 latency | Schema |
| ---: | :--- | ---: | :---: | ---: | ---: | ---: | ---: |
| 🥇 1 | Reference Agent A | **97** | A | 94 | 92% | 1.3s | 100% |
| 🥈 2 | Reference Agent B | **92** | A | 88 | 83% | 2.6s | 92% |
| 🥉 3 | Reference Agent C | **85** | B | 79 | 75% | 4.2s | 83% |
| 4 | Reference Agent D (local) | **54** | D | 66 | 58% | 9.1s | 67% |

<sub>Reliability = weighted blend of correctness (LLM-judged), structural validity (schema adherence), and performance (p95 latency). Grades: A 90+ &middot; B 75+ &middot; C 60+ &middot; D 40+ &middot; F &lt;40. Generated with [cane-eval](https://github.com/colingfly/cane-eval).</sub>
