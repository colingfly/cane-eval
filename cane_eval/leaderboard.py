"""
leaderboard.py -- Rank multiple AI systems by Agent Reliability Score.

Runs one benchmark suite against many competitors (models, agents, or
endpoints), scores each with a single fixed judge, and produces a ranked
leaderboard you can publish as Markdown, JSON, or a self-contained HTML page.

The scoring/presentation layer is decoupled from execution: build a
Leaderboard from any dict of run summaries, however they were produced.

Usage (programmatic):
    from cane_eval import ReliabilitySuite, ReliabilityRunner, Leaderboard
    from cane_eval.leaderboard import run_leaderboard, Competitor

    suite = ReliabilitySuite.from_yaml("benchmark.yaml")
    competitors = [
        Competitor(name="GPT-4o", provider="openai", model="gpt-4o"),
        Competitor(name="Claude Sonnet 4.5", provider="anthropic",
                    model="claude-sonnet-4-5-20250929"),
    ]
    board = run_leaderboard(suite, competitors,
                            judge_provider="anthropic",
                            judge_model="claude-sonnet-4-5-20250929")
    print(board.to_markdown())

Usage (CLI):
    cane-eval leaderboard benchmark.yaml --config competitors.yaml
    cane-eval leaderboard --demo
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from cane_eval.engine import ReliabilitySummary


def score_to_grade(score: float) -> str:
    """Map a 0-100 reliability score to an A-F grade.

    Mirrors the thresholds in reliability.compute_reliability so the
    leaderboard and the engine never disagree on a grade boundary.
    """
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


@dataclass
class Competitor:
    """A single system to score on the leaderboard.

    The competitor answers the benchmark questions; a fixed judge scores
    the answers. `provider`/`model`/`base_url` are passed to
    providers.get_provider, so anything the judge can talk to can compete:
    Anthropic, OpenAI, Gemini, or any OpenAI-compatible endpoint.
    """
    name: str
    provider: str = "anthropic"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    system: Optional[str] = None  # system prompt given to the competitor

    @classmethod
    def from_dict(cls, d: dict) -> "Competitor":
        return cls(
            name=d.get("name") or d.get("model") or d.get("provider", "unknown"),
            provider=d.get("provider", "anthropic"),
            model=d.get("model"),
            base_url=d.get("base_url"),
            api_key=d.get("api_key"),
            system=d.get("system"),
        )


@dataclass
class LeaderboardEntry:
    """One ranked row: a system's reliability results, flattened for display."""
    name: str
    provider: str = ""
    model: str = ""
    reliability_score: float = 0.0
    reliability_grade: str = "F"
    correctness: float = 0.0          # overall LLM-judged score
    pass_rate: float = 0.0            # % of cases that passed
    p95_ms: int = 0                  # p95 latency
    schema_pct: Optional[float] = None  # % schema-valid, if a schema was used
    total: int = 0                   # test cases run
    error: Optional[str] = None      # populated if the run failed
    rank: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def from_summary(
        cls,
        name: str,
        summary: ReliabilitySummary,
        provider: str = "",
        model: str = "",
    ) -> "LeaderboardEntry":
        """Flatten a ReliabilitySummary into a leaderboard row.

        Falls back to the overall judged score when a run has no explicit
        reliability score (e.g. no schema/latency pillars were configured).
        """
        rel = summary.reliability_score
        if rel is None:
            rel = summary.overall_score
        grade = summary.reliability_grade or score_to_grade(rel)

        schema_pct = None
        total_schema = summary.schema_pass + summary.schema_fail
        if total_schema:
            schema_pct = round(summary.schema_pass / total_schema * 100, 1)

        p95 = summary.latency.p95_ms if summary.latency else 0

        return cls(
            name=name,
            provider=provider,
            model=model,
            reliability_score=round(rel, 1),
            reliability_grade=grade,
            correctness=round(summary.overall_score, 1),
            pass_rate=round(summary.pass_rate, 1),
            p95_ms=p95,
            schema_pct=schema_pct,
            total=summary.total,
        )

    @classmethod
    def errored(cls, name: str, provider: str, model: str, error: str) -> "LeaderboardEntry":
        """Build a placeholder row for a competitor whose run failed."""
        return cls(
            name=name, provider=provider, model=model,
            reliability_score=0.0, reliability_grade="F", error=error,
        )

    def to_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "reliability_score": self.reliability_score,
            "reliability_grade": self.reliability_grade,
            "correctness": self.correctness,
            "pass_rate": self.pass_rate,
            "p95_ms": self.p95_ms,
            "total": self.total,
        }
        if self.schema_pct is not None:
            d["schema_pct"] = self.schema_pct
        if self.error is not None:
            d["error"] = self.error
        return d


def _fmt_ms(ms: int) -> str:
    if not ms:
        return "-"
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms}ms"


@dataclass
class Leaderboard:
    """A ranked set of systems scored on one benchmark suite."""
    suite_name: str
    entries: list[LeaderboardEntry] = field(default_factory=list)
    judge_model: str = ""
    judge_provider: str = ""
    total_cases: int = 0
    generated_at: Optional[str] = None

    @classmethod
    def from_summaries(
        cls,
        suite_name: str,
        summaries: dict,
        judge_model: str = "",
        judge_provider: str = "",
        generated_at: Optional[str] = None,
    ) -> "Leaderboard":
        """Build a ranked leaderboard from a dict of run summaries.

        `summaries` maps a display name to either a ReliabilitySummary or a
        (ReliabilitySummary, meta) tuple, where meta may carry
        {"provider": ..., "model": ...} for the display columns.

        This is the execution-agnostic entry point: real API runs, cached
        results, or fixtures all land here the same way.
        """
        entries: list[LeaderboardEntry] = []
        total_cases = 0
        for name, value in summaries.items():
            meta = {}
            summary = value
            if isinstance(value, tuple):
                summary, meta = value[0], (value[1] or {})
            entry = LeaderboardEntry.from_summary(
                name,
                summary,
                provider=meta.get("provider", ""),
                model=meta.get("model", ""),
            )
            total_cases = max(total_cases, summary.total)
            entries.append(entry)

        board = cls(
            suite_name=suite_name,
            entries=entries,
            judge_model=judge_model,
            judge_provider=judge_provider,
            total_cases=total_cases,
            generated_at=generated_at,
        )
        board.rank()
        return board

    def rank(self) -> "Leaderboard":
        """Sort entries best-first and assign 1-based ranks.

        Order: reliability desc, then pass-rate desc, then lower p95 latency.
        Errored competitors always sort to the bottom.
        """
        def key(e: LeaderboardEntry):
            return (
                0 if e.ok else 1,
                -e.reliability_score,
                -e.pass_rate,
                e.p95_ms if e.p95_ms else 10**9,
            )

        self.entries.sort(key=key)
        for i, e in enumerate(self.entries, start=1):
            e.rank = i
        return self

    @property
    def winner(self) -> Optional[LeaderboardEntry]:
        for e in self.entries:
            if e.ok:
                return e
        return None

    def _has_schema(self) -> bool:
        return any(e.schema_pct is not None for e in self.entries)

    # ---- Renderers ----

    def to_markdown(self) -> str:
        """Render a Markdown leaderboard, ready to paste into a README."""
        show_schema = self._has_schema()
        lines = []
        lines.append("# Agent Reliability Leaderboard")
        lines.append("")

        meta_bits = [f"**{self.total_cases}** test cases"]
        if self.judge_model:
            meta_bits.append(f"judged by `{self.judge_model}`")
        if self.generated_at:
            meta_bits.append(f"generated {self.generated_at}")
        lines.append(f"Suite: **{self.suite_name}** &middot; " + " &middot; ".join(meta_bits))
        lines.append("")

        header = ["Rank", "System", "Reliability", "Grade", "Correctness", "Pass rate", "p95 latency"]
        align = ["---:", ":---", "---:", ":---:", "---:", "---:", "---:"]
        if show_schema:
            header.append("Schema")
            align.append("---:")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(align) + " |")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for e in self.entries:
            if not e.ok:
                row = [str(e.rank), _md_escape(e.name), "—", "—", "—", "—", "error"]
                if show_schema:
                    row.append("—")
                lines.append("| " + " | ".join(row) + " |")
                continue
            rank_cell = f"{medals.get(e.rank, '')} {e.rank}".strip()
            row = [
                rank_cell,
                _md_escape(e.name),
                f"**{e.reliability_score:.0f}**",
                e.reliability_grade,
                f"{e.correctness:.0f}",
                f"{e.pass_rate:.0f}%",
                _fmt_ms(e.p95_ms),
            ]
            if show_schema:
                row.append(f"{e.schema_pct:.0f}%" if e.schema_pct is not None else "-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append(
            "<sub>Reliability = weighted blend of correctness (LLM-judged), "
            "structural validity (schema adherence), and performance (p95 "
            "latency). Grades: A 90+ &middot; B 75+ &middot; C 60+ &middot; "
            "D 40+ &middot; F &lt;40. Generated with "
            "[cane-eval](https://github.com/colingfly/cane-eval).</sub>")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider,
            "total_cases": self.total_cases,
            "generated_at": self.generated_at,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_html(self) -> str:
        """Render a self-contained, theme-aware HTML leaderboard page."""
        show_schema = self._has_schema()
        grade_colors = {
            "A": "#16a34a", "B": "#65a30d", "C": "#ca8a04",
            "D": "#ea580c", "F": "#dc2626",
        }

        rows = []
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for e in self.entries:
            if not e.ok:
                rows.append(
                    f'<tr class="errored"><td>{e.rank}</td>'
                    f'<td>{_html_escape(e.name)}</td>'
                    f'<td colspan="{6 if show_schema else 5}">'
                    f'run failed: {_html_escape(e.error or "")}</td></tr>')
                continue
            gcolor = grade_colors.get(e.reliability_grade, "#6b7280")
            schema_cell = (
                f"<td>{e.schema_pct:.0f}%</td>" if show_schema and e.schema_pct is not None
                else ("<td>-</td>" if show_schema else ""))
            rows.append(
                f"<tr>"
                f'<td class="rank">{medals.get(e.rank, "")} {e.rank}</td>'
                f'<td class="name">{_html_escape(e.name)}'
                f'{f"<span class=sub>{_html_escape(e.model)}</span>" if e.model else ""}</td>'
                f'<td class="score">{e.reliability_score:.0f}</td>'
                f'<td><span class="grade" style="background:{gcolor}">{e.reliability_grade}</span></td>'
                f"<td>{e.correctness:.0f}</td>"
                f"<td>{e.pass_rate:.0f}%</td>"
                f"<td>{_fmt_ms(e.p95_ms)}</td>"
                f"{schema_cell}"
                f"</tr>")

        schema_th = "<th>Schema</th>" if show_schema else ""
        meta_bits = [f"{self.total_cases} test cases"]
        if self.judge_model:
            meta_bits.append(f"judged by {self.judge_model}")
        if self.generated_at:
            meta_bits.append(f"generated {self.generated_at}")
        meta = " &middot; ".join(_html_escape(b) for b in meta_bits)

        return _HTML_TEMPLATE.format(
            suite=_html_escape(self.suite_name),
            meta=meta,
            schema_th=schema_th,
            rows="\n".join(rows),
        )


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def _html_escape(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;"))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Reliability Leaderboard &middot; {suite}</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #0f172a; --muted: #64748b;
    --line: #e2e8f0; --row: #f8fafc; --accent: #4f46e5;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #0b1120; --fg: #e2e8f0; --muted: #94a3b8;
      --line: #1e293b; --row: #111a2e; --accent: #818cf8;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2.5rem 1rem; background: var(--bg); color: var(--fg);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -0.02em; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }}
  .scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }}
  table {{ width: 100%; border-collapse: collapse; min-width: 620px; }}
  th, td {{ padding: .7rem .9rem; text-align: right; white-space: nowrap; }}
  th {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); border-bottom: 1px solid var(--line); font-weight: 600;
  }}
  th:nth-child(2), td.name {{ text-align: left; }}
  tbody tr:nth-child(even) {{ background: var(--row); }}
  td.rank {{ color: var(--muted); }}
  td.name {{ font-weight: 600; }}
  td.name .sub {{ display: block; font-weight: 400; font-size: .78rem; color: var(--muted); }}
  td.score {{ font-weight: 700; font-size: 1.05rem; }}
  .grade {{
    display: inline-block; min-width: 1.6rem; padding: .1rem .4rem; border-radius: 6px;
    color: #fff; font-weight: 700; font-size: .8rem; text-align: center;
  }}
  tr.errored td {{ color: var(--muted); font-style: italic; }}
  footer {{ color: var(--muted); font-size: .78rem; margin-top: 1.25rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Agent Reliability Leaderboard</h1>
    <div class="meta">{suite} &middot; {meta}</div>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>System</th><th>Reliability</th><th>Grade</th>
            <th>Correctness</th><th>Pass rate</th><th>p95</th>{schema_th}
          </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </div>
    <footer>
      Reliability blends correctness (LLM-judged), structural validity, and
      performance. Generated with
      <a href="https://github.com/colingfly/cane-eval">cane-eval</a>.
    </footer>
  </div>
</body>
</html>
"""


# ---- Execution ----

def model_agent(competitor: Competitor) -> Callable[[str], str]:
    """Wrap a competitor as an agent callable the runner can drive.

    The returned function takes a question and returns the model's answer
    string, using the competitor's provider/model/system prompt.
    """
    from cane_eval.providers import get_provider

    provider = get_provider(
        provider=competitor.provider,
        model=competitor.model,
        api_key=competitor.api_key,
        base_url=competitor.base_url,
    )
    system = competitor.system or (
        "You are the AI system under evaluation. Answer the user's question "
        "as accurately, completely, and concisely as you can.")

    def _agent(question: str) -> str:
        return provider.call(prompt=question, system=system)

    return _agent


def run_leaderboard(
    suite,
    competitors: list[Competitor],
    judge_provider: str = "anthropic",
    judge_model: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    schema: Optional[dict] = None,
    concurrency: int = 1,
    generated_at: Optional[str] = None,
    on_competitor_start: Optional[Callable[[Competitor, int, int], None]] = None,
    verbose: bool = False,
) -> Leaderboard:
    """Run a benchmark suite against every competitor and rank the results.

    A single fixed judge (judge_provider/judge_model) scores all competitors
    so the comparison is apples-to-apples. A competitor whose run raises is
    recorded as an errored entry rather than aborting the whole leaderboard.
    """
    from cane_eval.engine import ReliabilityRunner

    summaries: dict = {}
    total = len(competitors)
    for i, comp in enumerate(competitors):
        if on_competitor_start:
            on_competitor_start(comp, i + 1, total)

        model_label = comp.model or ""
        try:
            agent = model_agent(comp)
            runner = ReliabilityRunner(
                api_key=judge_api_key,
                model=judge_model,
                verbose=verbose,
                provider=judge_provider,
                base_url=judge_base_url,
                schema=schema,
                concurrency=concurrency,
            )
            summary = runner.run(suite, agent=agent)
            summaries[comp.name] = (summary, {"provider": comp.provider, "model": model_label})
        except Exception as e:  # noqa: BLE001 - one bad competitor shouldn't kill the board
            summaries[comp.name] = (
                _errored_summary(suite, str(e)),
                {"provider": comp.provider, "model": model_label, "error": str(e)},
            )

    board = Leaderboard.from_summaries(
        suite_name=getattr(suite, "name", "benchmark"),
        summaries={},
        judge_model=judge_model or "",
        judge_provider=judge_provider,
        generated_at=generated_at,
    )
    # Rebuild entries with error awareness (from_summaries flattens summaries;
    # errored competitors carry an "error" key in their meta).
    entries = []
    total_cases = 0
    for name, (summary, meta) in summaries.items():
        if meta.get("error"):
            entries.append(LeaderboardEntry.errored(
                name, meta.get("provider", ""), meta.get("model", ""), meta["error"]))
        else:
            entries.append(LeaderboardEntry.from_summary(
                name, summary, provider=meta.get("provider", ""), model=meta.get("model", "")))
            total_cases = max(total_cases, summary.total)
    board.entries = entries
    board.total_cases = total_cases
    board.rank()
    return board


def _errored_summary(suite, error: str) -> ReliabilitySummary:
    """A minimal summary standing in for a competitor that failed to run."""
    return ReliabilitySummary(suite_name=getattr(suite, "name", "benchmark"))


# ---- Demo (offline, no API key) ----

def _synthetic_summary(
    total: int,
    overall: float,
    pass_rate: float,
    p95_ms: int,
    schema_pct: Optional[float],
    latency_target: int = 5000,
) -> ReliabilitySummary:
    """Build a representative summary from target numbers (no LLM calls).

    Used only by demo_leaderboard() to show the leaderboard format with
    deterministic, illustrative data.
    """
    from cane_eval.engine import LatencyStats
    from cane_eval.reliability import compute_reliability

    passed = round(total * pass_rate / 100)
    failed = total - passed
    latency = LatencyStats(
        p50_ms=int(p95_ms * 0.4), p95_ms=p95_ms, p99_ms=int(p95_ms * 1.1),
        max_ms=int(p95_ms * 1.2), min_ms=int(p95_ms * 0.2), mean_ms=int(p95_ms * 0.5),
    )

    schema_pass = schema_fail = 0
    schema_score = None
    if schema_pct is not None:
        schema_pass = round(total * schema_pct / 100)
        schema_fail = total - schema_pass
        schema_score = schema_pct

    latency_score = 100.0 if p95_ms <= latency_target else max(
        0.0, 100.0 - ((p95_ms - latency_target) / latency_target) * 100)

    rel, grade = compute_reliability(
        accuracy_score=overall, schema_score=schema_score, latency_score=latency_score)

    return ReliabilitySummary(
        suite_name="Support Agent Benchmark v1",
        total=total, passed=passed, warned=0, failed=failed,
        overall_score=overall, latency=latency,
        schema_pass=schema_pass, schema_fail=schema_fail,
        reliability_score=rel, reliability_grade=grade,
    )


def demo_leaderboard(generated_at: Optional[str] = None) -> Leaderboard:
    """A ready-made leaderboard with illustrative (not real) systems.

    Runs no models and needs no API key -- it exists to show the output
    format and to back the sample LEADERBOARD.md. The names are deliberately
    generic so the numbers are never mistaken for a real benchmark claim.
    """
    demo = {
        "Reference Agent A": (
            _synthetic_summary(12, 94.0, 92.0, 1300, 100.0),
            {"provider": "anthropic", "model": "reference-a"}),
        "Reference Agent B": (
            _synthetic_summary(12, 88.0, 83.0, 2600, 92.0),
            {"provider": "openai", "model": "reference-b"}),
        "Reference Agent C": (
            _synthetic_summary(12, 79.0, 75.0, 4200, 83.0),
            {"provider": "gemini", "model": "reference-c"}),
        "Reference Agent D (local)": (
            _synthetic_summary(12, 66.0, 58.0, 9100, 67.0),
            {"provider": "openai-compatible", "model": "reference-d"}),
    }
    return Leaderboard.from_summaries(
        suite_name="Support Agent Benchmark v1 (illustrative sample)",
        summaries=demo,
        judge_model="claude-sonnet-4-5-20250929",
        judge_provider="anthropic",
        generated_at=generated_at,
    )
