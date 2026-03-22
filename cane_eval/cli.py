"""
cli.py -- Command-line interface for cane-eval.

Usage:
    cane-eval run tests.yaml
    cane-eval run tests.yaml --model claude-sonnet-4-5-20250929
    cane-eval run tests.yaml --tags policy,returns
    cane-eval run tests.yaml --export dpo --output training.jsonl
    cane-eval run tests.yaml --mine --mine-threshold 60
    cane-eval run tests.yaml -j 5 --reliability-weights 60:20:20
    cane-eval diff results_v1.json results_v2.json
"""

import argparse
import json
import sys
import os
from pathlib import Path


# ---- Color helpers (no dependency needed) ----

COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

def _supports_color():
    """Check if terminal supports color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

def c(text, color):
    if not USE_COLOR:
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# ---- Output formatters ----

def _status_badge(status):
    """Return colored status badge."""
    if status == "pass":
        return c(" PASS ", "green")
    if status == "warn":
        return c(" WARN ", "yellow")
    return c(" FAIL ", "red")


def _score_color(score):
    """Return score with appropriate color."""
    if score >= 80:
        return c(f"{score:.0f}", "green")
    if score >= 60:
        return c(f"{score:.0f}", "yellow")
    return c(f"{score:.0f}", "red")


def _bar(score, width=20):
    """Return a simple progress bar."""
    filled = int(score / 100 * width)
    empty = width - filled
    if score >= 80:
        color = "green"
    elif score >= 60:
        color = "yellow"
    else:
        color = "red"
    bar_str = c("=" * filled, color) + c("-" * empty, "dim")
    return f"[{bar_str}]"


def _format_ms(ms):
    """Format milliseconds to human-readable."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms}ms"


def _latency_color(ms, threshold=5000):
    """Color latency based on threshold."""
    if ms <= threshold:
        return c(_format_ms(ms), "green")
    if ms <= threshold * 1.5:
        return c(_format_ms(ms), "yellow")
    return c(_format_ms(ms), "red")


def print_result(result, index, total):
    """Print a single eval result."""
    q = result.question[:70] + "..." if len(result.question) > 70 else result.question
    badge = _status_badge(result.status)
    score = _score_color(result.score)
    bar = _bar(result.score)

    # Build suffix with latency + schema
    suffix = ""
    if result.response_time_ms > 0:
        suffix += f"  {c(_format_ms(result.response_time_ms), 'dim')}"
    if result.schema_result is not None:
        if result.schema_result.valid:
            suffix += f"  {c('schema:OK', 'green')}"
        else:
            suffix += f"  {c('schema:FAIL', 'red')}"

    print(f"  {badge} {c(f'{index}/{total}', 'dim')} {bar} {score}  {q}{suffix}")

    if result.status == "fail":
        reasoning = result.judge_result.overall_reasoning
        if reasoning:
            print(f"         {c(reasoning[:120], 'dim')}")

    if result.schema_result and not result.schema_result.valid:
        for err in result.schema_result.errors[:2]:
            print(f"         {c(f'schema: {err[:100]}', 'red')}")


def _grade_color(grade):
    """Color a reliability grade."""
    colors = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "red"}
    return c(grade, colors.get(grade, "dim"))


def print_summary(summary):
    """Print run summary."""
    print()
    print(c("  =" * 40, "dim"))
    print()
    print(f"  {c(summary.suite_name, 'bold')}  {c(f'{summary.duration_seconds:.1f}s', 'dim')}")
    print()

    # Score bar
    bar = _bar(summary.overall_score, 30)
    score = _score_color(summary.overall_score)
    print(f"  Overall: {bar} {score}")
    print()

    # Pass/Warn/Fail counts
    p = c(f"{summary.passed} passed", "green")
    w = c(f"{summary.warned} warned", "yellow")
    f_count = c(f"{summary.failed} failed", "red")
    print(f"  {p}  {w}  {f_count}  ({summary.total} total)")
    print(f"  Pass rate: {c(f'{summary.pass_rate:.0f}%', 'bold')}")

    # Latency stats
    if summary.latency and summary.latency.p50_ms > 0:
        print()
        lat = summary.latency
        print(f"  Latency:  p50: {_latency_color(lat.p50_ms)}  p95: {_latency_color(lat.p95_ms)}  max: {_latency_color(lat.max_ms)}")

    # Schema stats
    if summary.schema_pass or summary.schema_fail:
        total_schema = summary.schema_pass + summary.schema_fail
        pct = summary.schema_pass / total_schema * 100 if total_schema else 0
        schema_label = c(f"{summary.schema_pass}/{total_schema} valid", "green" if pct == 100 else "yellow" if pct >= 50 else "red")
        print(f"  Schema:   {schema_label} ({pct:.0f}%)")

    # Reliability score
    if summary.reliability_score is not None:
        grade = _grade_color(summary.reliability_grade)
        rel_bar = _bar(summary.reliability_score, 20)
        rel_score = _score_color(summary.reliability_score)
        print()
        print(f"  {c('Reliability:', 'bold')} {rel_bar} {rel_score} ({grade})")

    print()


def print_diff(old_results, new_results):
    """Print regression diff between two result sets."""
    old_by_q = {r["question"]: r for r in old_results}
    new_by_q = {r["question"]: r for r in new_results}

    all_questions = list(dict.fromkeys(list(old_by_q.keys()) + list(new_by_q.keys())))

    regressions = []
    improvements = []
    new_cases = []
    removed_cases = []

    for q in all_questions:
        old = old_by_q.get(q)
        new = new_by_q.get(q)

        if old and not new:
            removed_cases.append(q)
        elif new and not old:
            new_cases.append((q, new))
        else:
            old_score = old["overall_score"]
            new_score = new["overall_score"]
            delta = new_score - old_score

            if delta < -5:
                regressions.append((q, old_score, new_score, delta))
            elif delta > 5:
                improvements.append((q, old_score, new_score, delta))

    print()
    print(c("  Regression Diff", "bold"))
    print(c("  " + "-" * 60, "dim"))

    if regressions:
        print()
        print(f"  {c(f'{len(regressions)} Regressions', 'red')}")
        for q, old_s, new_s, delta in sorted(regressions, key=lambda x: x[3]):
            q_short = q[:55] + "..." if len(q) > 55 else q
            print(f"    {c(f'{delta:+.0f}', 'red')}  {old_s:.0f} -> {new_s:.0f}  {q_short}")

    if improvements:
        print()
        print(f"  {c(f'{len(improvements)} Improvements', 'green')}")
        for q, old_s, new_s, delta in sorted(improvements, key=lambda x: -x[3]):
            q_short = q[:55] + "..." if len(q) > 55 else q
            print(f"    {c(f'{delta:+.0f}', 'green')}  {old_s:.0f} -> {new_s:.0f}  {q_short}")

    if new_cases:
        print()
        print(f"  {c(f'{len(new_cases)} New', 'cyan')}")
        for q, r in new_cases:
            q_short = q[:55] + "..." if len(q) > 55 else q
            print(f"    {_score_color(r['overall_score'])}  {q_short}")

    if removed_cases:
        print()
        print(f"  {c(f'{len(removed_cases)} Removed', 'dim')}")
        for q in removed_cases:
            q_short = q[:55] + "..." if len(q) > 55 else q
            print(f"    {c('--', 'dim')}  {q_short}")

    if not regressions and not improvements and not new_cases and not removed_cases:
        print(f"\n  {c('No significant changes detected.', 'dim')}")

    print()


def print_mining_result(mining_result):
    """Print failure mining summary."""
    print()
    print(c("  Failure Mining", "bold"))
    print(c("  " + "-" * 40, "dim"))
    print(f"  Mined {c(str(mining_result.total_mined), 'bold')} examples from {mining_result.total_failures} failures")
    print()

    if mining_result.failure_distribution:
        print("  Failure types:")
        for ftype, count in sorted(mining_result.failure_distribution.items(), key=lambda x: -x[1]):
            pct = count / mining_result.total_mined * 100 if mining_result.total_mined else 0
            print(f"    {c(ftype, 'cyan'):30s} {count:3d}  ({pct:.0f}%)")
    print()


def _severity_badge(severity):
    """Return colored severity badge."""
    colors = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}
    color = colors.get(severity, "dim")
    return c(f" {severity.upper()} ", color)


def _category_label(category):
    """Return colored category label."""
    return c(category.replace("_", " "), "cyan")


def print_rca_result(rca_result):
    """Print batch root cause analysis result."""
    print()
    print(c("  Root Cause Analysis", "bold"))
    print(c("  " + "-" * 50, "dim"))
    print()
    print(f"  Analyzed {c(str(rca_result.total_analyzed), 'bold')} failures")
    if rca_result.avg_failure_score:
        print(f"  Avg failure score: {_score_color(rca_result.avg_failure_score)}")
    if rca_result.score_range and rca_result.score_range != [0, 0]:
        print(f"  Score range: {rca_result.score_range[0]:.0f} - {rca_result.score_range[1]:.0f}")
    print()

    if rca_result.summary:
        print(f"  {c('Summary:', 'bold')} {rca_result.summary}")
        print()

    if rca_result.top_recommendation:
        print(f"  {c('Top recommendation:', 'green')} {rca_result.top_recommendation}")
        print()

    if rca_result.root_causes:
        print(f"  {c(f'{len(rca_result.root_causes)} Root Causes Found', 'bold')}")
        print()
        for i, rc in enumerate(rca_result.root_causes):
            badge = _severity_badge(rc.severity)
            cat = _category_label(rc.category)
            print(f"  {i+1}. {badge} {cat}  {c(rc.title, 'bold')}")
            if rc.description:
                print(f"     {rc.description}")
            if rc.evidence:
                for ev in rc.evidence[:3]:
                    print(f"     {c('>', 'dim')} {ev}")
            if rc.recommendation:
                print(f"     {c('Fix:', 'green')} {rc.recommendation}")
            print()
    print()


def print_targeted_rca_result(targeted):
    """Print targeted (single result) RCA."""
    print()
    print(c("  Targeted Root Cause Analysis", "bold"))
    print(c("  " + "-" * 50, "dim"))
    print()

    q = targeted.question[:80] + "..." if len(targeted.question) > 80 else targeted.question
    print(f"  Question: {q}")
    print(f"  Score: {_score_color(targeted.score)}")
    print(f"  Likely cause: {c(targeted.likely_cause.replace('_', ' '), 'cyan')}")
    print(f"  Confidence: {c(f'{targeted.confidence}%', 'bold')}")
    print()

    if targeted.diagnosis:
        print(f"  {c('Diagnosis:', 'bold')}")
        print(f"    {targeted.diagnosis}")
        print()

    if targeted.contributing_factors:
        print(f"  {c('Contributing factors:', 'bold')}")
        for factor in targeted.contributing_factors:
            print(f"    {c('>', 'dim')} {factor}")
        print()

    if targeted.fix_actions:
        print(f"  {c('Fix actions:', 'bold')}")
        for fa in targeted.fix_actions:
            priority_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(fa.priority, "dim")
            print(f"    [{c(fa.priority.upper(), priority_color)}] {fa.action}  {c(f'({fa.effort})', 'dim')}")
        print()
    print()


# ---- Commands ----

def _parse_reliability_weights(weights_str: str):
    """Parse reliability weights from 'C:S:P' format (e.g. '60:20:20')."""
    from cane_eval.reliability import ReliabilityConfig

    parts = weights_str.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid weights format '{weights_str}'. Expected C:S:P (e.g. 60:20:20)"
        )
    try:
        c_w, s_w, p_w = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid weights format '{weights_str}'. Values must be integers."
        )
    total = c_w + s_w + p_w
    return ReliabilityConfig(
        correctness_weight=c_w / total,
        structural_weight=s_w / total,
        performance_weight=p_w / total,
    )


def cmd_run(args):
    """Run eval suite."""
    from cane_eval.suite import ReliabilitySuite
    from cane_eval.engine import ReliabilityRunner
    from cane_eval.export import Exporter

    # Load suite
    try:
        suite = ReliabilitySuite.from_yaml(args.suite)
    except FileNotFoundError:
        print(c(f"  Error: Suite file not found: {args.suite}", "red"))
        sys.exit(1)
    except Exception as e:
        print(c(f"  Error loading suite: {e}", "red"))
        sys.exit(1)

    # Override model if specified
    if args.model:
        suite.model = args.model

    print()
    print(f"  {c('cane-eval', 'cyan')} {c(suite.name, 'bold')}")
    print(f"  {len(suite.tests)} test cases | model: {suite.model}")
    print()

    # Parse tags
    tags = args.tags.split(",") if args.tags else None

    # Determine provider
    provider = getattr(args, "provider", "anthropic") or "anthropic"
    base_url = getattr(args, "base_url", None)

    # Pick the right API key env var for the provider
    from cane_eval.providers import PROVIDERS, PROVIDER_ALIASES
    resolved = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
    provider_cls = PROVIDERS.get(resolved)
    env_key = provider_cls.env_key() if provider_cls else "ANTHROPIC_API_KEY"

    # Load schema if specified
    schema = None
    if args.schema:
        try:
            with open(args.schema, "r") as f:
                schema = json.load(f)
        except FileNotFoundError:
            print(c(f"  Error: Schema file not found: {args.schema}", "red"))
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(c(f"  Error: Invalid JSON in schema file: {e}", "red"))
            sys.exit(1)

    # Also check for inline schema in suite
    if schema is None and hasattr(suite, 'schema') and suite.schema:
        schema = suite.schema

    # Parse reliability weights
    reliability_config = None
    if args.reliability_weights:
        reliability_config = _parse_reliability_weights(args.reliability_weights)
    elif suite.reliability_config:
        reliability_config = suite.get_reliability_config()

    # Determine concurrency
    concurrency = args.concurrency if args.concurrency else suite.concurrency

    # Run
    runner = ReliabilityRunner(
        api_key=args.api_key or os.environ.get(env_key),
        model=args.model,
        verbose=not args.quiet,
        on_result=print_result if not args.quiet else None,
        provider=provider,
        base_url=base_url,
        schema=schema,
        latency_p95=args.latency_p95,
        latency_target=args.latency_target,
        concurrency=concurrency,
        reliability_config=reliability_config,
    )

    summary = runner.run(suite, tags=tags)

    # Print summary
    if not args.quiet:
        print_summary(summary)

    # Save results JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"  Results saved to {args.output_json}")

    # Export training data
    if args.export:
        exporter = Exporter(summary)
        output_path = args.output or f"eval_{args.export}.jsonl"
        exporter.to_file(output_path, format=args.export)
        print(f"  Exported {args.export} data to {output_path}")

    # Mine failures
    if args.mine:
        from cane_eval.mining import FailureMiner

        miner = FailureMiner(
            api_key=args.api_key or os.environ.get(env_key),
            model=args.model or suite.model,
            verbose=not args.quiet,
            provider=provider,
            base_url=base_url,
        )

        mining_result = miner.mine(
            summary,
            max_score=args.mine_threshold,
            max_examples=args.mine_max,
        )

        if not args.quiet:
            print_mining_result(mining_result)

        if mining_result.total_mined > 0:
            mine_output = args.mine_output or "mined_dpo.jsonl"
            mining_result.to_file(mine_output, format=args.mine_format)
            print(f"  Mined data saved to {mine_output}")

    # Check latency threshold
    if args.latency_p95 and summary.latency and summary.latency.p95_ms > args.latency_p95:
        print(c(f"  FAIL: p95 latency {summary.latency.p95_ms}ms exceeds threshold {args.latency_p95}ms", "red"))
        print()
        sys.exit(1)

    # Check schema failures
    if args.fail_on_schema and summary.schema_fail > 0:
        print(c(f"  FAIL: {summary.schema_fail} response(s) failed schema validation", "red"))
        print()
        sys.exit(1)

    # Exit code based on failures
    if args.fail_on_warn:
        sys.exit(1 if (summary.failed > 0 or summary.warned > 0) else 0)
    else:
        sys.exit(1 if summary.failed > 0 else 0)


def cmd_diff(args):
    """Compare two eval result files."""
    try:
        with open(args.old, "r") as f:
            old_data = json.load(f)
        with open(args.new, "r") as f:
            new_data = json.load(f)
    except FileNotFoundError as e:
        print(c(f"  Error: {e}", "red"))
        sys.exit(1)

    old_results = old_data.get("results", old_data if isinstance(old_data, list) else [])
    new_results = new_data.get("results", new_data if isinstance(new_data, list) else [])

    print_diff(old_results, new_results)


def cmd_rca(args):
    """Run root cause analysis on eval failures."""
    from cane_eval.suite import ReliabilitySuite
    from cane_eval.engine import ReliabilityRunner
    from cane_eval.rca import RootCauseAnalyzer

    # Load suite
    try:
        suite = ReliabilitySuite.from_yaml(args.suite)
    except FileNotFoundError:
        print(c(f"  Error: Suite file not found: {args.suite}", "red"))
        sys.exit(1)
    except Exception as e:
        print(c(f"  Error loading suite: {e}", "red"))
        sys.exit(1)

    # Override model if specified
    if args.model:
        suite.model = args.model

    print()
    print(f"  {c('cane-eval rca', 'cyan')} {c(suite.name, 'bold')}")
    print(f"  {len(suite.tests)} test cases | model: {suite.model}")
    print()

    # Parse tags
    tags = args.tags.split(",") if args.tags else None

    # If results JSON provided, load from file instead of running
    if args.results:
        try:
            with open(args.results, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(c(f"  Error: Results file not found: {args.results}", "red"))
            sys.exit(1)

        from cane_eval.engine import ReliabilityResult
        from cane_eval.judge import JudgeResult, CriteriaScore

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

        from cane_eval.engine import ReliabilitySummary
        summary = ReliabilitySummary(
            suite_name=data.get("suite_name", "loaded"),
            total=len(eval_results),
            results=eval_results,
        )
    else:
        # Run the eval first
        print("  Running eval first...")
        print()

        # Determine provider
        provider = getattr(args, "provider", "anthropic") or "anthropic"
        base_url = getattr(args, "base_url", None)

        from cane_eval.providers import PROVIDERS, PROVIDER_ALIASES
        resolved = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
        provider_cls = PROVIDERS.get(resolved)
        env_key = provider_cls.env_key() if provider_cls else "ANTHROPIC_API_KEY"

        runner = ReliabilityRunner(
            api_key=args.api_key or os.environ.get(env_key),
            model=args.model,
            verbose=not args.quiet,
            on_result=print_result if not args.quiet else None,
            provider=provider,
            base_url=base_url,
        )

        summary = runner.run(suite, tags=tags)

        if not args.quiet:
            print_summary(summary)

    # Determine provider for RCA
    provider = getattr(args, "provider", "anthropic") or "anthropic"
    base_url = getattr(args, "base_url", None)

    from cane_eval.providers import PROVIDERS as _P, PROVIDER_ALIASES as _PA
    _resolved = _PA.get(provider.lower(), provider.lower())
    _pcls = _P.get(_resolved)
    _env_key = _pcls.env_key() if _pcls else "ANTHROPIC_API_KEY"

    # Now run RCA
    analyzer = RootCauseAnalyzer(
        api_key=args.api_key or os.environ.get(_env_key),
        model=args.model or suite.model,
        verbose=not args.quiet,
        provider=provider,
        base_url=base_url,
    )

    print(f"  {c('Running root cause analysis...', 'cyan')}")
    print()

    rca_result = analyzer.analyze(
        summary,
        max_score=args.threshold,
        max_failures=args.max_failures,
    )

    if not args.quiet:
        print_rca_result(rca_result)

    # Optionally run targeted analysis on each failure
    if args.targeted:
        failures = [r for r in summary.results if r.score <= args.threshold]
        failures.sort(key=lambda r: r.score)
        targeted_limit = min(len(failures), args.targeted_max)

        if failures:
            print(f"  {c(f'Running targeted analysis on {targeted_limit} worst failures...', 'cyan')}")
            print()

            for r in failures[:targeted_limit]:
                targeted = analyzer.analyze_result(r)
                if not args.quiet:
                    print_targeted_rca_result(targeted)

    # Save results JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rca_result.to_dict(), f, indent=2)
        print(f"  Results saved to {args.output}")

    # Exit code based on critical root causes
    critical_count = sum(1 for rc in rca_result.root_causes if rc.severity == "critical")
    if critical_count > 0:
        sys.exit(1)
    sys.exit(0)


def cmd_demo(args):
    """Run built-in demo."""
    from cane_eval.demo import run_demo
    run_demo(with_rca=args.with_rca, verbose=True)


def cmd_preflight(args):
    """Run pre-flight checks on a test suite."""
    from cane_eval.suite import ReliabilitySuite
    from cane_eval.engine import ReliabilityRunner

    try:
        suite = ReliabilitySuite.from_yaml(args.suite)
    except FileNotFoundError:
        print(c(f"  Error: Suite file not found: {args.suite}", "red"))
        sys.exit(1)
    except Exception as e:
        print(c(f"  Error loading suite: {e}", "red"))
        sys.exit(1)

    print()
    print(f"  {c('cane-eval preflight', 'cyan')} {c(suite.name, 'bold')}")
    print()

    result = ReliabilityRunner.preflight(suite, timeout=args.timeout, verbose=True)
    print()

    if result["ok"]:
        print(f"  {c('All checks passed. Ready to run.', 'green')}")
    else:
        err_count = len(result["errors"])
        print(f"  {c(f'{err_count} issue(s) found:', 'red')}")
        for err in result["errors"]:
            print(f"    {c('>', 'red')} {err}")

    print()
    sys.exit(0 if result["ok"] else 1)


def cmd_validate(args):
    """Validate a test suite YAML file."""
    from cane_eval.suite import ReliabilitySuite

    try:
        suite = ReliabilitySuite.from_yaml(args.suite)
        print(f"  {c('Valid', 'green')} {suite.name}")
        print(f"  {len(suite.tests)} test cases, {len(suite.criteria)} criteria")
        if suite.custom_rules:
            print(f"  {len(suite.custom_rules)} custom rules")
        if suite.target.type != "callable":
            print(f"  Target: {suite.target.type} ({suite.target.url or suite.target.command})")
    except Exception as e:
        print(c(f"  Invalid: {e}", "red"))
        sys.exit(1)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        prog="cane-eval",
        description="AI system reliability infrastructure",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Run a reliability eval suite")
    run_parser.add_argument("suite", help="Path to YAML test suite")
    run_parser.add_argument("--model", help="Override judge model")
    run_parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    run_parser.add_argument("--tags", help="Comma-separated tag filter")
    run_parser.add_argument("--export", choices=["dpo", "sft", "openai", "raw"], help="Export format")
    run_parser.add_argument("--output", help="Export output path")
    run_parser.add_argument("--output-json", help="Save full results as JSON")
    run_parser.add_argument("--mine", action="store_true", help="Run failure mining after eval")
    run_parser.add_argument("--mine-threshold", type=float, default=60, help="Max score for mining (default: 60)")
    run_parser.add_argument("--mine-max", type=int, default=100, help="Max examples to mine (default: 100)")
    run_parser.add_argument("--mine-format", choices=["dpo", "sft"], default="dpo", help="Mining export format")
    run_parser.add_argument("--mine-output", help="Mining export path")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    run_parser.add_argument("--fail-on-warn", action="store_true", help="Exit 1 on warnings too")
    run_parser.add_argument("--provider", default="anthropic", help="Judge provider: anthropic, openai, gemini, openai-compatible (default: anthropic)")
    run_parser.add_argument("--base-url", help="Base URL for OpenAI-compatible endpoints (e.g. http://localhost:11434/v1)")
    run_parser.add_argument("--schema", help="Path to JSON Schema file to validate agent responses against")
    run_parser.add_argument("--latency-p95", type=int, help="Fail if p95 latency exceeds this threshold (ms)")
    run_parser.add_argument("--latency-target", type=int, default=5000, help="Target latency for reliability scoring in ms (default: 5000)")
    run_parser.add_argument("--fail-on-schema", action="store_true", help="Exit 1 if any response fails schema validation")
    run_parser.add_argument("--concurrency", "-j", type=int, default=0, help="Number of parallel test executions (default: 1, or from suite YAML)")
    run_parser.add_argument("--reliability-weights", help="Reliability weights as C:S:P (e.g. 60:20:20 for correctness:structural:performance)")

    # diff
    diff_parser = subparsers.add_parser("diff", help="Compare two eval runs (regression diff)")
    diff_parser.add_argument("old", help="Path to older results JSON")
    diff_parser.add_argument("new", help="Path to newer results JSON")

    # rca
    rca_parser = subparsers.add_parser("rca", help="Run root cause analysis on eval failures")
    rca_parser.add_argument("suite", help="Path to YAML test suite")
    rca_parser.add_argument("--model", help="Override judge/analysis model")
    rca_parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    rca_parser.add_argument("--tags", help="Comma-separated tag filter")
    rca_parser.add_argument("--results", help="Path to existing results JSON (skip running eval)")
    rca_parser.add_argument("--threshold", type=float, default=60, help="Max score for analysis (default: 60)")
    rca_parser.add_argument("--max-failures", type=int, default=30, help="Max failures to analyze (default: 30)")
    rca_parser.add_argument("--targeted", action="store_true", help="Also run targeted analysis on each failure")
    rca_parser.add_argument("--targeted-max", type=int, default=5, help="Max results for targeted analysis (default: 5)")
    rca_parser.add_argument("--output", help="Save RCA results as JSON")
    rca_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    rca_parser.add_argument("--provider", default="anthropic", help="Judge provider: anthropic, openai, gemini, openai-compatible")
    rca_parser.add_argument("--base-url", help="Base URL for OpenAI-compatible endpoints")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate a test suite YAML")
    validate_parser.add_argument("suite", help="Path to YAML test suite")

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run built-in demo (no setup required)")
    demo_parser.add_argument("--with-rca", action="store_true", help="Include root cause analysis")

    # preflight
    preflight_parser = subparsers.add_parser("preflight", help="Check agent endpoints before running eval")
    preflight_parser.add_argument("suite", help="Path to YAML test suite")
    preflight_parser.add_argument("--timeout", type=int, default=5, help="Timeout per check in seconds (default: 5)")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "rca":
        cmd_rca(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "preflight":
        cmd_preflight(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
