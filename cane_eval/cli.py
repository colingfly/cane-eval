"""
cli.py -- Command-line interface for cane-eval.

Usage:
    cane-eval run tests.yaml
    cane-eval run tests.yaml --model claude-sonnet-4-5-20250929
    cane-eval run tests.yaml --tags policy,returns
    cane-eval run tests.yaml --export dpo --output training.jsonl
    cane-eval run tests.yaml --mine --mine-threshold 60
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


def print_result(result, index, total):
    """Print a single eval result."""
    q = result.question[:70] + "..." if len(result.question) > 70 else result.question
    badge = _status_badge(result.status)
    score = _score_color(result.score)
    bar = _bar(result.score)

    print(f"  {badge} {c(f'{index}/{total}', 'dim')} {bar} {score}  {q}")

    if result.status == "fail":
        reasoning = result.judge_result.overall_reasoning
        if reasoning:
            print(f"         {c(reasoning[:120], 'dim')}")


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


# ---- Commands ----

def cmd_run(args):
    """Run eval suite."""
    from cane_eval.suite import TestSuite
    from cane_eval.engine import EvalRunner
    from cane_eval.export import Exporter

    # Load suite
    try:
        suite = TestSuite.from_yaml(args.suite)
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

    # Run
    runner = EvalRunner(
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        model=args.model,
        verbose=not args.quiet,
        on_result=print_result if not args.quiet else None,
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
            api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=args.model or suite.model,
            verbose=not args.quiet,
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


def cmd_validate(args):
    """Validate a test suite YAML file."""
    from cane_eval.suite import TestSuite

    try:
        suite = TestSuite.from_yaml(args.suite)
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
        description="LLM-as-Judge evaluation for AI agents",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Run an eval suite")
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

    # diff
    diff_parser = subparsers.add_parser("diff", help="Compare two eval runs (regression diff)")
    diff_parser.add_argument("old", help="Path to older results JSON")
    diff_parser.add_argument("new", help="Path to newer results JSON")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate a test suite YAML")
    validate_parser.add_argument("suite", help="Path to YAML test suite")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
