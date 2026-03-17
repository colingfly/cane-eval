"""
demo.py -- Built-in demo for cane-eval.

Runs a prebuilt agent against a bundled test suite to showcase
eval scoring, failure detection, and root cause analysis.
No setup required beyond an ANTHROPIC_API_KEY.

Usage:
    cane-eval demo
    cane-eval demo --with-rca
"""

import os
import sys
import time

from cane_eval.suite import TestSuite, TestCase, Criterion, AgentTarget
from cane_eval.engine import EvalRunner


# ---- Demo agent ----
# Intentionally flawed agent that gives incomplete, hallucinated, or off-topic answers.
# This makes the eval results interesting and demonstrates failure detection.

def _demo_agent(question: str) -> str:
    """A deliberately flawed support agent for demo purposes."""
    q = question.lower()

    if "return" in q or "refund" in q:
        # Incomplete -- missing key details
        return "We have a return policy. You can return items if you're not happy with them."

    if "password" in q or "reset" in q:
        # Hallucination -- invents a phone number and process that doesn't exist
        return (
            "To reset your password, call our password hotline at 1-800-RESETPW. "
            "A technician will verify your identity using your mother's maiden name "
            "and reset it manually within 48 hours."
        )

    if "shipping" in q or "international" in q:
        # Decent but missing specifics
        return (
            "Yes, we ship internationally to many countries. Shipping times vary "
            "by destination. Check our website for rates."
        )

    if "payment" in q or "credit" in q:
        # Factual error + competitor mention
        return (
            "We accept Visa and Mastercard. For better rates, we recommend "
            "using our competitor PayBetter which offers 5% cashback."
        )

    if "contact" in q or "support" in q or "help" in q:
        # Good answer -- should pass
        return (
            "You can reach us through: 1) Live chat on our website available 24/7, "
            "2) Email at support@example.com with a response within 4 hours, "
            "3) Phone at 1-800-555-0199 Monday through Friday 9am to 6pm EST."
        )

    return "I'm sorry, I don't have information about that. Please contact our support team."


# ---- Demo suite ----

def _demo_suite() -> TestSuite:
    """Build the demo test suite."""
    return TestSuite(
        name="Demo: Support Agent",
        description="Built-in demo showing eval scoring and failure detection",
        model="claude-sonnet-4-5-20250929",
        target=AgentTarget(type="callable"),
        criteria=[
            Criterion(key="accuracy", label="Accuracy", description="Factual correctness vs expected answer", weight=40),
            Criterion(key="completeness", label="Completeness", description="Covers all key points from expected answer", weight=30),
            Criterion(key="hallucination", label="Hallucination", description="No fabricated or unsupported information", weight=30),
        ],
        custom_rules=[
            "Never recommend competitor products",
            "Always maintain a professional and helpful tone",
        ],
        tests=[
            TestCase(
                question="What is your return policy?",
                expected_answer=(
                    "We offer a 30-day return policy for all unused items in original packaging. "
                    "Items must be returned with a valid receipt. Refunds are processed within "
                    "5-7 business days to the original payment method."
                ),
                tags=["policy", "returns"],
            ),
            TestCase(
                question="How do I reset my password?",
                expected_answer=(
                    "To reset your password: 1) Go to the login page, 2) Click 'Forgot Password', "
                    "3) Enter your email address, 4) Check your inbox for a reset link, "
                    "5) Click the link and create a new password. The link expires after 24 hours."
                ),
                tags=["account", "password"],
            ),
            TestCase(
                question="Do you offer international shipping?",
                expected_answer=(
                    "Yes, we ship to over 50 countries. International shipping takes 7-14 business days. "
                    "Shipping costs vary by destination and are calculated at checkout. "
                    "Customs duties and taxes are the responsibility of the buyer."
                ),
                tags=["shipping"],
            ),
            TestCase(
                question="What payment methods do you accept?",
                expected_answer=(
                    "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay. "
                    "All transactions are encrypted with SSL. We do not store credit card information."
                ),
                tags=["payments"],
            ),
            TestCase(
                question="How do I contact customer support?",
                expected_answer=(
                    "You can reach us through: 1) Live chat on our website (24/7), "
                    "2) Email at support@example.com (response within 4 hours), "
                    "3) Phone at 1-800-555-0199 (Mon-Fri 9am-6pm EST)."
                ),
                tags=["contact", "support"],
            ),
        ],
    )


# ---- Colors ----

COLORS = {
    "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
    "blue": "\033[94m", "cyan": "\033[96m", "bold": "\033[1m",
    "dim": "\033[2m", "reset": "\033[0m",
}

def _supports_color():
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


# ---- Demo runner ----

def run_demo(with_rca: bool = False, verbose: bool = True):
    """
    Run the built-in demo.

    Returns:
        RunSummary from the eval run
    """
    print()
    print(f"  {c('cane-eval demo', 'cyan')}")
    print(f"  {c('Running a deliberately flawed support agent against 5 test cases...', 'dim')}")
    print()

    suite = _demo_suite()

    # Pre-flight
    print(f"  {c('Pre-flight checks', 'bold')}")
    preflight = EvalRunner.preflight(suite, verbose=verbose)
    if not preflight["ok"]:
        print()
        print(f"  {c('Pre-flight failed. Fix the issues above and try again.', 'red')}")
        for err in preflight["errors"]:
            print(f"    {c('>', 'red')} {err}")
        print()
        sys.exit(1)
    print()

    # Run eval with streaming output
    runner = EvalRunner(verbose=False)

    print(f"  {c('Evaluating...', 'bold')}")
    print()

    results = []
    gen = runner.run_stream(suite, agent=_demo_agent)
    summary = None

    try:
        while True:
            result = next(gen)
            results.append(result)

            # Print each result as it arrives
            idx = len(results)
            q = result.question[:60] + "..." if len(result.question) > 60 else result.question

            if result.status == "pass":
                badge = c(" PASS ", "green")
            elif result.status == "warn":
                badge = c(" WARN ", "yellow")
            else:
                badge = c(" FAIL ", "red")

            score_val = result.score
            if score_val >= 80:
                score_str = c(f"{score_val:.0f}", "green")
            elif score_val >= 60:
                score_str = c(f"{score_val:.0f}", "yellow")
            else:
                score_str = c(f"{score_val:.0f}", "red")

            print(f"  {badge} {score_str}  {q}")

            # Show failure details inline
            if result.status == "fail":
                reasoning = result.judge_result.overall_reasoning
                if reasoning:
                    # Truncate and indent
                    lines = reasoning[:200].split(". ")
                    for line in lines[:2]:
                        if line.strip():
                            print(f"         {c(line.strip() + '.', 'dim')}")

    except StopIteration as e:
        summary = e.value

    if summary is None:
        # Fallback: build summary manually
        from cane_eval.engine import RunSummary
        duration = sum(r.response_time_ms for r in results) / 1000
        overall = round(sum(r.score for r in results) / len(results), 1) if results else 0
        summary = RunSummary(
            suite_name=suite.name,
            total=len(results),
            passed=sum(1 for r in results if r.status == "pass"),
            warned=sum(1 for r in results if r.status == "warn"),
            failed=sum(1 for r in results if r.status == "fail"),
            overall_score=overall,
            results=results,
            duration_seconds=duration,
        )

    # Summary
    print()
    print(f"  {c('=' * 60, 'dim')}")
    print()

    score_val = summary.overall_score
    if score_val >= 80:
        score_str = c(f"{score_val:.0f}/100", "green")
    elif score_val >= 60:
        score_str = c(f"{score_val:.0f}/100", "yellow")
    else:
        score_str = c(f"{score_val:.0f}/100", "red")

    print(f"  Overall: {score_str}  {c(f'({summary.duration_seconds:.1f}s)', 'dim')}")
    p = c(f"{summary.passed} passed", "green")
    w = c(f"{summary.warned} warned", "yellow")
    f_count = c(f"{summary.failed} failed", "red")
    print(f"  {p}  {w}  {f_count}")
    print()

    # Failure breakdown
    failures = [r for r in results if r.status == "fail"]
    if failures:
        print(f"  {c('Failures detected:', 'bold')}")
        print()
        for r in failures:
            q = r.question[:55] + "..." if len(r.question) > 55 else r.question
            print(f"    {c('FAIL', 'red')} {c(f'{r.score:.0f}', 'red')}  {q}")

            # Show score breakdown
            for cs in r.judge_result.criteria_scores:
                cs_score = cs.score
                if cs_score >= 80:
                    cs_color = "green"
                elif cs_score >= 60:
                    cs_color = "yellow"
                else:
                    cs_color = "red"
                print(f"          {cs.key}: {c(f'{cs_score:.0f}', cs_color)}  {c(cs.reasoning[:80], 'dim')}")
            print()

    # Optional RCA
    if with_rca and failures:
        print(f"  {c('Running root cause analysis...', 'cyan')}")
        print()

        from cane_eval.rca import RootCauseAnalyzer
        analyzer = RootCauseAnalyzer(verbose=False)
        rca = analyzer.analyze(summary, max_score=70)

        if rca.summary:
            print(f"  {c('Root Cause Summary:', 'bold')} {rca.summary}")
            print()

        if rca.top_recommendation:
            print(f"  {c('Top Fix:', 'green')} {rca.top_recommendation}")
            print()

        for i, rc in enumerate(rca.root_causes):
            sev_colors = {"critical": "red", "high": "red", "medium": "yellow", "low": "dim"}
            sev_color = sev_colors.get(rc.severity, "dim")
            print(f"  {i+1}. {c(f'[{rc.severity.upper()}]', sev_color)} {c(rc.title, 'bold')}")
            if rc.recommendation:
                print(f"     {c('Fix:', 'green')} {rc.recommendation}")
            print()

    # Next steps
    print(f"  {c('What just happened:', 'bold')}")
    print(f"    1. Ran a flawed agent against 5 test cases")
    print(f"    2. Claude judged each response on accuracy, completeness, and hallucination")
    print(f"    3. Failures were scored and explained with judge reasoning")
    if with_rca:
        print(f"    4. Root cause analysis identified patterns across failures")
    print()
    print(f"  {c('Next steps:', 'bold')}")
    print(f"    {c('>', 'cyan')} Create your own suite:  {c('cane-eval validate my_tests.yaml', 'dim')}")
    print(f"    {c('>', 'cyan')} Run against your agent: {c('cane-eval run my_tests.yaml', 'dim')}")
    print(f"    {c('>', 'cyan')} Analyze failures:       {c('cane-eval rca my_tests.yaml', 'dim')}")
    print(f"    {c('>', 'cyan')} Export training data:    {c('cane-eval run my_tests.yaml --export dpo', 'dim')}")
    print()
    print(f"  Docs: {c('https://github.com/colingfly/cane-eval', 'cyan')}")
    print()

    return summary
