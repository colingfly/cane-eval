"""
Example: Run an eval suite against a simple agent function.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_eval.py
"""

from cane_eval import TestSuite, EvalRunner, Exporter, FailureMiner


# A simple dummy agent for demonstration.
# Replace this with your actual agent.
def my_agent(question: str) -> str:
    """Dummy agent that gives intentionally incomplete answers."""
    answers = {
        "return": "We have a return policy. You can return items.",
        "password": "Go to Settings and reset your password.",
        "shipping": "Yes, we ship internationally to many countries.",
        "payment": "We accept credit cards and PayPal.",
        "support": "You can contact us through our website.",
    }

    for keyword, answer in answers.items():
        if keyword in question.lower():
            return answer

    return "I'm not sure about that. Please contact support."


def main():
    # 1. Load the test suite
    suite = TestSuite.from_yaml("examples/support_agent.yaml")
    print(f"Loaded: {suite.name} ({len(suite.tests)} tests)")

    # 2. Run the eval
    runner = EvalRunner(verbose=True)
    summary = runner.run(suite, agent=my_agent)

    print(f"\nOverall score: {summary.overall_score}")
    print(f"Pass rate: {summary.pass_rate:.0f}%")
    print(f"Passed: {summary.passed}, Warned: {summary.warned}, Failed: {summary.failed}")

    # 3. Export results
    exporter = Exporter(summary)

    # Export failures as DPO training pairs (using expected_answer as "chosen")
    exporter.to_dpo("output_dpo.jsonl", max_score=60)
    print("\nDPO training data saved to output_dpo.jsonl")

    # Export passing results as SFT data
    exporter.to_sft("output_sft.jsonl", min_score=80, use_expected=False)
    print("SFT training data saved to output_sft.jsonl")

    # 4. Mine failures (optional -- generates improved answers via LLM)
    if summary.failed > 0:
        print("\nMining failures...")
        miner = FailureMiner(verbose=True)
        mined = miner.mine(summary, max_score=60)

        print(f"\nMined {mined.total_mined} examples")
        print(f"Failure types: {mined.failure_distribution}")

        # Export mined data with LLM-improved answers
        mined.to_file("output_mined_dpo.jsonl", format="dpo")
        print("Mined DPO data saved to output_mined_dpo.jsonl")


if __name__ == "__main__":
    main()
