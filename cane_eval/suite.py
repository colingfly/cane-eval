"""
suite.py -- YAML test suite loader.

Loads test cases from YAML files. Each suite defines:
- An agent target (callable, HTTP endpoint, or CLI command)
- Judge criteria with weights
- Custom rules
- Test cases with questions and expected answers
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestCase:
    """A single eval test case."""
    question: str
    expected_answer: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    context: Optional[str] = None  # optional source docs for grounded eval
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        q = self.question[:50] + "..." if len(self.question) > 50 else self.question
        return f"TestCase({q!r})"


@dataclass
class Criterion:
    """A scoring dimension the judge evaluates."""
    key: str
    label: str
    description: str = ""
    weight: int = 50

    def to_dict(self):
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "weight": self.weight,
            "is_enabled": True,
        }


@dataclass
class AgentTarget:
    """How to call the agent being evaluated."""
    type: str = "callable"  # callable | http | command
    url: Optional[str] = None
    method: str = "POST"
    headers: dict = field(default_factory=dict)
    payload_template: str = '{"message": "{{question}}"}'
    response_path: str = "response"
    command: Optional[str] = None  # for CLI agents: "python agent.py --query {{question}}"

    @classmethod
    def from_dict(cls, d: dict) -> "AgentTarget":
        return cls(
            type=d.get("type", "callable"),
            url=d.get("url"),
            method=d.get("method", "POST"),
            headers=d.get("headers", {}),
            payload_template=d.get("payload_template", '{"message": "{{question}}"}'),
            response_path=d.get("response_path", "response"),
            command=d.get("command"),
        )


DEFAULT_CRITERIA = [
    Criterion(key="accuracy", label="Accuracy", description="Factual correctness vs expected answer", weight=40),
    Criterion(key="completeness", label="Completeness", description="Covers all key points", weight=30),
    Criterion(key="hallucination", label="Hallucination Check", description="No fabricated information", weight=30),
]


class TestSuite:
    """
    A collection of test cases loaded from YAML.

    YAML format:
    ```yaml
    name: Customer Support Agent
    description: Eval suite for support bot
    model: claude-sonnet-4-5-20250929

    target:
      type: http
      url: https://my-agent.com/ask
      method: POST
      payload_template: '{"query": "{{question}}"}'
      response_path: data.answer

    criteria:
      - key: accuracy
        label: Accuracy
        description: Factual correctness
        weight: 40
      - key: completeness
        label: Completeness
        weight: 30
      - key: hallucination
        label: Hallucination Check
        weight: 30

    custom_rules:
      - Never recommend competitors
      - Always cite the source document

    tests:
      - question: What is the return policy?
        expected_answer: 30-day return policy for unused items
        tags: [policy, returns]

      - question: How do I reset my password?
        expected_answer: Go to Settings > Security > Reset Password
        tags: [account, password]
    ```
    """

    def __init__(
        self,
        name: str = "Untitled Suite",
        description: str = "",
        model: str = "claude-sonnet-4-5-20250929",
        target: Optional[AgentTarget] = None,
        criteria: Optional[list[Criterion]] = None,
        custom_rules: Optional[list[str]] = None,
        tests: Optional[list[TestCase]] = None,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.target = target or AgentTarget()
        self.criteria = criteria or list(DEFAULT_CRITERIA)
        self.custom_rules = custom_rules or []
        self.tests = tests or []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TestSuite":
        """Load a test suite from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Suite file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty suite file: {path}")

        # Parse target
        target = None
        if "target" in data:
            target = AgentTarget.from_dict(data["target"])

        # Parse criteria
        criteria = None
        if "criteria" in data:
            criteria = [
                Criterion(
                    key=c["key"],
                    label=c.get("label", c["key"].replace("_", " ").title()),
                    description=c.get("description", ""),
                    weight=c.get("weight", 50),
                )
                for c in data["criteria"]
            ]

        # Parse custom rules
        custom_rules = data.get("custom_rules", [])

        # Parse test cases
        tests = []
        for t in data.get("tests", []):
            tests.append(TestCase(
                question=t["question"],
                expected_answer=t.get("expected_answer"),
                tags=t.get("tags", []),
                context=t.get("context"),
                metadata=t.get("metadata", {}),
            ))

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            model=data.get("model", "claude-sonnet-4-5-20250929"),
            target=target,
            criteria=criteria,
            custom_rules=custom_rules,
            tests=tests,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TestSuite":
        """Create a suite from a dictionary (useful for programmatic creation)."""
        target = None
        if "target" in data:
            target = AgentTarget.from_dict(data["target"])

        criteria = None
        if "criteria" in data:
            criteria = [
                Criterion(
                    key=c["key"],
                    label=c.get("label", c["key"]),
                    description=c.get("description", ""),
                    weight=c.get("weight", 50),
                )
                for c in data["criteria"]
            ]

        tests = [
            TestCase(
                question=t["question"],
                expected_answer=t.get("expected_answer"),
                tags=t.get("tags", []),
                context=t.get("context"),
            )
            for t in data.get("tests", [])
        ]

        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            model=data.get("model", "claude-sonnet-4-5-20250929"),
            target=target,
            criteria=criteria,
            custom_rules=data.get("custom_rules", []),
            tests=tests,
        )

    def criteria_dicts(self) -> list[dict]:
        """Return criteria as list of dicts for judge consumption."""
        return [c.to_dict() for c in self.criteria]

    def filter_by_tags(self, tags: list[str]) -> list[TestCase]:
        """Return test cases matching any of the given tags."""
        tag_set = set(tags)
        return [t for t in self.tests if tag_set & set(t.tags)]

    def __len__(self):
        return len(self.tests)

    def __repr__(self):
        return f"TestSuite({self.name!r}, {len(self.tests)} tests)"
