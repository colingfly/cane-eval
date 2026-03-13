"""Tests for the YAML suite loader."""

import tempfile
import os
import pytest

from cane_eval.suite import TestSuite, TestCase, Criterion, AgentTarget, DEFAULT_CRITERIA


SAMPLE_YAML = """
name: Test Suite
description: A test suite
model: claude-sonnet-4-5-20250929

criteria:
  - key: accuracy
    label: Accuracy
    description: Factual correctness
    weight: 60
  - key: tone
    label: Tone
    weight: 40

custom_rules:
  - Be polite
  - Never say "I don't know"

tests:
  - question: What is 2+2?
    expected_answer: "4"
    tags: [math, basic]

  - question: What color is the sky?
    expected_answer: Blue
    tags: [science]

  - question: No expected answer here
"""


def _write_yaml(content: str) -> str:
    """Write content to a temp YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def test_load_from_yaml():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        assert suite.name == "Test Suite"
        assert suite.description == "A test suite"
        assert len(suite.tests) == 3
        assert len(suite.criteria) == 2
        assert len(suite.custom_rules) == 2
    finally:
        os.unlink(path)


def test_criteria_weights():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        assert suite.criteria[0].weight == 60
        assert suite.criteria[1].weight == 40
    finally:
        os.unlink(path)


def test_test_case_tags():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        assert suite.tests[0].tags == ["math", "basic"]
        assert suite.tests[2].tags == []
    finally:
        os.unlink(path)


def test_filter_by_tags():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        math_tests = suite.filter_by_tags(["math"])
        assert len(math_tests) == 1
        assert math_tests[0].question == "What is 2+2?"

        science_tests = suite.filter_by_tags(["science"])
        assert len(science_tests) == 1
    finally:
        os.unlink(path)


def test_filter_by_multiple_tags():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        filtered = suite.filter_by_tags(["math", "science"])
        assert len(filtered) == 2
    finally:
        os.unlink(path)


def test_expected_answer_optional():
    path = _write_yaml(SAMPLE_YAML)
    try:
        suite = TestSuite.from_yaml(path)
        assert suite.tests[2].expected_answer is None
    finally:
        os.unlink(path)


def test_default_criteria():
    assert len(DEFAULT_CRITERIA) == 3
    keys = [c.key for c in DEFAULT_CRITERIA]
    assert "accuracy" in keys
    assert "completeness" in keys
    assert "hallucination" in keys


def test_criteria_dicts():
    suite = TestSuite(
        name="Test",
        criteria=[Criterion(key="accuracy", label="Accuracy", weight=100)],
    )
    dicts = suite.criteria_dicts()
    assert len(dicts) == 1
    assert dicts[0]["key"] == "accuracy"
    assert dicts[0]["is_enabled"] is True


def test_from_dict():
    data = {
        "name": "Dict Suite",
        "tests": [
            {"question": "Hello?", "expected_answer": "Hi"},
        ],
    }
    suite = TestSuite.from_dict(data)
    assert suite.name == "Dict Suite"
    assert len(suite.tests) == 1


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        TestSuite.from_yaml("nonexistent.yaml")


def test_empty_file():
    path = _write_yaml("")
    try:
        with pytest.raises(ValueError):
            TestSuite.from_yaml(path)
    finally:
        os.unlink(path)


def test_http_target():
    yaml_content = """
name: HTTP Test
target:
  type: http
  url: https://example.com/api
  method: POST
  payload_template: '{"q": "{{question}}"}'
  response_path: data.answer
tests:
  - question: Test
"""
    path = _write_yaml(yaml_content)
    try:
        suite = TestSuite.from_yaml(path)
        assert suite.target.type == "http"
        assert suite.target.url == "https://example.com/api"
        assert suite.target.response_path == "data.answer"
    finally:
        os.unlink(path)


def test_suite_len():
    suite = TestSuite(tests=[TestCase(question="Q1"), TestCase(question="Q2")])
    assert len(suite) == 2
