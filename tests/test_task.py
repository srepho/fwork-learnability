"""Tests for task definitions."""

import pytest

from src.tasks.tier1.claims_classification import ClaimsClassificationTask


class TestClaimsClassificationTask:
    @pytest.fixture
    def task(self):
        return ClaimsClassificationTask()

    def test_task_has_dev_tests(self, task):
        assert len(task.development_tests) == 5

    def test_task_has_hidden_tests(self, task):
        assert len(task.hidden_tests) >= 10

    def test_dev_test_structure(self, task):
        for test in task.development_tests:
            # Check input has required fields
            assert "claim_id" in test.input_data
            assert "description" in test.input_data
            assert "damage_estimate" in test.input_data
            assert "parties_involved" in test.input_data
            assert "has_injury" in test.input_data
            assert "liability_clear" in test.input_data
            assert "vehicle_type" in test.input_data

            # Check expected output
            assert "classification" in test.expected_output
            assert test.expected_output["classification"] in ["simple", "complex"]

    def test_task_prompt_includes_framework(self, task):
        prompt = task.get_task_prompt("pydantic-ai")
        assert "pydantic-ai" in prompt
        assert "classify" in prompt.lower()

    def test_generate_dev_test_code(self, task):
        code = task.generate_dev_test_code()
        assert "from solution import" in code
        assert "classify_claim" in code
        assert "assert" in code
        # Should test all 5 dev cases
        for test in task.development_tests:
            assert test.case_id in code

    def test_generate_hidden_test_code(self, task):
        code = task.generate_hidden_test_code()
        assert "from solution import" in code
        assert "passed" in code
        assert "total" in code

    def test_simple_claims_criteria(self, task):
        """Verify simple claims match expected criteria."""
        simple_tests = [t for t in task.development_tests + task.hidden_tests
                       if t.expected_output["classification"] == "simple"]

        for test in simple_tests:
            data = test.input_data
            # Simple claims should have: low damage, clear liability, no injury, personal vehicle
            # At least some of these conditions should be true
            conditions = [
                data["damage_estimate"] < 5000,
                data["liability_clear"] is True,
                data["has_injury"] is False,
                data["parties_involved"] <= 2,
            ]
            # Most conditions should be true for simple cases
            assert sum(conditions) >= 2, f"Simple claim {test.case_id} has too few simple indicators"

    def test_complex_claims_criteria(self, task):
        """Verify complex claims match expected criteria."""
        complex_tests = [t for t in task.development_tests + task.hidden_tests
                        if t.expected_output["classification"] == "complex"]

        for test in complex_tests:
            data = test.input_data
            # Complex claims should have at least one of: high damage, injury, unclear liability,
            # commercial vehicle, or multiple parties
            conditions = [
                data["damage_estimate"] > 10000,
                data["has_injury"] is True,
                data["liability_clear"] is False,
                data["vehicle_type"] == "commercial",
                data["parties_involved"] > 2,
            ]
            assert any(conditions), f"Complex claim {test.case_id} has no complexity indicators"
