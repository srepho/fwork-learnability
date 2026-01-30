"""Base task definition classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskTier(Enum):
    """Task complexity tiers."""

    TIER_1 = 1  # Simple classification - basic framework setup
    TIER_2 = 2  # Tool use - tool definition, schema handling
    TIER_3 = 3  # Agent-native - routing, state, multi-step


@dataclass
class TestCase:
    """A single test case for a task."""

    case_id: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    description: str = ""


@dataclass
class Task:
    """Definition of a benchmark task."""

    task_id: str
    name: str
    tier: TaskTier
    description: str
    expected_interface: str  # Description of expected function/class signature
    development_tests: list[TestCase] = field(default_factory=list)
    hidden_tests: list[TestCase] = field(default_factory=list)

    def get_task_prompt(self, framework: str) -> str:
        """Generate the task prompt for the LLM.

        Args:
            framework: Target framework name.

        Returns:
            Task prompt string.
        """
        return f"""Implement a solution using the {framework} framework.

## Task: {self.name}

{self.description}

## Expected Interface

{self.expected_interface}

## Requirements
- Use {framework} for the implementation
- The solution must be a complete, runnable Python module
- Include all necessary imports
"""

    def generate_dev_test_code(self) -> str:
        """Generate test code for development tests."""
        test_lines = ["from solution import *", ""]

        for test in self.development_tests:
            test_lines.append(f"# Test: {test.case_id}")
            test_lines.append(self._generate_single_test(test))
            test_lines.append("")

        test_lines.append("print('All development tests passed!')")
        return "\n".join(test_lines)

    def generate_hidden_test_code(self) -> str:
        """Generate test code for hidden tests."""
        test_lines = ["from solution import *", "", "passed = 0", "total = 0", ""]

        for test in self.hidden_tests:
            test_lines.append(f"# Test: {test.case_id}")
            test_lines.append("total += 1")
            test_lines.append("try:")
            test_lines.append(f"    {self._generate_single_test(test)}")
            test_lines.append("    passed += 1")
            test_lines.append("except AssertionError as e:")
            test_lines.append(f"    print(f'FAILED {test.case_id}: {{e}}')")
            test_lines.append("")

        test_lines.append("print(f'Hidden tests: {passed}/{total} passed')")
        test_lines.append("assert passed == total, f'Hidden tests failed: {passed}/{total}'")
        return "\n".join(test_lines)

    def _generate_single_test(self, test: TestCase) -> str:
        """Generate code for a single test case. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _generate_single_test")
