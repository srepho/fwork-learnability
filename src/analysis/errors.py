"""Error categorization and analysis."""

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors from the methodology."""

    IMPORT = "import"  # Module not found, wrong import path
    DEPENDENCY = "dependency"  # Missing extras, version conflicts
    TYPE = "type"  # Type mismatches, wrong argument types
    RUNTIME = "runtime"  # Execution failures, exceptions
    LOGIC = "logic"  # Code runs but wrong output
    HALLUCINATION = "hallucination"  # Non-existent API called
    SYNTAX = "syntax"  # Python syntax errors


class ErrorOrigin(Enum):
    """Where the error originated."""

    FRAMEWORK = "framework"  # Custom exception from the framework
    FRAMEWORK_CONTEXT = "framework_context"  # Standard exception inside framework code
    USER_CODE = "user_code"  # Exception from generated solution


@dataclass
class NormalizedError:
    """Normalized error signature for clustering and analysis."""

    category: ErrorCategory
    origin: ErrorOrigin
    exception_type: str
    signature: str  # Normalized signature for clustering
    is_actionable: bool  # Whether the error message is actionable
    original_message: str
    original_traceback: str | None = None


class ErrorAnalyzer:
    """Analyze and categorize errors from code execution."""

    # Patterns for error classification
    IMPORT_PATTERNS = [
        r"ModuleNotFoundError",
        r"ImportError",
        r"No module named",
    ]

    DEPENDENCY_PATTERNS = [
        r"pip install",
        r"extras_require",
        r"version.*conflict",
        r"incompatible.*version",
    ]

    TYPE_PATTERNS = [
        r"TypeError",
        r"expected.*got",
        r"argument.*type",
    ]

    HALLUCINATION_PATTERNS = [
        r"AttributeError.*has no attribute",
        r"AttributeError.*has no method",
        r"NameError.*is not defined",
        r"module.*has no attribute",
    ]

    SYNTAX_PATTERNS = [
        r"SyntaxError",
        r"IndentationError",
    ]

    # Framework package names for origin detection
    FRAMEWORK_PACKAGES = [
        "pydantic_ai",
        "pydantic-ai",
        "haystack",
        "langgraph",
        "langchain",
        "openai_agents",
        "openai-agents",
        "autogen",
        "semantic_kernel",
    ]

    def __init__(self, framework: str | None = None):
        self.framework = framework

    def analyze(
        self,
        exception_type: str | None,
        exception_message: str | None,
        traceback: str | None,
    ) -> NormalizedError:
        """Analyze an error and return normalized representation.

        Args:
            exception_type: The exception class name.
            exception_message: The error message.
            traceback: The full traceback string.

        Returns:
            NormalizedError with categorization and normalization.
        """
        exc_type = exception_type or "Unknown"
        message = exception_message or ""
        tb = traceback or ""

        # Determine category
        category = self._categorize(exc_type, message, tb)

        # Determine origin
        origin = self._determine_origin(exc_type, tb)

        # Normalize signature
        signature = self._normalize_signature(exc_type, message)

        # Check if actionable
        is_actionable = self._is_actionable(category, message, tb)

        return NormalizedError(
            category=category,
            origin=origin,
            exception_type=exc_type,
            signature=signature,
            is_actionable=is_actionable,
            original_message=message,
            original_traceback=tb,
        )

    def _categorize(self, exc_type: str, message: str, tb: str) -> ErrorCategory:
        """Determine the error category."""
        combined = f"{exc_type} {message} {tb}"

        if exc_type == "SyntaxError" or any(
            re.search(p, combined, re.I) for p in self.SYNTAX_PATTERNS
        ):
            return ErrorCategory.SYNTAX

        if any(re.search(p, combined, re.I) for p in self.IMPORT_PATTERNS):
            return ErrorCategory.IMPORT

        if any(re.search(p, combined, re.I) for p in self.DEPENDENCY_PATTERNS):
            return ErrorCategory.DEPENDENCY

        if any(re.search(p, combined, re.I) for p in self.HALLUCINATION_PATTERNS):
            return ErrorCategory.HALLUCINATION

        if exc_type == "TypeError" or any(
            re.search(p, combined, re.I) for p in self.TYPE_PATTERNS
        ):
            return ErrorCategory.TYPE

        if exc_type == "AssertionError" or "assertion" in message.lower():
            return ErrorCategory.LOGIC

        return ErrorCategory.RUNTIME

    def _determine_origin(self, exc_type: str, tb: str) -> ErrorOrigin:
        """Determine where the error originated."""
        # Check if it's a framework-specific exception class
        for pkg in self.FRAMEWORK_PACKAGES:
            if pkg.replace("-", "_") in exc_type.lower() or pkg.replace("_", "-") in exc_type.lower():
                return ErrorOrigin.FRAMEWORK

        # Check if the traceback points to framework code
        for pkg in self.FRAMEWORK_PACKAGES:
            if f"/{pkg.replace('-', '_')}/" in tb or f"\\{pkg.replace('-', '_')}\\" in tb:
                return ErrorOrigin.FRAMEWORK_CONTEXT

        # Check for solution.py in traceback
        if "solution.py" in tb:
            return ErrorOrigin.USER_CODE

        return ErrorOrigin.USER_CODE

    def _normalize_signature(self, exc_type: str, message: str) -> str:
        """Create a normalized signature for clustering.

        Removes specific values while preserving error structure.
        """
        # Extract first line of message
        first_line = message.split("\n")[0] if message else ""

        # Remove line numbers
        normalized = re.sub(r"line \d+", "line N", first_line)

        # Remove file paths
        normalized = re.sub(r'["\']?/[\w/.-]+\.py["\']?', "module.py", normalized)
        normalized = re.sub(r'["\']?[\w/\\.-]+\.py["\']?', "module.py", normalized)

        # Remove specific string values in quotes
        normalized = re.sub(r"'[^']{20,}'", "'...'", normalized)
        normalized = re.sub(r'"[^"]{20,}"', '"..."', normalized)

        # Truncate long messages
        if len(normalized) > 100:
            normalized = normalized[:100] + "..."

        return f"{exc_type}:{normalized}"

    def _is_actionable(self, category: ErrorCategory, message: str, tb: str) -> bool:
        """Determine if the error message is actionable.

        Actionable errors clearly indicate what to fix.
        """
        # Syntax errors with line numbers are actionable
        if category == ErrorCategory.SYNTAX and "line" in message.lower():
            return True

        # Import errors with module names are actionable
        if category == ErrorCategory.IMPORT and ("No module named" in message or "cannot import" in message.lower()):
            return True

        # Type errors with expected/got are actionable
        if category == ErrorCategory.TYPE and ("expected" in message.lower() or "got" in message.lower()):
            return True

        # Attribute errors with specific attribute names are actionable
        if category == ErrorCategory.HALLUCINATION and "has no attribute" in message:
            return True

        # Assertion errors with clear messages are actionable
        if category == ErrorCategory.LOGIC and "assert" in message.lower():
            return True

        # Short tracebacks pointing to user code are more actionable
        if "solution.py" in tb and tb.count("\n") < 15:
            return True

        return False


def compute_meaningful_error_rate(
    errors: list[NormalizedError],
    early_turn_threshold: int = 2,
) -> float:
    """Compute the meaningful error rate for early failures.

    Args:
        errors: List of normalized errors with turn information.
        early_turn_threshold: What counts as "early" (turns 1-N).

    Returns:
        Fraction of early errors that were actionable.
    """
    if not errors:
        return 0.0

    actionable_count = sum(1 for e in errors if e.is_actionable)
    return actionable_count / len(errors)
