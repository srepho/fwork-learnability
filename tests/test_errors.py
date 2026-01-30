"""Tests for error analysis."""

import pytest

from src.analysis.errors import ErrorAnalyzer, ErrorCategory, ErrorOrigin


@pytest.fixture
def analyzer():
    return ErrorAnalyzer(framework="pydantic-ai")


class TestErrorAnalyzer:
    def test_categorize_import_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="ModuleNotFoundError",
            exception_message="No module named 'pydantic_ai'",
            traceback="",
        )
        assert result.category == ErrorCategory.IMPORT

    def test_categorize_syntax_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="SyntaxError",
            exception_message="invalid syntax (line 5)",
            traceback="",
        )
        assert result.category == ErrorCategory.SYNTAX

    def test_categorize_type_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="TypeError",
            exception_message="expected str, got int",
            traceback="",
        )
        assert result.category == ErrorCategory.TYPE

    def test_categorize_hallucination(self, analyzer):
        result = analyzer.analyze(
            exception_type="AttributeError",
            exception_message="'Agent' object has no attribute 'verify'",
            traceback="",
        )
        assert result.category == ErrorCategory.HALLUCINATION

    def test_categorize_logic_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="AssertionError",
            exception_message="Expected 'simple', got 'complex'",
            traceback="",
        )
        assert result.category == ErrorCategory.LOGIC

    def test_origin_user_code(self, analyzer):
        result = analyzer.analyze(
            exception_type="KeyError",
            exception_message="'model'",
            traceback="File solution.py, line 10\n  result['model']",
        )
        assert result.origin == ErrorOrigin.USER_CODE

    def test_origin_framework(self, analyzer):
        result = analyzer.analyze(
            exception_type="KeyError",
            exception_message="'model'",
            traceback="File /path/to/pydantic_ai/agent.py, line 100",
        )
        assert result.origin == ErrorOrigin.FRAMEWORK_CONTEXT

    def test_normalize_signature(self, analyzer):
        result = analyzer.analyze(
            exception_type="TypeError",
            exception_message="expected str but got int at line 42 in /very/long/path/to/file.py",
            traceback="",
        )
        # Signature should have normalized line numbers and paths
        assert "line N" in result.signature or "line 42" not in result.signature
        assert "/very/long/path" not in result.signature

    def test_actionable_type_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="TypeError",
            exception_message="Agent() got an unexpected keyword argument 'model_name'",
            traceback="",
        )
        assert result.is_actionable

    def test_actionable_import_error(self, analyzer):
        result = analyzer.analyze(
            exception_type="ImportError",
            exception_message="cannot import name 'Agent' from 'pydantic_ai'",
            traceback="",
        )
        assert result.is_actionable
