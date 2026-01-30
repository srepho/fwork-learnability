"""Tests for compliance checking."""

import pytest

from src.compliance.static import StaticComplianceChecker, get_checker


class TestStaticComplianceChecker:
    def test_pydantic_ai_compliant(self):
        checker = get_checker("pydantic-ai")
        code = """
from pydantic_ai import Agent

agent = Agent(model="gpt-4")
result = agent.run_sync("Hello")
"""
        result = checker.check(code)
        assert result.passed
        assert result.has_import
        assert result.has_framework_symbols

    def test_pydantic_ai_missing_import(self):
        checker = get_checker("pydantic-ai")
        code = """
# No imports
def classify(claim):
    return {"classification": "simple"}
"""
        result = checker.check(code)
        assert not result.passed
        assert not result.has_import

    def test_pydantic_ai_wrong_framework(self):
        checker = get_checker("pydantic-ai")
        code = """
from langchain import LLMChain

chain = LLMChain()
"""
        result = checker.check(code)
        assert not result.passed
        assert not result.has_import

    def test_direct_api_always_passes(self):
        checker = get_checker("direct-api")
        code = """
import httpx

response = httpx.post("https://api.example.com")
"""
        result = checker.check(code)
        # Direct API has no requirements
        assert result.passed

    def test_syntax_error_fails(self):
        checker = get_checker("pydantic-ai")
        code = """
from pydantic_ai import Agent

def broken(
"""
        result = checker.check(code)
        assert not result.passed
        assert "syntax" in result.missing_requirements[0].lower()

    def test_alias_import(self):
        checker = get_checker("pydantic-ai")
        code = """
from pydantic_ai import Agent as MyAgent

agent = MyAgent(model="gpt-4")
"""
        result = checker.check(code)
        assert result.passed
        assert "MyAgent" in result.used_symbols or "Agent" in str(result.imported_modules)

    def test_submodule_import(self):
        checker = get_checker("pydantic-ai")
        code = """
from pydantic_ai.agent import Agent

agent = Agent()
"""
        result = checker.check(code)
        assert result.passed
        assert result.has_import


class TestCheckerFactory:
    def test_get_known_checker(self):
        checker = get_checker("pydantic-ai")
        assert checker.framework == "pydantic-ai"
        assert "pydantic_ai" in checker.required_imports

    def test_get_unknown_checker(self):
        checker = get_checker("unknown-framework")
        assert checker.framework == "unknown-framework"
        # Should still work with default settings
