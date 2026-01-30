"""Static compliance checking - verify framework imports and symbol usage."""

import ast
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    """Result of compliance check."""

    passed: bool
    has_import: bool
    has_framework_symbols: bool
    imported_modules: list[str]
    used_symbols: list[str]
    missing_requirements: list[str]


class StaticComplianceChecker:
    """Check that code uses the target framework (not bypassing it).

    Performs static analysis to verify:
    1. Framework is imported
    2. At least one framework-specific symbol is used
    """

    def __init__(
        self,
        framework: str,
        required_imports: list[str] | None = None,
        allowlist_symbols: list[str] | None = None,
    ):
        """Initialize checker.

        Args:
            framework: Framework package name.
            required_imports: Import paths that must be present.
            allowlist_symbols: Framework symbols that must be used (at least one).
        """
        self.framework = framework
        self.package_name = framework.replace("-", "_")
        self.required_imports = required_imports or [self.package_name]
        self.allowlist_symbols = allowlist_symbols or []

    def check(self, code: str) -> ComplianceResult:
        """Check code for framework compliance.

        Args:
            code: Python source code to check.

        Returns:
            ComplianceResult with details of the check.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ComplianceResult(
                passed=False,
                has_import=False,
                has_framework_symbols=False,
                imported_modules=[],
                used_symbols=[],
                missing_requirements=["Code has syntax errors"],
            )

        # Find all imports
        imported_modules = self._find_imports(tree)

        # Check for required imports
        has_import = any(
            imp.startswith(self.package_name) for imp in imported_modules
        )

        # Find used symbols
        used_symbols = self._find_used_symbols(tree, imported_modules)

        # Check for framework symbols
        has_framework_symbols = self._has_framework_symbols(used_symbols)

        # Determine what's missing
        missing = []
        if not has_import:
            missing.append(f"Must import from {self.package_name}")
        if not has_framework_symbols and self.allowlist_symbols:
            missing.append(f"Must use at least one of: {self.allowlist_symbols}")

        passed = has_import and (has_framework_symbols or not self.allowlist_symbols)

        return ComplianceResult(
            passed=passed,
            has_import=has_import,
            has_framework_symbols=has_framework_symbols,
            imported_modules=imported_modules,
            used_symbols=used_symbols,
            missing_requirements=missing,
        )

    def _find_imports(self, tree: ast.AST) -> list[str]:
        """Find all imported module names."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    # Also track specific imports
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        return imports

    def _find_used_symbols(
        self,
        tree: ast.AST,
        imported_modules: list[str],
    ) -> list[str]:
        """Find symbols used from framework modules."""
        used = []

        # Build mapping of aliases to modules
        alias_map = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    alias_map[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname or alias.name
                        alias_map[name] = f"{node.module}.{alias.name}"

        # Find all Name and Attribute nodes that reference framework
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in alias_map:
                    module = alias_map[node.id]
                    if module.startswith(self.package_name):
                        used.append(node.id)
            elif isinstance(node, ast.Attribute):
                chain = self._get_attribute_chain(node)
                first = chain.split(".")[0]
                if first in alias_map:
                    module = alias_map[first]
                    if module.startswith(self.package_name):
                        used.append(chain)

        return list(set(used))

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get full attribute chain as string."""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))

    def _has_framework_symbols(self, used_symbols: list[str]) -> bool:
        """Check if any allowlisted symbols are used."""
        if not self.allowlist_symbols:
            # No allowlist means any framework import is sufficient
            return len(used_symbols) > 0

        for symbol in used_symbols:
            for allowed in self.allowlist_symbols:
                if allowed in symbol:
                    return True

        return False


# Framework-specific checker configurations
FRAMEWORK_CHECKERS = {
    "pydantic-ai": StaticComplianceChecker(
        framework="pydantic-ai",
        required_imports=["pydantic_ai"],
        allowlist_symbols=["Agent", "RunContext", "Tool", "ModelRetry"],
    ),
    "haystack": StaticComplianceChecker(
        framework="haystack",
        required_imports=["haystack"],
        allowlist_symbols=["Pipeline", "Component", "Document"],
    ),
    "langgraph": StaticComplianceChecker(
        framework="langgraph",
        required_imports=["langgraph"],
        allowlist_symbols=["StateGraph", "Graph", "END", "START"],
    ),
    "openai-agents": StaticComplianceChecker(
        framework="openai-agents",
        required_imports=["openai_agents", "agents"],
        allowlist_symbols=["Agent", "Runner", "function_tool"],
    ),
    "autogen": StaticComplianceChecker(
        framework="autogen",
        required_imports=["autogen"],
        allowlist_symbols=["AssistantAgent", "UserProxyAgent", "ConversableAgent"],
    ),
    "direct-api": StaticComplianceChecker(
        framework="direct-api",
        required_imports=[],  # No framework required
        allowlist_symbols=[],  # Any code is fine
    ),
}


def get_checker(framework: str) -> StaticComplianceChecker:
    """Get the appropriate compliance checker for a framework."""
    if framework in FRAMEWORK_CHECKERS:
        return FRAMEWORK_CHECKERS[framework]

    # Default checker for unknown frameworks
    return StaticComplianceChecker(framework=framework)
