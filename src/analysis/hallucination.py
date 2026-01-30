"""Hallucination detection via API surface extraction."""

import ast
import importlib
import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class HallucinationType(Enum):
    """Types of API hallucinations."""

    VALID = "valid"  # Symbol exists in current version
    INVENTED = "invented"  # Symbol never existed in any version
    VERSION_CONFLICT = "version_conflict"  # Symbol existed in old version


@dataclass
class APISymbol:
    """A symbol from a package's API surface."""

    full_name: str  # e.g., "pydantic_ai.Agent.run"
    symbol_type: str  # "class", "method", "function", "attribute"
    parent: str | None = None  # Parent class/module


@dataclass
class HallucinationResult:
    """Result of hallucination analysis for a code submission."""

    total_api_calls: int = 0
    valid_calls: int = 0
    invented_calls: int = 0
    version_conflict_calls: int = 0
    hallucinated_symbols: list[str] = field(default_factory=list)
    classifications: dict[str, HallucinationType] = field(default_factory=dict)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of API calls that were hallucinated."""
        if self.total_api_calls == 0:
            return 0.0
        return (self.invented_calls + self.version_conflict_calls) / self.total_api_calls

    @property
    def invented_rate(self) -> float:
        """Fraction of API calls that were invented."""
        if self.total_api_calls == 0:
            return 0.0
        return self.invented_calls / self.total_api_calls

    @property
    def version_conflict_rate(self) -> float:
        """Fraction of API calls with version conflicts."""
        if self.total_api_calls == 0:
            return 0.0
        return self.version_conflict_calls / self.total_api_calls


class HallucinationDetector:
    """Detect hallucinated API calls by comparing against known surfaces."""

    def __init__(
        self,
        framework: str,
        current_surface: set[str] | None = None,
        old_surfaces: dict[str, set[str]] | None = None,
    ):
        """Initialize detector.

        Args:
            framework: Framework package name.
            current_surface: Set of valid symbols in current version.
            old_surfaces: Dict mapping version strings to symbol sets.
        """
        self.framework = framework
        self.current_surface = current_surface or set()
        self.old_surfaces = old_surfaces or {}

    @classmethod
    def from_installed_package(cls, package_name: str) -> "HallucinationDetector":
        """Create detector from an installed package.

        Args:
            package_name: Name of the installed package.

        Returns:
            HallucinationDetector with current API surface extracted.
        """
        surface = extract_api_surface(package_name)
        return cls(framework=package_name, current_surface=surface)

    def analyze_code(self, code: str) -> HallucinationResult:
        """Analyze code for hallucinated API calls.

        Args:
            code: Python source code to analyze.

        Returns:
            HallucinationResult with classification of all API calls.
        """
        # Extract API calls from code
        api_calls = self._extract_api_calls(code)

        result = HallucinationResult(total_api_calls=len(api_calls))

        for call in api_calls:
            classification = self._classify_symbol(call)
            result.classifications[call] = classification

            if classification == HallucinationType.VALID:
                result.valid_calls += 1
            elif classification == HallucinationType.INVENTED:
                result.invented_calls += 1
                result.hallucinated_symbols.append(call)
            elif classification == HallucinationType.VERSION_CONFLICT:
                result.version_conflict_calls += 1
                result.hallucinated_symbols.append(call)

        return result

    def _extract_api_calls(self, code: str) -> list[str]:
        """Extract framework API calls from code using AST analysis."""
        calls = []
        framework_aliases = self._find_framework_aliases(code)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return calls

        for node in ast.walk(tree):
            # Function/method calls
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node.func)
                if call_name and self._is_framework_call(call_name, framework_aliases):
                    calls.append(call_name)

            # Attribute access (even without call)
            elif isinstance(node, ast.Attribute):
                attr_name = self._get_attribute_chain(node)
                if attr_name and self._is_framework_call(attr_name, framework_aliases):
                    calls.append(attr_name)

        return list(set(calls))  # Deduplicate

    def _find_framework_aliases(self, code: str) -> set[str]:
        """Find import aliases for the framework."""
        aliases = {self.framework, self.framework.replace("-", "_")}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return aliases

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(self.framework.replace("-", "_")):
                        aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(self.framework.replace("-", "_")):
                    for alias in node.names:
                        aliases.add(alias.asname or alias.name)

        return aliases

    def _get_call_name(self, node: ast.expr) -> str | None:
        """Get the full name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_chain(node)
        return None

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get the full attribute chain (e.g., 'module.Class.method')."""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))

    def _is_framework_call(self, name: str, aliases: set[str]) -> bool:
        """Check if a call is to the framework."""
        first_part = name.split(".")[0]
        return first_part in aliases

    def _classify_symbol(self, symbol: str) -> HallucinationType:
        """Classify a symbol as valid, invented, or version conflict."""
        # Normalize the symbol name
        normalized = self._normalize_symbol(symbol)

        # Check current version
        if self._symbol_exists(normalized, self.current_surface):
            return HallucinationType.VALID

        # Check old versions
        for version, surface in self.old_surfaces.items():
            if self._symbol_exists(normalized, surface):
                return HallucinationType.VERSION_CONFLICT

        return HallucinationType.INVENTED

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison."""
        # Remove framework prefix variations
        for prefix in [self.framework, self.framework.replace("-", "_")]:
            if symbol.startswith(prefix + "."):
                symbol = symbol[len(prefix) + 1:]
                break
        return symbol

    def _symbol_exists(self, symbol: str, surface: set[str]) -> bool:
        """Check if a symbol exists in an API surface."""
        # Direct match
        if symbol in surface:
            return True

        # Check partial matches (for method calls on instances)
        parts = symbol.split(".")
        for i in range(len(parts)):
            partial = ".".join(parts[i:])
            if partial in surface:
                return True

        return False


def extract_api_surface(package_name: str, max_depth: int = 3) -> set[str]:
    """Extract all public symbols from a package.

    Args:
        package_name: Name of the package to analyze.
        max_depth: Maximum recursion depth for submodules.

    Returns:
        Set of symbol names (e.g., "Agent", "Agent.run", "run_sync").
    """
    symbols = set()

    try:
        module = importlib.import_module(package_name.replace("-", "_"))
    except ImportError:
        return symbols

    _extract_from_module(module, "", symbols, max_depth, 0)
    return symbols


def _extract_from_module(
    module,
    prefix: str,
    symbols: set[str],
    max_depth: int,
    current_depth: int,
):
    """Recursively extract symbols from a module."""
    if current_depth > max_depth:
        return

    try:
        members = inspect.getmembers(module)
    except Exception:
        return

    for name, obj in members:
        if name.startswith("_"):
            continue

        full_name = f"{prefix}.{name}" if prefix else name
        symbols.add(full_name)

        # Recurse into classes
        if inspect.isclass(obj):
            try:
                for method_name, _ in inspect.getmembers(obj):
                    if not method_name.startswith("_"):
                        symbols.add(f"{full_name}.{method_name}")
            except Exception:
                pass

        # Recurse into submodules
        elif inspect.ismodule(obj) and hasattr(obj, "__package__"):
            _extract_from_module(obj, full_name, symbols, max_depth, current_depth + 1)
