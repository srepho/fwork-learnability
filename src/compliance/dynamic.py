"""Dynamic compliance checking - verify framework entrypoints are called at runtime."""

from dataclasses import dataclass


@dataclass
class DynamicComplianceResult:
    """Result of dynamic compliance check."""

    passed: bool
    entrypoint_called: bool
    call_count: int
    call_details: list[dict]


class DynamicComplianceChecker:
    """Verify framework entrypoints are called at runtime.

    Injects monitoring code to track whether the framework's main
    entry points (e.g., Agent.run(), Pipeline.run()) are actually invoked.
    """

    def __init__(
        self,
        framework: str,
        entrypoints: list[str] | None = None,
    ):
        """Initialize checker.

        Args:
            framework: Framework package name.
            entrypoints: List of method/function names to monitor.
        """
        self.framework = framework
        self.package_name = framework.replace("-", "_")
        self.entrypoints = entrypoints or []

    def generate_wrapper_code(self) -> str:
        """Generate code to wrap framework entrypoints for monitoring.

        Returns:
            Python code string to inject into the execution environment.
        """
        if not self.entrypoints:
            return ""

        wrapper_parts = [
            "# Dynamic compliance monitoring",
            "_compliance_calls = []",
            "",
        ]

        for ep in self.entrypoints:
            parts = ep.rsplit(".", 1)
            if len(parts) == 2:
                class_name, method_name = parts
                wrapper_parts.append(self._wrap_method(class_name, method_name))
            else:
                # Function wrapping
                wrapper_parts.append(self._wrap_function(ep))

        return "\n".join(wrapper_parts)

    def _wrap_method(self, class_name: str, method_name: str) -> str:
        """Generate wrapper for a class method."""
        return f'''
try:
    from {self.package_name} import {class_name}
    _orig_{class_name}_{method_name} = {class_name}.{method_name}
    def _wrapped_{class_name}_{method_name}(self, *args, **kwargs):
        _compliance_calls.append({{
            "entrypoint": "{class_name}.{method_name}",
            "args": repr(args)[:100],
            "kwargs": repr(kwargs)[:100],
        }})
        return _orig_{class_name}_{method_name}(self, *args, **kwargs)
    {class_name}.{method_name} = _wrapped_{class_name}_{method_name}
except ImportError:
    pass
'''

    def _wrap_function(self, func_name: str) -> str:
        """Generate wrapper for a module-level function."""
        return f'''
try:
    import {self.package_name}
    _orig_{func_name} = {self.package_name}.{func_name}
    def _wrapped_{func_name}(*args, **kwargs):
        _compliance_calls.append({{
            "entrypoint": "{func_name}",
            "args": repr(args)[:100],
            "kwargs": repr(kwargs)[:100],
        }})
        return _orig_{func_name}(*args, **kwargs)
    {self.package_name}.{func_name} = _wrapped_{func_name}
except (ImportError, AttributeError):
    pass
'''

    def generate_check_code(self) -> str:
        """Generate code to check compliance after execution.

        Returns:
            Python code that outputs compliance result.
        """
        return '''
import json
compliance_result = {
    "passed": len(_compliance_calls) > 0,
    "entrypoint_called": len(_compliance_calls) > 0,
    "call_count": len(_compliance_calls),
    "call_details": _compliance_calls,
}
print("__COMPLIANCE_START__")
print(json.dumps(compliance_result))
print("__COMPLIANCE_END__")
'''

    def parse_result(self, output: str) -> DynamicComplianceResult:
        """Parse compliance result from execution output.

        Args:
            output: Stdout from execution containing compliance markers.

        Returns:
            DynamicComplianceResult.
        """
        import json

        try:
            start = output.index("__COMPLIANCE_START__") + len("__COMPLIANCE_START__")
            end = output.index("__COMPLIANCE_END__")
            result_json = output[start:end].strip()
            data = json.loads(result_json)

            return DynamicComplianceResult(
                passed=data.get("passed", False),
                entrypoint_called=data.get("entrypoint_called", False),
                call_count=data.get("call_count", 0),
                call_details=data.get("call_details", []),
            )
        except (ValueError, json.JSONDecodeError):
            return DynamicComplianceResult(
                passed=False,
                entrypoint_called=False,
                call_count=0,
                call_details=[],
            )


# Framework-specific dynamic checkers
DYNAMIC_CHECKERS = {
    "pydantic-ai": DynamicComplianceChecker(
        framework="pydantic-ai",
        entrypoints=["Agent.run", "Agent.run_sync"],
    ),
    "haystack": DynamicComplianceChecker(
        framework="haystack",
        entrypoints=["Pipeline.run"],
    ),
    "langgraph": DynamicComplianceChecker(
        framework="langgraph",
        entrypoints=["StateGraph.compile", "Graph.invoke"],
    ),
    "openai-agents": DynamicComplianceChecker(
        framework="openai-agents",
        entrypoints=["Runner.run", "Runner.run_sync"],
    ),
}


def get_dynamic_checker(framework: str) -> DynamicComplianceChecker | None:
    """Get the dynamic compliance checker for a framework."""
    return DYNAMIC_CHECKERS.get(framework)
