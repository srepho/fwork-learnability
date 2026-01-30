"""Isolated code execution sandbox."""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExecutionStatus(Enum):
    """Status of code execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exception_type: str | None = None
    exception_message: str | None = None
    traceback: str | None = None
    return_value: dict | None = None
    execution_time_ms: float = 0
    exit_code: int = 0


@dataclass
class Sandbox:
    """Isolated execution environment for framework code.

    Each sandbox uses a separate Python environment (venv) with only
    the required framework dependencies installed.
    """

    framework: str
    venv_path: Path
    timeout_seconds: int = 60
    working_dir: Path | None = None

    def __post_init__(self):
        if self.working_dir is None:
            self.working_dir = Path(tempfile.mkdtemp(prefix=f"sandbox_{self.framework}_"))

    @property
    def python_executable(self) -> Path:
        """Path to the Python executable in the venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    def execute(
        self,
        code: str,
        test_code: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute code in the sandbox.

        Args:
            code: The solution code to execute.
            test_code: Optional test code to run after the solution.
            env_vars: Additional environment variables.

        Returns:
            ExecutionResult with execution details.
        """
        # Write the solution code
        solution_file = self.working_dir / "solution.py"
        solution_file.write_text(code)

        # Build the runner script
        runner_code = self._build_runner(test_code)
        runner_file = self.working_dir / "_runner.py"
        runner_file.write_text(runner_code)

        # Set up environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Execute in subprocess
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                [str(self.python_executable), str(runner_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.working_dir),
                env=env,
            )
            execution_time_ms = (time.time() - start_time) * 1000

            # Parse the structured output
            return self._parse_result(result, execution_time_ms)

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Execution timed out after {self.timeout_seconds} seconds",
                execution_time_ms=self.timeout_seconds * 1000,
            )

    def _build_runner(self, test_code: str | None) -> str:
        """Build the runner script that wraps execution."""
        return f'''
import json
import sys
import traceback

def main():
    result = {{
        "status": "success",
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "return_value": None,
    }}

    try:
        # Import and run the solution
        import solution

        # Run tests if provided
        {"exec(" + repr(test_code) + ")" if test_code else "pass"}

        result["return_value"] = {{"tests_passed": True}}

    except SyntaxError as e:
        result["status"] = "syntax_error"
        result["exception_type"] = "SyntaxError"
        result["exception_message"] = str(e)
        result["traceback"] = traceback.format_exc()

    except Exception as e:
        result["status"] = "error"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["traceback"] = traceback.format_exc()

    # Output structured result
    print("__RESULT_START__")
    print(json.dumps(result))
    print("__RESULT_END__")

if __name__ == "__main__":
    main()
'''

    def _parse_result(self, proc_result: subprocess.CompletedProcess, execution_time_ms: float) -> ExecutionResult:
        """Parse the structured output from the runner."""
        stdout = proc_result.stdout
        stderr = proc_result.stderr

        # Extract structured result
        try:
            start_marker = "__RESULT_START__"
            end_marker = "__RESULT_END__"

            if start_marker in stdout and end_marker in stdout:
                start = stdout.index(start_marker) + len(start_marker)
                end = stdout.index(end_marker)
                result_json = stdout[start:end].strip()
                result_data = json.loads(result_json)

                # Clean stdout (remove result markers)
                clean_stdout = stdout[:stdout.index(start_marker)].strip()

                status = ExecutionStatus(result_data["status"])
                return ExecutionResult(
                    status=status,
                    stdout=clean_stdout,
                    stderr=stderr,
                    exception_type=result_data.get("exception_type"),
                    exception_message=result_data.get("exception_message"),
                    traceback=result_data.get("traceback"),
                    return_value=result_data.get("return_value"),
                    execution_time_ms=execution_time_ms,
                    exit_code=proc_result.returncode,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse as unstructured output
        if proc_result.returncode == 0:
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time_ms,
                exit_code=0,
            )
        else:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time_ms,
                exit_code=proc_result.returncode,
            )

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        if self.working_dir and self.working_dir.exists():
            shutil.rmtree(self.working_dir)


class SandboxManager:
    """Manage sandboxes for different frameworks."""

    def __init__(self, venvs_base_dir: Path):
        self.venvs_base_dir = venvs_base_dir
        self._sandboxes: dict[str, Sandbox] = {}

    def get_sandbox(self, framework: str, timeout_seconds: int = 60) -> Sandbox:
        """Get or create a sandbox for the given framework."""
        venv_path = self.venvs_base_dir / framework

        if not venv_path.exists():
            raise ValueError(
                f"Virtual environment for '{framework}' not found at {venv_path}. "
                "Run setup script to create framework environments."
            )

        sandbox = Sandbox(
            framework=framework,
            venv_path=venv_path,
            timeout_seconds=timeout_seconds,
        )
        self._sandboxes[framework] = sandbox
        return sandbox

    def cleanup_all(self):
        """Clean up all sandbox working directories."""
        for sandbox in self._sandboxes.values():
            sandbox.cleanup()
