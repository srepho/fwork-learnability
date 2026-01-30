"""Conversation loop implementing "Fresh Start with Error" context management."""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from ..llm.base import LLMClient, LLMResponse
from .extractor import CodeExtractor, ExtractedCode
from .sandbox import ExecutionResult, ExecutionStatus, Sandbox


class TurnOutcome(Enum):
    """Outcome of a single turn."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    EXTRACTION_FAILED = "extraction_failed"


@dataclass
class Turn:
    """Record of a single conversation turn."""

    turn_number: int
    outcome: TurnOutcome
    code: str | None = None
    code_hash: str | None = None
    code_loc: int = 0
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    llm_response: str | None = None
    tokens_generated: int = 0
    execution_time_ms: float = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.code and not self.code_hash:
            self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]
        if self.code and not self.code_loc:
            self.code_loc = len(self.code.splitlines())


@dataclass
class TrialResult:
    """Result of a complete trial (up to max_turns)."""

    trial_id: str
    framework: str
    doc_level: str
    task_tier: int
    model: str
    turns: list[Turn] = field(default_factory=list)
    outcome: str = "incomplete"  # success, failure, max_turns_reached
    final_turn: int = 0
    dev_set_pass: bool = False
    hidden_set_pass: bool = False
    hidden_set_score: float = 0.0
    compliance_check_pass: bool = True
    total_tokens: int = 0
    start_time: str = ""
    end_time: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.utcnow().isoformat()


class ConversationLoop:
    """Implement the "Turns to Working Code" loop with Fresh Start context management."""

    def __init__(
        self,
        llm_client: LLMClient,
        sandbox: Sandbox,
        max_turns: int = 10,
        extractor: CodeExtractor | None = None,
    ):
        self.llm_client = llm_client
        self.sandbox = sandbox
        self.max_turns = max_turns
        self.extractor = extractor or CodeExtractor()

    def run(
        self,
        trial_id: str,
        task_description: str,
        documentation: str,
        test_code: str,
        framework: str,
        doc_level: str,
        task_tier: int,
        compliance_checker: Callable[[str], bool] | None = None,
        hidden_test_runner: Callable[[str], tuple[bool, float]] | None = None,
    ) -> TrialResult:
        """Run the conversation loop for a single trial.

        Uses "Fresh Start with Error" context management:
        - Each turn sees only: task + docs + previous code + previous error
        - Does NOT accumulate history of failed attempts

        Args:
            trial_id: Unique identifier for this trial.
            task_description: The task to implement.
            documentation: Framework documentation at specified level.
            test_code: Development test code for error feedback.
            framework: Name of the framework being tested.
            doc_level: Documentation level (none, minimal, moderate, full).
            task_tier: Task tier (1, 2, or 3).
            compliance_checker: Optional function to verify framework usage.
            hidden_test_runner: Optional function to run hidden tests, returns (pass, score).

        Returns:
            TrialResult with all turn records and final outcome.
        """
        result = TrialResult(
            trial_id=trial_id,
            framework=framework,
            doc_level=doc_level,
            task_tier=task_tier,
            model=self.llm_client.name,
        )

        previous_code: str | None = None
        previous_error: str | None = None

        for turn_num in range(1, self.max_turns + 1):
            # Build prompt using Fresh Start with Error strategy
            prompt = self._build_prompt(
                task_description=task_description,
                documentation=documentation,
                previous_code=previous_code,
                previous_error=previous_error,
                turn_number=turn_num,
            )

            # Get LLM response
            llm_response = self.llm_client.generate(prompt)

            # Extract code
            extracted = self.extractor.extract_python(llm_response.content)

            if not extracted:
                turn = Turn(
                    turn_number=turn_num,
                    outcome=TurnOutcome.EXTRACTION_FAILED,
                    llm_response=llm_response.content,
                    tokens_generated=llm_response.completion_tokens,
                    error_message="No Python code block found in response",
                )
                result.turns.append(turn)
                result.total_tokens += llm_response.total_tokens
                previous_error = "Your response did not contain a Python code block. Please provide your implementation in a ```python code block."
                continue

            if not extracted.is_valid_syntax:
                turn = Turn(
                    turn_number=turn_num,
                    outcome=TurnOutcome.SYNTAX_ERROR,
                    code=extracted.code,
                    llm_response=llm_response.content,
                    tokens_generated=llm_response.completion_tokens,
                    error_type="SyntaxError",
                    error_message=extracted.syntax_error,
                )
                result.turns.append(turn)
                result.total_tokens += llm_response.total_tokens
                previous_code = extracted.code
                previous_error = f"SyntaxError: {extracted.syntax_error}"
                continue

            # Check compliance if checker provided
            if compliance_checker and not compliance_checker(extracted.code):
                result.compliance_check_pass = False

            # Execute code
            exec_result = self.sandbox.execute(extracted.code, test_code)

            turn = self._create_turn(turn_num, extracted, exec_result, llm_response)
            result.turns.append(turn)
            result.total_tokens += llm_response.total_tokens

            if exec_result.status == ExecutionStatus.SUCCESS:
                result.outcome = "success"
                result.final_turn = turn_num
                result.dev_set_pass = True

                # Run hidden tests if available
                if hidden_test_runner:
                    hidden_pass, hidden_score = hidden_test_runner(extracted.code)
                    result.hidden_set_pass = hidden_pass
                    result.hidden_set_score = hidden_score

                break

            # Prepare for next turn
            previous_code = extracted.code
            previous_error = self._format_error(exec_result)

        else:
            # Max turns reached without success
            result.outcome = "max_turns_reached"
            result.final_turn = self.max_turns

        result.end_time = datetime.utcnow().isoformat()
        return result

    def _build_prompt(
        self,
        task_description: str,
        documentation: str,
        previous_code: str | None,
        previous_error: str | None,
        turn_number: int,
    ) -> str:
        """Build the prompt for a turn using Fresh Start with Error strategy."""
        parts = []

        # Always include task
        parts.append("# Task\n")
        parts.append(task_description)
        parts.append("\n\n")

        # Include documentation if provided
        if documentation:
            parts.append("# Framework Documentation\n")
            parts.append(documentation)
            parts.append("\n\n")

        # Include previous attempt and error (only from turn N-1)
        if previous_code and previous_error:
            parts.append("# Your Previous Attempt\n")
            parts.append("```python\n")
            parts.append(previous_code)
            parts.append("\n```\n\n")
            parts.append("# Error from Previous Attempt\n")
            parts.append("```\n")
            parts.append(previous_error)
            parts.append("\n```\n\n")
            parts.append("Please fix the error and provide a corrected implementation.\n")
        elif turn_number == 1:
            parts.append("Please implement the solution in a Python code block.\n")

        return "".join(parts)

    def _create_turn(
        self,
        turn_number: int,
        extracted: ExtractedCode,
        exec_result: ExecutionResult,
        llm_response: LLMResponse,
    ) -> Turn:
        """Create a Turn record from execution results."""
        if exec_result.status == ExecutionStatus.SUCCESS:
            outcome = TurnOutcome.SUCCESS
        elif exec_result.status == ExecutionStatus.TIMEOUT:
            outcome = TurnOutcome.TIMEOUT
        elif exec_result.status == ExecutionStatus.SYNTAX_ERROR:
            outcome = TurnOutcome.SYNTAX_ERROR
        else:
            outcome = TurnOutcome.ERROR

        return Turn(
            turn_number=turn_number,
            outcome=outcome,
            code=extracted.code,
            llm_response=llm_response.content,
            tokens_generated=llm_response.completion_tokens,
            error_type=exec_result.exception_type,
            error_message=exec_result.exception_message,
            error_traceback=exec_result.traceback,
            execution_time_ms=exec_result.execution_time_ms,
        )

    def _format_error(self, exec_result: ExecutionResult) -> str:
        """Format execution error for feedback."""
        parts = []

        if exec_result.exception_type:
            parts.append(f"{exec_result.exception_type}: {exec_result.exception_message}")

        if exec_result.traceback:
            parts.append("\nTraceback:")
            parts.append(exec_result.traceback)

        if exec_result.stderr:
            parts.append("\nStderr:")
            parts.append(exec_result.stderr)

        return "\n".join(parts) if parts else "Unknown error occurred"
