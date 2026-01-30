"""Metrics computation for learnability benchmark."""

from dataclasses import dataclass, field
from typing import Any

from ..harness.conversation import TrialResult, TurnOutcome


@dataclass
class TrialMetrics:
    """Computed metrics for a single trial."""

    # Primary metrics
    turns_to_success: int | None  # None if failed
    success: bool
    first_attempt_success: bool
    total_tokens: int

    # Error metrics
    error_types: dict[str, int] = field(default_factory=dict)
    meaningful_error_rate: float = 0.0
    self_correction_ratio: float = 0.0

    # Hallucination metrics
    hallucination_rate: float = 0.0
    invented_rate: float = 0.0
    version_conflict_rate: float = 0.0

    # Compliance
    compliance_pass: bool = True
    framework_bypass: bool = False

    # Hidden test performance
    hidden_set_pass: bool = False
    hidden_set_score: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple trials."""

    # Sample size
    n_trials: int = 0

    # Success metrics
    success_rate: float = 0.0
    first_attempt_success_rate: float = 0.0
    median_turns_to_success: float | None = None

    # Error patterns
    error_category_distribution: dict[str, float] = field(default_factory=dict)
    mean_meaningful_error_rate: float = 0.0
    mean_self_correction_ratio: float = 0.0

    # Hallucination
    mean_hallucination_rate: float = 0.0
    mean_invented_rate: float = 0.0
    mean_version_conflict_rate: float = 0.0

    # Cost
    mean_tokens: float = 0.0
    total_tokens: int = 0

    # Compliance
    bypass_rate: float = 0.0

    # Hidden test
    hidden_set_success_rate: float = 0.0
    mean_hidden_set_score: float = 0.0


class MetricsCalculator:
    """Calculate metrics from trial results."""

    def compute_trial_metrics(
        self,
        result: TrialResult,
        error_analyses: list[Any] | None = None,
        hallucination_result: Any | None = None,
    ) -> TrialMetrics:
        """Compute metrics for a single trial.

        Args:
            result: The trial result.
            error_analyses: List of NormalizedError objects for each turn.
            hallucination_result: HallucinationResult from final code.

        Returns:
            TrialMetrics for the trial.
        """
        success = result.outcome == "success"
        turns_to_success = result.final_turn if success else None

        # First attempt success
        first_success = (
            len(result.turns) > 0 and result.turns[0].outcome == TurnOutcome.SUCCESS
        )

        # Count error types
        error_types: dict[str, int] = {}
        for turn in result.turns:
            if turn.error_type:
                error_types[turn.error_type] = error_types.get(turn.error_type, 0) + 1

        # Meaningful error rate (for turns 1-2)
        early_errors = [t for t in result.turns[:2] if t.outcome != TurnOutcome.SUCCESS]
        meaningful_rate = 0.0
        if error_analyses and early_errors:
            actionable = sum(1 for e in error_analyses[:2] if e and e.is_actionable)
            meaningful_rate = actionable / len(early_errors) if early_errors else 0.0

        # Self-correction ratio
        correction_ratio = self._compute_self_correction_ratio(result)

        # Hallucination metrics
        halluc_rate = 0.0
        invented = 0.0
        version_conflict = 0.0
        if hallucination_result:
            halluc_rate = hallucination_result.hallucination_rate
            invented = hallucination_result.invented_rate
            version_conflict = hallucination_result.version_conflict_rate

        return TrialMetrics(
            turns_to_success=turns_to_success,
            success=success,
            first_attempt_success=first_success,
            total_tokens=result.total_tokens,
            error_types=error_types,
            meaningful_error_rate=meaningful_rate,
            self_correction_ratio=correction_ratio,
            hallucination_rate=halluc_rate,
            invented_rate=invented,
            version_conflict_rate=version_conflict,
            compliance_pass=result.compliance_check_pass,
            framework_bypass=not result.compliance_check_pass,
            hidden_set_pass=result.hidden_set_pass,
            hidden_set_score=result.hidden_set_score,
        )

    def compute_aggregate_metrics(
        self,
        trial_metrics: list[TrialMetrics],
    ) -> AggregateMetrics:
        """Compute aggregate metrics across multiple trials.

        Args:
            trial_metrics: List of trial metrics.

        Returns:
            AggregateMetrics across all trials.
        """
        if not trial_metrics:
            return AggregateMetrics()

        n = len(trial_metrics)

        # Success metrics
        successes = [m for m in trial_metrics if m.success]
        success_rate = len(successes) / n
        first_success_rate = sum(1 for m in trial_metrics if m.first_attempt_success) / n

        # Median turns (among successes only - per survival analysis methodology)
        turns = sorted([m.turns_to_success for m in successes if m.turns_to_success])
        median_turns = None
        if turns:
            mid = len(turns) // 2
            median_turns = turns[mid] if len(turns) % 2 else (turns[mid - 1] + turns[mid]) / 2

        # Error distribution
        all_errors: dict[str, int] = {}
        for m in trial_metrics:
            for err_type, count in m.error_types.items():
                all_errors[err_type] = all_errors.get(err_type, 0) + count
        total_errors = sum(all_errors.values())
        error_dist = {k: v / total_errors for k, v in all_errors.items()} if total_errors else {}

        # Mean metrics
        mean_meaningful = sum(m.meaningful_error_rate for m in trial_metrics) / n
        mean_correction = sum(m.self_correction_ratio for m in trial_metrics) / n
        mean_halluc = sum(m.hallucination_rate for m in trial_metrics) / n
        mean_invented = sum(m.invented_rate for m in trial_metrics) / n
        mean_version = sum(m.version_conflict_rate for m in trial_metrics) / n

        # Token metrics
        total_tokens = sum(m.total_tokens for m in trial_metrics)
        mean_tokens = total_tokens / n

        # Compliance
        bypass_rate = sum(1 for m in trial_metrics if m.framework_bypass) / n

        # Hidden test
        hidden_success = sum(1 for m in trial_metrics if m.hidden_set_pass) / n
        mean_hidden = sum(m.hidden_set_score for m in trial_metrics) / n

        return AggregateMetrics(
            n_trials=n,
            success_rate=success_rate,
            first_attempt_success_rate=first_success_rate,
            median_turns_to_success=median_turns,
            error_category_distribution=error_dist,
            mean_meaningful_error_rate=mean_meaningful,
            mean_self_correction_ratio=mean_correction,
            mean_hallucination_rate=mean_halluc,
            mean_invented_rate=mean_invented,
            mean_version_conflict_rate=mean_version,
            mean_tokens=mean_tokens,
            total_tokens=total_tokens,
            bypass_rate=bypass_rate,
            hidden_set_success_rate=hidden_success,
            mean_hidden_set_score=mean_hidden,
        )

    def _compute_self_correction_ratio(self, result: TrialResult) -> float:
        """Compute self-correction ratio based on code changes.

        Measures whether the LLM adapts or repeats mistakes.
        """
        if len(result.turns) < 2:
            return 1.0  # No opportunity to self-correct

        # Count distinct code attempts (by hash)
        hashes = [t.code_hash for t in result.turns if t.code_hash]
        unique_hashes = set(hashes)

        if len(hashes) == 0:
            return 0.0

        return len(unique_hashes) / len(hashes)


def compute_contamination_score(
    success_at_none: float,
    symbol_exactness: float,
) -> dict[str, Any]:
    """Compute contamination interpretation from two-axis metrics.

    Args:
        success_at_none: Success rate with framework name only (no docs).
        symbol_exactness: Fraction of API calls that exactly match installed symbols.

    Returns:
        Dict with interpretation and confidence.
    """
    if success_at_none >= 0.5 and symbol_exactness >= 0.8:
        interpretation = "memorized"
        confidence = "high"
        learnability_valid = False
    elif success_at_none >= 0.5 and symbol_exactness < 0.8:
        interpretation = "guessable_api"
        confidence = "medium"
        learnability_valid = True
    else:
        interpretation = "clean_slate"
        confidence = "high"
        learnability_valid = True

    return {
        "success_at_none": success_at_none,
        "symbol_exactness": symbol_exactness,
        "interpretation": interpretation,
        "confidence": confidence,
        "learnability_metric_valid": learnability_valid,
    }
