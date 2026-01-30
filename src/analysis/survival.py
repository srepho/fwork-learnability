"""Survival analysis for turns-to-success metric."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SurvivalData:
    """Data prepared for survival analysis."""

    time: np.ndarray  # Turn number
    event: np.ndarray  # 1 if success, 0 if censored (failure/max turns)
    group: np.ndarray | None = None  # Optional grouping variable


@dataclass
class SurvivalCurve:
    """Kaplan-Meier survival curve results."""

    times: np.ndarray  # Time points
    survival_prob: np.ndarray  # P(success by time t)
    confidence_lower: np.ndarray  # Lower confidence bound
    confidence_upper: np.ndarray  # Upper confidence bound
    n_at_risk: np.ndarray  # Number at risk at each time
    median_time: float | None  # Median time to success


class SurvivalAnalyzer:
    """Survival analysis for turns-to-success data.

    Treats failures (max turns reached) as right-censored observations.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def prepare_data(
        self,
        trials: list[dict],
        group_by: str | None = None,
    ) -> SurvivalData:
        """Prepare trial results for survival analysis.

        Args:
            trials: List of trial result dicts with 'outcome' and 'final_turn'.
            group_by: Optional field to group by (e.g., 'framework', 'doc_level').

        Returns:
            SurvivalData ready for analysis.
        """
        times = []
        events = []
        groups = []

        for trial in trials:
            turn = trial.get("final_turn", self.max_turns)
            success = trial.get("outcome") == "success"

            times.append(turn)
            events.append(1 if success else 0)  # 1 = event, 0 = censored

            if group_by:
                groups.append(trial.get(group_by, "unknown"))

        return SurvivalData(
            time=np.array(times),
            event=np.array(events),
            group=np.array(groups) if group_by else None,
        )

    def kaplan_meier(
        self,
        data: SurvivalData,
        confidence: float = 0.95,
    ) -> SurvivalCurve:
        """Compute Kaplan-Meier survival curve.

        Note: This is actually computing P(success by turn t), not P(survival).
        We invert the traditional survival interpretation for clarity.

        Args:
            data: Prepared survival data.
            confidence: Confidence level for intervals.

        Returns:
            SurvivalCurve with probabilities and confidence bounds.
        """
        try:
            from lifelines import KaplanMeierFitter

            kmf = KaplanMeierFitter()
            kmf.fit(data.time, event_observed=data.event)

            # Get survival function (P(not succeeded by t))
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_survival_function_

            # Convert to P(success by t) = 1 - P(not succeeded by t)
            success_prob = 1 - sf.values.flatten()
            ci_lower = 1 - ci.iloc[:, 1].values  # Swap bounds
            ci_upper = 1 - ci.iloc[:, 0].values

            return SurvivalCurve(
                times=sf.index.values,
                survival_prob=success_prob,
                confidence_lower=ci_lower,
                confidence_upper=ci_upper,
                n_at_risk=kmf.event_table["at_risk"].values,
                median_time=kmf.median_survival_time_,
            )

        except ImportError:
            # Fallback: simple empirical calculation
            return self._empirical_curve(data)

    def _empirical_curve(self, data: SurvivalData) -> SurvivalCurve:
        """Simple empirical success curve without lifelines."""
        times = sorted(set(data.time))
        success_by_t = []
        n_at_risk = []

        total = len(data.time)

        for t in times:
            # Count successes by time t
            successes = np.sum((data.time <= t) & (data.event == 1))
            success_by_t.append(successes / total)
            n_at_risk.append(np.sum(data.time >= t))

        return SurvivalCurve(
            times=np.array(times),
            survival_prob=np.array(success_by_t),
            confidence_lower=np.zeros(len(times)),  # No CI without lifelines
            confidence_upper=np.ones(len(times)),
            n_at_risk=np.array(n_at_risk),
            median_time=self._compute_median(times, success_by_t),
        )

    def _compute_median(
        self,
        times: list,
        success_prob: list,
    ) -> float | None:
        """Find median time (first time when P(success) >= 0.5)."""
        for t, p in zip(times, success_prob):
            if p >= 0.5:
                return t
        return None

    def compare_groups(
        self,
        data: SurvivalData,
    ) -> dict:
        """Compare survival curves between groups using log-rank test.

        Args:
            data: Survival data with group assignments.

        Returns:
            Dict with test statistic, p-value, and per-group curves.
        """
        if data.group is None:
            raise ValueError("Data must have group assignments for comparison")

        try:
            from lifelines.statistics import logrank_test

            groups = np.unique(data.group)
            results = {"groups": {}}

            # Compute curve for each group
            for group in groups:
                mask = data.group == group
                group_data = SurvivalData(
                    time=data.time[mask],
                    event=data.event[mask],
                )
                results["groups"][group] = self.kaplan_meier(group_data)

            # Pairwise comparisons
            if len(groups) == 2:
                g1, g2 = groups
                mask1 = data.group == g1
                mask2 = data.group == g2

                lr = logrank_test(
                    data.time[mask1],
                    data.time[mask2],
                    event_observed_A=data.event[mask1],
                    event_observed_B=data.event[mask2],
                )
                results["logrank_statistic"] = lr.test_statistic
                results["logrank_pvalue"] = lr.p_value

            return results

        except ImportError:
            # Fallback without lifelines
            groups = np.unique(data.group)
            results = {"groups": {}}

            for group in groups:
                mask = data.group == group
                group_data = SurvivalData(
                    time=data.time[mask],
                    event=data.event[mask],
                )
                results["groups"][group] = self._empirical_curve(group_data)

            return results
