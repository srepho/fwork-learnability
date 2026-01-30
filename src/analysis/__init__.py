"""Analysis and metrics modules."""

from .errors import ErrorCategory, ErrorAnalyzer, NormalizedError
from .hallucination import HallucinationType, HallucinationDetector
from .metrics import MetricsCalculator, TrialMetrics
from .contamination import (
    ContaminationDetector,
    ContaminationLevel,
    ContaminationResult,
    VersionAlignment,
    DocContradictionTest,
    compute_enhanced_contamination_score,
)

__all__ = [
    "ErrorCategory",
    "ErrorAnalyzer",
    "NormalizedError",
    "HallucinationType",
    "HallucinationDetector",
    "MetricsCalculator",
    "TrialMetrics",
    "ContaminationDetector",
    "ContaminationLevel",
    "ContaminationResult",
    "VersionAlignment",
    "DocContradictionTest",
    "compute_enhanced_contamination_score",
]
