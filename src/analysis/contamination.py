"""Contamination detection for LLM framework learnability.

This module provides clever approaches to detect whether an LLM is using
training data knowledge vs actually learning from provided documentation.

Key Approaches:
1. Version Alignment - Which framework version does generated code match?
2. Doc Adherence - Does the model follow modified docs or use training knowledge?
3. Code Fingerprinting - Compare code structure across doc levels
4. Symbol Recency - Are the APIs used from before or after training cutoff?
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum


class ContaminationLevel(Enum):
    """Interpretation of contamination evidence."""

    HIGH = "high"  # Strong evidence of training data usage
    MEDIUM = "medium"  # Some evidence, but ambiguous
    LOW = "low"  # Little to no evidence of contamination
    UNKNOWN = "unknown"  # Insufficient data to determine


@dataclass
class VersionAlignment:
    """Result of version alignment analysis."""

    framework: str
    detected_version: str | None = None  # Best matching version
    version_scores: dict[str, float] = field(default_factory=dict)
    v1_symbols_used: list[str] = field(default_factory=list)
    v2_symbols_used: list[str] = field(default_factory=list)
    alignment_confidence: float = 0.0

    @property
    def uses_old_api(self) -> bool:
        """True if code predominantly uses older API."""
        v1_score = self.version_scores.get("v1", 0)
        v2_score = self.version_scores.get("v2", 0)
        return v1_score > v2_score and v1_score > 0.5


@dataclass
class DocAdherenceResult:
    """Result of documentation adherence test."""

    # Did model use renamed API from modified docs?
    used_modified_api: bool = False
    # Did model use real API despite modified docs?
    used_real_api: bool = False
    # Count of times each was used
    modified_api_count: int = 0
    real_api_count: int = 0

    @property
    def adherence_score(self) -> float:
        """Score from 0 (ignores docs) to 1 (follows docs perfectly)."""
        total = self.modified_api_count + self.real_api_count
        if total == 0:
            return 0.5  # No data
        return self.modified_api_count / total


@dataclass
class CodeFingerprint:
    """Structural fingerprint of generated code."""

    # Structural features
    num_imports: int = 0
    num_functions: int = 0
    num_classes: int = 0
    num_method_calls: int = 0
    avg_function_length: float = 0.0

    # Framework-specific
    framework_symbols: list[str] = field(default_factory=list)
    api_call_sequence: list[str] = field(default_factory=list)

    # Hash for quick comparison
    structure_hash: str = ""

    def similarity_to(self, other: "CodeFingerprint") -> float:
        """Compute similarity to another fingerprint (0-1)."""
        # Compare symbol sequences
        if not self.api_call_sequence or not other.api_call_sequence:
            return 0.0

        matcher = SequenceMatcher(
            None,
            self.api_call_sequence,
            other.api_call_sequence
        )
        return matcher.ratio()


@dataclass
class ContaminationResult:
    """Comprehensive contamination analysis result."""

    framework: str
    training_cutoff: str  # e.g., "2024-07"

    # Version alignment
    version_alignment: VersionAlignment | None = None

    # Doc adherence (if tested)
    doc_adherence: DocAdherenceResult | None = None

    # Code fingerprints across doc levels
    fingerprints_by_level: dict[str, CodeFingerprint] = field(default_factory=dict)

    # Cross-level similarity
    none_vs_full_similarity: float = 0.0
    none_vs_minimal_similarity: float = 0.0

    # Overall interpretation
    contamination_level: ContaminationLevel = ContaminationLevel.UNKNOWN
    evidence: list[str] = field(default_factory=list)

    def compute_overall_score(self) -> float:
        """Compute overall contamination score (0-1, higher = more contaminated)."""
        scores = []

        # Version alignment evidence
        if self.version_alignment and self.version_alignment.uses_old_api:
            scores.append(0.8)

        # Doc adherence evidence
        if self.doc_adherence:
            # Low adherence = high contamination
            scores.append(1.0 - self.doc_adherence.adherence_score)

        # Code similarity across doc levels
        if self.none_vs_full_similarity > 0.8:
            scores.append(0.9)  # Very similar = contaminated
        elif self.none_vs_full_similarity > 0.5:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.0


# Known API differences between Haystack v1 and v2
HAYSTACK_V1_APIS = {
    # Deprecated in v2
    "BaseComponent",
    "BaseReader",
    "BaseRetriever",
    "BaseGenerator",
    "BaseRanker",
    "BaseSummarizer",
    "BaseTranslator",
    "BaseExtractor",
    "DocumentStore",
    "ElasticsearchDocumentStore",
    "FAISSDocumentStore",
    "InMemoryDocumentStore",  # Different location in v2
    "Pipeline.add_node",
    "Pipeline.get_node",
    "Pipeline.draw",
    "Finder",
    "Reader",
    "Retriever",
    "DensePassageRetriever",
    "EmbeddingRetriever",
    "BM25Retriever",
    "TfidfRetriever",
    # v1 style imports
    "from haystack.document_stores",
    "from haystack.nodes",
    "from haystack.pipelines import Pipeline",
}

HAYSTACK_V2_APIS = {
    # New in v2
    "Pipeline",  # Simple import
    "component",
    "@component",
    "Component",
    "PromptBuilder",
    "OpenAIGenerator",
    "HuggingFaceLocalGenerator",
    "AzureOpenAIGenerator",
    "CohereGenerator",
    "AnthropicGenerator",
    "OutputAdapter",
    "AnswerBuilder",
    "DocumentJoiner",
    "DocumentSplitter",
    "MetaFieldRanker",
    "TransformersSimilarityRanker",
    "Secret.from_token",
    "Secret.from_env_var",
    # v2 style imports
    "from haystack.components",
    "from haystack.components.builders",
    "from haystack.components.generators",
    "from haystack.components.retrievers",
    "from haystack.components.rankers",
    "haystack.Pipeline",
}

# LangGraph v0.x vs v1.x differences
LANGGRAPH_V0_APIS = {
    "StateGraph",
    "END",
    "add_edge",
    "add_node",
    "add_conditional_edges",
    "set_entry_point",
    "set_finish_point",
    "compile",
}

LANGGRAPH_V1_APIS = {
    # v1 additions
    "StateGraph",  # Still exists
    "START",  # New constant
    "END",
    "add_edge",
    "add_node",
    "add_conditional_edges",
    "Checkpointer",
    "MemorySaver",
    "SqliteSaver",
    "PostgresSaver",
    "interrupt",
    "Command",
    "Send",
}


class ContaminationDetector:
    """Detect training data contamination in LLM-generated code."""

    def __init__(
        self,
        framework: str,
        training_cutoff: str = "2024-07",
    ):
        """Initialize detector.

        Args:
            framework: Framework being tested.
            training_cutoff: Model's training data cutoff (YYYY-MM format).
        """
        self.framework = framework
        self.training_cutoff = training_cutoff
        self._v1_apis = self._get_v1_apis()
        self._v2_apis = self._get_v2_apis()

    def _get_v1_apis(self) -> set[str]:
        """Get v1 API patterns for the framework."""
        if self.framework == "haystack":
            return HAYSTACK_V1_APIS
        elif self.framework == "langgraph":
            return LANGGRAPH_V0_APIS
        return set()

    def _get_v2_apis(self) -> set[str]:
        """Get v2 API patterns for the framework."""
        if self.framework == "haystack":
            return HAYSTACK_V2_APIS
        elif self.framework == "langgraph":
            return LANGGRAPH_V1_APIS
        return set()

    def analyze_version_alignment(self, code: str) -> VersionAlignment:
        """Analyze which framework version the code aligns with.

        This detects if the model is using old APIs from training data
        despite being given new documentation.
        """
        result = VersionAlignment(framework=self.framework)

        # Count matches against each version
        v1_matches = []
        v2_matches = []

        for pattern in self._v1_apis:
            if pattern in code:
                v1_matches.append(pattern)

        for pattern in self._v2_apis:
            if pattern in code:
                v2_matches.append(pattern)

        result.v1_symbols_used = v1_matches
        result.v2_symbols_used = v2_matches

        # Compute scores
        total = len(v1_matches) + len(v2_matches)
        if total > 0:
            result.version_scores["v1"] = len(v1_matches) / total
            result.version_scores["v2"] = len(v2_matches) / total

            # Determine detected version
            if result.version_scores["v1"] > result.version_scores["v2"]:
                result.detected_version = "v1"
            else:
                result.detected_version = "v2"

            result.alignment_confidence = max(
                result.version_scores["v1"],
                result.version_scores["v2"]
            )

        return result

    def extract_fingerprint(self, code: str) -> CodeFingerprint:
        """Extract structural fingerprint from code."""
        fp = CodeFingerprint()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return fp

        # Count structural elements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                fp.num_imports += 1
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                fp.num_functions += 1
            elif isinstance(node, ast.ClassDef):
                fp.num_classes += 1
            elif isinstance(node, ast.Call):
                fp.num_method_calls += 1
                # Extract call name
                call_name = self._get_call_name(node)
                if call_name:
                    fp.api_call_sequence.append(call_name)

        # Extract framework symbols from imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if self.framework.replace("-", "_") in node.module:
                    for alias in node.names:
                        fp.framework_symbols.append(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if self.framework.replace("-", "_") in alias.name:
                        fp.framework_symbols.append(alias.name)

        # Compute structure hash
        structure = f"{fp.num_imports}:{fp.num_functions}:{fp.num_classes}:{':'.join(sorted(fp.framework_symbols))}"
        fp.structure_hash = hashlib.md5(structure.encode()).hexdigest()[:8]

        return fp

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def compare_across_doc_levels(
        self,
        code_by_level: dict[str, str],
    ) -> ContaminationResult:
        """Compare code generated at different doc levels.

        If code at "none" is very similar to code at "full",
        the model is likely using training data, not docs.

        Args:
            code_by_level: Dict mapping doc level to generated code.

        Returns:
            ContaminationResult with cross-level analysis.
        """
        result = ContaminationResult(
            framework=self.framework,
            training_cutoff=self.training_cutoff,
        )

        # Extract fingerprints
        for level, code in code_by_level.items():
            fp = self.extract_fingerprint(code)
            result.fingerprints_by_level[level] = fp

        # Compare none vs full
        if "none" in result.fingerprints_by_level and "full" in result.fingerprints_by_level:
            none_fp = result.fingerprints_by_level["none"]
            full_fp = result.fingerprints_by_level["full"]
            result.none_vs_full_similarity = none_fp.similarity_to(full_fp)

        # Compare none vs minimal
        if "none" in result.fingerprints_by_level and "minimal" in result.fingerprints_by_level:
            none_fp = result.fingerprints_by_level["none"]
            min_fp = result.fingerprints_by_level["minimal"]
            result.none_vs_minimal_similarity = none_fp.similarity_to(min_fp)

        # Analyze version alignment on all code
        all_code = "\n\n".join(code_by_level.values())
        result.version_alignment = self.analyze_version_alignment(all_code)

        # Determine overall contamination level
        result.contamination_level = self._interpret_contamination(result)

        return result

    def _interpret_contamination(self, result: ContaminationResult) -> ContaminationLevel:
        """Interpret contamination level from evidence."""
        evidence = []

        # High similarity across doc levels suggests contamination
        if result.none_vs_full_similarity > 0.8:
            evidence.append("Code at 'none' docs is >80% similar to 'full' docs")
            result.evidence.append("high_cross_level_similarity")

        # Using old API version suggests training data
        if result.version_alignment and result.version_alignment.uses_old_api:
            evidence.append(f"Uses deprecated {self.framework} v1 APIs: {result.version_alignment.v1_symbols_used[:3]}")
            result.evidence.append("old_api_usage")

        # Interpret
        if len(result.evidence) >= 2:
            return ContaminationLevel.HIGH
        elif len(result.evidence) == 1:
            return ContaminationLevel.MEDIUM
        elif result.none_vs_full_similarity > 0.5:
            return ContaminationLevel.MEDIUM
        else:
            return ContaminationLevel.LOW


class DocContradictionTest:
    """Test whether model follows docs or uses training knowledge.

    Provides modified documentation with renamed APIs, then checks
    if the model uses the modified names (following docs) or real
    names (using training data).
    """

    # API renaming rules for testing
    RENAMING_RULES = {
        "haystack": {
            "PromptBuilder": "TemplateBuilder",
            "OpenAIGenerator": "OpenAITextGenerator",
            "Pipeline": "Workflow",
            "Secret.from_token": "Secret.create",
        },
        "langgraph": {
            "StateGraph": "StateMachine",
            "add_node": "register_node",
            "add_edge": "connect_nodes",
            "compile": "build",
        },
        "pydantic-ai": {
            "Agent": "Assistant",
            "run_sync": "execute",
            "tool": "function",
        },
    }

    def __init__(self, framework: str):
        """Initialize with framework."""
        self.framework = framework
        self.rules = self.RENAMING_RULES.get(framework, {})

    def create_modified_docs(self, original_docs: str) -> str:
        """Create documentation with renamed APIs.

        Args:
            original_docs: Original documentation text.

        Returns:
            Modified documentation with renamed APIs.
        """
        modified = original_docs
        for old_name, new_name in self.rules.items():
            modified = modified.replace(old_name, new_name)
        return modified

    def analyze_adherence(self, code: str) -> DocAdherenceResult:
        """Analyze whether code follows modified docs or real APIs.

        Args:
            code: Generated code to analyze.

        Returns:
            DocAdherenceResult with adherence metrics.
        """
        result = DocAdherenceResult()

        for old_name, new_name in self.rules.items():
            # Count uses of modified (doc) names
            modified_count = code.count(new_name)
            result.modified_api_count += modified_count
            if modified_count > 0:
                result.used_modified_api = True

            # Count uses of real (training) names
            real_count = code.count(old_name)
            result.real_api_count += real_count
            if real_count > 0:
                result.used_real_api = True

        return result


def compute_enhanced_contamination_score(
    success_at_none: float,
    version_alignment: VersionAlignment | None,
    doc_adherence: DocAdherenceResult | None,
    code_similarity: float,
) -> dict:
    """Compute enhanced contamination score with multiple signals.

    Args:
        success_at_none: Success rate with no documentation.
        version_alignment: Version alignment analysis result.
        doc_adherence: Doc adherence test result (if run).
        code_similarity: Similarity between none and full doc code.

    Returns:
        Dict with interpretation and confidence.
    """
    signals = []

    # Signal 1: Success without docs
    if success_at_none >= 0.66:  # 2/3 or better
        signals.append(("success_at_none", 0.8))
    elif success_at_none >= 0.33:
        signals.append(("success_at_none", 0.4))

    # Signal 2: Uses old API version
    if version_alignment and version_alignment.uses_old_api:
        signals.append(("old_api_usage", 0.9))

    # Signal 3: Ignores modified docs
    if doc_adherence and doc_adherence.adherence_score < 0.3:
        signals.append(("ignores_docs", 0.95))
    elif doc_adherence and doc_adherence.adherence_score < 0.5:
        signals.append(("partial_doc_ignorance", 0.6))

    # Signal 4: High code similarity across doc levels
    if code_similarity > 0.8:
        signals.append(("high_code_similarity", 0.85))
    elif code_similarity > 0.5:
        signals.append(("moderate_code_similarity", 0.5))

    # Compute overall score
    if not signals:
        overall_score = 0.0
        interpretation = "clean_slate"
        confidence = "low"
    else:
        # Weighted combination
        overall_score = sum(s[1] for s in signals) / len(signals)

        if overall_score >= 0.7:
            interpretation = "memorized"
            confidence = "high"
        elif overall_score >= 0.4:
            interpretation = "partial_contamination"
            confidence = "medium"
        else:
            interpretation = "likely_clean"
            confidence = "medium"

    return {
        "overall_score": overall_score,
        "interpretation": interpretation,
        "confidence": confidence,
        "signals": [s[0] for s in signals],
        "signal_scores": dict(signals),
        "success_at_none": success_at_none,
        "uses_old_api": version_alignment.uses_old_api if version_alignment else None,
        "doc_adherence_score": doc_adherence.adherence_score if doc_adherence else None,
        "code_similarity": code_similarity,
        "learnability_metric_valid": overall_score < 0.5,
    }
