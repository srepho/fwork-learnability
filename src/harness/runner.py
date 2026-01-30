"""Main experiment runner with trial logging."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..analysis.errors import ErrorAnalyzer, NormalizedError
from ..analysis.hallucination import HallucinationDetector
from ..analysis.metrics import MetricsCalculator, TrialMetrics
from ..compliance.static import StaticComplianceChecker, get_checker
from ..docs.corpus import DocumentCorpus, DocumentLevel
from ..llm.base import LLMClient
from ..tasks.base import Task
from .conversation import ConversationLoop, TrialResult
from .sandbox import Sandbox, SandboxManager


@dataclass
class ExperimentConfig:
    """Configuration for a benchmark experiment."""

    experiment_id: str
    frameworks: list[str]
    doc_levels: list[DocumentLevel]
    task_tiers: list[int]
    models: list[str]
    runs_per_condition: int = 5
    max_turns: int = 10
    timeout_seconds: int = 60

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls(
            experiment_id=f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            frameworks=["pydantic-ai", "haystack", "langgraph", "openai-agents", "direct-api"],
            doc_levels=[DocumentLevel.NONE, DocumentLevel.MINIMAL, DocumentLevel.MODERATE, DocumentLevel.FULL],
            task_tiers=[1, 2, 3],
            models=["deepseek-v3"],
            runs_per_condition=5,
            max_turns=10,
            timeout_seconds=60,
        )


@dataclass
class TrialLog:
    """Complete log of a single trial for reproducibility."""

    trial_id: str
    experiment_id: str
    timestamp: str

    # Condition
    framework: str
    framework_version: str
    doc_level: str
    doc_hash: str
    doc_token_count: int
    task_tier: int
    task_id: str
    model: str
    run_number: int

    # Result
    result: dict
    metrics: dict

    # Error analysis
    error_analyses: list[dict] = field(default_factory=list)

    # Hallucination analysis
    hallucination_result: dict | None = None

    # Compliance
    static_compliance: dict | None = None

    # Metadata
    python_version: str = ""
    dependency_versions: dict = field(default_factory=dict)


class ExperimentRunner:
    """Run benchmark experiments and log results."""

    def __init__(
        self,
        results_dir: Path,
        corpus_dir: Path,
        venvs_dir: Path,
        llm_clients: dict[str, LLMClient],
        tasks: dict[int, Task],
    ):
        """Initialize experiment runner.

        Args:
            results_dir: Directory to store results.
            corpus_dir: Directory with documentation corpus.
            venvs_dir: Directory with framework virtual environments.
            llm_clients: Mapping of model names to LLM clients.
            tasks: Mapping of tier numbers to Task objects.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.corpus = DocumentCorpus(corpus_dir)
        self.sandbox_manager = SandboxManager(venvs_dir)
        self.llm_clients = llm_clients
        self.tasks = tasks

        self.metrics_calculator = MetricsCalculator()
        self._current_experiment_id: str | None = None

    def run_experiment(
        self,
        config: ExperimentConfig,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TrialLog]:
        """Run a full experiment.

        Args:
            config: Experiment configuration.
            progress_callback: Optional callback (current, total, description).

        Returns:
            List of trial logs.
        """
        self._current_experiment_id = config.experiment_id
        all_logs = []

        # Calculate total trials
        total_trials = (
            len(config.frameworks)
            * len(config.doc_levels)
            * len(config.task_tiers)
            * len(config.models)
            * config.runs_per_condition
        )

        current_trial = 0

        for framework in config.frameworks:
            for doc_level in config.doc_levels:
                for tier in config.task_tiers:
                    for model in config.models:
                        for run in range(config.runs_per_condition):
                            current_trial += 1

                            if progress_callback:
                                desc = f"{framework}/{doc_level.value}/tier{tier}/{model}/run{run+1}"
                                progress_callback(current_trial, total_trials, desc)

                            trial_log = self.run_single_trial(
                                framework=framework,
                                doc_level=doc_level,
                                task_tier=tier,
                                model=model,
                                run_number=run + 1,
                                experiment_id=config.experiment_id,
                                max_turns=config.max_turns,
                                timeout_seconds=config.timeout_seconds,
                            )

                            all_logs.append(trial_log)
                            self._save_trial_log(trial_log)

        return all_logs

    def run_single_trial(
        self,
        framework: str,
        doc_level: DocumentLevel,
        task_tier: int,
        model: str,
        run_number: int,
        experiment_id: str | None = None,
        max_turns: int = 10,
        timeout_seconds: int = 60,
        framework_version: str = "latest",
    ) -> TrialLog:
        """Run a single trial.

        Args:
            framework: Target framework.
            doc_level: Documentation level.
            task_tier: Task tier (1, 2, or 3).
            model: Model name.
            run_number: Run number within condition.
            experiment_id: Parent experiment ID.
            max_turns: Maximum conversation turns.
            timeout_seconds: Execution timeout.
            framework_version: Framework version string.

        Returns:
            TrialLog with complete trial data.
        """
        trial_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()

        # Get task
        task = self.tasks.get(task_tier)
        if not task:
            raise ValueError(f"No task defined for tier {task_tier}")

        # Get documentation
        doc_content = self.corpus.get_documentation(framework, framework_version, doc_level)
        doc_info = self.corpus.get_snapshot_info(framework, framework_version, doc_level) or {}

        # Get LLM client
        llm_client = self.llm_clients.get(model)
        if not llm_client:
            raise ValueError(f"No LLM client for model {model}")

        # Get sandbox
        try:
            sandbox = self.sandbox_manager.get_sandbox(framework, timeout_seconds)
        except ValueError:
            # For direct-api or if venv doesn't exist, use a basic sandbox
            from .sandbox import Sandbox
            import tempfile
            sandbox = Sandbox(
                framework=framework,
                venv_path=Path(tempfile.gettempdir()) / "fallback_venv",
                timeout_seconds=timeout_seconds,
            )

        # Get compliance checker
        compliance_checker = get_checker(framework)

        # Build conversation loop
        loop = ConversationLoop(
            llm_client=llm_client,
            sandbox=sandbox,
            max_turns=max_turns,
        )

        # Run trial
        task_prompt = task.get_task_prompt(framework)
        test_code = task.generate_dev_test_code()

        def compliance_check(code: str) -> bool:
            result = compliance_checker.check(code)
            return result.passed

        def hidden_test_runner(code: str) -> tuple[bool, float]:
            hidden_test_code = task.generate_hidden_test_code()
            result = sandbox.execute(code, hidden_test_code)
            passed = result.status.value == "success"
            # Parse score from output if available
            score = 1.0 if passed else 0.0
            return passed, score

        result = loop.run(
            trial_id=trial_id,
            task_description=task_prompt,
            documentation=doc_content,
            test_code=test_code,
            framework=framework,
            doc_level=doc_level.value,
            task_tier=task_tier,
            compliance_checker=compliance_check,
            hidden_test_runner=hidden_test_runner,
        )

        # Analyze errors
        error_analyzer = ErrorAnalyzer(framework)
        error_analyses = []
        for turn in result.turns:
            if turn.error_type:
                analysis = error_analyzer.analyze(
                    turn.error_type,
                    turn.error_message,
                    turn.error_traceback,
                )
                error_analyses.append(asdict(analysis))

        # Analyze hallucinations (if we have final code)
        hallucination_result = None
        if result.turns and result.turns[-1].code:
            try:
                detector = HallucinationDetector.from_installed_package(framework.replace("-", "_"))
                halluc_analysis = detector.analyze_code(result.turns[-1].code)
                hallucination_result = {
                    "total_api_calls": halluc_analysis.total_api_calls,
                    "valid_calls": halluc_analysis.valid_calls,
                    "invented_calls": halluc_analysis.invented_calls,
                    "version_conflict_calls": halluc_analysis.version_conflict_calls,
                    "hallucinated_symbols": halluc_analysis.hallucinated_symbols,
                }
            except (ImportError, Exception):
                pass  # Framework not installed or analysis failed

        # Compute metrics
        metrics = self.metrics_calculator.compute_trial_metrics(
            result,
            error_analyses=[None] * len(result.turns),  # Simplified for now
            hallucination_result=None,
        )

        # Check static compliance for final code
        static_compliance = None
        if result.turns and result.turns[-1].code:
            comp_result = compliance_checker.check(result.turns[-1].code)
            static_compliance = asdict(comp_result)

        # Build trial log
        return TrialLog(
            trial_id=trial_id,
            experiment_id=experiment_id or self._current_experiment_id or "unknown",
            timestamp=timestamp,
            framework=framework,
            framework_version=framework_version,
            doc_level=doc_level.value,
            doc_hash=doc_info.get("content_hash", ""),
            doc_token_count=doc_info.get("token_count", 0),
            task_tier=task_tier,
            task_id=task.task_id,
            model=model,
            run_number=run_number,
            result=self._serialize_result(result),
            metrics=asdict(metrics),
            error_analyses=error_analyses,
            hallucination_result=hallucination_result,
            static_compliance=static_compliance,
        )

    def _serialize_result(self, result: TrialResult) -> dict:
        """Serialize TrialResult to dict."""
        return {
            "trial_id": result.trial_id,
            "framework": result.framework,
            "doc_level": result.doc_level,
            "task_tier": result.task_tier,
            "model": result.model,
            "outcome": result.outcome,
            "final_turn": result.final_turn,
            "dev_set_pass": result.dev_set_pass,
            "hidden_set_pass": result.hidden_set_pass,
            "hidden_set_score": result.hidden_set_score,
            "compliance_check_pass": result.compliance_check_pass,
            "total_tokens": result.total_tokens,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "outcome": t.outcome.value,
                    "code_hash": t.code_hash,
                    "code_loc": t.code_loc,
                    "error_type": t.error_type,
                    "error_message": t.error_message,
                    "tokens_generated": t.tokens_generated,
                    "execution_time_ms": t.execution_time_ms,
                }
                for t in result.turns
            ],
        }

    def _save_trial_log(self, log: TrialLog):
        """Save trial log to disk."""
        exp_dir = self.results_dir / log.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        log_file = exp_dir / f"{log.trial_id}.json"
        log_file.write_text(json.dumps(asdict(log), indent=2, default=str))

    def load_experiment_results(self, experiment_id: str) -> list[TrialLog]:
        """Load all trial logs for an experiment."""
        exp_dir = self.results_dir / experiment_id
        if not exp_dir.exists():
            return []

        logs = []
        for log_file in exp_dir.glob("*.json"):
            data = json.loads(log_file.read_text())
            # Convert back to TrialLog (simplified - would need proper deserialization)
            logs.append(data)

        return logs
