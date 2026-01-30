#!/usr/bin/env python3
"""Main experiment runner script."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.docs.corpus import DocumentLevel
from src.harness.runner import ExperimentConfig, ExperimentRunner
from src.llm.deepseek import DeepSeekClient
from src.tasks.tier1.claims_classification import ClaimsClassificationTask


def main():
    parser = argparse.ArgumentParser(description="Run LLM framework learnability experiment")

    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["pydantic-ai"],
        help="Frameworks to test",
    )
    parser.add_argument(
        "--doc-levels",
        nargs="+",
        choices=["none", "minimal", "moderate", "full"],
        default=["none", "minimal", "moderate", "full"],
        help="Documentation levels",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=[1],
        help="Task tiers to run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per condition",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per trial",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store results",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("docs_corpus"),
        help="Directory with documentation corpus",
    )
    parser.add_argument(
        "--venvs-dir",
        type=Path,
        default=Path("venvs"),
        help="Directory with framework virtual environments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--contradiction-test",
        action="store_true",
        help="Run contradiction test with modified docs (renamed APIs)",
    )
    parser.add_argument(
        "--contradiction-level",
        type=str,
        choices=["moderate", "full"],
        default="moderate",
        help="Doc level to use for contradiction test (default: moderate)",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    console = Console()

    # Validate API keys
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        console.print("[red]Error: DEEPSEEK_API_KEY not found in environment[/red]")
        sys.exit(1)

    # Convert doc levels
    doc_levels = [DocumentLevel(level) for level in args.doc_levels]

    # Build config
    config = ExperimentConfig(
        experiment_id=args.experiment_id or ExperimentConfig.default().experiment_id,
        frameworks=args.frameworks,
        doc_levels=doc_levels,
        task_tiers=args.tiers,
        models=["deepseek-v3"],
        runs_per_condition=args.runs,
        max_turns=args.max_turns,
    )

    # Calculate total trials
    total = (
        len(config.frameworks)
        * len(config.doc_levels)
        * len(config.task_tiers)
        * len(config.models)
        * config.runs_per_condition
    )

    # Add contradiction test trials if requested
    if args.contradiction_test:
        contradiction_trials = (
            len(config.frameworks)
            * len(config.task_tiers)
            * len(config.models)
            * config.runs_per_condition
        )
        total += contradiction_trials

    console.print(f"\n[bold]LLM Framework Learnability Benchmark[/bold]")
    console.print(f"Experiment ID: {config.experiment_id}")
    console.print(f"Frameworks: {', '.join(config.frameworks)}")
    console.print(f"Doc levels: {', '.join(l.value for l in config.doc_levels)}")
    console.print(f"Task tiers: {config.task_tiers}")
    console.print(f"Runs per condition: {config.runs_per_condition}")
    if args.contradiction_test:
        console.print(f"[yellow]Contradiction test enabled at {args.contradiction_level} level[/yellow]")
    console.print(f"Total trials: {total}")
    console.print()

    if args.dry_run:
        console.print("[yellow]Dry run - not executing[/yellow]")
        return

    # Initialize components
    llm_clients = {
        "deepseek-v3": DeepSeekClient(api_key=deepseek_key),
    }

    tasks = {
        1: ClaimsClassificationTask(),
        # Add tier 2 and 3 tasks when implemented
    }

    # Use absolute paths
    script_dir = Path(__file__).parent.parent.resolve()
    results_dir = args.results_dir if args.results_dir.is_absolute() else script_dir / args.results_dir
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else script_dir / args.corpus_dir
    venvs_dir = args.venvs_dir if args.venvs_dir.is_absolute() else script_dir / args.venvs_dir

    runner = ExperimentRunner(
        results_dir=results_dir,
        corpus_dir=corpus_dir,
        venvs_dir=venvs_dir,
        llm_clients=llm_clients,
        tasks=tasks,
    )

    # Run experiment with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running trials...", total=total)

        def progress_callback(current: int, total: int, description: str):
            progress.update(task, completed=current, description=f"[{current}/{total}] {description}")

        try:
            logs = runner.run_experiment(config, progress_callback)

            # Run contradiction tests if requested
            if args.contradiction_test:
                console.print("\n[yellow]Running contradiction tests...[/yellow]")
                contradiction_level = DocumentLevel(args.contradiction_level)

                for framework in config.frameworks:
                    for tier in config.task_tiers:
                        for model in config.models:
                            for run in range(config.runs_per_condition):
                                desc = f"{framework}/{args.contradiction_level}_contradiction/tier{tier}/{model}/run{run+1}"
                                progress.update(task, description=f"[contradiction] {desc}")

                                trial_log = runner.run_single_trial(
                                    framework=framework,
                                    doc_level=contradiction_level,
                                    task_tier=tier,
                                    model=model,
                                    run_number=run + 1,
                                    experiment_id=config.experiment_id,
                                    max_turns=config.max_turns,
                                    timeout_seconds=config.timeout_seconds,
                                    use_contradiction_docs=True,
                                )
                                logs.append(trial_log)
                                runner._save_trial_log(trial_log)

            console.print(f"\n[green]Experiment complete![/green]")
            console.print(f"Results saved to: {args.results_dir / config.experiment_id}")

            # Print summary
            successes = sum(1 for log in logs if log.result.get("outcome") == "success")
            console.print(f"Success rate: {successes}/{len(logs)} ({100*successes/len(logs):.1f}%)")

            # Print contradiction test summary if run
            if args.contradiction_test:
                contradiction_logs = [log for log in logs if getattr(log, 'is_contradiction_test', False) or
                                     (isinstance(log, dict) and log.get('is_contradiction_test'))]
                if contradiction_logs:
                    console.print(f"\n[yellow]Contradiction Test Results:[/yellow]")
                    for log in contradiction_logs:
                        analysis = log.contradiction_analysis if hasattr(log, 'contradiction_analysis') else log.get('contradiction_analysis')
                        if analysis:
                            framework = log.framework if hasattr(log, 'framework') else log.get('framework')
                            console.print(f"  {framework}: {analysis.get('interpretation', 'unknown')} "
                                        f"(adherence: {analysis.get('adherence_score', 0):.0%})")

        except KeyboardInterrupt:
            console.print("\n[yellow]Experiment interrupted[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise


if __name__ == "__main__":
    main()
