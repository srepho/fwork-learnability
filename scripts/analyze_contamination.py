#!/usr/bin/env python3
"""Analyze contamination signals in experiment results.

This script detects whether the LLM is using training data knowledge
vs actually learning from provided documentation.

Usage:
    python scripts/analyze_contamination.py --results-dir results/exp_20260129_234834
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.contamination import (
    ContaminationDetector,
    ContaminationLevel,
    VersionAlignment,
    compute_enhanced_contamination_score,
)


def load_trial_results(results_dir: Path) -> list[dict]:
    """Load all trial results from a directory."""
    results = []
    for json_file in results_dir.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        try:
            with open(json_file) as f:
                results.append(json.load(f))
        except json.JSONDecodeError:
            continue
    return results


def group_by_framework_and_level(results: list[dict]) -> dict:
    """Group results by framework and doc level."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        framework = r.get("framework", "unknown")
        doc_level = r.get("doc_level", "unknown")
        grouped[framework][doc_level].append(r)
    return grouped


def extract_code_from_trial(trial: dict) -> str | None:
    """Extract the final successful code from a trial."""
    result = trial.get("result", {})
    turns = result.get("turns", [])

    # Find the last successful turn or the last turn
    for turn in reversed(turns):
        if turn.get("outcome") == "success":
            # We don't store the actual code in JSON, but we can
            # use the code_hash to identify uniqueness
            return turn.get("code_hash")
    return None


def compute_success_at_none(results_by_level: dict) -> float:
    """Compute success rate at 'none' doc level."""
    none_results = results_by_level.get("none", [])
    if not none_results:
        return 0.0
    successes = sum(1 for r in none_results if r.get("result", {}).get("outcome") == "success")
    return successes / len(none_results)


def analyze_version_alignment(results_by_level: dict, framework: str) -> VersionAlignment | None:
    """Analyze version alignment across all trials."""
    detector = ContaminationDetector(framework)

    # We can analyze the static compliance data which contains used_symbols
    all_v1_symbols = []
    all_v2_symbols = []

    for level, trials in results_by_level.items():
        for trial in trials:
            static_compliance = trial.get("static_compliance") or {}
            used_symbols = static_compliance.get("used_symbols") or []

            # Check against version patterns
            for symbol in used_symbols:
                if symbol in detector._v1_apis:
                    all_v1_symbols.append(symbol)
                if symbol in detector._v2_apis:
                    all_v2_symbols.append(symbol)

    if not all_v1_symbols and not all_v2_symbols:
        return None

    result = VersionAlignment(framework=framework)
    result.v1_symbols_used = list(set(all_v1_symbols))
    result.v2_symbols_used = list(set(all_v2_symbols))

    total = len(all_v1_symbols) + len(all_v2_symbols)
    if total > 0:
        result.version_scores["v1"] = len(all_v1_symbols) / total
        result.version_scores["v2"] = len(all_v2_symbols) / total

        if result.version_scores["v1"] > result.version_scores["v2"]:
            result.detected_version = "v1"
        else:
            result.detected_version = "v2"

        result.alignment_confidence = max(result.version_scores.values())

    return result


def compute_code_similarity_across_levels(results_by_level: dict) -> float:
    """Compute code similarity between 'none' and 'full' doc levels."""
    none_hashes = set()
    full_hashes = set()

    for trial in results_by_level.get("none", []):
        for turn in trial.get("result", {}).get("turns", []):
            if turn.get("code_hash"):
                none_hashes.add(turn["code_hash"])

    for trial in results_by_level.get("full", []):
        for turn in trial.get("result", {}).get("turns", []):
            if turn.get("code_hash"):
                full_hashes.add(turn["code_hash"])

    if not none_hashes or not full_hashes:
        return 0.0

    # Jaccard similarity of code hashes
    intersection = none_hashes & full_hashes
    union = none_hashes | full_hashes

    return len(intersection) / len(union) if union else 0.0


def main():
    parser = argparse.ArgumentParser(description="Analyze contamination in experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--framework",
        type=str,
        help="Analyze specific framework only",
    )

    args = parser.parse_args()
    console = Console()

    if not args.results_dir.exists():
        console.print(f"[red]Results directory not found: {args.results_dir}[/red]")
        return

    # Load results
    results = load_trial_results(args.results_dir)
    console.print(f"Loaded {len(results)} trial results from {args.results_dir}")

    # Group by framework
    grouped = group_by_framework_and_level(results)

    # Filter to specific framework if requested
    if args.framework:
        if args.framework not in grouped:
            console.print(f"[red]Framework '{args.framework}' not found in results[/red]")
            return
        grouped = {args.framework: grouped[args.framework]}

    # Create results table
    table = Table(title="Contamination Analysis")
    table.add_column("Framework", style="cyan")
    table.add_column("Success@None", style="yellow")
    table.add_column("Version Used", style="green")
    table.add_column("V1/V2 Ratio", style="magenta")
    table.add_column("Code Similarity", style="blue")
    table.add_column("Contamination", style="red")
    table.add_column("Evidence", style="white")

    for framework, results_by_level in sorted(grouped.items()):
        # Compute metrics
        success_at_none = compute_success_at_none(results_by_level)
        version_alignment = analyze_version_alignment(results_by_level, framework)
        code_similarity = compute_code_similarity_across_levels(results_by_level)

        # Compute enhanced score
        score = compute_enhanced_contamination_score(
            success_at_none=success_at_none,
            version_alignment=version_alignment,
            doc_adherence=None,  # Would need modified doc test
            code_similarity=code_similarity,
        )

        # Format results
        version_str = "-"
        ratio_str = "-"
        if version_alignment:
            version_str = version_alignment.detected_version or "-"
            v1 = version_alignment.version_scores.get("v1", 0)
            v2 = version_alignment.version_scores.get("v2", 0)
            ratio_str = f"{v1:.0%} / {v2:.0%}"

        # Determine contamination level color
        interp = score["interpretation"]
        if interp == "memorized":
            contam_style = "[bold red]"
        elif interp == "partial_contamination":
            contam_style = "[yellow]"
        else:
            contam_style = "[green]"

        evidence_str = ", ".join(score["signals"][:3]) if score["signals"] else "-"

        table.add_row(
            framework,
            f"{success_at_none:.0%}",
            version_str,
            ratio_str,
            f"{code_similarity:.0%}",
            f"{contam_style}{interp}[/]",
            evidence_str,
        )

    console.print(table)

    # Print detailed analysis
    console.print("\n[bold]Detailed Analysis[/bold]\n")

    for framework, results_by_level in sorted(grouped.items()):
        console.print(f"\n[bold cyan]--- {framework} ---[/bold cyan]")

        # Success by doc level
        console.print("\nSuccess rate by doc level:")
        for level in ["none", "minimal", "moderate", "full"]:
            trials = results_by_level.get(level, [])
            if trials:
                successes = sum(1 for r in trials if r.get("result", {}).get("outcome") == "success")
                console.print(f"  {level}: {successes}/{len(trials)} ({successes/len(trials):.0%})")

        # Version alignment details
        va = analyze_version_alignment(results_by_level, framework)
        if va and (va.v1_symbols_used or va.v2_symbols_used):
            console.print(f"\nAPI Version Analysis:")
            if va.v1_symbols_used:
                console.print(f"  [yellow]Old (v1) APIs used:[/yellow] {', '.join(va.v1_symbols_used[:5])}")
            if va.v2_symbols_used:
                console.print(f"  [green]New (v2) APIs used:[/green] {', '.join(va.v2_symbols_used[:5])}")

            if va.uses_old_api:
                console.print(f"  [red]WARNING: Using deprecated APIs suggests training data contamination[/red]")

        # Contamination interpretation
        success_at_none = compute_success_at_none(results_by_level)
        if success_at_none >= 0.66:
            console.print(f"\n[red]High contamination signal:[/red] {success_at_none:.0%} success with NO documentation")
            console.print("  This suggests the model has this framework in training data.")
        elif success_at_none >= 0.33:
            console.print(f"\n[yellow]Moderate contamination signal:[/yellow] {success_at_none:.0%} success with NO documentation")
        else:
            console.print(f"\n[green]Low contamination signal:[/green] {success_at_none:.0%} success with NO documentation")
            console.print("  This suggests the model is actually learning from documentation.")


if __name__ == "__main__":
    main()
