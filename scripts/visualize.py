#!/usr/bin/env python3
"""Visualization scripts for experiment results."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: Path, experiment_id: str) -> pd.DataFrame:
    """Load experiment results into a DataFrame."""
    exp_dir = results_dir / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    records = []
    for log_file in exp_dir.glob("*.json"):
        data = json.loads(log_file.read_text())
        # Flatten nested structure
        record = {
            "trial_id": data["trial_id"],
            "framework": data["framework"],
            "doc_level": data["doc_level"],
            "task_tier": data["task_tier"],
            "model": data["model"],
            "run_number": data["run_number"],
            "outcome": data["result"]["outcome"],
            "final_turn": data["result"]["final_turn"],
            "total_tokens": data["result"]["total_tokens"],
            "dev_set_pass": data["result"]["dev_set_pass"],
            "hidden_set_pass": data["result"]["hidden_set_pass"],
            "hidden_set_score": data["result"]["hidden_set_score"],
            "success": data["result"]["outcome"] == "success",
        }
        records.append(record)

    return pd.DataFrame(records)


def plot_survival_curves(df: pd.DataFrame, output_path: Path):
    """Plot Kaplan-Meier style success probability curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    frameworks = df["framework"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(frameworks)))

    for framework, color in zip(frameworks, colors):
        fw_data = df[df["framework"] == framework]

        # Calculate success probability by turn
        turns = range(1, 11)
        success_by_turn = []
        for t in turns:
            success_count = ((fw_data["final_turn"] <= t) & fw_data["success"]).sum()
            success_by_turn.append(success_count / len(fw_data))

        ax.plot(turns, success_by_turn, label=framework, color=color, marker="o")

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("P(Success by Turn)")
    ax.set_title("Success Probability by Turn (Survival Analysis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_learning_curves(df: pd.DataFrame, output_path: Path):
    """Plot success rate vs documentation level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    doc_order = ["none", "minimal", "moderate", "full"]
    pivot = df.groupby(["framework", "doc_level"])["success"].mean().unstack()
    pivot = pivot.reindex(columns=doc_order)

    pivot.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Framework")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Documentation Level")
    ax.legend(title="Doc Level")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_friction_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot error category heatmap by framework."""
    # This would require error categorization data
    # Placeholder for when error analysis is integrated

    fig, ax = plt.subplots(figsize=(10, 8))

    # Group by framework and calculate failure rate
    failure_rates = df.groupby("framework").apply(
        lambda x: 1 - x["success"].mean()
    )

    # Create simple heatmap placeholder
    frameworks = failure_rates.index.tolist()
    categories = ["Import", "Type", "Runtime", "Logic", "Hallucination"]

    # Placeholder data - would be populated from error analysis
    data = np.random.rand(len(frameworks), len(categories))
    data = data / data.sum(axis=1, keepdims=True)  # Normalize

    sns.heatmap(
        data,
        xticklabels=categories,
        yticklabels=frameworks,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
    )

    ax.set_title("Error Category Distribution by Framework")
    ax.set_xlabel("Error Category")
    ax.set_ylabel("Framework")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_contamination_scatter(df: pd.DataFrame, output_path: Path):
    """Plot contamination matrix: success@none vs symbol exactness."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate success at none for each framework
    none_success = df[df["doc_level"] == "none"].groupby("framework")["success"].mean()

    # Placeholder for symbol exactness (would come from hallucination analysis)
    symbol_exactness = pd.Series({fw: np.random.uniform(0.5, 1.0) for fw in none_success.index})

    frameworks = none_success.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(frameworks)))

    for fw, color in zip(frameworks, colors):
        ax.scatter(
            none_success[fw],
            symbol_exactness[fw],
            s=200,
            c=[color],
            label=fw,
            alpha=0.7,
        )
        ax.annotate(fw, (none_success[fw], symbol_exactness[fw]), fontsize=9)

    # Add quadrant lines
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # Add quadrant labels
    ax.text(0.75, 0.9, "Memorized", fontsize=10, alpha=0.7)
    ax.text(0.75, 0.7, "Guessable API", fontsize=10, alpha=0.7)
    ax.text(0.25, 0.9, "Clean Slate\n(Valid)", fontsize=10, alpha=0.7)
    ax.text(0.25, 0.7, "Clean Slate\n(Valid)", fontsize=10, alpha=0.7)

    ax.set_xlabel("Success Rate @ None (Prior Knowledge)")
    ax.set_ylabel("Symbol Exactness")
    ax.set_title("Contamination Detection Matrix")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_tier_comparison(df: pd.DataFrame, output_path: Path):
    """Plot success rate by task tier for each framework."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df.groupby(["framework", "task_tier"])["success"].mean().unstack()

    pivot.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Framework")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Task Tier")
    ax.legend(title="Task Tier", labels=["Tier 1 (Classification)", "Tier 2 (Tool Use)", "Tier 3 (Agent)"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table."""
    summary = df.groupby("framework").agg({
        "success": "mean",
        "final_turn": lambda x: x[df.loc[x.index, "success"]].median() if df.loc[x.index, "success"].any() else np.nan,
        "total_tokens": "mean",
        "hidden_set_score": "mean",
    }).round(3)

    summary.columns = ["Success Rate", "Median Turns (Success)", "Avg Tokens", "Hidden Set Score"]

    # Calculate first-attempt success
    first_success = df.groupby("framework").apply(
        lambda x: (x["final_turn"] == 1).mean()
    )
    summary["First Attempt Success"] = first_success.round(3)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")

    parser.add_argument(
        "experiment_id",
        type=str,
        help="Experiment ID to visualize",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir / args.experiment_id}")
    df = load_results(args.results_dir, args.experiment_id)
    print(f"Loaded {len(df)} trials")

    # Generate visualizations
    print("Generating survival curves...")
    plot_survival_curves(df, args.output_dir / "survival_curves.png")

    print("Generating learning curves...")
    plot_learning_curves(df, args.output_dir / "learning_curves.png")

    print("Generating friction heatmap...")
    plot_friction_heatmap(df, args.output_dir / "friction_heatmap.png")

    print("Generating contamination scatter...")
    plot_contamination_scatter(df, args.output_dir / "contamination_scatter.png")

    print("Generating tier comparison...")
    plot_tier_comparison(df, args.output_dir / "tier_comparison.png")

    # Generate summary table
    print("\n=== Summary Statistics ===")
    summary = generate_summary_table(df)
    print(summary.to_string())

    # Save summary
    summary.to_csv(args.output_dir / "summary.csv")

    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
