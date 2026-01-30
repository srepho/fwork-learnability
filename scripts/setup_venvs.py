#!/usr/bin/env python3
"""Set up isolated virtual environments for each framework."""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console


# Framework dependencies
FRAMEWORK_DEPS = {
    "pydantic-ai": ["pydantic-ai>=1.0.0", "pydantic>=2.0.0"],
    "haystack": ["haystack-ai>=2.0.0"],
    "langgraph": ["langgraph>=1.0.0", "langchain>=0.1.0"],
    "openai-agents": ["openai>=1.0.0"],
    "anthropic-agents": ["anthropic>=0.39.0"],
    "direct-api": ["httpx>=0.27.0"],
}


def create_venv(venv_path: Path, console: Console) -> bool:
    """Create a virtual environment."""
    try:
        console.print(f"  Creating venv at: {venv_path}")
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Failed to create venv: {e.stderr.decode()}[/red]")
        return False


def install_deps(venv_path: Path, deps: list[str], console: Console) -> bool:
    """Install dependencies in a venv."""
    if sys.platform == "win32":
        pip = venv_path / "Scripts" / "pip.exe"
    else:
        pip = venv_path / "bin" / "pip"

    try:
        # Upgrade pip first
        subprocess.run(
            [str(pip), "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )

        # Install deps
        console.print(f"  Installing: {', '.join(deps)}")
        result = subprocess.run(
            [str(pip), "install"] + deps,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"  [yellow]Warning: Some packages may have failed[/yellow]")
            console.print(f"  {result.stderr[:500]}")

        return True

    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Failed to install deps: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up framework virtual environments")

    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=list(FRAMEWORK_DEPS.keys()),
        help="Frameworks to set up",
    )
    parser.add_argument(
        "--venvs-dir",
        type=Path,
        default=Path("venvs"),
        help="Directory to create venvs in",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate existing venvs",
    )

    args = parser.parse_args()
    console = Console()

    args.venvs_dir.mkdir(parents=True, exist_ok=True)

    for framework in args.frameworks:
        if framework not in FRAMEWORK_DEPS:
            console.print(f"[yellow]Unknown framework: {framework}[/yellow]")
            continue

        console.print(f"\n[bold]Setting up: {framework}[/bold]")

        venv_path = args.venvs_dir / framework

        if venv_path.exists():
            if args.force:
                console.print(f"  Removing existing venv")
                import shutil
                shutil.rmtree(venv_path)
            else:
                console.print(f"  [yellow]Venv already exists (use --force to recreate)[/yellow]")
                continue

        if create_venv(venv_path, console):
            deps = FRAMEWORK_DEPS[framework]
            if install_deps(venv_path, deps, console):
                console.print(f"  [green]Successfully set up {framework}[/green]")
            else:
                console.print(f"  [red]Failed to install dependencies[/red]")
        else:
            console.print(f"  [red]Failed to create venv[/red]")

    console.print(f"\n[bold green]Virtual environments created in: {args.venvs_dir}[/bold green]")


if __name__ == "__main__":
    main()
