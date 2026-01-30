#!/usr/bin/env python3
"""Fetch and snapshot framework documentation."""

import argparse
import sys
from pathlib import Path

import httpx
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.docs.corpus import DocumentCorpus, DocumentLevel, generate_minimal_doc


# Documentation sources for each framework
DOC_SOURCES = {
    "pydantic-ai": {
        "minimal": {
            "description": "PydanticAI is a Python agent framework designed to make it easy to build production-grade applications with Generative AI. It provides a type-safe, Pythonic interface for building agents with tool use capabilities.",
            "install": "pip install pydantic-ai",
        },
        "moderate": [
            "https://ai.pydantic.dev/",
            "https://ai.pydantic.dev/agents/",
        ],
        "full": [
            "https://ai.pydantic.dev/",
            "https://ai.pydantic.dev/agents/",
            "https://ai.pydantic.dev/tools/",
            "https://ai.pydantic.dev/api/agent/",
        ],
    },
    "haystack": {
        "minimal": {
            "description": "Haystack is an open-source framework for building production-ready LLM applications, RAG pipelines, and agent systems. It provides composable components for retrieval, generation, and tool use.",
            "install": "pip install haystack-ai",
        },
        "moderate": [
            "https://docs.haystack.deepset.ai/docs/intro",
            "https://docs.haystack.deepset.ai/docs/pipelines",
        ],
        "full": [
            "https://docs.haystack.deepset.ai/docs/intro",
            "https://docs.haystack.deepset.ai/docs/pipelines",
            "https://docs.haystack.deepset.ai/docs/agents",
            "https://docs.haystack.deepset.ai/reference/",
        ],
    },
    "langgraph": {
        "minimal": {
            "description": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains or actors across multiple steps of computation.",
            "install": "pip install langgraph",
        },
        "moderate": [
            "https://langchain-ai.github.io/langgraph/",
            "https://langchain-ai.github.io/langgraph/concepts/",
        ],
        "full": [
            "https://langchain-ai.github.io/langgraph/",
            "https://langchain-ai.github.io/langgraph/concepts/",
            "https://langchain-ai.github.io/langgraph/how-tos/",
            "https://langchain-ai.github.io/langgraph/reference/",
        ],
    },
    "openai-agents": {
        "minimal": {
            "description": "OpenAI Agents SDK provides tools for building AI agents that can use tools, maintain conversation state, and handle complex multi-step tasks using OpenAI models.",
            "install": "pip install openai-agents",
        },
        "moderate": [
            "https://platform.openai.com/docs/guides/agents",
        ],
        "full": [
            "https://platform.openai.com/docs/guides/agents",
            "https://platform.openai.com/docs/api-reference/agents",
        ],
    },
    "direct-api": {
        "minimal": {
            "description": "Direct API baseline uses raw HTTP requests to LLM APIs without any framework abstraction. This serves as a control condition to measure framework overhead.",
            "install": "pip install httpx",
        },
        "moderate": [],
        "full": [],
    },
}


def fetch_url(url: str, console: Console) -> str | None:
    """Fetch content from a URL."""
    try:
        console.print(f"  Fetching: {url}")
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        return response.text
    except Exception as e:
        console.print(f"  [yellow]Warning: Failed to fetch {url}: {e}[/yellow]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch and snapshot framework documentation")

    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=list(DOC_SOURCES.keys()),
        help="Frameworks to fetch docs for",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Version string for the snapshot",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("docs_corpus"),
        help="Directory to store documentation",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["minimal", "moderate", "full"],
        default=["minimal", "moderate", "full"],
        help="Documentation levels to fetch",
    )

    args = parser.parse_args()
    console = Console()

    corpus = DocumentCorpus(args.corpus_dir)

    for framework in args.frameworks:
        if framework not in DOC_SOURCES:
            console.print(f"[yellow]Unknown framework: {framework}[/yellow]")
            continue

        console.print(f"\n[bold]Fetching docs for: {framework}[/bold]")
        sources = DOC_SOURCES[framework]

        # Minimal level
        if "minimal" in args.levels and "minimal" in sources:
            minimal_info = sources["minimal"]
            content = generate_minimal_doc(
                framework,
                minimal_info["description"],
                minimal_info["install"],
            )
            snapshot = corpus.add_document(
                framework=framework,
                version=args.version,
                level=DocumentLevel.MINIMAL,
                content=content,
                source_urls=["generated"],
            )
            console.print(f"  [green]Created minimal doc: {snapshot.content_hash}[/green]")

        # Moderate level
        if "moderate" in args.levels and sources.get("moderate"):
            contents = []
            for url in sources["moderate"]:
                html = fetch_url(url, console)
                if html:
                    contents.append(html)

            if contents:
                combined = "\n\n---\n\n".join(contents)
                snapshot = corpus.add_document(
                    framework=framework,
                    version=args.version,
                    level=DocumentLevel.MODERATE,
                    content=combined,
                    source_urls=sources["moderate"],
                )
                console.print(f"  [green]Created moderate doc: {snapshot.content_hash}[/green]")

        # Full level
        if "full" in args.levels and sources.get("full"):
            contents = []
            for url in sources["full"]:
                html = fetch_url(url, console)
                if html:
                    contents.append(html)

            if contents:
                combined = "\n\n---\n\n".join(contents)
                snapshot = corpus.add_document(
                    framework=framework,
                    version=args.version,
                    level=DocumentLevel.FULL,
                    content=combined,
                    source_urls=sources["full"],
                )
                console.print(f"  [green]Created full doc: {snapshot.content_hash}[/green]")

    console.print(f"\n[bold green]Documentation saved to: {args.corpus_dir}[/bold green]")


if __name__ == "__main__":
    main()
