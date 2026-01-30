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
# IMPORTANT: URLs should point to actual tutorials with code examples,
# NOT landing pages with marketing content.
DOC_SOURCES = {
    "pydantic-ai": {
        "minimal": {
            "description": "PydanticAI is a Python agent framework designed to make it easy to build production-grade applications with Generative AI. It provides a type-safe, Pythonic interface for building agents with tool use capabilities.",
            "install": "pip install pydantic-ai",
        },
        "moderate": [
            # Quickstart with actual code examples
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
            # These URLs have actual code tutorials, not landing pages
            "https://docs.haystack.deepset.ai/docs/creating-pipelines",
            "https://docs.haystack.deepset.ai/docs/prompt-builder",
        ],
        "full": [
            "https://docs.haystack.deepset.ai/docs/creating-pipelines",
            "https://docs.haystack.deepset.ai/docs/prompt-builder",
            "https://docs.haystack.deepset.ai/docs/generators",
            "https://docs.haystack.deepset.ai/docs/openaigemeerator",  # Generator component docs
        ],
    },
    "langgraph": {
        "minimal": {
            "description": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains or actors across multiple steps of computation.",
            "install": "pip install langgraph",
        },
        "moderate": [
            # Actual tutorials with code, not redirecting landing pages
            "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
            "https://langchain-ai.github.io/langgraph/concepts/low_level/",
        ],
        "full": [
            "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
            "https://langchain-ai.github.io/langgraph/concepts/low_level/",
            "https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/",
            "https://langchain-ai.github.io/langgraph/reference/graphs/",
        ],
    },
    "openai-agents": {
        "minimal": {
            "description": "OpenAI Agents SDK provides tools for building AI agents that can use tools, maintain conversation state, and handle complex multi-step tasks using OpenAI models.",
            "install": "pip install openai",
        },
        "moderate": [
            "https://platform.openai.com/docs/guides/agents",
        ],
        "full": [
            "https://platform.openai.com/docs/guides/agents",
            "https://platform.openai.com/docs/api-reference/agents",
        ],
    },
    "anthropic-agents": {
        "minimal": {
            "description": "Anthropic API provides Claude models for building AI agents. Use the Messages API with tools for agent-like behavior.",
            "install": "pip install anthropic",
        },
        "moderate": [
            "https://docs.anthropic.com/en/docs/build-with-claude/tool-use",
        ],
        "full": [
            "https://docs.anthropic.com/en/docs/build-with-claude/tool-use",
            "https://docs.anthropic.com/en/api/messages",
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


# Manual curated documentation snippets for frameworks where
# web scraping is unreliable. These contain actual code examples.
CURATED_DOCS = {
    "haystack": {
        "moderate": """# Haystack v2 Documentation

## Creating Pipelines

Haystack v2 uses a component-based pipeline architecture:

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

# Create components
prompt_builder = PromptBuilder(template=\"\"\"
Given the following context, answer the question.
Context: {{context}}
Question: {{question}}
Answer:
\"\"\")

generator = OpenAIGenerator(
    model="gpt-3.5-turbo",
    api_key=Secret.from_env_var("OPENAI_API_KEY")
)

# Build pipeline
pipeline = Pipeline()
pipeline.add_component("prompt", prompt_builder)
pipeline.add_component("generator", generator)
pipeline.connect("prompt", "generator")

# Run pipeline
result = pipeline.run({
    "prompt": {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?"
    }
})
print(result["generator"]["replies"][0])
```

## Key Components

- **PromptBuilder**: Builds prompts from templates with variables
- **OpenAIGenerator**: Generates text using OpenAI models
- **Pipeline**: Connects components together
- **Secret**: Securely handles API keys
""",
        "full": """# Haystack v2 Complete Documentation

## Installation

```bash
pip install haystack-ai
```

## Creating Pipelines

Haystack v2 uses a component-based pipeline architecture:

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

# Create components
prompt_builder = PromptBuilder(template=\"\"\"
Given the following context, answer the question.
Context: {{context}}
Question: {{question}}
Answer:
\"\"\")

generator = OpenAIGenerator(
    model="gpt-3.5-turbo",
    api_key=Secret.from_env_var("OPENAI_API_KEY")
)

# Build pipeline
pipeline = Pipeline()
pipeline.add_component("prompt", prompt_builder)
pipeline.add_component("generator", generator)
pipeline.connect("prompt", "generator")

# Run pipeline
result = pipeline.run({
    "prompt": {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?"
    }
})
print(result["generator"]["replies"][0])
```

## Component Reference

### PromptBuilder
Creates prompts from Jinja2 templates:
```python
from haystack.components.builders import PromptBuilder

builder = PromptBuilder(template="Classify this: {{text}}")
result = builder.run(text="Hello world")
# result["prompt"] contains the rendered template
```

### OpenAIGenerator
Generates text using OpenAI's chat models:
```python
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

generator = OpenAIGenerator(
    model="gpt-3.5-turbo",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    generation_kwargs={"temperature": 0}
)
result = generator.run(prompt="Say hello")
# result["replies"] is a list of generated responses
```

### Secret
Handles API keys securely:
```python
from haystack.utils import Secret

# From environment variable (recommended)
api_key = Secret.from_env_var("OPENAI_API_KEY")

# From token string (not recommended for production)
api_key = Secret.from_token("sk-...")
```

## Pipeline Connections

Connect components with `pipeline.connect()`:
```python
pipeline = Pipeline()
pipeline.add_component("a", ComponentA())
pipeline.add_component("b", ComponentB())
# Connect output of "a" to input of "b"
pipeline.connect("a.output_name", "b.input_name")
# Or use default names
pipeline.connect("a", "b")
```

## Running Pipelines

```python
result = pipeline.run({
    "component_name": {
        "input_param": value
    }
})
```
""",
    },
    "anthropic-agents": {
        "moderate": """# Anthropic Agents SDK Documentation

## Overview

The Anthropic Agents SDK provides a simple interface for building AI agents with Claude.

## Installation

```bash
pip install anthropic
```

## Basic Agent

```python
import anthropic

client = anthropic.Anthropic()

def classify_claim(claim: dict) -> dict:
    \"\"\"Classify an insurance claim using Claude.\"\"\"
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f\"\"\"Classify this insurance claim as 'simple' or 'complex'.

Claim: {claim}

Respond with JSON: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "..."}}\"\"\"
            }
        ]
    )

    import json
    return json.loads(response.content[0].text)
```

## With Tools

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "classify_claim",
        "description": "Classify an insurance claim",
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {"type": "string", "enum": ["simple", "complex"]},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"}
            },
            "required": ["classification", "confidence", "reasoning"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Classify this claim..."}]
)

# Extract tool use
for block in response.content:
    if block.type == "tool_use":
        result = block.input  # Contains classification, confidence, reasoning
```
""",
        "full": """# Anthropic Agents SDK Complete Documentation

## Installation

```bash
pip install anthropic
```

## Basic Usage

```python
import anthropic
import json

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

def classify_claim(claim: dict) -> dict:
    \"\"\"Classify an insurance claim using Claude.\"\"\"

    prompt = f\"\"\"Classify this insurance claim as 'simple' or 'complex'.

A claim is SIMPLE if ALL of these are true:
- Low damage (< $5000)
- Clear liability
- No injuries
- 2 or fewer parties

A claim is COMPLEX if ANY of these are true:
- High damage (>= $5000)
- Unclear liability
- Has injuries
- Commercial vehicle
- 3+ parties

Claim details:
- Description: {claim['description']}
- Damage estimate: ${claim['damage_estimate']}
- Parties involved: {claim['parties_involved']}
- Has injury: {claim['has_injury']}
- Liability clear: {claim['liability_clear']}
- Vehicle type: {claim['vehicle_type']}

Respond with JSON only: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}\"\"\"

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

## Tool Use Pattern

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "classify_claim",
        "description": "Classify an insurance claim as simple or complex",
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["simple", "complex"],
                    "description": "The classification result"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score 0-1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the classification"
                }
            },
            "required": ["classification", "confidence", "reasoning"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "classify_claim"},
    messages=[{
        "role": "user",
        "content": "Classify: Minor fender bender, $1200 damage, liability clear"
    }]
)

# Extract result from tool use
for block in response.content:
    if block.type == "tool_use":
        result = block.input
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']}")
```

## Key Classes

- `anthropic.Anthropic()` - Main client
- `client.messages.create()` - Send messages
- `tools` parameter - Define available tools
- `tool_choice` - Force specific tool usage
""",
    },
    "openai-agents": {
        "moderate": """# OpenAI Agents Documentation

## Overview

Use OpenAI's API with function calling for agent-like behavior.

## Installation

```bash
pip install openai
```

## Basic Classification

```python
from openai import OpenAI
import json

client = OpenAI()  # Uses OPENAI_API_KEY env var

def classify_claim(claim: dict) -> dict:
    \"\"\"Classify an insurance claim.\"\"\"
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": f\"\"\"Classify this insurance claim as 'simple' or 'complex'.

Claim: {claim}

Respond with JSON: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "..."}}\"\"\"
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

## With Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_claim",
            "description": "Classify an insurance claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {"type": "string", "enum": ["simple", "complex"]},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"}
                },
                "required": ["classification", "confidence", "reasoning"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[{"role": "user", "content": "Classify this claim..."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "classify_claim"}}
)

# Extract function call result
tool_call = response.choices[0].message.tool_calls[0]
result = json.loads(tool_call.function.arguments)
```
""",
        "full": """# OpenAI Agents Complete Documentation

## Installation

```bash
pip install openai
```

## Basic Usage

```python
from openai import OpenAI
import json

client = OpenAI()  # Uses OPENAI_API_KEY env var

def classify_claim(claim: dict) -> dict:
    \"\"\"Classify an insurance claim using GPT-4.\"\"\"

    prompt = f\"\"\"Classify this insurance claim as 'simple' or 'complex'.

A claim is SIMPLE if ALL of these are true:
- Low damage (< $5000)
- Clear liability
- No injuries
- 2 or fewer parties

A claim is COMPLEX if ANY of these are true:
- High damage (>= $5000)
- Unclear liability
- Has injuries
- Commercial vehicle
- 3+ parties

Claim details:
- Description: {claim['description']}
- Damage estimate: ${claim['damage_estimate']}
- Parties involved: {claim['parties_involved']}
- Has injury: {claim['has_injury']}
- Liability clear: {claim['liability_clear']}
- Vehicle type: {claim['vehicle_type']}

Respond with JSON only: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}\"\"\"

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

## Function Calling Pattern

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "classify_claim",
            "description": "Classify an insurance claim as simple or complex",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["simple", "complex"],
                        "description": "The classification result"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for classification"
                    }
                },
                "required": ["classification", "confidence", "reasoning"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[{
        "role": "user",
        "content": "Classify: Minor fender bender, $1200 damage, liability clear"
    }],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "classify_claim"}}
)

# Extract function call result
tool_call = response.choices[0].message.tool_calls[0]
result = json.loads(tool_call.function.arguments)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")
```

## Key Classes

- `OpenAI()` - Main client
- `client.chat.completions.create()` - Chat completion
- `tools` parameter - Define function tools
- `tool_choice` - Force specific function
- `response_format` - Request JSON output
""",
    },
    "langgraph": {
        "moderate": """# LangGraph Documentation

## Introduction

LangGraph is a library for building stateful, multi-step AI applications.

## Basic StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define state schema
class State(TypedDict):
    messages: list[str]
    result: str

# Create graph
graph = StateGraph(State)

# Define node functions
def process(state: State) -> dict:
    messages = state["messages"]
    return {"result": f"Processed {len(messages)} messages"}

def finalize(state: State) -> dict:
    return {"result": state["result"] + " - done"}

# Add nodes
graph.add_node("process", process)
graph.add_node("finalize", finalize)

# Add edges
graph.add_edge(START, "process")
graph.add_edge("process", "finalize")
graph.add_edge("finalize", END)

# Compile and run
app = graph.compile()
result = app.invoke({"messages": ["hello", "world"], "result": ""})
print(result["result"])  # "Processed 2 messages - done"
```

## Key Concepts

- **StateGraph**: The main graph class
- **START/END**: Special nodes for entry and exit
- **add_node**: Register a function as a node
- **add_edge**: Connect nodes
- **compile**: Build the runnable graph
""",
        "full": """# LangGraph Complete Documentation

## Installation

```bash
pip install langgraph
```

## Basic StateGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define state schema
class State(TypedDict):
    messages: list[str]
    result: str

# Create graph
graph = StateGraph(State)

# Define node functions
def process(state: State) -> dict:
    messages = state["messages"]
    return {"result": f"Processed {len(messages)} messages"}

def finalize(state: State) -> dict:
    return {"result": state["result"] + " - done"}

# Add nodes
graph.add_node("process", process)
graph.add_node("finalize", finalize)

# Add edges
graph.add_edge(START, "process")
graph.add_edge("process", "finalize")
graph.add_edge("finalize", END)

# Compile and run
app = graph.compile()
result = app.invoke({"messages": ["hello", "world"], "result": ""})
print(result["result"])  # "Processed 2 messages - done"
```

## Conditional Edges

Route to different nodes based on state:

```python
from langgraph.graph import StateGraph, START, END

def router(state: State) -> str:
    if state["result"] == "error":
        return "handle_error"
    return "continue"

graph.add_conditional_edges(
    "check",  # Source node
    router,   # Function that returns next node name
    {
        "handle_error": "error_node",
        "continue": "next_node",
    }
)
```

## With LLM Integration

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(model="gpt-4")

def call_llm(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

app = graph.compile()
result = app.invoke({"messages": ["Hello!"]})
```

## Checkpointing (Memory)

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Each thread_id maintains separate state
config = {"configurable": {"thread_id": "user-1"}}
result = app.invoke({"messages": ["Hello"]}, config)
```
""",
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
    parser.add_argument(
        "--prefer-curated",
        action="store_true",
        default=True,
        help="Prefer curated docs over fetched docs (default: True)",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetching from URLs even if curated docs exist",
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
        curated = CURATED_DOCS.get(framework, {})

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

        # Moderate level - prefer curated docs
        if "moderate" in args.levels:
            if not args.force_fetch and args.prefer_curated and "moderate" in curated:
                # Use curated documentation
                content = curated["moderate"]
                snapshot = corpus.add_document(
                    framework=framework,
                    version=args.version,
                    level=DocumentLevel.MODERATE,
                    content=content,
                    source_urls=["curated"],
                )
                console.print(f"  [green]Created moderate doc (curated): {snapshot.content_hash} ({snapshot.token_count} tokens)[/green]")
            elif sources.get("moderate"):
                # Fetch from URLs
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
                    console.print(f"  [green]Created moderate doc (fetched): {snapshot.content_hash} ({snapshot.token_count} tokens)[/green]")

        # Full level - prefer curated docs
        if "full" in args.levels:
            if not args.force_fetch and args.prefer_curated and "full" in curated:
                # Use curated documentation
                content = curated["full"]
                snapshot = corpus.add_document(
                    framework=framework,
                    version=args.version,
                    level=DocumentLevel.FULL,
                    content=content,
                    source_urls=["curated"],
                )
                console.print(f"  [green]Created full doc (curated): {snapshot.content_hash} ({snapshot.token_count} tokens)[/green]")
            elif sources.get("full"):
                # Fetch from URLs
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
                    console.print(f"  [green]Created full doc (fetched): {snapshot.content_hash} ({snapshot.token_count} tokens)[/green]")

    console.print(f"\n[bold green]Documentation saved to: {args.corpus_dir}[/bold green]")


if __name__ == "__main__":
    main()
