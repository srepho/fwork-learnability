# Anthropic Agents SDK Documentation

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
    """Classify an insurance claim using Claude."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Classify this insurance claim as 'simple' or 'complex'.

Claim: {claim}

Respond with JSON: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "..."}}"""
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
