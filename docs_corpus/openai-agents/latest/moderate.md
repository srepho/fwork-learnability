# OpenAI Agents Documentation

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
    """Classify an insurance claim."""
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": f"""Classify this insurance claim as 'simple' or 'complex'.

Claim: {claim}

Respond with JSON: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "..."}}"""
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
