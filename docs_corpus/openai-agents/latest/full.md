# OpenAI Agents Complete Documentation

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
    """Classify an insurance claim using GPT-4."""

    prompt = f"""Classify this insurance claim as 'simple' or 'complex'.

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

Respond with JSON only: {{"classification": "simple" or "complex", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

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
