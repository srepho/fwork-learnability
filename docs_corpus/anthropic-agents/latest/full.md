# Anthropic Agents SDK Complete Documentation

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
    """Classify an insurance claim using Claude."""

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
