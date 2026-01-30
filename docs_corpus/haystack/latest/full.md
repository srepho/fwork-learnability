# Haystack v2 Complete Documentation

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
prompt_builder = PromptBuilder(template="""
Given the following context, answer the question.
Context: {{context}}
Question: {{question}}
Answer:
""")

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
