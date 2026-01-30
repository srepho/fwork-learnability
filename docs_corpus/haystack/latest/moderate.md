# Haystack v2 Documentation

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

## Key Components
- **PromptBuilder**: Builds prompts from templates with variables
- **OpenAIGenerator**: Generates text using OpenAI models
- **Pipeline**: Connects components together
- **Secret**: Securely handles API keys
