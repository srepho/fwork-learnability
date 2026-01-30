# LangGraph Documentation

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
