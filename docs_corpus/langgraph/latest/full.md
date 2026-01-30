# LangGraph Complete Documentation

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
