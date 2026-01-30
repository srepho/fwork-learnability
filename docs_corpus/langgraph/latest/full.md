# Reference[¶](#reference "Permanent link")

Welcome to the LangGraph reference docs! These pages detail the core interfaces you will use when building with LangGraph. Each section covers a different part of the ecosystem.

Tip

If you are just getting started, see [LangGraph basics](../concepts/why-langgraph/) for an introduction to the main concepts and usage patterns.

## LangGraph[¶](#langgraph "Permanent link")

The core APIs for the LangGraph open source library.
- [Graphs](graphs/): Main graph abstraction and usage.
- [Functional API](func/): Functional programming interface for graphs.
- [Pregel](pregel/): Pregel-inspired computation model.
- [Checkpointing](checkpoints/): Saving and restoring graph state.
- [Storage](store/): Storage backends and options.
- [Caching](cache/): Caching mechanisms for performance.
- [Types](types/): Type definitions for graph components.
- [Config](config/): Configuration options.
- [Errors](errors/): Error types and handling.
- [Constants](constants/): Global constants.
- [Channels](channels/): Message passing and channels.

## Prebuilt components[¶](#prebuilt-components "Permanent link")

Higher-level abstractions for common workflows, agents, and other patterns.
- [Agents](agents/): Built-in agent patterns.
- [Supervisor](supervisor/): Orchestration and delegation.
- [Swarm](swarm/): Multi-agent collaboration.
- [MCP Adapters](mcp/): Integrations with external systems.

## LangGraph Platform[¶](#langgraph-platform "Permanent link")

Tools for deploying and connecting to the LangGraph Platform.
- [SDK (Python)](../cloud/reference/sdk/python_sdk_ref/): Python SDK for interacting with instances of the LangGraph Server.
- [SDK (JS/TS)](../cloud/reference/sdk/js_ts_sdk_ref/): JavaScript/TypeScript SDK for interacting with instances of the LangGraph Server.
- [RemoteGraph](remote_graph/): `Pregel` abstraction for connecting to LangGraph Server instances.

See the [LangGraph Platform reference](https://docs.langchain.com/langgraph-platform/reference-overview) for more reference documentation.

Back to top
