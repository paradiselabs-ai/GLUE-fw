# GLUE to Smolagents Abstraction Mapping & Adapter Design

This document maps GLUE framework concepts to their Smolagents equivalents and outlines adapter design notes for implementation.

## Mapping Table

| GLUE Concept            | Smolagents Equivalent            | Adapter/Wrapper Design Notes |
|-------------------------|----------------------------------|------------------------------|
| Model/Agent             | `Agent` / `CodeAgent`            | - Implement `GlueSmolAgent` wrapping Smolagents Agent
- Inject GLUE adhesives, flow configurations, and team context
- Override `run` methods to manage GLUE tool loops and memory hooks |
| Tool                    | `Tool` / `RetrieverTool`         | - Create `GlueSmolTool` subclass of Smolagents Tool
- Bridge GLUE `ToolConfig` & `ToolResult` to Smolagents schema
- Apply adhesive and permission logic around `forward` |
| Adhesive Persistence    | Smolagents `Memory` or custom    | - Define `GlueMemory` (persistent), `VelcroMemory` (session), `TapeMemory` (ephemeral)
- Implement Smolagents Memory interface for each adhesive
- Attach appropriate memory to agents based on adhesive type |
| Magnetic Flows          | Managed Agents & message passing | - Build `GlueTeam` orchestrator registering managed agents
- Represent flows as callbacks/message hooks between agents
- Wrap Smolagents managed agents to enforce flow rules |
| Team                    | Group of managed agents          | - Implement `GlueTeam` grouping `GlueSmolAgent` instances
- Register team members in a lead agent's `managed_agents`
- Expose delegation APIs using Smolagents agent-as-tool pattern |
| Prompt Templates        | Smolagents system prompt         | - Use templates with placeholders: `{{tool_descriptions}}`, `{{managed_agents_description}}`, `{{authorized_imports}}`
- Extend prompt generation to inject GLUE-specific context and placeholders |
| DSL & CLI               | Smolagents config generation     | - Refactor DSL parser to emit Smolagents agent/tool JSON configs
- Update CLI (`cli.py`) to instantiate `GlueSmolAgent`/`GlueTeam` instead of BaseModel/GlueApp |
| Planning & Reflection   | `planning_interval` in agent     | - Surface GLUE self-evaluation as Smolagents extra planning step
- Pass GLUE reflection frequency to `GlueSmolAgent.planning_interval` |
| Persistent Memory Store | Smolagents RetrieverTool pattern | - Implement `VectorDBRetrieverTool` subclassing Smolagents Tool
- Expose schema compatible with both GLUE and Smolagents | 