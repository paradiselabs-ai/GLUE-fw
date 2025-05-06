# Inventory of GLUE Features to Preserve/Adapt

This document lists all GLUE features and logic that must be preserved, adapted, or removed in the Smolagents integration.

## Features to Preserve/Adapt

### 1. Adhesives
- GLUE: Team-wide persistent memory
- VELCRO: Session-based, agent-specific memory
- TAPE: Ephemeral, one-time-use memory

### 2. Magnetic Flows
- Push, pull, repel, attract flows between teams
- Flow configuration and routing logic

### 3. Agent Loops & Orchestration
- `TeamLeadAgentLoop` and `TeamMemberAgentLoop`
- `GlueApp` orchestration layer
- `BaseModel` generation and tool call loop

### 4. Tools & Tool Wrappers
- GLUE tool abstraction (`add_tool`, `add_tool_sync`)
- Tool schemas and execution logic
- `ToolResult` handling

### 5. Teams & Delegation
- `Team` structure: members, lead, communication flows
- Team context injection into prompts

### 6. Prompt Engineering & Templates
- System prompt generation (`_generate_system_prompt`)
- Placeholders and formatting functions

### 7. Knowledge Store & Memory
- Cross-team persistent store
- Queryable memory interfaces

### 8. DSL & CLI
- Declarative DSL parsing logic
- CLI commands for agents, teams, and tools

### 9. Self-Learning & Reflection
- Layered memory scoring and self-evaluation loops

### 10. Logging, Tracing & Debugging
- Logging (`logger`) and trace ID usage
- Development mode and debug flags
