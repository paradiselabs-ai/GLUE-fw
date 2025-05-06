# GLUE + Smolagents Integration: Super Comprehensive Plan

---

## 1. Background & Context

### 1.1. GLUE Framework Overview
- **Purpose:** Modular, extensible framework for multi-model, multi-agent AI applications.
- **Core Concepts:**
  - **GlueApp:** Orchestrates models, teams, tools, flows, and memory.
  - **Models/Agents:** Use adhesives (GLUE, VELCRO, TAPE) to control persistence/sharing of tool outputs. *(Note: Adhesives are a GLUE-specific concept and not native to Smolagents; see mapping section for adaptation details.)*
  - **Teams:** Manage models, tool sharing, and result persistence.
  - **MagneticField/Flows:** Control team-to-team information flows (push, pull, repel, attract). *(Note: Magnetic flows are a GLUE-specific concept and not native to Smolagents; see mapping section for adaptation details.)*
  - **Knowledge Store:** Persistent, queryable memory for cross-team awareness.
  - **Self-Learning:** Layered memory (agent, team, global) with scoring.
  - **DSL & CLI:** Declarative language and CLI for configuration, onboarding, and extensibility.

### 1.2. Smolagents Overview
- **Purpose:** Lightweight, composable agentic framework for LLM-powered tool use, planning, and multi-step workflows.
- **Core Concepts:**
  - **Agent:** LLM-powered, can call tools, manage memory, and run multi-step loops.
  - **Tool:** Callable function with schema, description, and input/output types.
  - **Prompt Templating:** System prompt must include placeholders for tool/agent descriptions.
  - **Planning:** Optional extra planning step for fact reflection and next-step reasoning.
  - **Team/Managed Agents:** Agents can delegate to other agents (including humans) as tools.

---

## 2. Integration Goals

- **Replace or wrap GLUE's agent loop, tool, and (optionally) team abstractions with smolagents' equivalents.**
- **Preserve GLUE's unique features:** adhesives, magnetic flows, persistent memory, DSL, and CLI.
- **Remove or refactor obsolete code (agent loops, tool wrappers, etc.).**
- **Update documentation, CLI, and DSL to reflect the new architecture.**
- **Ensure all core workflows (agent creation, team orchestration, tool usage, information flow) are preserved and tested.**

---

## 3. Mapping GLUE to Smolagents

| GLUE Concept         | Smolagents Equivalent/Mapping                | Notes/Actions Needed                                  |
|----------------------|----------------------------------------------|-------------------------------------------------------|
| Model/Agent          | `Agent`/`CodeAgent`/`ToolCallingAgent`       | Replace or wrap GLUE Model/BaseModel with Smolagents. |
| Team                 | Managed agents (agents as tools)             | Map Team to a group of agents, possibly as tools.     |
| Tool                 | Smolagents Tool (callable + schema)          | Migrate to Smolagents tool format.                    |
| Agent Loop           | Smolagents agent run loop                    | Replace TeamMemberAgentLoop/TeamLeadAgentLoop.        |
| Adhesives            | Custom memory/persistence logic              | *(GLUE-specific: implement as wrappers or hooks in tool/agent memory; not native to Smolagents.)* |
| Magnetic Flows       | Message passing/agent delegation             | *(GLUE-specific: map to agent-to-agent calls or custom comms; not native to Smolagents.)* |
| Working Memory       | Smolagents memory or custom memory           | Map or wrap as needed.                                |
| Persistent Memory    | Custom (file-based)             | Integrate with smolagents memory if possible.         |
| Prompts              | Smolagents prompt templates                  | Use required placeholders (`{{tool_descriptions}}`, etc.) |
| CLI/DSL              | Update to instantiate smolagents agents      | Update parsing, config, and examples.                 |

---

## 4. Integration Plan: Actionable Steps

### Step 1: Research and Requirements
- Deeply review smolagents docs (see attached, e.g. [v1.14.0 tutorial](https://huggingface.co/docs/smolagents/v1.14.0/en/tutorials/building_good_agents)).
- Identify all required prompt placeholders and agent/tool APIs.
- List all GLUE features that must be preserved or adapted.

### Step 2: Abstraction Mapping & Adapter Design
- Map all GLUE agent, team, and tool abstractions to smolagents equivalents.
- Design adapters/wrappers for:
  - Adhesive persistence (GLUE/VELCRO/TAPE) as memory hooks or tool wrappers. *(GLUE-specific: this is a custom extension, not a Smolagents feature.)*
  - Magnetic flows as agent-to-agent calls or message passing. *(GLUE-specific: this is a custom extension, not a Smolagents feature.)*
  - Team orchestration as managed agents or agent groups.
- Document all mappings and any gaps.

### Step 3: Smolagents Integration Layer
- Implement a `GlueSmolAgent` class that wraps or extends smolagents' `Agent`/`CodeAgent`.
- Implement a `GlueSmolTool` class that wraps smolagents tools, adding adhesive and permission logic.
- Implement a `GlueTeam` abstraction if needed, as a group of managed agents.
- Implement memory adapters to support adhesives (team-wide, session, ephemeral). *(GLUE-specific: adhesives are implemented as custom memory logic, not a Smolagents feature.)*
- Implement prompt template generation to inject tool/agent descriptions as required by smolagents.

### Step 4: Refactor/Remove Obsolete Code
- Remove or refactor:
  - `TeamMemberAgentLoop`, `TeamLeadAgentLoop`
  - Old tool base classes and wrappers
  - Redundant memory/adhesive logic now handled by smolagents
  - Any orchestration code replaced by smolagents' planning/loop
- Update all references in teams, app, and CLI.

### Step 5: CLI, DSL, and Documentation Update
- Update CLI to instantiate and run smolagents-based agents and teams.
- Update DSL parser to generate smolagents agent/team/tool configs.
- Update documentation and examples to show smolagents-based workflows.
- Ensure prompt templates in code and DSL include required placeholders.

### Step 6: Testing and Validation
- Develop comprehensive tests for:
  - Agent creation, tool usage, team orchestration, information flow, adhesives.
  - Backward compatibility (where possible).
- Remove or update obsolete tests.
- Document test results and any issues.

### Step 7: Final Cleanup and Optimization
- Remove dead code, optimize integration layer.
- Ensure all new/modified components are well-documented.
- Update README and developer docs to explain the new architecture, usage, and extension points.

---

## 5. Key Technical Considerations

### 5.1. Prompt Engineering
- All smolagents system prompts **must** include:
  - `{{tool_descriptions}}`
  - `{{managed_agents_description}}` (if using managed agents/teams)
  - For `CodeAgent`: `{{authorized_imports}}`
- Prompts can be customized, but must retain these placeholders for tool/agent injection.

### 5.2. Tool/Agent Registration
- Tools must be registered with input/output schemas.
- Agents can be registered as tools for delegation (managed agents).
- Team orchestration can be implemented as a "lead" agent with managed agents as tools.

### 5.3. Adhesive Persistence
- Implement adhesive logic as memory hooks or wrappers:
  - **GLUE:** Team-wide persistent memory (shared context).
  - **VELCRO:** Session-based, agent-specific memory.
  - **TAPE:** Ephemeral, one-time-use memory.
- *(Note: Adhesives are a GLUE-specific extension; Smolagents does not provide this natively, so custom memory classes or wrappers are required.)*
- Map these to smolagents' memory or implement custom memory classes.

### 5.4. Magnetic Flows
- Map push/pull/repel/attract flows to agent-to-agent calls, message passing, or managed agent delegation.
- *(Note: Magnetic flows are a GLUE-specific extension; Smolagents does not provide this natively, so this is implemented as custom logic.)*
- Ensure flows are respected in team orchestration logic.

### 5.5. Planning and Reflection
- Use smolagents' planning interval and extra planning step for fact reflection and next-step reasoning.
- Integrate with GLUE's self-evaluation and refinement logic.

### 5.6. Backward Compatibility
- Where possible, provide adapters or migration guides for existing GLUE DSL/CLI users.
- Clearly document any breaking changes.

#### Adapting for GLUE
- Add logic to embed documents and queries using a selected embedding model (e.g., from Hugging Face or sentence-transformers).
- Ensure the tool exposes a schema compatible with GLUE's tool system and can be registered via the tool factory.

#### Requirements for Integration
- Provide a utility for ingesting and embedding documents.
- Ensure the retriever tool can be used by agents in both GLUE and Smolagents workflows.
- Document the RAG workflow and provide example usage in the docs and CLI/DSL.

#### Benefits
- Enables agentic RAG workflows in GLUE, allowing LLM agents to ground their answers in retrieved, semantically relevant documents.
- Follows Smolagents best practices for tool and agent design.

### 5.8. Smolagents Best Practices: Simplicity, Grouping, and Error Handling

- **Simplicity First:** Smolagents strongly recommends keeping agentic workflows as simple as possible. Whenever feasible, group related tool logic into a single tool to reduce the number of LLM calls, which lowers cost, latency, and error risk.
- **Deterministic Logic:** Prefer deterministic functions and direct tool calls over agentic decision-making when possible. This reduces the risk of LLM errors and makes the system more robust.
- **Minimize LLM Calls:** The main guideline is to reduce the number of LLM calls as much as you can. Combine tool logic and avoid unnecessary agentic steps.
- **Error Logging and Retry:** Well-programmed agentic systems should include error logging and retry mechanisms, so the LLM engine has a chance to self-correct mistakes. Integrate these mechanisms into your agent and tool wrappers.
- **Prompt Guidance:** Only change the system prompt when necessary, and always retain the required placeholders for tool and agent descriptions.

These best practices should guide all integration and extension work to ensure robust, efficient, and maintainable agentic systems.

---

## 6. Implementation Checklist

### A. Preparation
- [ ] Review all smolagents documentation and examples.
- [ ] List all GLUE features to preserve/adapt.

### B. Core Integration
- [ ] Implement `GlueSmolAgent` and `GlueSmolTool` wrappers.
- [ ] Implement adhesive memory adapters.
- [ ] Implement team/managed agent orchestration.
- [ ] Implement prompt template generation with required placeholders.

### C. Refactoring
- [ ] Remove obsolete agent loops and tool wrappers.
- [ ] Update all references in teams, app, and CLI.

### D. CLI/DSL/Docs
- [ ] Update CLI and DSL parser for smolagents.
- [ ] Update documentation and examples.

### E. Testing
- [ ] Develop and run comprehensive tests.
- [ ] Remove/update obsolete tests.

### F. Finalization
- [ ] Cleanup and optimize codebase.
- [ ] Update README and developer docs.

---

## 7. References & Further Reading

- [Smolagents v1.14.0 Tutorial](https://huggingface.co/docs/smolagents/v1.14.0/en/tutorials/building_good_agents)
- [Smolagents Conceptual Guide](https://huggingface.co/docs/smolagents/v1.14.0/en/conceptual_guides/intro_agents)
- [GLUE Framework Deep Analysis](../GLUE_Framework_Deep_Analysis.md)
- [GLUE Agent Loops](Agent_loops.md)
- [GLUE README](../README.md)

---

## 8. Appendix: Example Smolagents Integration Snippet

```python
from smolagents import InferenceClientModel
from glue.core.glue_smoltool import GlueSmolTool
from glue.core.glue_smolagent import GlueSmolAgent

# Wrap existing GLUE tools as Smolagents tools
tools = [GlueSmolTool(glue_tool1), GlueSmolTool(glue_tool2)]

# Create a Smolagents-based GLUE agent
agent = GlueSmolAgent(
    tools=tools,
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct"),
    planning_interval=3,
)
# The agent's system prompt template now includes these placeholders:
#   {{tool_descriptions}}
#   {{managed_agents_description}}
#   {{authorized_imports}}

result = agent.run("Your task here")
```

---

## 9. Summary Table: GLUE → Smolagents Migration

| Area                | Action/Mapping                                    |
|---------------------|--------------------------------------------------|
| Agent Loop          | Replace with smolagents agent run loop           |
| Tool                | Wrap as smolagents tool, add adhesive logic      |
| Team                | Managed agents or agent groups                   |
| Memory              | Map adhesives to smolagents/custom memory        |
| Flows               | Agent-to-agent calls or message passing          |
| Prompts             | Use smolagents templates with required placeholders |
| CLI/DSL             | Update to instantiate smolagents agents/teams    |
| Docs/Examples       | Update for new architecture                      |

---
- `src/glue/core/agent_loop.py`: Implements the main agentic loops for both team leads and team members, including task delegation, reporting, and orchestration logic.
- `src/glue/core/teams.py`: Manages team composition, starts agent loops, and provides task fetching and orchestration at the team level.
- `src/glue/tools/delegate_task_tool.py`: Implements the tool for delegating tasks from leads to members.
- `src/glue/tools/report_task_completion_tool.py`: Implements the tool for reporting task completion from members to leads.
- `src/glue/core/working_memory.py`, `src/glue/core/types.py`, `src/glue/core/schemas.py`: Define memory, enums, and data models used throughout the agentic loop.
- `src/glue/core/adhesive.py`: Implements adhesives (GLUE, VELCRO, TAPE) for memory and result persistence.
- `src/glue/core/flow.py`: Implements magnetic flows (push, pull, bidirectional, repel) for inter-team communication.

---

### 1. Core Features in GLUE Agentic Loop, Delegation, and Reporting

#### a. Agentic Loop
- **TeamLeadAgentLoop**: Orchestrates task decomposition, delegation, report evaluation, retry/escalation, and final synthesis.
- **TeamMemberAgentLoop**: Handles task fetching, context gathering, planning, tool invocation, self-evaluation, reporting, and memory management.

#### b. Delegation & Reporting Tools
- **delegate_task_tool**: Assigns tasks to team members, persists assignments, and notifies agents.
- **report_task_completion_tool**: Signals task completion, provides results to the lead, and terminates the agent's loop.

#### c. Memory & Persistence
- **WorkingMemory**: In-memory, per-agent working memory for context and planning.
- **PersistentMemory**: File-based, team-level persistent memory for leads.
- **Adhesive System**: Controls result persistence (GLUE = team-wide, VELCRO = session/agent, TAPE = ephemeral).

#### d. Magnetic Flows
- **Flow System**: Implements push, pull, bidirectional, and repel flows for inter-team communication.

#### e. Tool Registration & Execution
- **Tool Factory/Registry**: Dynamic tool creation, registration, and permission/adhesive management.
- **Tool Base Classes**: Define tool schemas, permissions, and execution logic.

#### f. Orchestration & Team Management
- **Team Class**: Manages team composition, agent loop startup, tool registration, and task fetching.

#### g. Prompt Engineering
- **Prompt Templates**: Customizable, but must include placeholders for tool/agent descriptions.

---

### 2. Features to Preserve, Adapt, or Remove for Smolagents Integration

| Feature/Logic                | Preserve | Adapt | Remove/Refactor | Notes/Actions Needed |
|------------------------------|----------|-------|-----------------|---------------------|
| TeamLeadAgentLoop            |          |   ✔   |        ✔        | Replace with Smolagents agent loop; preserve orchestration logic as needed. |
| TeamMemberAgentLoop          |          |   ✔   |        ✔        | Replace with Smolagents agent loop; preserve planning, self-eval, and reporting logic. |
| delegate_task_tool           |    ✔     |   ✔   |                 | Adapt to Smolagents tool format; preserve delegation logic. |
| report_task_completion_tool  |    ✔     |   ✔   |                 | Adapt to Smolagents tool format; preserve reporting logic. |
| WorkingMemory                |    ✔     |   ✔   |                 | Map to Smolagents memory or wrap as needed. |
| PersistentMemory             |    ✔     |   ✔   |                 | Integrate with Smolagents memory if possible. |
| Adhesive System (GLUE/VELCRO/TAPE) | ✔   |   ✔   |                 | Implement as custom memory hooks/wrappers in Smolagents. |
| Magnetic Flows               |          |   ✔   |                 | Map to agent-to-agent calls or custom comms in Smolagents. |
| Tool Factory/Registry        |    ✔     |   ✔   |                 | Adapt for Smolagents tool registration and schema. |
| Tool Base Classes            |          |   ✔   |        ✔        | Replace with Smolagents Tool base; preserve schema/permission logic. |
| Team Class                   |    ✔     |   ✔   |                 | Adapt team orchestration to use Smolagents managed agents. |
| Prompt Templates             |    ✔     |   ✔   |                 | Ensure required placeholders for Smolagents. |

---

### 3. Checklist of Features and Logic to Migrate or Refactor

- [ ] Replace TeamLeadAgentLoop and TeamMemberAgentLoop with Smolagents-compatible agent loop(s).
- [ ] Adapt delegate_task_tool and report_task_completion_tool to Smolagents Tool format.
- [ ] Map or wrap WorkingMemory and PersistentMemory for Smolagents compatibility.
- [ ] Implement adhesive memory logic (GLUE/VELCRO/TAPE) as custom memory hooks/wrappers in Smolagents.
- [ ] Map magnetic flows to agent-to-agent calls or custom communication logic in Smolagents.
- [ ] Adapt tool factory/registry for Smolagents tool registration and schema requirements.
- [ ] Replace old tool base classes with Smolagents Tool base, preserving schema and permission logic.
- [ ] Adapt Team class to use Smolagents managed agents for orchestration.
- [ ] Update prompt templates to include required Smolagents placeholders.

### Step 5: CLI, DSL, and Documentation Update
- Update CLI to instantiate and run smolagents-based agents and teams.
- Update DSL parser to generate smolagents agent/team/tool configs.
- Update documentation and examples to show smolagents-based workflows.
- *Example DSL snippet with Smolagents-specific options:*
  ```glue
  model researcher {
      provider = openrouter
      adhesives = [glue, velcro]
      config {
          model = "meta-llama/llama-4-maverick:free"
          planning_interval = 3     // Smolagents extra planning interval
          system_prompt     = "You are an expert researcher who summarizes findings."
      }
  }
  ```
- Ensure prompt templates in code and DSL include required placeholders.

