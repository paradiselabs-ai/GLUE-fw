# GLUE + Smolagents Integration: Sequential Implementation Tasks

This document lists the actionable, sequential tasks required to migrate the GLUE agentic loop, delegation, and reporting to a Smolagents-based architecture, as described in the integration plan. Tasks are grouped by phase and ordered for dependency, with checkboxes for progress tracking.

---

## A. Preparation
- [DONE] - **Review Smolagents Documentation**
  - Deeply review Smolagents docs and tutorials to ensure full understanding of required APIs, prompt placeholders, and best practices.
- [DONE] - **Inventory GLUE Features to Preserve/Adapt** Read docs\GLUE_Smolagents_Features_Inventory.md
  - List all GLUE features and logic that must be preserved, adapted, or removed for Smolagents integration.

---

## B. Core Integration
- [DONE] - **Design Abstraction Mapping & Adapters** Read docs\GLUE_Smolagents_Abstraction_Mapping.md
  - Map all GLUE agent, team, and tool abstractions to Smolagents equivalents.
  - Design adapters/wrappers for adhesives, magnetic flows, and team orchestration.
- [DONE] - **Implement `GlueSmolAgent` Wrapper**
  - Create a class that wraps or extends Smolagents' `Agent`/`CodeAgent` to support GLUE-specific features.
- [DONE] - **Implement `GlueSmolTool` Wrapper**
  - Create a class that wraps Smolagents tools, adding adhesive and permission logic.
- [DONE] - **Implement Adhesive Memory Adapters**
  - Implement custom memory classes or hooks for GLUE, VELCRO, and TAPE adhesives in the Smolagents context.
- [DONE] - **Implement Team/Managed Agent Orchestration**
  - Map GLUE teams to Smolagents managed agents or agent groups, supporting delegation and reporting.
- [DONE] - **Implement Prompt Template Generation**
  - Ensure all system prompts include required Smolagents placeholders (`{{tool_descriptions}}`, `{{managed_agents_description}}`, etc.).

---

## C. Refactoring
- [DONE] - **Remove Obsolete Agent Loops and Tool Wrappers**
  - Remove or refactor `TeamLeadAgentLoop`, `TeamMemberAgentLoop`, and old tool base classes/wrappers.
- [DONE] - **Update References in Teams, App, and CLI**
  - Refactor all references to use the new Smolagents-based abstractions.

---

## D. CLI/DSL/Docs
- [DONE] - **Update CLI to Use Smolagents Agents/Teams**
  - Refactor CLI logic to instantiate and run Smolagents-based agents and teams.
- [DONE] - **Update DSL Parser for Smolagents**
  - Update DSL parsing logic to generate Smolagents agent/team/tool configs.
- [ ] **Update Documentation and Examples**
  - Revise documentation and code examples to reflect the new architecture and workflows.
- [DONE] - **Ensure Prompt Templates in Code and DSL**
  - Verify that all prompt templates include required Smolagents placeholders.

---

## E. Testing
- [ ] **Develop Comprehensive Tests**
  - Write tests for agent creation, tool usage, team orchestration, information flow, and adhesives in the new architecture.
- [ ] **Remove/Update Obsolete Tests**
  - Remove or update tests that target obsolete logic.
- [ ] **Document Test Results and Issues**
  - Record test outcomes and any issues encountered during migration.

---

## F. Finalization
- [ ] **Cleanup and Optimize Codebase**
  - Remove dead code, optimize the integration layer, and ensure maintainability.
- [ ] **Update README and Developer Docs**
  - Document the new architecture, usage, and extension points for developers.

---

This checklist should be used to guide and track the iterative implementation of the GLUE + Smolagents integration. Each task should be completed and checked off in order, with code reviews and testing at each phase.
