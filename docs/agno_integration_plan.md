# Agno Integration Plan for GLUE Framework

## Overview

This plan outlines the steps to integrate the Agno framework as the core engine for the GLUE framework while retaining GLUE's unique features: the StickyScript DSL, CLI, team structures, adhesives, and magnetic flow operators. The integration will follow Test-Driven Development (TDD) principles throughout.

## Integration Architecture

We will use the Adapter pattern to bridge GLUE's unique features with Agno's core components. This approach:

- Preserves GLUE's unique features and interfaces
- Leverages Agno's robust core components
- Maintains backward compatibility
- Allows for clean separation of concerns

## Implementation Tasks

1. TDD: Create Agno Dependency Integration (Complexity: 5)
   - RED: Write failing tests that verify Agno can be imported and initialized
   - GREEN: Add Agno as a project dependency and implement minimal initialization code
   - REFACTOR: Clean up the integration code
   - **Status**: Completed. Agno installed as dependency via pip.

2. TDD: Setup Agno Base Integration in CLI (Complexity: 6)
   - RED: Write failing test(s) for basic Agno Workflow/Team initialization and execution within `cli.py`
   - GREEN: Modify `cli.py` to add minimal Agno initialization/execution while maintaining the existing CLI interface
   - REFACTOR: Clean up the initialization code
   - **Status**: Completed. CLI now supports `--engine agno` flag and executes minimal Agno workflow.

3. TDD: Create GlueAgnoAdapter Class (Complexity: 8)
   - RED: Write failing tests for the adapter class that verify it can translate between GLUE and Agno concepts
   - GREEN: Implement the adapter class with minimal functionality to pass the tests
   - REFACTOR: Clean up the implementation
   - **Status**: Completed. Created adapter class in `src/glue/core/adapters/agno/adapter.py` with unit tests.

4. TDD: Integrate GLUE DSL Parser with Agno Config (Complexity: 8)
   - RED: Write failing test(s) for translating a basic GLUE DSL AST into Agno Workflow/Team/Agent config
   - GREEN: Implement the translation logic in or called by `dsl/parser.py` to make tests pass
   - REFACTOR: Clean up translation code
   - **Status**: Completed.

5. TDD: Implement Agno Team Integration (Complexity: 7)
   - RED: Write failing test(s) for translating GLUE teams to Agno teams
   - GREEN: Implement team translation logic
   - REFACTOR: Clean up team integration code
   - **Status**: Completed

6. **TDD: Implement Agno Tool Integration** (Complexity: 7)
   - RED: Write failing test(s) for translating GLUE tools to Agno tools
   - GREEN: Implement tool translation logic
   - REFACTOR: Clean up tool integration code
   - Status: ✅ Completed

7. **TDD: Implement Agno Adhesive Integration** (Complexity: 9)
   - RED: Write failing test(s) for translating GLUE adhesives to Agno persistence
   - GREEN: Implement adhesive translation logic
   - REFACTOR: Clean up adhesive integration code
   - **Status**: ✅ Completed

8. **TDD: Implement Agno Magnetic Flow Integration** (Complexity: 9)
   - RED: Write failing test(s) for translating GLUE magnetic flows to Agno team connections
   - GREEN: Implement flow translation logic
   - REFACTOR: Clean up flow integration code
   - **Status**: ✅ Completed

9. **TDD: Update CLI to Use Agno as Default Engine** (Complexity: 5)
   - RED: Write failing test(s) for CLI using Agno as default engine
   - GREEN: Update CLI to use Agno by default
   - REFACTOR: Remove engine option and clean up CLI code
   - **Status**: ✅ Completed

10. **TDD: Create End-to-End Integration Tests** (Complexity: 8)
    - RED: Write failing end-to-end test(s) for complete GLUE app running on Agno
    - GREEN: Implement any missing functionality to make tests pass
    - REFACTOR: Clean up integration code
    - Status: 📝 Planned
    - Description: Write and run comprehensive integration tests covering end-to-end scenarios involving multiple components (DSL parsing, Agno execution, Team structure, Tools, Adhesives, Magnetic Flow). Ensure all retained GLUE features work correctly together on the Agno base. Refine the integrated system based on test results.

## Implementation Approach

For each task:

1. Start with the RED phase by writing failing tests
2. Implement the minimal code needed to make tests pass (GREEN phase)
3. Refactor the code while keeping tests passing
4. Update the pass list with the status of each test

## Directory Structure

The integration will add new adapter classes in a new directory:

```python
src/glue/core/adapters/agno/
├── __init__.py
├── adhesive.py        # Adhesive adapter for Agno memory
├── cli.py             # CLI adapter for Agno execution
├── magnetic.py        # Magnetic field adapter for Agno teams
├── model.py           # Model adapter for Agno agents
├── team.py            # Team adapter for Agno teams
└── tool.py            # Tool adapter for Agno tools
```

## Testing Strategy

- Unit tests for each adapter class
- Integration tests for combinations of adapters
- End-to-end tests for complete workflows
- All tests must follow TDD principles and use actual implementations (no mocks)