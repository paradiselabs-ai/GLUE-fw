# GLUE-Agno Team Integration Strategy

## Current GLUE Team Implementation

The GLUE `Team` class acts as a container and manager for a collection of `Model` instances (agents). Key characteristics:

### Core Components
- **Models**: Dictionary mapping model names to `Model` instances
- **Lead**: Optional `Model` instance representing the team lead
- **Tools**: Dictionary mapping tool names to tool instances
- **Shared Results**: Dictionary storing tool results for sharing
- **Conversation History**: List of `Message` objects

### Flow Management
- **Message Queue**: Asynchronous queue for handling messages
- **Incoming/Outgoing Flows**: Lists for managing message flows between teams
- **Agent Loops**: Dictionary tracking active TeamMember and TeamLead loops

### Key Methods
- **add_member/add_member_sync**: Adds a GLUE `Model` to the team
- **add_tool**: Adds a tool to the team's tools dictionary
- **process_message**: Processes messages within the team, including tool execution
- **direct_communication**: Handles communication between two models within the team
- **start_agent_loops**: Starts TeamLead and TeamMember agent loops
- **send_information/receive_information**: Handles inter-team communication

## Agno Team Implementation

Agno's `Team` class acts as an agent itself with its own AI model. Key characteristics:

### Core Components
- **Model**: The "lead" AI model (e.g., GPT-4) that orchestrates the team
- **Members**: Collection of `Agent` or `Team` instances
- **Mode**: Collaboration mode (collaborate, coordinate, route)
- **Instructions**: Directives for the lead model on how to manage the team

### Key Methods
- **run/arun**: Executes the team's task, orchestrating member contributions
- **add_member**: Adds an agent or sub-team to the team
- **forward_task_to_member**: Tool for the lead model to delegate tasks

## Integration Strategy

### 1. Refactor GLUE Team to Wrap Agno Team

```python
class Team:
    def __init__(self, name, config=None, lead=None, members=None):
        # Initialize GLUE Team components
        self.name = name
        self.config = config or TeamConfig(name=name, lead="", members=[], tools=[])
        self._tools = {}
        self.shared_results = {}
        self.conversation_history = []
        
        # Create Agno Team
        self.agno_team = agno.Team(
            name=name,
            model="gpt-4",  # Default model, can be configured
            mode="coordinate",  # Default mode, can be configured
            instructions=self._generate_instructions()
        )
        
        # Handle backward compatibility
        if lead is not None:
            self.add_member_sync(lead, role="lead")
        if members is not None:
            for member in members:
                self.add_member_sync(member)
```

### 2. Adapt GLUE Models to Agno Agents

Create adapter classes that wrap GLUE `Model` instances to be compatible with Agno's `Agent` interface:

```python
class GlueModelAdapter(agno.Agent):
    def __init__(self, glue_model):
        self.glue_model = glue_model
        super().__init__(
            name=glue_model.name,
            model=glue_model.model_name,
            instructions=self._get_model_instructions()
        )
        
    async def arun(self, task):
        # Convert Agno task to GLUE format
        response = await self.glue_model.generate(task)
        # Process tool calls if present
        return response
```

### 3. Tool Integration

Adapt GLUE tools to be usable by Agno:

```python
def create_agno_tool_from_glue_tool(glue_tool, name):
    async def tool_wrapper(*args, **kwargs):
        # Call GLUE tool
        result = await glue_tool.execute(*args, **kwargs)
        return result
    
    # Set metadata
    tool_wrapper.__name__ = name
    tool_wrapper.__doc__ = getattr(glue_tool, "__doc__", f"Tool: {name}")
    
    return tool_wrapper
```

### 4. Message Flow Handling

Adapt GLUE's message flow system to work with Agno's direct communication approach:

```python
async def send_information(self, target_team, content):
    # Find the target Agno team
    target = self.app.teams.get(target_team)
    if not target:
        return {"success": False, "error": f"Team {target_team} not found"}
    
    # Use Agno's communication mechanism
    result = await target.agno_team.arun(content)
    return {"success": True, "result": result}
```

### 5. Agent Loop Integration

Replace GLUE's TeamLead and TeamMember loops with Agno's orchestration:

```python
async def start_agent_loops(self, initial_input=None):
    if initial_input:
        # Start the Agno team with the initial input
        asyncio.create_task(self.agno_team.arun(initial_input))
```

## Migration Path

1. **Phase 1**: Create adapter classes without changing GLUE's core functionality
2. **Phase 2**: Gradually replace GLUE's orchestration with Agno's
3. **Phase 3**: Refactor GLUE Teams to be thin wrappers around Agno Teams
4. **Phase 4**: Update DSL and CLI to work with the new implementation

## Benefits of Integration

1. **Simplified Orchestration**: Leverage Agno's proven orchestration patterns
2. **Reduced Complexity**: Remove custom message passing and agent loop logic
3. **Improved Reliability**: Use Agno's tested approach to agent collaboration
4. **Enhanced Capabilities**: Gain access to Agno's collaboration modes
5. **Maintainability**: Reduce custom code that needs to be maintained

## Challenges and Considerations

- **Backward Compatibility**: While current GLUE applications may not yet function under Agno, once all outlined steps are implemented, they will continue to work seamlessly.
- **Performance**: Monitor for any performance impacts during the transition
- **Testing**: Comprehensive testing to ensure equivalent functionality
- **Documentation**: Update documentation to reflect the new architecture

## Next Steps and Roadmap

1. **Write Tests (RED Phase)**: Define minimal failing tests for each adapter and integration point (e.g., `GlueModelAdapter`, tool wrappers).
2. **Implement Adapters (GREEN Phase)**: Develop minimal code to satisfy the tests for model and tool adapters.
3. **Refactor & Integrate (REFACTOR Phase)**: Clean up code, integrate adapters into CLI/DSL, and ensure backward compatibility.
4. **Update CLI/DSL**: Extend `glue run` and StickyScript syntax to support the Agno engine and new adapter classes.
5. **Finalize Documentation**: Update `projectOverview.md`, `currentPhase.md`, and the pass list in `progressFiles/pass list.md` to reflect completed tasks.

## Accelerated Timeline & Milestones

- **Day 1**: Write failing tests for model and tool adapters (RED Phase) and implement minimal adapters (GREEN Phase).
- **Day 2**: Write tests and implement tool wrappers for GLUE tools.
- **Day 3**: Integrate adapters into CLI/DSL and execute end-to-end minimal workflows.
- **Day 4**: Refactor implementation, update `pass list.md`, `projectOverview.md`, and `progressFiles` files; revise documentation.
- **Day 5**: Final QA—ensure all tests pass, CI pipeline is green, and prepare for alpha release.

## References

- Migration Path (above)
- GLUE TDD Guidelines (`CLAUDE.md`)
- `progressFiles/` project file hierarchy
