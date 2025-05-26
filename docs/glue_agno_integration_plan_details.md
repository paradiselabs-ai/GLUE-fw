# GLUE-Agno Integration Implementation Plan - Part 1

## Overview
This plan outlines the steps needed to enhance the GLUE-Agno integration by properly mapping GLUE Models to Agno Agents with system prompts and implementing workflow integration, while preserving GLUE's unique features including team lead orchestration, magnetic flows, polarity-based communication, and StickyScript DSL support. The implementation follows Test-Driven Development (TDD) principles, ensuring high-quality code with real implementations rather than mocks.

## Implementation Tasks

### Phase 1: System Prompt Integration

#### Task 1: Create test for system prompt mapping to AgnoAgent
**Description:** Create a new test method in TestGlueTeam class that verifies GLUE Model system prompts are correctly mapped to Agno Agent system_prompt parameters. This test will:
1. Create a GLUE Model with a specific system prompt
2. Add it to a GlueTeam
3. Call _initialize_agno_team
4. Verify the created AgnoAgent has the correct system prompt

This follows TDD principles by creating the test before implementing the feature.

**Complexity:** 3

**Code Example:**
```python
@pytest.mark.asyncio
async def test_initialize_agno_team_maps_system_prompts(self, basic_glue_team: GlueTeam):
    """
    Tests that _initialize_agno_team correctly maps GLUE Model system prompts
    to Agno Agent system_prompt parameters.
    """
    # Create a model with a specific system prompt
    test_system_prompt = "You are a specialized test agent with custom instructions."
    model_config = {
        "name": "test_model_with_prompt",
        "provider": "test",
        "model_name": "test_model_variant_1",
        "system_prompt": test_system_prompt
    }
    model_with_prompt = Model(config=model_config)
    
    # Add the model to the team
    await basic_glue_team.add_member(model_with_prompt)
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None
    assert hasattr(basic_glue_team.agno_team, 'members')
    
    # Find the agent corresponding to our test model
    test_agent = None
    for agent in basic_glue_team.agno_team.members:
        if agent.name == f"{model_with_prompt.name}_agno_proxy":
            test_agent = agent
            break
    
    assert test_agent is not None, f"Agent for {model_with_prompt.name} not found in Agno team"
    assert hasattr(test_agent, 'system_prompt'), "Agno Agent does not have system_prompt attribute"
    assert test_agent.system_prompt == test_system_prompt, "System prompt was not correctly mapped"
```

#### Task 2: Enhance AgnoAgent creation with system prompts from GLUE Models
**Description:** Update the _initialize_agno_team method in GlueTeam to properly map GLUE Model system prompts to Agno Agent system_prompt parameter. This involves:
1. Extract system prompts from GLUE Model instances
2. Pass these prompts to AgnoAgent constructor
3. Handle cases where system prompts might not be available

This builds on the existing placeholder implementation that currently only sets the agent name.

**Complexity:** 4

**Code Example:**
```python
# In _initialize_agno_team method:
for model_name, glue_model_instance in self.models.items():
    try:
        # Extract system prompt from GLUE Model
        system_prompt = glue_model_instance.config.get("system_prompt", "")
        if not system_prompt and hasattr(glue_model_instance, "get_system_prompt"):
            system_prompt = glue_model_instance.get_system_prompt()
        
        # Create Agno Agent with proper system prompt
        agno_agent = AgnoAgent(
            name=f"{glue_model_instance.name}_agno_proxy",
            system_prompt=system_prompt or f"You are {glue_model_instance.name}, a helpful assistant in the {self.name} team."
        )
        agno_agents_list.append(agno_agent)
        logger.info(f"Created AgnoAgent: {agno_agent.name} with system prompt for GLUE model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to create AgnoAgent for GLUE model {model_name}: {e}")
```

### Phase 2: LLM Configuration Integration

#### Task 3: Create test for LLM configuration mapping to AgnoAgent
**Description:** Create a new test method in TestGlueTeam class that verifies GLUE Model LLM configuration (model_name, temperature, max_tokens) is correctly mapped to Agno Agent parameters. This test will:
1. Create a GLUE Model with specific LLM configuration
2. Add it to a GlueTeam
3. Call _initialize_agno_team
4. Verify the created AgnoAgent has the correct LLM configuration

This follows TDD principles by creating the test before implementing the feature, ensuring we're using real implementations rather than mocks.

**Complexity:** 3

**Code Example:**
```python
@pytest.mark.asyncio
async def test_initialize_agno_team_maps_llm_configuration(self, basic_glue_team: GlueTeam):
    """
    Tests that _initialize_agno_team correctly maps GLUE Model LLM configuration
    to Agno Agent parameters.
    """
    # Create a model with specific LLM configuration
    test_model_name = "gpt-4"
    test_temperature = 0.5
    test_max_tokens = 2000
    model_config = {
        "name": "test_model_with_config",
        "provider": "test",
        "model": test_model_name,
        "temperature": test_temperature,
        "max_tokens": test_max_tokens
    }
    model_with_config = Model(config=model_config)
    
    # Add the model to the team
    await basic_glue_team.add_member(model_with_config)
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None
    assert hasattr(basic_glue_team.agno_team, 'members')
    
    # Find the agent corresponding to our test model
    test_agent = None
    for agent in basic_glue_team.agno_team.members:
        if agent.name == f"{model_with_config.name}_agno_proxy":
            test_agent = agent
            break
    
    assert test_agent is not None, f"Agent for {model_with_config.name} not found in Agno team"
    
    # Verify LLM configuration was mapped correctly
    assert hasattr(test_agent, 'model'), "Agno Agent does not have model attribute"
    assert test_agent.model == test_model_name, "Model name was not correctly mapped"
    
    assert hasattr(test_agent, 'temperature'), "Agno Agent does not have temperature attribute"
    assert test_agent.temperature == test_temperature, "Temperature was not correctly mapped"
    
    assert hasattr(test_agent, 'max_tokens'), "Agno Agent does not have max_tokens attribute"
    assert test_agent.max_tokens == test_max_tokens, "Max tokens was not correctly mapped"
```

#### Task 4: Map GLUE Model LLM configuration to AgnoAgent parameters
**Description:** Update the _initialize_agno_team method to map GLUE Model LLM configuration (like model_name, temperature, max_tokens) to corresponding AgnoAgent constructor parameters. This involves:
1. Extract LLM configuration from GLUE Model instances
2. Map these parameters to the appropriate AgnoAgent constructor parameters
3. Handle default values when specific parameters aren't available

This ensures that the Agno Agents use the same LLM configuration as their corresponding GLUE Models.

**Complexity:** 5

**Code Example:**
```python
# In _initialize_agno_team method:
for model_name, glue_model_instance in self.models.items():
    try:
        # Extract system prompt from GLUE Model
        system_prompt = glue_model_instance.config.get("system_prompt", "")
        if not system_prompt and hasattr(glue_model_instance, "get_system_prompt"):
            system_prompt = glue_model_instance.get_system_prompt()
        
        # Extract LLM configuration
        model_name = glue_model_instance.model_name
        temperature = glue_model_instance.temperature
        max_tokens = glue_model_instance.max_tokens
        
        # Create Agno Agent with proper configuration
        agno_agent = AgnoAgent(
            name=f"{glue_model_instance.name}_agno_proxy",
            model=model_name,  # Use the same model as GLUE
            system_prompt=system_prompt or f"You are {glue_model_instance.name}, a helpful assistant in the {self.name} team.",
            temperature=temperature,
            max_tokens=max_tokens,
            # Other parameters as needed
        )
        agno_agents_list.append(agno_agent)
        logger.info(f"Created AgnoAgent: {agno_agent.name} with model {model_name} for GLUE model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to create AgnoAgent for GLUE model {model_name}: {e}")
```

### Phase 3: Team Lead Orchestration

#### Task 5: Create test for team lead orchestration in Agno integration
**Description:** Create a test that verifies the GLUE team lead orchestration model is preserved in the Agno integration. This test will:
1. Create a GlueTeam with a designated lead model and multiple member models
2. Verify the lead model is properly designated as the orchestrator in the Agno team
3. Test that the lead model can properly coordinate the member models
4. Ensure the hierarchical team structure is maintained in the Agno implementation

This preserves GLUE's core team structure where the team lead acts as an orchestrator for team members.

**Complexity:** 5

**Code Example:**
```python
@pytest.mark.asyncio
async def test_agno_integration_preserves_team_lead_orchestration(self, basic_glue_team: GlueTeam):
    """
    Tests that the Agno integration preserves GLUE's team lead orchestration model.
    """
    # Create additional member models
    member1_config = {
        "name": "member1_model",
        "provider": "test",
        "model_name": "test_model_variant_1"
    }
    member1 = Model(config=member1_config)
    
    member2_config = {
        "name": "member2_model",
        "provider": "test",
        "model_name": "test_model_variant_1"
    }
    member2 = Model(config=member2_config)
    
    # Add members to the team
    await basic_glue_team.add_member(member1)
    await basic_glue_team.add_member(member2)
    
    # Verify the team structure
    assert basic_glue_team.lead is not None, "Team lead is not set"
    assert len(basic_glue_team.models) == 3, "Team should have 3 models (1 lead + 2 members)"
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None, "Agno team was not created"
    
    # Verify the team lead is properly designated in the Agno team
    # This will depend on how we implement the lead designation in Agno
    # For example, it might be the first agent in the members list, or have a special role
    agno_team = basic_glue_team.agno_team
    
    # Check if the lead model is properly designated
    # This might be through a 'lead' attribute, a 'role' property, or another mechanism
    lead_found = False
    for agent in agno_team.members:
        if agent.name == f"{basic_glue_team.lead.name}_agno_proxy":
            # Check if this agent has a lead designation
            if hasattr(agent, 'role') and agent.role == 'lead':
                lead_found = True
                break
    
    assert lead_found, "Team lead not properly designated in Agno team"
    
    # Test orchestration by simulating a run
    # This might require more setup depending on how Agno handles orchestration
    # The key is to verify that the lead agent coordinates the member agents
```

#### Task 6: Implement team lead orchestration in Agno integration
**Description:** Update the _initialize_agno_team method to properly designate the GLUE team lead as the orchestrator in the Agno team. This involves:
1. Identify the lead model in the GLUE team
2. Create a special AgnoAgent for the lead with appropriate orchestration capabilities
3. Configure the AgnoWorkflow to use the lead agent as the coordinator
4. Ensure the hierarchical team structure is maintained in the Agno implementation

This preserves GLUE's core team structure where the team lead acts as an orchestrator for team members.

**Complexity:** 6

**Code Example:**
```python
# In _initialize_agno_team method:
# After creating basic AgnoAgents for all models

# Identify the lead agent and designate it as the orchestrator
lead_agent = None
member_agents = []

for model_name, glue_model_instance in self.models.items():
    # Find the corresponding AgnoAgent we created earlier
    agent = next((a for a in agno_agents_list if a.name == f"{glue_model_instance.name}_agno_proxy"), None)
    if agent and glue_model_instance == self.lead:
        # This is the lead agent
        lead_agent = agent
        # Set special properties for the lead agent
        lead_agent.role = "lead"
        # Configure lead agent with orchestration capabilities
        lead_agent.instructions = f"""You are the lead agent for the {self.name} team.
Your role is to coordinate the team members and orchestrate their activities.
Analyze the task, break it down into subtasks, and delegate to appropriate team members.
Synthesize the results from team members into a coherent response."""
    elif agent:
        # This is a member agent
        member_agents.append(agent)
        agent.role = "member"
        agent.instructions = f"""You are a member of the {self.name} team.
Your role is to assist the lead agent by completing tasks assigned to you.
Focus on your specific expertise and provide detailed responses to the lead agent."""

# Configure the workflow to use the lead agent as the orchestrator
workflow_mode = "coordinate"  # This mode designates a lead agent as coordinator
workflow = AgnoWorkflow(
    name=f"{self.name}_workflow",
    description=f"Workflow for {self.name} team with lead orchestration",
    mode=workflow_mode,
    lead_agent=lead_agent.name if lead_agent else None
)

# Store the workflow for later use
self.agno_workflow = workflow

# Create the AgnoTeam with the lead-orchestrated structure
self.agno_team = AgnoTeam(
    name=f"{self.name}_agno",
    members=agno_agents_list,
    mode="coordinate",  # This mode designates a lead agent as coordinator
    # Other parameters as needed
)

# Designate the lead agent in the team if needed
if lead_agent and hasattr(self.agno_team, 'set_lead'):
    self.agno_team.set_lead(lead_agent.name)
```

### Phase 4: Workflow Integration

#### Task 7: Create test for AgnoWorkflow creation and configuration
**Description:** Create a new test method in TestGlueTeam class that verifies proper AgnoWorkflow creation and configuration. This test will:
1. Set up a GlueTeam with multiple models
2. Call _initialize_agno_team
3. Verify the created AgnoWorkflow has the correct configuration
4. Verify the workflow is properly associated with the team

This follows TDD principles by creating the test before implementing the feature, ensuring we're using real implementations rather than mocks.

**Complexity:** 4

**Code Example:**
```python
@pytest.mark.asyncio
async def test_initialize_agno_team_creates_proper_workflow(self, basic_glue_team: GlueTeam, basic_agent_instance: Agent):
    """
    Tests that _initialize_agno_team correctly creates and configures an AgnoWorkflow
    that matches the GLUE team's structure.
    """
    # Add a member to the team to create a multi-agent team
    await basic_glue_team.add_member(basic_agent_instance)
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None
    
    # Verify the workflow was created and stored
    assert hasattr(basic_glue_team, 'agno_workflow'), "AgnoWorkflow not stored in GlueTeam"
    assert basic_glue_team.agno_workflow is not None, "AgnoWorkflow is None"
    
    # Verify workflow configuration
    workflow = basic_glue_team.agno_workflow
    assert workflow.name == f"{basic_glue_team.name}_workflow", "Workflow name is incorrect"
    assert hasattr(workflow, 'description'), "Workflow has no description"
    assert basic_glue_team.name in workflow.description, "Team name not in workflow description"
    
    # Verify the workflow is properly associated with the team or its agents
    # The exact verification will depend on how Agno associates workflows with teams
    # This might involve checking team.workflow, or checking if agents are added to the workflow
```

#### Task 8: Implement proper AgnoWorkflow creation and configuration
**Description:** Update the _initialize_agno_team method to create and configure a proper AgnoWorkflow instance. Currently, we're creating a placeholder workflow with just a name, but we need to:
1. Understand the Agno Workflow API and required parameters
2. Create a workflow that matches GLUE team's orchestration patterns
3. Configure the workflow with appropriate parameters (mode, description, etc.)
4. Handle the relationship between workflow and agents

This will enable proper team orchestration when using Agno as the execution engine.

**Complexity:** 6

**Code Example:**
```python
# In _initialize_agno_team method:
try:
    # Create a workflow appropriate for this team's structure
    # The workflow type might depend on team configuration or role relationships
    workflow_mode = "coordinate"  # Default mode, could be "collaborate" or "route" based on team config
    
    # Create the workflow with appropriate configuration
    workflow = AgnoWorkflow(
        name=f"{self.name}_workflow",
        description=f"Workflow for {self.name} team",
        # Other parameters as needed by Agno Workflow API
    )
    
    logger.info(f"Created AgnoWorkflow: {workflow.name} with mode: {workflow_mode}")
    
    # Store the workflow for later use when initializing the AgnoTeam
    # Note: We'll need to determine if workflow is passed to AgnoTeam constructor
    # or associated with it after creation
    self.agno_workflow = workflow
except Exception as e:
    logger.error(f"Failed to create AgnoWorkflow for {self.name}: {e}")
    return None
```

### Phase 5: Tool Integration

#### Task 9: Create test for GLUE tools mapping to Agno tools
**Description:** Create a new test method in TestGlueTeam class that verifies GLUE tools are correctly mapped to Agno tools. This test will:
1. Create a GlueTeam with specific tools
2. Call _initialize_agno_team
3. Verify the created Agno Agents have access to these tools
4. Verify tool execution works and results are properly shared

This follows TDD principles by creating the test before implementing the feature, ensuring we're using real implementations rather than mocks.

**Complexity:** 5

**Code Example:**
```python
@pytest.mark.asyncio
async def test_initialize_agno_team_maps_tools(self, basic_glue_team: GlueTeam):
    """
    Tests that _initialize_agno_team correctly maps GLUE tools to Agno tools.
    """
    # Create a simple test tool
    async def test_tool_function(param1: str, param2: int = 0):
        return f"Tool executed with {param1} and {param2}"
    
    test_tool = {
        "name": "test_tool",
        "description": "A test tool for verification",
        "parameters": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter", "default": 0}
        },
        "execute": test_tool_function
    }
    
    # Add the tool to the team
    basic_glue_team.add_tool("test_tool", test_tool)
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None
    assert hasattr(basic_glue_team.agno_team, 'members')
    
    # Get the first agent
    assert len(basic_glue_team.agno_team.members) > 0, "No agents in Agno team"
    agno_agent = basic_glue_team.agno_team.members[0]
    
    # Verify tools were mapped correctly
    assert hasattr(agno_agent, 'tools'), "Agno Agent does not have tools attribute"
    assert len(agno_agent.tools) > 0, "No tools assigned to Agno Agent"
    
    # Find our test tool
    test_agno_tool = None
    for tool in agno_agent.tools:
        if tool['function']['name'] == 'test_tool':
            test_agno_tool = tool
            break
    
    assert test_agno_tool is not None, "Test tool not found in Agno Agent tools"
    assert test_agno_tool['function']['description'] == "A test tool for verification", "Tool description not mapped correctly"
    
    # Test tool execution if possible
    # This might require more setup or mocking depending on how Agno executes tools
    # For now, we'll just verify the tool structure is correct
    assert 'function' in test_agno_tool, "Tool does not have function attribute"
    assert callable(test_agno_tool['function']['function']), "Tool function is not callable"
```

#### Task 10: Implement GLUE tools mapping to Agno tools
**Description:** Update the _initialize_agno_team method to map GLUE tools to Agno tools. This involves:
1. Extract tools from the GlueTeam's _tools dictionary
2. Convert each GLUE tool to an Agno-compatible tool format
3. Assign these tools to the appropriate Agno Agents
4. Handle tool execution and result sharing

This ensures that tools available to GLUE Models are also available to their corresponding Agno Agents, maintaining feature parity.

**Complexity:** 7

**Code Example:**
```python
# In _initialize_agno_team method:
# After creating Agno agents but before creating the team

# Convert GLUE tools to Agno tools
agno_tools = []
for tool_name, glue_tool in self._tools.items():
    try:
        # Create an Agno-compatible tool wrapper
        async def tool_wrapper(*args, **kwargs):
            # Execute the GLUE tool and return its result
            result = await glue_tool.execute(*args, **kwargs)
            # Store result in shared_results for access by other agents
            self.shared_results[tool_name] = result
            return result
            
        # Create Agno tool with proper metadata
        agno_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": getattr(glue_tool, "description", f"Tool: {tool_name}"),
                "parameters": getattr(glue_tool, "parameters", {}),
                "function": tool_wrapper
            }
        }
        agno_tools.append(agno_tool)
        logger.info(f"Created Agno tool wrapper for GLUE tool: {tool_name}")
    except Exception as e:
        logger.error(f"Failed to create Agno tool for GLUE tool {tool_name}: {e}")

# Assign tools to Agno agents
# This might vary based on how tools are assigned in GLUE (team-level vs agent-level)
for agno_agent in agno_agents_list:
    agno_agent.tools = agno_tools
    logger.info(f"Assigned {len(agno_tools)} tools to agent: {agno_agent.name}")
```

### Phase 6: Adhesive Integration

#### Task 11: Create test for Adhesive integration with Agno memory
**Description:** Create a new test method in TestGlueTeam class that verifies GLUE Adhesives are correctly mapped to Agno memory settings. This test will:
1. Create GLUE Models with specific adhesives (MEMORY, SESSION)
2. Add them to a GlueTeam
3. Call _initialize_agno_team
4. Verify the created AgnoTeam has the correct memory settings

This follows TDD principles by creating the test before implementing the feature, ensuring we're using real implementations rather than mocks.

**Complexity:** 4

**Code Example:**
```python
@pytest.mark.asyncio
async def test_initialize_agno_team_maps_adhesives_to_memory(self, basic_glue_team: GlueTeam):
    """
    Tests that _initialize_agno_team correctly maps GLUE Adhesives to Agno memory settings.
    """
    # Create a model with memory adhesives
    from glue.core.types import AdhesiveType
    
    model_config = {
        "name": "test_model_with_adhesives",
        "provider": "test",
        "model_name": "test_model_variant_1",
        "adhesives": [AdhesiveType.MEMORY, AdhesiveType.SESSION]
    }
    model_with_adhesives = Model(config=model_config)
    
    # Add the model to the team
    await basic_glue_team.add_member(model_with_adhesives)
    
    # Initialize the Agno team
    basic_glue_team._initialize_agno_team()
    
    # Verify the Agno team was created
    assert basic_glue_team.agno_team is not None
    
    # Verify memory settings were mapped correctly
    assert hasattr(basic_glue_team.agno_team, 'enable_user_memories'), "AgnoTeam does not have enable_user_memories attribute"
    assert basic_glue_team.agno_team.enable_user_memories is True, "User memories not enabled despite MEMORY adhesive"
    
    assert hasattr(basic_glue_team.agno_team, 'enable_session_summaries'), "AgnoTeam does not have enable_session_summaries attribute"
    assert basic_glue_team.agno_team.enable_session_summaries is True, "Session summaries not enabled despite SESSION adhesive"
```

#### Task 12: Implement Adhesive integration with Agno memory system
**Description:** Update the _initialize_agno_team method to map GLUE Adhesives to Agno's memory/state management system. This involves:
1. Extract adhesive configuration from GLUE Models and the GlueTeam
2. Map these adhesives to Agno's memory settings (enable_user_memories, enable_session_summaries, etc.)
3. Configure the AgnoTeam with appropriate memory parameters
4. Ensure persistence of memory across sessions if required by adhesives

This ensures that GLUE's persistence system is properly integrated with Agno's memory management.

**Complexity:** 6

**Code Example:**
```python
# In _initialize_agno_team method:
# Before creating the AgnoTeam

# Map GLUE adhesives to Agno memory settings
enable_user_memories = False
enable_session_summaries = False

# Check team-level adhesives first (from models)
for model_name, model in self.models.items():
    if hasattr(model, 'adhesives') and model.adhesives:
        # Check for memory-related adhesives
        for adhesive in model.adhesives:
            if adhesive == AdhesiveType.MEMORY or adhesive == AdhesiveType.USER_MEMORY:
                enable_user_memories = True
                logger.info(f"Enabling user memories for Agno team based on {model_name}'s adhesives")
            if adhesive == AdhesiveType.SESSION or adhesive == AdhesiveType.SESSION_SUMMARY:
                enable_session_summaries = True
                logger.info(f"Enabling session summaries for Agno team based on {model_name}'s adhesives")

# Create the AgnoTeam with memory settings
try:
    self.agno_team = AgnoTeam(
        name=f"{self.name}_agno",
        members=agno_agents_list,
        enable_user_memories=enable_user_memories,
        enable_session_summaries=enable_session_summaries,
        # Other memory-related parameters
    )
    logger.info(f"Successfully initialized Agno team: {self.agno_team.name} with memory settings")
    return self.agno_team
except Exception as e:
    logger.error(f"Failed to instantiate AgnoTeam for {self.name}: {e}")
    self.agno_team = None
    return None
```

### Phase 7: Magnetic Flow Integration

#### Task 13: Create test for magnetic flow operators in Agno integration
**Description:** Create a test that verifies GLUE's magnetic flow operators (PUSH, PULL, BIDIRECTIONAL, REPEL) are properly implemented in the Agno integration. This test will:
1. Create multiple GlueTeams with different magnetic relationships
2. Configure push/pull/bidirectional/repel relationships between teams
3. Verify these communication patterns are preserved in the Agno implementation
4. Test information flow between teams using the magnetic operators

This ensures GLUE's unique inter-team communication system is maintained when using Agno as the execution engine.

**Complexity:** 7

**Code Example:**
```python
@pytest.mark.asyncio
async def test_agno_integration_preserves_magnetic_flow_operators(self):
    """
    Tests that the Agno integration preserves GLUE's magnetic flow operators.
    """
    # Create multiple teams with Agno integration
    team1_model = Model(config={"name": "team1_lead", "provider": "test"})
    team1 = GlueTeam(name="Team1", lead=team1_model, use_agno_team=True)
    
    team2_model = Model(config={"name": "team2_lead", "provider": "test"})
    team2 = GlueTeam(name="Team2", lead=team2_model, use_agno_team=True)
    
    team3_model = Model(config={"name": "team3_lead", "provider": "test"})
    team3 = GlueTeam(name="Team3", lead=team3_model, use_agno_team=True)
    
    # Configure magnetic relationships
    # Team1 PUSHES to Team2
    team1.add_magnetic_relationship(team2.name, FlowType.PUSH)
    
    # Team2 PULLS from Team3
    team2.add_magnetic_relationship(team3.name, FlowType.PULL)
    
    # Team1 and Team3 have BIDIRECTIONAL communication
    team1.add_magnetic_relationship(team3.name, FlowType.BIDIRECTIONAL)
    team3.add_magnetic_relationship(team1.name, FlowType.BIDIRECTIONAL)
    
    # Initialize Agno teams
    team1._initialize_agno_team()
    team2._initialize_agno_team()
    team3._initialize_agno_team()
    
    # Verify all teams have Agno teams
    assert team1.agno_team is not None
    assert team2.agno_team is not None
    assert team3.agno_team is not None
    
    # Verify magnetic relationships are preserved
    # This will depend on how we implement magnetic flows in Agno
    # For example, we might store them in the AgnoTeam or in a separate structure
    
    # Test PUSH relationship: Team1 -> Team2
    test_data = "Data from Team1 to Team2"
    
    # Mock the direct_communication method to track calls
    original_direct_comm = team1.direct_communication
    direct_comm_called = False
    
    async def mock_direct_communication(target_team, message):
        nonlocal direct_comm_called
        direct_comm_called = True
        return await original_direct_comm(target_team, message)
    
    team1.direct_communication = mock_direct_communication
    
    try:
        # Simulate a communication that should use magnetic flow
        await team1.send_information(team2.name, test_data)
        
        # Verify magnetic flow was used
        assert direct_comm_called, "Magnetic flow was not used"
        
        # Verify the message was received by Team2
        # This might require checking team2's conversation history or a mock
        # The exact verification will depend on how we implement magnetic flows
        
    finally:
        # Restore original method
        team1.direct_communication = original_direct_comm
```

#### Task 14: Implement magnetic flow operators in Agno integration
**Description:** Implement GLUE's magnetic flow operators (PUSH, PULL, BIDIRECTIONAL, REPEL) in the Agno integration. This involves:
1. Create a mechanism to map GLUE's magnetic relationships to Agno's communication system
2. Implement handlers for each flow type (PUSH, PULL, BIDIRECTIONAL, REPEL)
3. Ensure information can flow between teams according to their magnetic relationships
4. Maintain the unique semantics of each flow type

This preserves GLUE's distinctive inter-team communication patterns when using Agno as the execution engine.

**Complexity:** 8

**Code Example:**
```python
# Add to GlueTeam class:

def _register_magnetic_flows_with_agno(self):
    """
    Register GLUE's magnetic flow operators with the Agno team.
    This enables inter-team communication according to magnetic relationships.
    """
    if not self.agno_team:
        logger.warning(f"Cannot register magnetic flows: No Agno team for {self.name}")
        return
        
    # Create communication channels based on magnetic relationships
    for target_team_name, flow_type in self.relationships.items():
        logger.info(f"Registering {flow_type} relationship from {self.name} to {target_team_name}")
        
        if flow_type == FlowType.PUSH:
            # Register a tool for the team to push information to the target
            self._register_push_tool(target_team_name)
            
        elif flow_type == FlowType.PULL:
            # Register a tool for the team to pull information from the target
            self._register_pull_tool(target_team_name)
            
        elif flow_type == FlowType.BIDIRECTIONAL:
            # Register tools for bidirectional communication
            self._register_push_tool(target_team_name)
            self._register_pull_tool(target_team_name)
            
        elif flow_type == FlowType.REPEL:
            # Register a blocker that prevents communication
            self._register_repel_blocker(target_team_name)
    
    # Register handlers for incoming information
    self._register_information_handlers()

def _register_push_tool(self, target_team_name):
    """Register a tool that allows pushing information to a target team."""
    async def push_to_team(message: str):
        """Push information to another team."""
        logger.info(f"Agno team {self.name} pushing information to {target_team_name}: {message}")
        # Find the target team in the app
        target_team = None
        if self.app:
            target_team = self.app.get_team(target_team_name)
        
        if not target_team:
            return f"Error: Team {target_team_name} not found"
            
        # Send the information using GLUE's native mechanism
        await self.send_information(target_team_name, message)
        return f"Information sent to {target_team_name}"
    
    # Register this as a tool with the Agno team
    push_tool = {
        "type": "function",
        "function": {
            "name": f"push_to_{target_team_name}",
            "description": f"Push information to the {target_team_name} team",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The information to send"
                    }
                },
                "required": ["message"]
            },
            "function": push_to_team
        }
    }
    
    # Add the tool to all agents in the Agno team
    for agent in self.agno_team.members:
        if not hasattr(agent, 'tools'):
            agent.tools = []
        agent.tools.append(push_tool)

# Similar implementations for _register_pull_tool, _register_repel_blocker, etc.
```

### Phase 8: Polarity-Based Communication

#### Task 15: Create test for polarity-based communication in Agno integration
**Description:** Create a test that verifies GLUE's polarity-based communication (attract >< ) is properly implemented in the Agno integration. This test will:
1. Create multiple GlueTeams with polarity-based relationships
2. Configure attraction relationships between team leads
3. Test direct communication between team leads without sending data through magnetic fields
4. Verify polarity-based communication works correctly in the Agno implementation

This ensures GLUE's unique team lead communication system is maintained when using Agno as the execution engine.

**Complexity:** 6

**Code Example:**
```python
@pytest.mark.asyncio
async def test_agno_integration_preserves_polarity_communication(self):
    """
    Tests that the Agno integration preserves GLUE's polarity-based communication between team leads.
    """
    # Create multiple teams with Agno integration
    team1_model = Model(config={"name": "team1_lead", "provider": "test"})
    team1 = GlueTeam(name="Team1", lead=team1_model, use_agno_team=True)
    
    team2_model = Model(config={"name": "team2_lead", "provider": "test"})
    team2 = GlueTeam(name="Team2", lead=team2_model, use_agno_team=True)
    
    # Configure polarity relationships (attraction)
    # Team1 lead ATTRACTS Team2 lead (bidirectional communication without magnetic flow)
    team1.add_polarity_relationship(team2.name, PolarityType.ATTRACT)
    team2.add_polarity_relationship(team1.name, PolarityType.ATTRACT)
    
    # Initialize Agno teams
    team1._initialize_agno_team()
    team2._initialize_agno_team()
    
    # Verify all teams have Agno teams
    assert team1.agno_team is not None
    assert team2.agno_team is not None
    
    # Test direct communication between team leads (without magnetic flow)
    # This should use the polarity-based communication channel
    
    # Team1 lead communicates directly with Team2 lead
    test_message = "Direct message from Team1 lead to Team2 lead"
    
    # Mock the direct_communication method to track calls
    original_direct_comm = team1.direct_communication
    direct_comm_called = False
    
    async def mock_direct_communication(target_team, message):
        nonlocal direct_comm_called
        direct_comm_called = True
        return await original_direct_comm(target_team, message)
    
    team1.direct_communication = mock_direct_communication
    
    try:
        # Simulate a communication that should use polarity
        await team1.communicate_via_polarity(team2.name, test_message)
        
        # Verify direct communication was used (polarity) rather than magnetic flow
        assert direct_comm_called, "Polarity-based direct communication was not used"
        
        # Verify the message was received by Team2's lead
        # This might require checking team2's conversation history or a mock
        # The exact verification will depend on how we implement polarity in Agno
        
    finally:
        # Restore original method
        team1.direct_communication = original_direct_comm
```

#### Task 16: Implement polarity-based communication in Agno integration
**Description:** Implement GLUE's polarity-based communication (attract >< ) in the Agno integration. This involves:
1. Create a mechanism for direct communication between team leads without using magnetic flows
2. Implement handlers for attraction relationships between team leads
3. Configure Agno agents to use direct communication channels based on polarity relationships
4. Ensure team leads can communicate directly without sending data through the magnetic field

This preserves GLUE's unique team lead communication system where leads can communicate directly when attracted, but not when repelled.

**Complexity:** 7

**Code Example:**
```python
# Add to GlueTeam class:

def _register_polarity_relationships_with_agno(self):
    """
    Register GLUE's polarity-based communication with the Agno team.
    This enables direct communication between team leads based on polarity.
    """
    if not self.agno_team or not self.lead:
        logger.warning(f"Cannot register polarity relationships: No Agno team or lead for {self.name}")
        return
        
    # Find the lead agent in the Agno team
    lead_agent = None
    for agent in self.agno_team.members:
        if agent.name == f"{self.lead.name}_agno_proxy":
            lead_agent = agent
            break
            
    if not lead_agent:
        logger.warning(f"Cannot register polarity relationships: Lead agent not found in Agno team for {self.name}")
        return
        
    # Get all teams that this team's lead is attracted to
    attracted_teams = []
    for team_name, polarity_type in self.polarity_relationships.items():
        if polarity_type == PolarityType.ATTRACT:
            attracted_teams.append(team_name)
            
    if not attracted_teams:
        logger.info(f"No attraction relationships to register for {self.name}")
        return
        
    # Register direct communication tools for the lead agent
    for target_team_name in attracted_teams:
        self._register_direct_communication_tool(lead_agent, target_team_name)
            
def _register_direct_communication_tool(self, lead_agent, target_team_name):
    """Register a tool that allows direct communication with another team's lead."""
    async def communicate_directly(message: str):
        """Communicate directly with another team's lead."""
        logger.info(f"Agno team {self.name} lead communicating directly with {target_team_name} lead: {message}")
        
        # Find the target team in the app
        target_team = None
        if self.app:
            target_team = self.app.get_team(target_team_name)
        
        if not target_team:
            return f"Error: Team {target_team_name} not found"
            
        # Use GLUE's direct communication mechanism
        await self.direct_communication(target_team_name, message)
        return f"Direct communication sent to {target_team_name} lead"
    
    # Register this as a tool with the lead agent only
    direct_comm_tool = {
        "type": "function",
        "function": {
            "name": f"communicate_directly_with_{target_team_name}_lead",
            "description": f"Communicate directly with the lead of {target_team_name} team (polarity-based)",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send directly to the other team lead"
                    }
                },
                "required": ["message"]
            },
            "function": communicate_directly
        }
    }
    
    # Add the tool to the lead agent only
    if not hasattr(lead_agent, 'tools'):
        lead_agent.tools = []
    lead_agent.tools.append(direct_comm_tool)

# Similar implementations for other polarity types
```

### Phase 9: DSL Integration

#### Task 17: Create test for DSL integration with Agno
**Description:** Create a test that verifies GLUE's StickyScript DSL can properly configure and use Agno-integrated teams. This test will:
1. Parse a simple StickyScript DSL file that defines teams with tools and adhesives
2. Verify the DSL correctly configures GlueTeams with Agno integration
3. Test that tools assigned at the team level in the DSL are properly available to Agno agents
4. Verify that adhesives assigned to agents in the DSL are properly mapped to Agno memory settings

This ensures GLUE's unique DSL-based configuration system works seamlessly with the Agno integration.

**Complexity:** 7

**Code Example:**
```python
@pytest.mark.asyncio
async def test_dsl_integration_with_agno():
    """
    Tests that GLUE's StickyScript DSL properly configures and uses Agno-integrated teams.
    """
    # Create a simple DSL script as a string
    dsl_script = """
    // Define a team with Agno integration
    team ResearchTeam {
        use_agno_team: true;
        
        // Team lead with memory adhesive
        agent LeadResearcher {
            role: "lead";
            model: "gpt-4";
            provider: "test";
            adhesives: ["memory", "session"];
        }
        
        // Team member
        agent AssistantResearcher {
            role: "member";
            model: "gpt-3.5-turbo";
            provider: "test";
        }
        
        // Tools assigned at team level
        tools: ["web_search", "document_analyzer"];
    }
    
    // Define magnetic relationships
    flow ResearchTeam -> SummaryTeam {
        type: "push";
    }
    """
    
    # Parse the DSL script
    from glue.dsl.parser import parse_glue_script
    app_config = parse_glue_script(dsl_script)
    
    # Create a GlueApp from the config
    from glue.app import GlueApp
    app = GlueApp(config=app_config)
    
    # Initialize the app
    await app.initialize()
    
    # Get the ResearchTeam
    research_team = app.get_team("ResearchTeam")
    
    # Verify the team was created with Agno integration
    assert research_team is not None, "ResearchTeam not created"
    assert research_team.use_agno_team is True, "use_agno_team not set to True"
    
    # Initialize the Agno team
    if not research_team.agno_team:
        research_team._initialize_agno_team()
    
    assert research_team.agno_team is not None, "Agno team not initialized"
    
    # Verify the team structure
    assert len(research_team.models) == 2, "Team should have 2 models"
    assert research_team.lead is not None, "Team lead not set"
    assert research_team.lead.name == "LeadResearcher", "Wrong lead model"
    
    # Verify tools were assigned
    assert len(research_team._tools) == 2, "Team should have 2 tools"
    assert "web_search" in research_team._tools, "web_search tool not found"
    assert "document_analyzer" in research_team._tools, "document_analyzer tool not found"
    
    # Verify tools are available to Agno agents
    for agent in research_team.agno_team.members:
        assert hasattr(agent, 'tools'), f"Agent {agent.name} has no tools"
        assert len(agent.tools) >= 2, f"Agent {agent.name} should have at least 2 tools"
        
        # Check if the tool names are in the agent's tools
        tool_names = [t['function']['name'] for t in agent.tools if 'function' in t and 'name' in t['function']]
        assert any("web_search" in name for name in tool_names), f"web_search tool not found in {agent.name}'s tools"
        assert any("document_analyzer" in name for name in tool_names), f"document_analyzer tool not found in {agent.name}'s tools"
    
    # Verify adhesives were mapped to memory settings
    assert hasattr(research_team.agno_team, 'enable_user_memories'), "AgnoTeam does not have enable_user_memories attribute"
    assert research_team.agno_team.enable_user_memories is True, "User memories not enabled despite MEMORY adhesive"
    
    assert hasattr(research_team.agno_team, 'enable_session_summaries'), "AgnoTeam does not have enable_session_summaries attribute"
    assert research_team.agno_team.enable_session_summaries is True, "Session summaries not enabled despite SESSION adhesive"
    
    # Verify magnetic relationships
    assert "SummaryTeam" in research_team.relationships, "Magnetic relationship not set"
    assert research_team.relationships["SummaryTeam"] == FlowType.PUSH, "Wrong flow type"
```

#### Task 18: Implement DSL integration with Agno
**Description:** Update the GLUE DSL parser and GlueApp initialization to properly configure Agno-integrated teams from StickyScript. This involves:
1. Modify the DSL parser to recognize and process the 'use_agno_team' parameter
2. Ensure tools assigned at the team level in the DSL are properly mapped to Agno agents
3. Map adhesives assigned to agents in the DSL to Agno memory settings
4. Configure magnetic and polarity relationships defined in the DSL for Agno teams

This ensures GLUE's unique DSL-based configuration system works seamlessly with the Agno integration, preserving the ability to configure complex agent systems through StickyScript.

**Complexity:** 8

**Code Example:**
```python
# Update the DSL parser to handle Agno integration
def parse_team_config(team_config):
    """Parse a team configuration from the DSL."""
    # Existing parsing code...
    
    # Add support for use_agno_team parameter
    use_agno_team = team_config.get('use_agno_team', False)
    if isinstance(use_agno_team, str):
        use_agno_team = use_agno_team.lower() == 'true'
    
    # Create the team config
    team = {
        'name': team_config['name'],
        'use_agno_team': use_agno_team,
        'agents': [],
        'tools': team_config.get('tools', []),
        'relationships': {},
        'polarity': {}
    }
    
    # Parse agents...
    # Parse tools...
    # Parse relationships...
    
    return team

# Update GlueApp initialization to handle Agno integration
async def initialize(self):
    """Initialize the GlueApp with all teams and relationships."""
    # Create all teams
    for team_config in self.config['teams']:
        team_name = team_config['name']
        use_agno_team = team_config.get('use_agno_team', False)
        
        # Create the team
        team = GlueTeam(name=team_name, use_agno_team=use_agno_team)
        self.teams[team_name] = team
        
        # Add agents to the team
        for agent_config in team_config['agents']:
            role = agent_config.get('role', 'member')
            model_config = {
                'name': agent_config['name'],
                'provider': agent_config.get('provider', 'openai'),
                'model': agent_config.get('model', 'gpt-3.5-turbo'),
                'adhesives': agent_config.get('adhesives', [])
            }
            
            model = Model(config=model_config)
            
            if role == 'lead':
                team.set_lead(model)
            else:
                await team.add_member(model)
        
        # Add tools to the team
        for tool_name in team_config.get('tools', []):
            tool = self.get_tool(tool_name)
            if tool:
                team.add_tool(tool_name, tool)
    
    # Configure relationships
    for flow in self.config.get('flows', []):
        source_team = self.teams.get(flow['source'])
        target_team = self.teams.get(flow['target'])
        
        if source_team and target_team:
            flow_type = flow.get('type', 'push')
            source_team.add_magnetic_relationship(target_team.name, flow_type)
    
    # Configure polarity
    for polarity in self.config.get('polarity', []):
        team1 = self.teams.get(polarity['team1'])
        team2 = self.teams.get(polarity['team2'])
        
        if team1 and team2:
            polarity_type = polarity.get('type', 'attract')
            team1.add_polarity_relationship(team2.name, polarity_type)
            team2.add_polarity_relationship(team1.name, polarity_type)
    
    # Initialize Agno teams for teams with use_agno_team=True
    for team in self.teams.values():
        if team.use_agno_team:
            team._initialize_agno_team()
            
            # Register magnetic flows with Agno
            team._register_magnetic_flows_with_agno()
            
            # Register polarity relationships with Agno
            team._register_polarity_relationships_with_agno()
```

### Phase 10: Execution Integration

#### Task 19: Create test for GlueTeam.run delegation to Agno
**Description:** Create a new test method in TestGlueTeam class that verifies the GlueTeam.run method properly delegates to Agno and processes the response. This test will:
1. Set up a GlueTeam with use_agno_team=True
2. Initialize the Agno team
3. Call team.run with a test input
4. Verify the input is properly delegated to Agno
5. Verify the response is correctly processed and returned

This follows TDD principles by creating the test before implementing the feature, ensuring we're using real implementations rather than mocks.

**Complexity:** 5

**Code Example:**
```python
@pytest.mark.asyncio
async def test_glue_team_run_delegates_to_agno(self, basic_glue_team: GlueTeam):
    """
    Tests that GlueTeam.run properly delegates to Agno and processes the response.
    """
    # Ensure Agno team is initialized
    basic_glue_team._initialize_agno_team()
    assert basic_glue_team.agno_team is not None, "Agno team was not initialized"
    
    # Store the original arun method to verify it's called
    original_arun = basic_glue_team.agno_team.arun
    
    # Create a flag to track if arun was called
    arun_called = False
    test_response_content = "This is a test response from Agno"
    
    # Replace arun with a test implementation
    async def test_arun(input_text, **kwargs):
        nonlocal arun_called
        arun_called = True
        # Create a simple TeamRunResponse-like object
        return type('TestTeamRunResponse', (), {
            'content': test_response_content,
            'session_id': 'test_session_id'
        })
    
    # Temporarily replace the arun method
    basic_glue_team.agno_team.arun = test_arun
    
    try:
        # Call run with a test input
        test_input = "Test input for Agno team"
        response = await basic_glue_team.run(test_input)
        
        # Verify arun was called
        assert arun_called, "Agno team's arun method was not called"
        
        # Verify response was processed correctly
        assert response is not None, "Run method returned None"
        assert hasattr(response, 'content'), "Response has no content attribute"
        assert response.content == test_response_content, "Response content does not match expected"
        
        # Verify GLUE state was updated
        assert len(basic_glue_team.conversation_history) > 0, "Conversation history was not updated"
        assert basic_glue_team.conversation_history[-1]['content'] == test_response_content, "Conversation history content incorrect"
        assert hasattr(basic_glue_team, 'session_id'), "Session ID was not stored"
        assert basic_glue_team.session_id == 'test_session_id', "Session ID was not stored correctly"
        
    finally:
        # Restore the original arun method
        basic_glue_team.agno_team.arun = original_arun
```

#### Task 20: Enhance GlueTeam.run method to properly delegate to Agno
**Description:** Update the GlueTeam.run method to properly delegate tasks to the AgnoTeam when use_agno_team is true. This involves:
1. Ensure input is properly formatted for Agno (converting Message objects if needed)
2. Handle session management for continuity across runs
3. Process the TeamRunResponse from Agno and update GLUE's state (conversation_history, shared_results)
4. Implement proper error handling and fallback to native GLUE if Agno execution fails

This ensures seamless execution through Agno while maintaining GLUE's expected behavior and state management.

**Complexity:** 7

**Code Example:**
```python
async def run(self, initial_input: Any) -> Optional[TeamRunResponse]: 
    """
    Runs the GLUE team. If an Agno team is integrated, it delegates to Agno.
    Otherwise, it uses the native GLUE agent loops.
    
    Args:
        initial_input: The input to start the team's execution.

    Returns:
        An Agno TeamRunResponse if Agno path is taken, otherwise None or GLUE-specific output.
    """
    from .agent_loop import TeamMemberAgentLoop, TeamLeadAgentLoop # Keep for GLUE native path

    logger.info(f"Team {self.name} run method called with input type: {type(initial_input)}")
    
    if self.use_agno_team and self.agno_team and initial_input is not None:
        logger.info(f"Delegating task to Agno team for {self.name} with input: {initial_input}")
        try:
            # Initialize Agno team if not already done
            if not self.agno_team:
                self._initialize_agno_team()
                if not self.agno_team:
                    logger.warning(f"Failed to initialize Agno team for {self.name}, falling back to native GLUE")
                    return await self._run_native(initial_input)
            
            # Prepare input for Agno team
            agno_input = initial_input
            if isinstance(initial_input, dict) and "content" in initial_input:
                agno_input = initial_input["content"]
            elif hasattr(initial_input, "content"):
                agno_input = initial_input.content
                
            # Configure session for continuity
            session_id = getattr(self, "session_id", None)
            
            # Run the Agno team
            response: TeamRunResponse = await self.agno_team.arun(
                agno_input, 
                session_id=session_id,
            )
            
            # Process the response
            if response:
                # Update GLUE state from Agno response
                if hasattr(response, "content") and response.content:
                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response.content,
                        "name": self.name
                    })
                
                # Store session ID for continuity
                if hasattr(response, "session_id"):
                    self.session_id = response.session_id
                    
                logger.info(f"Agno team {self.name} completed run. Response content type: {type(response.content if response else None)}")
                return response
            else:
                logger.warning(f"Agno team {self.name} returned None response")
                return None
                
        except Exception as e:
            logger.error(f"Error during Agno team run for {self.name}: {e}", exc_info=True)
            logger.warning(f"Falling back to native GLUE execution for {self.name}")
            return await self._run_native(initial_input)
    else:
        logger.info(f"Using native GLUE agent loops for team {self.name} (Agno team not available or no input for Agno path)")
        return await self._run_native(initial_input)
        
async def _run_native(self, initial_input: Any) -> None:
    """Native GLUE execution path (extracted for clarity)"""
    # Fallback to existing GLUE agent loop logic
    if self.config.lead and self.models.get(self.config.lead) and initial_input:
        logger.info(f"Processing with native GLUE lead model: {self.config.lead}")
        # Ensure initial_input is a string for process_message if it's the GLUE native path
        message_content = str(initial_input) if not isinstance(initial_input, str) else initial_input
        await self.process_message(content=message_content, source_model=self.config.lead)
        # Native GLUE path does not currently produce a TeamRunResponse.
        logger.warning(f"Native GLUE run for {self.name} finished. No Agno TeamRunResponse produced.")
        return None 
    else:
        logger.warning(f"Native GLUE run for {self.name} skipped: No lead/model or no initial_input for native path.")
        return None
```

### Phase 11: Documentation

#### Task 21: Document Agno integration in GLUE framework
**Description:** Create comprehensive documentation for the Agno integration in the GLUE framework. This involves:
1. Update existing documentation to explain the Agno integration
2. Document how to enable/disable Agno integration
3. Explain the mapping between GLUE concepts and Agno concepts
4. Provide examples of using Agno-powered GLUE teams
5. Document any limitations or differences in behavior

This ensures that users of the GLUE framework understand how to use the Agno integration and what to expect.

**Complexity:** 3

**Code Example:**
```markdown
# GLUE-Agno Integration

## Overview
The GLUE framework now supports integration with the Agno framework for enhanced team orchestration and execution. This integration allows GLUE teams to leverage Agno's powerful agent coordination capabilities while maintaining GLUE's unique features.

## Enabling Agno Integration
To use Agno as the execution engine for a GLUE team, set `use_agno_team=True` when creating the team:

```python
from glue.core.teams import GlueTeam
from glue.core.model import Model

# Create a lead model
lead_model = Model(config={
    "name": "lead_model",
    "provider": "openai",
    "model": "gpt-4"
})

# Create a team with Agno integration enabled
team = GlueTeam(
    name="MyTeam",
    lead=lead_model,
    use_agno_team=True
)
```

## Concept Mapping
The following GLUE concepts are mapped to Agno concepts:

| GLUE Concept | Agno Concept | Notes |
|--------------|--------------|-------|
| Model | Agent | GLUE Models are wrapped as Agno Agents |
| Team | Team | GLUE Teams delegate to Agno Teams |
| Tools | Tools | GLUE Tools are wrapped as Agno Tools |
| Adhesives | Memory | GLUE Adhesives map to Agno memory settings |

## Behavior Differences
When using Agno integration:
- Team execution is delegated to Agno's orchestration
- Responses are returned as TeamRunResponse objects
- Session continuity is maintained via Agno's session management
```

## Implementation Approach
- Follow strict TDD principles: write tests first, then implement
- Use real implementations rather than mocks
- Maintain backward compatibility with native GLUE functionality
- Ensure proper error handling and logging
- Document all integration decisions
- Preserve GLUE's unique features while leveraging Agno's capabilities

---

# GLUE-Agno Integration Implementation Plan - Part 2

## Overview
This plan outlines the steps needed to further enhance the GLUE-Agno integration by mapping GLUE's core loops and structures to Agno's agent and team execution, while preserving GLUE's unique features.

## Implementation Tasks

### Phase A: Core Loop and Structure Mapping

#### Part 2 Task 1: Analyze and Document Agno Agent/Team Execution Flow
**Description:** Conduct a detailed analysis of Agno's `Agent.run`/`arun` and `Team.run`/`arun` (including modes: `collaborate`, `coordinate`, `route`) internal logic. Document the execution flow, context propagation, and how member agents are invoked. This forms the basis for mapping GLUE loops.
**Complexity:** 7

#### Part 2 Task 2: Map GLUE TeamMemberAgentLoop to Agno Agent Execution
**Description:** Based on P2T1, design and document how GLUE's `TeamMemberAgentLoop` functionalities (task fetching, context gathering, parse/analyze, plan substeps, tool selection/execution, memory decision, self-eval, report) will be implemented or driven by an Agno `Agent` instance operating within an Agno `Team`. Create tests for key interaction points. Implement the mapping.
**Complexity:** 8

#### Part 2 Task 3: Map GLUE TeamLeadAgentLoop to Agno Team Orchestration
**Description:** Based on P2T1, design and document how GLUE's `TeamLeadAgentLoop` functionalities (goal decomposition, subtask delegation, report evaluation, synthesis) will be implemented using an Agno `Team`. This includes how the Agno `Team.model` acts as the GLUE Team Lead and utilizes Agno modes. Create tests for core orchestration logic. Implement the mapping.
**Complexity:** 9

#### Part 2 Task 4: Define and Implement GLUE Intra-Team Communication within Agno
**Description:** Design and implement mechanisms for GLUE's 'natural' intra-team communication philosophy when using Agno `Team` modes. Detail how Agno `Agent`s (GLUE members) access shared team context, history (`team_context_str`, `team_member_interactions_str`), and potentially communicate more directly if Agno's modes allow, while being orchestrated by the GLUE Team Lead (Agno `Team.model`). Test communication patterns.
**Complexity:** 7

### Phase B: Advanced Feature Integration & Research

#### Part 2 Task 5: Research Agno RAG and Vector DB Capabilities
**Description:** Investigate Agno's codebase and documentation for built-in RAG (Retrieval Augmented Generation) or vector database integration features. Document findings.
**Complexity:** 6

#### Part 2 Task 6: Integrate Agno RAG/VectorDB into GLUE (if applicable)
**Description:** If P2T5 identifies RAG/VectorDB features in Agno, design and implement their integration into GLUE. This includes configuration via StickyScript, interaction with GLUE's memory/adhesive system, and making it available as a GLUE feature (e.g., specialized tool, memory configuration). Create tests for the RAG-enabled workflow.
**Complexity:** 8

#### Part 2 Task 7: Detail GLUE Adhesive Type Mapping to Agno Memory Operations
**Description:** Elaborate on 'Part 1, Task 9-10'. Specifically define how GLUE's adhesive types (`GLUE`, `VELCRO`, `TAPE`) translate to concrete operations on Agno's `memory.v2.Memory` system (e.g., `create_user_memories`, `create_session_summary`, direct `RunResponse` storage/retrieval, interaction with `MemoryDb`). Test persistence and retrieval for each adhesive type.
**Complexity:** 7

#### Part 2 Task 8: Research Agno Self-Learning/Adapting Mechanisms
**Description:** Investigate Agno's codebase and documentation for any explicit self-learning or self-adapting features (e.g., automated prompt tuning, dynamic strategy adjustment). Document findings.
**Complexity:** 6

#### Part 2 Task 9: Integrate Agno Self-Learning into GLUE (if applicable)
**Description:** If P2T8 identifies self-learning/adapting features in Agno, design and implement their integration into GLUE. Define how these capabilities are exposed and controlled within the GLUE framework. Test the adaptive behaviors.
**Complexity:** 8

### Phase C: Tooling and Operational Modes

#### Part 2 Task 10: Identify and List Agno's Native External Tools
**Description:** Review Agno's codebase and documentation to identify all standard and external tools that come bundled or are natively supported by Agno (e.g., web search, code execution, API tools). Create a comprehensive list.
**Complexity:** 5

#### Part 2 Task 11: Adapt and Integrate Agno's Native Tools into GLUE
**Description:** For each tool identified in P2T10, design and implement an adapter or wrapper to make it compatible with GLUE's tool definition (`Tool` base class) and registration system (`ToolRegistry`). Ensure these tools can be assigned to GLUE Teams (Agno `Team`) and are selectable/usable by GLUE members (Agno `Agent`s) according to GLUE's tool management philosophy. Test each integrated tool.
**Complexity:** 8

#### Part 2 Task 12: Implement and Test GLUE's Dual Operational Modes (Interactive/Continuous) with Agno
**Description:** Design and implement support for GLUE's interactive (human-in-the-loop) and continuous non-interactive operational modes when using Agno as the backend. Define how HITL points are managed. Test both modes with a sample GLUE application running on Agno.
**Complexity:** 7

### Phase D: Documentation

#### Part 2 Task 13: Document GLUE's Agent Loop Philosophy and Agno Mapping
**Description:** Create content for `docs/Agent_loops.md` (currently empty). This document should explain GLUE's conceptual agent loop(s) and team orchestration strategies, and then detail how these are mapped to and implemented using Agno's core Agent and Team functionalities and operational modes.
**Complexity:** 6

## Part 3: Agno v2 Memory System Analysis and GLUE Adhesive/Persistence Mapping

Our investigation into Agno's memory capabilities, informed by GLUE's core concept documents (`01_core_concepts.md`, `03_tool_system.md`, `sticky apps.md`), focuses on Agno's `v2` memory system. This system, particularly `agno.memory.v2.memory.Memory` and its components, is best suited for GLUE.

### Core Agno v2 Memory Components:

1.  **`agno.memory.v2.memory.Memory`**: The central class. Manages:
    *   `runs`: Dict storing `RunResponse`/`TeamRunResponse` lists per session (raw interaction history).
    *   `memories`: Dict storing `UserMemory` objects (long-term, cross-session persistence).
    *   `summaries`: Dict storing `SessionSummary` objects (session overviews).
    *   `team_context`: Dict storing `TeamContext` objects (team interaction details per session).
    *   Utilizes `MemoryDb` for persistent `UserMemory` storage.

2.  **`agno.memory.v2.schema.UserMemory`**: Dataclass for individual persistent memories (content, topics, context, timestamps).

3.  **`agno.memory.v2.schema.SessionSummary`**: Dataclass for session summaries (summary, topics, timestamps).

4.  **`agno.memory.v2.manager.MemoryManager`**: LLM-driven component for intelligent CRUD operations on `UserMemory` objects via `MemoryDb`.

5.  **`agno.memory.v2.summarizer.SessionSummarizer`**: LLM-driven component generating `SessionSummaryResponse` from conversation history.

### GLUE Concepts and Agno v2 Memory Mapping:

**A. GLUE Adhesive Bindings (Agent-Tool Interaction Persistence):**

*   **`tape` (GLUE Adhesive):**
    *   **GLUE Meaning:** Tool output is one-time, used, and discarded. No specific persistence beyond the immediate interaction.
    *   **Agno Mapping:** The tool's `RunResponse` is captured in `Memory.runs`.
    *   **Nature:** In-memory, session-specific, raw interaction data. Available for immediate use within the current operational scope of an agent or team.

*   **`velcro` (GLUE Adhesive):**
    *   **GLUE Meaning:** Tool output is session-based and private to the agent using it. Persists for the agent's current session/task, allowing the agent to refer back to its own recent tool uses.
    *   **Agno Mapping:** The tool's `RunResponse` is stored in `Memory.runs`. The agent can access its recent history from these runs. This might contribute to an agent-specific `SessionSummary` if the agent's work over a session is summarized. It does *not* automatically become a team-wide `UserMemory`.
    *   **Nature:** In-memory, session-specific, raw interaction data. Available for immediate use within the current operational scope of an agent or team.

*   **`glue` (GLUE Adhesive):**
    *   **GLUE Meaning:** Tool output is permanently bound, and results are automatically shared with the team, becoming part of team-wide persistent knowledge.
    *   **Agno Mapping:** The tool's `RunResponse` is recorded in `Memory.runs`. Crucially, the output (or a processed/summarized version of it) should be transformed into a `UserMemory` object via `MemoryManager`. This `UserMemory` would be associated with the team or the overarching user/task, making it available across sessions for the team.
    *   **Nature:** Persistent, cross-session, shared knowledge. Available for use by the team across multiple sessions.

**B. GLUE App-Level Persistence (`sticky = true`):**

*   **GLUE Meaning (Current):** As defined in `sticky apps.md` and `01_core_concepts.md`, this flag in the `glue app {}` block enables global app persistence between runs. This includes:
    *   Last user interactions (to a certain point).
    *   Output of tools used with `glue` adhesive (persisting within the team between runs).
    *   Inner-team, agent-to-agent communication and context memory for each team.
    *   Previous tasks completed per model/team.
*   **Agno Mapping (Current):** Achieving this requires saving and reloading the entire state of the Agno `Memory` object (e.g., via `Memory.to_dict()` and `Memory.from_dict()`). This ensures that `Memory.runs` (for recent interaction context), `Memory.summaries` (for session overviews and summarized team context), and `Memory.memories` (containing `UserMemory` from `glue` adhesive use and other persistent knowledge) are all preserved and reloaded across application runs.

*   **GLUE Meaning (Future Vision from `sticky apps.md`):** Expand `sticky` persistence to include a structured, app-wide task execution log. This log would detail task breakdown among teams, summaries of team outputs, `glue` tool usage outputs, lists of all tools used by teams, and the overall task's final output. This is intended to feed a self-learning mechanism.
*   **Agno Mapping (Future Vision):** This enhanced log could be implemented as specialized `UserMemory` objects (created and managed by `MemoryManager`) or a separate structured logging system that GLUE integrates. `SessionSummarizer` could assist in generating the concise team output summaries for this log.

### Summary for GLUE Implementation:

GLUE will primarily use Agno's `v2.memory.Memory` system. The `AdhesiveType` used by a GLUE agent when calling `model.use_tool()` will determine how the tool's output is processed by Agno's memory components:
*   `TAPE` -> Stays in `Memory.runs`.
*   `VELCRO` -> Stays in `Memory.runs`, potentially contributes to agent's `SessionSummary`.
*   `GLUE` -> Stored in `Memory.runs`, then processed by `MemoryManager` into a `UserMemory` for team-wide, cross-session persistence.

The `sticky = true` app setting will trigger the serialization and deserialization of the Agno `Memory` object's state to persist GLUE sessions.

This refined understanding provides a clear path for implementing GLUE's adhesive and persistence mechanisms using Agno's v2 memory system.
