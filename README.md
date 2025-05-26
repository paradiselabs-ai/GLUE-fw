
# GLUE Framework

## What is GLUE?

GLUE (GenAI Linking & Unification Engine) is a powerful framework designed for the development of sophisticated multi-agent autonomous AI systems. Our vision is to streamline and standardize the creation of agentic applications by providing:

* An intuitive **StickyScript DSL** for defining agent behaviors and workflows, which is pythonically interpreted for advanced customization.
* Robust **Python extensibility**, allowing developers to integrate custom components seamlessly.
* The **GLUE Forge CLI** for managing GLUE projects, agents, tools, and other components (under development).
* A clear structure for **team-based agent organization** and **inter-team communication** through Magnetic Flow Operators.
* A system for **persistent memory and output management** via Adhesives.

GLUE was initially built leveraging core components from the [Agno](https://github.com/agno-ai/agno) framework. While we are grateful for Agno's foundational contributions (the Agno source code included in this repository under `agno/` retains its original MPL-2.0 license), GLUE is now an independent project, evolving with its own unique features and roadmap to meet the specific needs of the autonomous AI development community.

## Read the full docs [here](https://applyglue.com)

*Note: Documentation is actively being developed alongside the framework.*

## Features

* **Dynamic Tool Creation**: GLUE agents can autonomously create custom tools on-the-fly based on task requirements
* **MCP Server Integration**: Seamless connection and creation of MCP servers for enhanced collaboration
* **Natural Team Structure**: Organize AI models into teams with clear roles and responsibilities
* **Intuitive Tool Usage**: Use tools with different adhesive bindings (GLUE, VELCRO, TAPE) for flexible persistence
* **Magnetic Information Flow**: Control how information flows between teams with push and pull patterns
* **Self-Learning Capabilities**: Agents continuously adapt to environmental changes and optimize workflows
* **Simple Expression Language**: Write clear, declarative AI applications with the GLUE DSL
* **Built-in Tools**: Web search, file handling, and code interpretation out of the box
* **Extensible Design**: Create custom tools and add new model providers easily
* **API Key Management**: Integration with [Portkey.ai](https://portkey.ai) for secure API key management

## Core Concepts

### 1. Models and Adhesive Tool Usage

Models are AI agents that can use tools with different adhesive bindings:

* **GLUE**: Team-wide persistent results
* **VELCRO**: Session-based persistence
* **TAPE**: One-time use, no persistence

```glue
model researcher {
    adhesives = [glue, velcro]  // Available binding types
}
```

### 2. Teams and Communication

Teams organize models and their tools:

```glue
magnetize {
    research {
        lead = researcher
        members = [assistant]
        tools = [web_search]
    }
}
```

### 3. Information Flow

Control how teams share information:

```glue
magnetize {
    research {
        lead = researcher
    }
    
    docs {
        lead = writer
    }
    
    flow {
        research -> docs  // Push results
        docs <- pull     // Pull when needed
    }
}
```

## Getting Started & Contributing

We welcome contributions to GLUE! To get started with the development version or to contribute:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/paradiselabs-ai/GLUE-fw.git
    cd GLUE-fw
    ```

2. **Set up your Python environment:**

    We recommend using a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies in editable mode:**

    This allows you to make changes and test them immediately.

    ```bash
    pip install -e .[dev]
    ```

    *(Ensure you have a `[project.optional-dependencies]` section in your `pyproject.toml` for `dev` extras, typically including `pytest`, `ruff`, `black`, etc.)*

4. **Set up API Keys (If using tools/models requiring them):**

    GLUE can integrate with various model providers and tools that may require API keys. Export them as environment variables as needed. For example:

    ```bash
    # For a model provider like OpenRouter
    # export OPENROUTER_API_KEY='your_key_here'
    
    # For a search tool like Tavily
    # export TAVILY_API_KEY='your_key_here'
    ```

    Refer to the documentation for specific tools or models you intend to use.

5. **Run Tests:**

    Ensure your setup is correct by running the test suite:

    ```bash
    pytest
    ```

For more details on our development process, coding standards, and how to submit pull requests, please see our [Contributing Guide](CONTRIBUTING.md) (to be updated).

## Example Applications

### 1. Research Assistant

* Multi-model research system
* Fact-checking and verification
* Documentation generation

### 2. Code Generator

* Architecture design
* Code generation and review
* Testing and validation

### 3. Content Pipeline

* Content research and creation
* Editing and improvement
* Fact verification

## Roadmap (Vision)

### Modular Plugins

* **Orchestration Module**: Advanced agent orchestration with real-time workflow optimization
* **Multi-Role Module**: Multi-agent collaboration with dynamic role assignment
* **Re-Cache Module**: Integrated prompt caching, model capability caching, and enhanced reasoning

### GLUE Studio

* Visual development environment for creating GLUE applications
* Drag-and-drop interface with advanced node-based workflows
* Real-time collaboration and version control

### Deployment

* Simplified deployment through CLI and Studio interfaces
* Cloud-native architecture with auto-scaling capabilities
* Integrated monitoring and analytics dashboard

### StickyScript

* Evolve the GLUE DSL into a certified AI Agent Programming Language
* Fully packaged programming language for rapid AI agent development
* Built-in debugging and optimization tools

## License

The GLUE framework code (excluding the `agno/` directory and its contents) is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The `agno/` directory contains a copy of the Agno framework, which is licensed under the Mozilla Public License Version 2.0. A copy of the MPL-2.0 can be found in `agno/LICENSE`. Any use, modification, or distribution of the code within the `agno/` directory must comply with the terms of the MPL-2.0.
