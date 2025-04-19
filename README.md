# GLUE Framework

# ⚠️ ALPHA RELEASE NOTICE ⚠️

This is an alpha release (0.1.0-alpha) of the GLUE framework. The API may change in future releases. We welcome your feedback and bug reports as we work toward the beta release.

GLUE (GenAI Linking & Unification Engine) is a powerful framework for building multi-model AI applications with natural communication patterns and intuitive tool usage.

## Status Update - April 2025

**All tests are now passing!** We've successfully fixed the flow management and adhesive system issues. The framework is now stable and ready for an alpha release.

### Immediate Alpha Release

We're releasing an alpha version of the GLUE framework immediately:

- **Alpha Release Date**: April 5th, 2025
- **Version**: 0.1.0-alpha
- **Focus**: Core functionality and stability

The alpha release includes all core components with passing tests, but with limited documentation and partial CLI implementation. See the [CHANGELOG.md](./CHANGELOG.md) for details.

### Release Roadmap

- **Alpha**: April 5th, 2025
- **Beta**: April 15th, 2025
- **Release**: May 1st, 2025

## Features

- **Dynamic Tool Creation**: GLUE agents can autonomously create custom tools on-the-fly based on task requirements
- **MCP Server Integration**: Seamless connection and creation of MCP servers for enhanced collaboration
- **Natural Team Structure**: Organize AI models into teams with clear roles and responsibilities
- **Intuitive Tool Usage**: Use tools with different adhesive bindings (GLUE, VELCRO, TAPE) for flexible persistence
- **Magnetic Information Flow**: Control how information flows between teams with push and pull patterns
- **Self-Learning Capabilities**: Agents continuously adapt to environmental changes and optimize workflows
- **Simple Expression Language**: Write clear, declarative AI applications with the GLUE DSL
- **Built-in Tools**: Web search, file handling, and code interpretation out of the box
- **Extensible Design**: Create custom tools and add new model providers easily
- **API Key Management**: Integration with [Portkey.ai](https://portkey.ai) for secure API key management

## Quick Start

1. Install GLUE:
```bash
pip install glue-fw
```

2. Create a GLUE application using `glue new`

3. Set up your API keys:
```bash
# Required
export OPENROUTER_API_KEY=your_key_here

# Optional (for web search)
export SERP_API_KEY=your_key_here
export TAVILY_API_KEY=your_key_here

# Enable Portkey integration
export PORTKEY_ENABLED=true

# Set your Portkey API key
export PORTKEY_API_KEY=your_portkey_api_key
```


4. Run your application:
 Interactive mode:
```bash
glue run app.glue -I
```
 Non-Interactive mode:
 ```bash
glue run app.glue --input "..Define your task.."
```

## Core Concepts

### 1. Models and Adhesive Tool Usage

Models are AI agents that can use tools with different adhesive bindings:

- **GLUE**: Team-wide persistent results
- **VELCRO**: Session-based persistence
- **TAPE**: One-time use, no persistence

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
## Read the full docs [here](https://applyglue.com)

## Example Applications

### 1. Research Assistant
- Multi-model research system
- Fact-checking and verification
- Documentation generation

### 2. Code Generator
- Architecture design
- Code generation and review
- Testing and validation

### 3. Content Pipeline
- Content research and creation
- Editing and improvement
- Fact verification

## Roadmap

### Modular Plugins
- **Orchestration Module**: Advanced agent orchestration with real-time workflow optimization
- **Multi-Role Module**: Multi-agent collaboration with dynamic role assignment
- **Re-Cache Module**: Integrated prompt caching, model capability caching, and enhanced reasoning (cut 40% of your API costs!)

### GLUE Studio
- Visual development environment for creating GLUE applications
- Drag-and-drop interface with advanced node-based workflows
- Real-time collaboration and version control

### Deployment
- Simplified deployment through CLI and Studio interfaces
- Cloud-native architecture with auto-scaling capabilities
- Integrated monitoring and analytics dashboard

### StickyScript
- Evolve the GLUE DSL into a certified AI Agent Programming Language
- Fully packaged programming language for rapid AI agent development with syntax nearly the same level as natural language
- Built-in debugging and optimization tools
- Comprehensive documentation and examples

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
