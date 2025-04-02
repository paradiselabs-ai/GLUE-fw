# GLUE Forge

GLUE Forge is a powerful tool within the GLUE Framework that helps you create custom components with AI assistance. It leverages Google's Gemini 2.5 Pro model to generate code templates for tools, MCP integrations, and API clients.

## Getting Started

To use GLUE Forge, simply run:

```bash
glue forge
```

This will start an interactive session that guides you through the process of creating custom components.

## API Key Setup

GLUE Forge requires an API key to access the AI model. You have two options:

1. **Google AI Studio** - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **OpenRouter** - Get a free API key from [OpenRouter](https://openrouter.ai/keys)

During your first run, GLUE Forge will prompt you to choose your API key source and enter your key. This key will be securely stored for future use.

## Creating Custom Components

GLUE Forge can help you create three types of custom components:

### 1. Custom Tools

Tools are components that provide specific functionality to your GLUE applications. Examples include web search, file handling, and code interpretation.

To create a custom tool:

```bash
glue forge tool <name> --description "<description>" [--template <template>]
```

Available templates:
- `basic` - A simple tool with a single function (default)
- `api` - A tool that makes external API calls
- `data` - A tool that processes data files

Example:
```bash
glue forge tool weather --description "Get weather information for a location" --template api
```

### 2. Custom MCP Integrations

MCP (Model Control Protocol) integrations allow you to connect to external AI services and use them within your GLUE applications.

To create a custom MCP integration:

```bash
glue forge mcp <name> --description "<description>"
```

Example:
```bash
glue forge mcp custom_llm --description "Integration with a custom LLM API"
```

### 3. Custom API Integrations

API integrations allow you to connect to external services and use them within your GLUE applications.

To create a custom API integration:

```bash
glue forge api <name> --description "<description>"
```

Example:
```bash
glue forge api weather_service --description "Integration with a weather service API"
```

## Using Custom Components

After creating a custom component, you can use it in your GLUE applications by adding it to your `.glue` file:

### Using a Custom Tool

```
tool weather {
    custom = true
}
```

### Using a Custom MCP

```
mcp custom_llm {
    custom = true
}
```

## Advanced Usage

For more advanced usage, you can modify the generated code to add custom functionality. The generated code is designed to be a starting point that you can extend and customize as needed.

## Troubleshooting

If you encounter any issues with GLUE Forge:

1. Check that your API key is valid and has not expired
2. Ensure you have an active internet connection
3. Check the logs in `~/.glue/logs/glue.log` for more information

To reset your API key configuration:

```bash
rm ~/.glue/forge_config.json
```

Then run `glue forge` again to set up a new API key.
