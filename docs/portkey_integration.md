# Portkey.ai Integration

## Overview

The GLUE framework includes integration with [Portkey.ai](https://portkey.ai) for API key management, usage tracking, and cost optimization for AI services. This integration is particularly valuable for applications that use multiple model providers or need to monitor and optimize their AI usage costs.

## Features

- **API Key Management**: Securely manage API keys for different model providers
- **Usage Tracking**: Track API usage across different models and providers
- **Cost Optimization**: Optimize costs by routing requests to the most cost-effective providers
- **Request Tracing**: Trace requests across your application with unique trace IDs
- **Fallback Mechanism**: Gracefully fall back to direct provider calls if Portkey is unavailable

## Setup

### 1. Get a Portkey API Key

Sign up at [Portkey.ai](https://portkey.ai) and obtain an API key.

### 2. Set Environment Variables

```bash
# Enable Portkey integration
export PORTKEY_ENABLED=true

# Set your Portkey API key
export PORTKEY_API_KEY=your_portkey_api_key
```

### 3. Use GLUE Framework as Normal

The Portkey integration is designed to be non-intrusive. Once enabled via environment variables, it will automatically wrap your model providers with Portkey functionality.

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORTKEY_ENABLED` | Enable/disable Portkey integration | `false` |
| `PORTKEY_API_KEY` | Your Portkey API key | None |
| `PORTKEY_BASE_URL` | Portkey API base URL | `https://api.portkey.ai/v1` |

### Custom Configuration

For more advanced configuration, you can create a `PortkeyConfig` object:

```python
from glue.utils.portkey_client import PortkeyConfig, PortkeyClient

# Create custom configuration
config = PortkeyConfig(
    api_key="your_portkey_api_key",
    tags={"environment": "production", "app": "my-glue-app"}
)

# Create client with custom configuration
client = PortkeyClient(config)
```

## Usage Examples

### Basic Usage

The integration is automatically applied when you create models with the GLUE framework:

```python
from glue import GlueApp

# Create a GLUE app
app = GlueApp(
    models=[
        {
            "name": "assistant",
            "provider": "openai",
            "model": "gpt-4",
            "role": "assistant"
        }
    ]
)

# Portkey integration is automatically applied if PORTKEY_ENABLED=true
```

### Accessing Usage Data

You can access usage data from Portkey programmatically:

```python
import asyncio
from glue.utils.portkey_client import get_portkey_client

async def get_usage_data():
    client = get_portkey_client()
    usage = await client.get_usage("2025-04-01", "2025-04-30")
    print(f"Total tokens used: {usage['total_tokens']}")
    print(f"Total cost: ${usage['total_cost']}")

asyncio.run(get_usage_data())
```

## Troubleshooting

### Logs

The Portkey integration logs information about its operations. You can enable debug logging to see more details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **Portkey Not Working**: Ensure `PORTKEY_ENABLED` is set to `true` and `PORTKEY_API_KEY` is set correctly.

2. **API Key Invalid**: Verify your Portkey API key is correct and has the necessary permissions.

3. **Provider Not Supported**: Check if your model provider is supported by Portkey. The integration will fall back to direct provider calls if not supported.

## CI/CD Integration

For CI/CD pipelines, you can set the Portkey API key as a secret in your GitHub repository:

1. Go to your repository settings
2. Navigate to Secrets and variables > Actions
3. Add a new repository secret named `PORTKEY_API_KEY`

Then in your GitHub Actions workflow:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      PORTKEY_ENABLED: true
      PORTKEY_API_KEY: ${{ secrets.PORTKEY_API_KEY }}
    steps:
      # Your workflow steps
```

## Additional Resources

- [Portkey Documentation](https://docs.portkey.ai)
- [GLUE Framework Documentation](https://github.com/paradiselabs-ai/GLUE-fw)
