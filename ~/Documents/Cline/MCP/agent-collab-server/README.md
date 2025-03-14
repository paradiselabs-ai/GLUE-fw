# Agent Collaboration MCP Server

The Agent Collaboration MCP Server enables multiple AI agents (CASCADE, Cline, Blackbox-ai, etc.) to communicate and collaborate autonomously on project development. It provides visibility into the agents' communications and decisions, and includes a human-in-the-loop mechanism for confirming file changes and terminal commands.

## Features

- **Agent Collaboration**: Facilitates communication and collaboration between multiple AI agents.
- **Decision Visibility**: Provides visibility into the agents' discussions and decisions.
- **Human-in-the-Loop**: Requires human confirmation for file changes, terminal commands, and other critical actions.
- **Session Management**: Manages active agent collaboration sessions and their details.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/agent-collab-server.git
   cd agent-collab-server
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure the server by editing `config.json`.

4. Start the server:
   ```bash
   npm start
   ```

## Usage

### Starting a New Session

To start a new agent collaboration session, use the `start_session` tool:

```json
{
  "tool": "start_session",
  "arguments": {
    "agents": ["CASCADE", "Cline", "Blackbox-ai"]
  }
}
```

### Proposing an Action

To propose an action for human approval, use the `propose_action` tool:

```json
{
  "tool": "propose_action",
  "arguments": {
    "sessionId": "session-id",
    "action": "create_file",
    "details": {
      "path": "src/index.ts",
      "content": "console.log('Hello, world!');"
    }
  }
}
```

### Approving an Action

To approve a proposed action, use the `approve_action` tool:

```json
{
  "tool": "approve_action",
  "arguments": {
    "actionId": "action-id"
  }
}
```

## Configuration

The server can be configured by editing the `config.json` file. The following options are available:

- **server.port**: The port number the server will listen on.
- **server.logLevel**: The logging level (e.g., "info", "debug").
- **agents.defaultAgents**: The default list of agents to include in new sessions.
- **approval.timeout**: The timeout for pending actions (in milliseconds).
- **approval.maxPendingActions**: The maximum number of pending actions allowed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
