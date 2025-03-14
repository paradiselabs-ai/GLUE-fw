#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import axios from 'axios';

class AgentCollaborationServer {
  private server: Server;
  private agentSessions: Map<string, any> = new Map();
  private pendingActions: Map<string, any> = new Map();

  constructor() {
    this.server = new Server(
      {
        name: 'agent-collab-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupResourceHandlers();
    this.setupToolHandlers();
    
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupResourceHandlers() {
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        {
          uri: 'agent-collab://sessions',
          name: 'Active Agent Sessions',
          mimeType: 'application/json',
          description: 'List of active agent collaboration sessions',
        },
      ],
    }));

    this.server.setRequestHandler(
      ListResourceTemplatesRequestSchema,
      async () => ({
        resourceTemplates: [
          {
            uriTemplate: 'agent-collab://session/{sessionId}',
            name: 'Agent Session Details',
            mimeType: 'application/json',
            description: 'Details of a specific agent collaboration session',
          },
        ],
      })
    );

    this.server.setRequestHandler(
      ReadResourceRequestSchema,
      async (request) => {
        const match = request.params.uri.match(
          /^agent-collab:\/\/session\/([^\/]+)$/
        );
        if (!match) {
          throw new McpError(
            ErrorCode.InvalidRequest,
            `Invalid URI format: ${request.params.uri}`
          );
        }
        const sessionId = match[1];
        const session = this.agentSessions.get(sessionId);
        if (!session) {
          throw new McpError(
            ErrorCode.NotFound,
            `Session ${sessionId} not found`
          );
        }
        return {
          contents: [
            {
              uri: request.params.uri,
              mimeType: 'application/json',
              text: JSON.stringify(session, null, 2),
            },
          ],
        };
      }
    );
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'start_session',
          description: 'Start a new agent collaboration session',
          inputSchema: {
            type: 'object',
            properties: {
              agents: {
                type: 'array',
                items: { type: 'string' },
                description: 'List of agent names to include in the session',
              },
            },
            required: ['agents'],
          },
        },
        {
          name: 'propose_action',
          description: 'Propose an action for human approval',
          inputSchema: {
            type: 'object',
            properties: {
              sessionId: { type: 'string' },
              action: { type: 'string' },
              details: { type: 'object' },
            },
            required: ['sessionId', 'action'],
          },
        },
        {
          name: 'approve_action',
          description: 'Approve a proposed action',
          inputSchema: {
            type: 'object',
            properties: {
              actionId: { type: 'string' },
            },
            required: ['actionId'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      switch (request.params.name) {
        case 'start_session':
          return this.handleStartSession(request.params.arguments);
        case 'propose_action':
          return this.handleProposeAction(request.params.arguments);
        case 'approve_action':
          return this.handleApproveAction(request.params.arguments);
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${request.params.name}`
          );
      }
    });
  }

  private async handleStartSession(args: any) {
    if (!Array.isArray(args.agents) || args.agents.length === 0) {
      throw new McpError(
        ErrorCode.InvalidParams,
        'Invalid agents list'
      );
    }

    const sessionId = crypto.randomUUID();
    const session = {
      id: sessionId,
      agents: args.agents,
      createdAt: new Date().toISOString(),
      status: 'active',
      messages: [],
    };

    this.agentSessions.set(sessionId, session);

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({ sessionId }, null, 2),
        },
      ],
    };
  }

  private async handleProposeAction(args: any) {
    const session = this.agentSessions.get(args.sessionId);
    if (!session) {
      throw new McpError(
        ErrorCode.NotFound,
        `Session ${args.sessionId} not found`
      );
    }

    const actionId = crypto.randomUUID();
    const action = {
      id: actionId,
      sessionId: args.sessionId,
      action: args.action,
      details: args.details || {},
      status: 'pending',
      proposedAt: new Date().toISOString(),
    };

    this.pendingActions.set(actionId, action);

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({ actionId }, null, 2),
        },
      ],
    };
  }

  private async handleApproveAction(args: any) {
    const action = this.pendingActions.get(args.actionId);
    if (!action) {
      throw new McpError(
        ErrorCode.NotFound,
        `Action ${args.actionId} not found`
      );
    }

    action.status = 'approved';
    action.approvedAt = new Date().toISOString();

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({ status: 'approved' }, null, 2),
        },
      ],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Agent Collaboration MCP server running on stdio');
  }
}

const server = new AgentCollaborationServer();
server.run().catch(console.error);
