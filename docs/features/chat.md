# Chat Interface

AI-powered conversational interface with session management and memory integration.

## Overview

The Chat interface provides multi-turn conversations with the AgentX agent, featuring:

- **Session Management**: Persistent conversation sessions
- **Context Awareness**: Sliding window context with summarization
- **Memory Integration**: Access to episodic and semantic memory
- **Reasoning Traces**: Visibility into agent's thought process

## Usage

### Starting a Conversation

Send a message to begin or continue a chat session:

```bash
curl -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you help me with?"}'
```

### Session Continuity

Include a `session_id` to continue an existing conversation:

```bash
curl -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me more about that",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

## Features

### Context Management

The agent automatically manages conversation context:

- **Token Estimation**: Tracks context window usage
- **Sliding Window**: Keeps recent messages in full detail
- **Summarization**: Compresses older context when needed

### Reasoning Strategies

Chat interactions can use various reasoning patterns:

| Strategy | Use Case |
|----------|----------|
| Chain-of-Thought | Step-by-step reasoning |
| Tree-of-Thought | Exploring multiple approaches |
| ReAct | Tasks requiring tool use |
| Reflection | Self-critique and improvement |

### Tool Integration

The chat interface can access tools from connected MCP servers:

- File system operations
- Web search
- Database queries
- Custom tools

## Client Integration

The Tauri client provides a dedicated Chat tab with:

- Message history display
- Session selector
- Reasoning trace viewer
- Tool usage visualization

## API Reference

See [Agent Endpoints](../api/endpoints.md#agent-endpoints) for full API documentation.
