# API Layer

The Django REST API provides AI-powered services for translation and memory management.

## Overview

- Framework: Django 5.2.8
- Port: 12319
- Base URL: `http://localhost:12319/api/`

## Key Components

### Translation Kit (`api/agentx_ai/kit/translation.py`)

Handles language detection and translation using HuggingFace models.

### Conversation System (`api/agentx_ai/kit/conversation.py`)

Manages conversation state and context (planned).

### Memory Graph (`api/agentx_ai/kit/lib/memory_graph.py`)

Graph-based memory system using Neo4j (in development).

## API Endpoints

See [API Reference](../api/endpoints.md) for complete endpoint documentation.

## Next Steps

- [API Endpoints](../api/endpoints.md) - Complete API reference
- [Database Integration](databases.md) - How API connects to databases
