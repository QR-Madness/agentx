# Architecture Overview

AgentX follows a two-tier architecture with clear separation between API and client layers.

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Tauri[Tauri v2 Desktop App]
        React[React 19 + TypeScript]
        Vite[Vite Build System]
    end

    subgraph "API Layer"
        Django[Django 5.2.8 REST API]
        ML[HuggingFace Transformers]
    end

    subgraph "Data Layer"
        Neo4j[Neo4j Graph DB]
        Postgres[PostgreSQL + pgvector]
        Redis[Redis Cache]
    end

    React --> Tauri
    Tauri -->|HTTP| Django
    Django --> ML
    Django --> Neo4j
    Django --> Postgres
    Django --> Redis
```

## Core Components

- **Client Layer**: Desktop application built with Tauri and React
- **API Layer**: Django REST API providing AI services
- **Data Layer**: Multi-database stack for different use cases

See [API Layer](api.md), [Client Layer](client.md), and [Database Stack](databases.md) for details.
