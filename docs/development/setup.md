# Development Setup

Set up your development environment for contributing to AgentX.

## Prerequisites

Install required tools:

- Python 3.10+ with uv
- Node.js 18+ with bun
- Docker & Docker Compose
- Task runner
- Git

See [Installation](../getting-started/installation.md) for details.

## Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agentx-source.git
cd agentx-source

# Install dependencies
task install

# Initialize databases
task db:init

# Start development environment
task dev
```

## Development Workflow

1. Create feature branch
2. Make changes with hot reload
3. Run tests
4. Commit and push
5. Create pull request

See [Contributing Guide](contributing.md) for detailed workflow.
