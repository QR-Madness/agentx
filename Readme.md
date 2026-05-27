<div align="center">

<a href="https://agentx.thejpnet.net/docs">
  <img src="AgentX-Logo-v3-banner.png" alt="AgentX — The Glassbox AI Framework" width="720">
</a>

**An AI Agent Platform for Extreme Customization**

[![Get Started](https://img.shields.io/badge/Get_Started-Read_the_Docs-6366f1?style=for-the-badge&logo=astro&logoColor=white)](https://agentx.thejpnet.net/docs/getting-started/quickstart)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=for-the-badge)]()

---

*Multi-model orchestration • MCP tool integration • Reasoning frameworks • Memory systems*

</div>

## ✨ Features

- **🤖 Agent Core** — Task planning, execution, and context management
- **🔧 MCP Client** — Connect to external tool servers (filesystem, GitHub, databases)
- **🧠 Reasoning Framework** — Chain-of-Thought, Tree-of-Thought, ReAct, Reflection
- **📝 Drafting Models** — Speculative decoding, multi-model pipelines, candidate generation
- **🌐 Multi-Provider** — OpenAI, Anthropic, Ollama support with unified interface
- **💾 Memory System** — Neo4j graphs, PostgreSQL vectors, Redis caching
- **🌍 Translation** — 200+ languages via NLLB-200

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/QR-Madness/agentx.git
cd agentx

# First-time setup (installs deps, initializes databases)
task setup

# Start development environment
task dev
```

**Prerequisites:** [Task](https://taskfile.dev), [uv](https://docs.astral.sh/uv/), [Bun](https://bun.sh), [Docker](https://docker.com)

## 📖 Documentation

<div align="center">

### **[📚 Explore the AgentX Docs →](https://agentx.thejpnet.net/docs)**

New here? Start with the **[Quick Start guide](https://agentx.thejpnet.net/docs/getting-started/quickstart)** to get up and running in minutes.

</div>

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Client["🖥️ Tauri Desktop App"]
        UI["Dashboard · Translation · Agent · Tools · Settings"]
    end
    
    subgraph API["⚡ Django API Layer"]
        Agent["🤖 Agent Core"]
        Reasoning["🧠 Reasoning Framework"]
        Drafting["📝 Drafting Models"]
        MCP["🔧 MCP Client"]
    end
    
    subgraph Data["💾 Data Layer"]
        Neo4j[("Neo4j<br/>Graphs")]
        Postgres[("PostgreSQL<br/>Vectors")]
        Redis[("Redis<br/>Cache")]
    end
    
    subgraph External["🌐 External MCP Servers"]
        FS["Filesystem"]
        GH["GitHub"]
        DB["Databases"]
        Custom["Custom..."]
    end
    
    Client --> API
    Agent --> Reasoning
    Agent --> Drafting
    Agent --> MCP
    API --> Data
    MCP --> External
    
    style Client fill:#1a1a2e,stroke:#a855f7,color:#fff
    style API fill:#0a0d17,stroke:#06b6d4,color:#fff
    style Data fill:#0f0f1a,stroke:#ec4899,color:#fff
    style External fill:#111827,stroke:#6b7280,color:#fff
```

## 🛠️ Development

| Command | Description |
|---------|-------------|
| `task dev` | Start full environment (Docker + API + Client) |
| `task test` | Run backend tests |
| `task lint` | Run linters |
| `task check` | Verify environment ready |
| `task --list` | Show all available tasks |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](https://agentx.thejpnet.net/docs)** · **[Issues](https://github.com/QR-Madness/agentx/issues)** · **[Discussions](https://github.com/QR-Madness/agentx/discussions)**

</div>