# AgentX Documentation

AgentX is a self-hosted platform for AI agents that remember what matters, reason in the open and
hand work to each other. You run the server yourself, bring your own model (a provider key or a
model on your own machine), and chat through the desktop app or a browser. Every step an agent
takes is visible as it happens: what it recalled, which tools it used and how it reasoned.

> **Current release:** v0.21.302 (the v0.20 milestone — "Mobile-Ready Alpha").

## Start here

| You want to… | Start with |
|---|---|
| **Try AgentX:** run your own server and chat with your agents | [Quick Start](getting-started/quickstart.md) |
| **Learn the terms:** agents, memory, teams, tools | [Concepts](getting-started/concepts.md) |
| **Use it day to day** | [Chat](features/chat.md), then the other pages under *Using the Platform* |
| **Build on it:** call AgentX from your own code | [Integration Overview](integrate/overview.md) and the [API Reference](api/endpoints.md) |
| **Run it for other people:** a public address, several instances | [Self-Hosting](deployment/self-hosting.md) and *Deploy & Operate* |
| **Work on AgentX itself** | [Development Setup](development/setup.md) and [Architecture](architecture/overview.md) |

## What AgentX does

| Feature | What it means for you |
|---|---|
| **[Agents you define](features/agent-profiles.md)** | Give each agent its own model, instructions, tools and memory: a careful planner, a researcher, a writer. |
| **[Memory that lasts](features/memory.md)** | Agents keep what you discussed, facts about you and your work, and approaches that worked, and recall them on every turn. |
| **[Visible thinking](features/reasoning.md)** | Choose how an agent reasons, such as step by step or draft-then-critique, and watch it happen. |
| **[Teams of agents](features/multi-agent.md)** | Agents hand focused work to teammates, or work in a chain of command. |
| **[Connectors & tools](features/mcp.md)** | Connect web search, file storage, code hosts and more, and choose which agents may use them. |
| **[Any model](features/providers.md)** | Anthropic, OpenAI, OpenRouter, the Vercel AI Gateway, LM Studio, or your own OpenAI-compatible endpoint. |
| **[Projects](features/workspaces.md)** | Give agents the files and instructions behind a piece of work. |
| **[Translation](features/translation.md)** | Translate between more than 200 languages, offline, on your own server. |

## License

AgentX is released under the [MIT License](https://github.com/QR-Madness/agentx/blob/master/LICENSE).
Development history and plans are on the [Roadmap](roadmap.md).
