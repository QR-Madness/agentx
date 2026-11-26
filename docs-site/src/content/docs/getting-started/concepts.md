# Concepts

AgentX uses a handful of terms throughout the app and these docs. Each one is explained here in
a sentence or two, with a link to the page that covers it in full. Read this once and the rest of
the docs will be easier to follow.

## What you run

**Server.** The part that does the work. It runs your agents, keeps their memory, holds your
provider keys and talks to models and tools. You run it yourself with Docker; see the
[Quick Start](quickstart.md).

**Client.** How you use AgentX: the desktop app, or the web client in a browser. Your agents,
conversations and keys stay on the server, so any client connected to it shows the same things.

**Dashboard.** The small web page that comes with the server package, also called the deployment
manager. It starts, stops and watches the server; you don't chat there.
See [Deployment Manager](../deployment/manager.md).

## Agents

**Agent and agent profile.** An agent is an assistant with its own name, model, instructions and
abilities. Its profile is the set of settings that defines it. Every install starts with three
agents, and you can create as many as you like, each suited to a different kind of work.
See [Agent Profiles](../features/agent-profiles.md).

**Model and provider.** The model is the AI that writes an agent's replies. The provider is the
service or program that runs it: Anthropic, OpenAI, OpenRouter, the Vercel AI Gateway, LM Studio
on your own machine, or any OpenAI-compatible endpoint you add. You bring your own key, and each
agent can use a different model. See [Providers](../features/providers.md).

**Thinking pattern.** How an agent works through a problem before it answers: step by step,
from first principles, drafting and then critiquing its own answer, and so on. **Auto** chooses
one for each message. See [Thinking Patterns](../features/reasoning.md).

**Prompt.** The standing instructions an agent follows on every turn. AgentX builds them from
layers: rules that apply to every agent, then each agent's own voice and instructions.
See [Prompts](../features/prompts.md).

## Conversations

**Chat.** A conversation with an agent. While it works, the agent can recall memory and use
tools, and you watch each step as it happens. See [Chat](../features/chat.md).

**The Relay.** The control panel for the conversation in front of you, opened from the button
beside the message box. It holds the thinking mode, memory on or off, team delegation, the model,
the project, attachments, and any runs still working in the background.
See [Chat](../features/chat.md#the-relay--the-conversations-command-center).

## Memory

**Memory.** What an agent keeps between conversations: what you discussed, facts it learned about
you and your work, and approaches that worked. On every turn it recalls the pieces that matter
and works them into its context, without being asked. See [Memory](../features/memory.md).

**Memory channel.** A scope that keeps related memories together, so a work project and a
personal thread don't blur into each other. Recall always draws on your global channel as well,
so an agent keeps your preferences wherever it works. Each project has its own channel, and each
agent has a private one for what it learns about itself.
See [Channels & Projects](../features/memory.md#channels--projects).

## More than one agent

**Agent Teams.** Agents handing work to each other. An agent can pass a focused task to any
teammate that has joined the team roster, and a saved team puts a lead in charge of its members.
See [Agent Teams](../features/multi-agent.md).

**Organization.** A chain of command for your agents: managers direct leads, and each lead runs a
team. See [Organizations](../features/multi-agent.md#organizations).

**Ambassador.** A companion that briefs you about a conversation on the side, by text or voice,
without posting into it. See [Ambassador](../features/multi-agent.md#the-ambassador).

## Tools and knowledge

**Connectors and tools.** A connector links AgentX to an outside service, such as web search,
file storage or a code host, and gives agents tools to use it. You decide which agents can use
which tools. See [Connectors & Tools](../features/mcp.md).

**Skills.** Reusable instructions for a kind of task. An agent sees a short list of the skills it
may use and loads one when a task calls for it. See [Connectors & Tools](../features/mcp.md).

**Projects.** The files and instructions behind a piece of work. Attach a project to a
conversation and the agent can search and cite its documents, follows its instructions, and keeps
what it learns in the project's own memory. See [Projects](../features/workspaces.md).

**Translation.** Built-in translation between more than 200 languages, running on your own
server. See [Translation](../features/translation.md).
