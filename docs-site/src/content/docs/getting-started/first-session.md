# Your First Session

This walkthrough picks up where the [Quick Start](quickstart.md) ends: your server is running and
the app is open. In about fifteen minutes you'll connect a model, have a first conversation, make
an agent of your own, and watch AgentX remember what you told it. The screenshots come from a
fresh install.

![The Start page of a fresh install](/images/first-session/start.webp)

## 1 · Connect a model provider

AgentX doesn't come with a model; you bring your own. Open the command palette (`⌘K` / `Ctrl K`)
and choose **Settings: Model Providers**. Open a provider, paste its API key and choose
**Save to server**. For OpenRouter you can choose **Link account** instead, which signs you in once
and gives AgentX its own key without any copying. **Test connection** checks that the provider
answers.

![Model Providers, with the OpenRouter card open](/images/first-session/providers-openrouter.webp)

OpenRouter is marked **Recommended** because one key reaches hundreds of models from many
companies. To run models on your own machine instead, add your LM Studio address in the same place;
if LM Studio runs on the same machine as the server, see [Troubleshooting](troubleshooting.md#models).

## 2 · Choose your default model

Next, tell AgentX which model to use when nothing else says otherwise. Open
**Settings: Model Roles** and set the **Global Default Model**. Every agent without a model of its
own uses it, so a fast, inexpensive model is a sensible start; you can give particular agents
stronger models later.

![Model Roles, with a global default model set](/images/first-session/global-default-model.webp)

!!! warning "Don't skip this step"
    A fresh install has no global default, and its fallback is a model on LM Studio. If you only
    connected a cloud provider, new conversations stop with "No model chosen" until you set one
    here. AgentX won't pick a cloud model for you, since that would spend money you didn't approve.

Choosing a model opens the **Choose model** dialog. Search by name, filter by what a model can do
(tools, vision, image generation), and compare each model's context size and price per million
tokens before you pick one.

![The Choose model dialog, searching for a model](/images/first-session/choose-model.webp)

## 3 · Have your first conversation

Type into the box on the Start page, or open **Agents** in the top bar and start a new
conversation. You're talking to **AgentX**, the general assistant every install starts with.

![A first conversation](/images/first-session/first-conversation.webp)

A few things to notice:

- **Each step is visible.** Tool calls and memory lookups appear as lines in the conversation,
  and you can open one to see what happened. Here, the agent recorded your goal in the
  conversation's working state.
- **The chips under the message box** set this conversation's agent, model, memory, team
  delegation and [thinking pattern](../features/reasoning.md). Choosing a model there changes it
  for this conversation only.
- **Every reply shows what it cost:** the model, the tokens in and out, an estimated price and how
  long it took.

## 4 · Make an agent your own

Agents are defined by **profiles**. Open the command palette, choose **Agent profiles**, then
**New Profile**. Give it a name, a one-line description and its **instructions**, the standing
guidance it follows in every conversation. You can also choose its model (or leave
**System default**) and its thinking pattern. Then choose **Create Profile**.

![Creating a "Reading Coach" agent](/images/first-session/new-profile.webp)

To talk to your new agent, pick it from the agent chip under the message box. Here the coach
follows its instructions (short and practical, ending with a question), and its badge shows it
reasoned with the **Step-back** pattern, which works out the governing principles before it
answers:

![The new agent's first answer](/images/first-session/reading-coach.webp)

## 5 · Watch memory work

AgentX remembers across conversations. Start a new conversation and ask about something you
mentioned in an earlier one:

![Recall in a new conversation](/images/first-session/memory-recall.webp)

The **Memory** and **User history recall** lines show the agent looking back before it answers.
In the background, AgentX also turns conversations into lasting **facts**. Open **Memory** in the
top bar to see them. Consolidation runs every 15 minutes, or you can choose **Consolidate** to run
it now; if the list hasn't updated afterwards, use the refresh button beside **Extract**.

![What AgentX learned, in the Memory panel](/images/first-session/memory-facts.webp)

Each fact shows how confident AgentX is in it and where it came from. [Memory](../features/memory.md)
explains how recall and consolidation work, and how to export or move what it has learned.

## Where next

| You want to… | Go to |
|---|---|
| Understand the terms used here | [Concepts](concepts.md) |
| Give agents tools: web search, files, code hosts and more | [Connectors & Tools](../features/mcp.md) |
| Let agents hand work to each other | [Agent Teams](../features/multi-agent.md) |
| Work from a set of documents | [Projects](../features/workspaces.md) |
| Choose how agents reason | [Thinking Patterns](../features/reasoning.md) |
| Fix something that isn't working | [Troubleshooting](troubleshooting.md) |
