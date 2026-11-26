# Agent Teams

**Agent Teams** lets your agents hand work to each other. While it answers you, an agent can pass a
focused task to a teammate that is better suited to it, then fold the result into its reply. You
watch the handoff happen in the conversation.

There are two ways to set it up:

- **The team roster.** Any agent you put on the roster can be handed tasks by the agent you're
  chatting with. That agent still handles most things itself and passes work on only when a
  teammate is clearly the better fit.
- **A saved team.** A lead runs the conversation and directs a fixed group of members, for work you
  repeat.

AgentX ships two agents already on the roster: the **Researcher** (web search, cited answers) and
the **Deluxe Image Creator** (hand it a visual brief and the finished image lands in the
conversation). Edit or delete them freely; they won't come back.

## Set it up

Two ways to get agents working together — one takes seconds, the other gives you a repeatable team.

### The quick path — an ad-hoc roster

For teammates you reach for now and then, with no setup ceremony:

1. Open each agent you want available ([Agent Profiles](/docs/features/agent-profiles)) →
   **Core → Team membership** → turn on **Join the team roster** and write a one-line
   **Specialty**.
2. That's it. In any ordinary chat, your active agent now *sees* those teammates in its prompt
   and can hand one a subtask when it's clearly the better fit. It still handles most things
   itself — delegation is an option, not a reflex.
3. Want a chat to stay single-agent? Flip the **Solo/Team** toggle in the Relay to **Solo**.

AgentX ships two agents already on the roster — the **Researcher** and the **Deluxe Image
Creator** — so "research this" or "make me an image of…" delegates out of the box.

### The structured path — a saved Team

For a workflow you run repeatedly (a lead that always orchestrates a fixed set of specialists):

1. Command palette (`⌘K`) → **Open Organization** → **New team**.
2. Pick a **Lead** — the agent that owns the conversation and decides when to delegate. Give it
   a capable model; the orchestration is only as good as the lead.
3. **Add member** for each specialist, each with an optional hint about what to send it.
4. Save, then attach the Team to a conversation. The lead now runs the show under an
   "orchestrate, don't execute" brief, delegating to members over a shared memory channel and
   synthesizing their results into one answer.

!!! note "Lead vs. Members"
    A Team has exactly one **Lead** (the supervisor) and any number of **Members** (specialists).
    Members run in isolation — they see only the task you route them, not the whole conversation —
    and they can't re-delegate. Ambassadors are never team members.

### What you'll see

When an agent delegates, its teammate's work renders **inline** as a delegation card in the
transcript — you watch the subtask run, not just its result. A **Trace** on the conversation opens
the **Team Run Trace**: who delegated to whom, and each step's tokens, cost, and wall-clock time.
Nothing happens off-screen.

## Organizations

A team can belong to an **organization**: give it a **manager**, and the manager delegates to the
team's lead, the lead works its own members, and a member can escalate to its lead. For everyone on
a managed team, this **chain of command** replaces the flat roster as the list of agents they can
hand work to: one step down, or for a member one step up to its lead, never sideways and never past
a level. Teams without a manager, and agents on no team, work exactly as described above.

**Roles** live on the profile: [Agent Profiles](/docs/features/agent-profiles) → **Team
membership** → **Role** (Agent, Lead or Manager). The teams decide the structure; the role decides
how an agent behaves in it. **Manager** adds the manager charter to the agent's prompt and blocks its
hands-on tools (document writes, shell, media generation), so it directs and reports rather than
doing the work itself. You can re-enable any of those tools on the **Tools** tab.

### The Organization surface

Command palette (`⌘K`) → **Open Organization**, or **Manage Teams…** in the agent picker's
**Teams** tab. It draws the whole organization as one chart:

- the **ambassador** at the top, above the chain and outside it (it can dispatch to anyone);
- a reserved **executive** band, empty for now;
- each **manager** above the teams it owns, with a line to each team's lead;
- each **team** as a column, lead first and members below, with a chevron where the line leaves the
  lead: its members can escalate back to it;
- teams **without a manager**, then the **independent** agents that are on no team.

Select a line to see what it means: who directs whom or hands whom work, on which team, and who can
escalate back, with links to both agents and the team. Select a team to edit it in the side panel: name, lead, manager, members and their hints. The
member list shows where each candidate already sits, so putting an agent on a second team is a
choice, not an accident. Select an agent to see its teams, who it reports to and who it directs,
with a link to its profile, and to change the three things the organization reads from it in place:
its **role** (Agent, Lead or Manager; Manager also applies the report-only tool limits), whether it
**joins the team roster**, and its **specialty**. Role and roster save at once; the specialty saves
when you press Save.

**Needs attention** lists anything that won't work as intended: a member id that no longer matches
an agent, a team with no members, or an agent that manages a team without the Manager role (so the
charter and tool limits don't apply to it).

Drag the ambassador, a manager or a whole team to arrange the chart; a team's lead and members
move with it. The arrangement is saved on the server and keeps its place when an agent or team is
renamed. **Reset the chart layout** (the circular-arrow button in the header) puts every box back
in its automatic place. Dragging never changes the structure itself; that is edited in the side
panel. On a narrow screen the same structure appears as an outline, where a small share icon marks
agents on the team roster (other agents can hand them subtasks).

## The Ambassador

The **Ambassador** is a companion that briefs you about your conversations without ever posting
into them. Ask it what your agents decided, for a summary, or to look into a particular turn, and
it answers from what it can read: your conversations, agents, active runs, usage and memory.

- **Beside a chat:** open the command palette (`⌘K` / `Ctrl K`) and choose **Open Ambassador**. It
  docks next to the conversation, so you can watch your agents work and talk to the Ambassador at
  once. Its switcher moves it to another conversation.
- **Across conversations:** the **Command Deck** is the same Ambassador on a full page. Open it
  from the **Deck** pill in the top bar, or with **Open Command Deck** in the palette.
- **Inquiries:** each line of questioning is an **Inquiry**, an ordered thread of briefings and
  answers that you can rename, switch between, or start afresh.
- **Voice:** it can speak its briefings, and you can ask it questions by holding to talk.
- **Actions only with your say-so:** when it suggests renaming, archiving or deleting a
  conversation, or handing a task to an agent, it shows a confirm strip, and nothing happens until
  you confirm. It can also draft a message for you to review and send into a conversation as your
  own turn.

Choose the default Ambassador in **Settings → Ambassador**. An Ambassador is a kind of agent
profile, so you shape its personas and voices in [Agent Profiles](agent-profiles.md).

## Under the hood

How delegation works inside the server (the data model, the runtime, the streaming events,
storage and the API) is on [Agent Teams Internals](../architecture/agent-teams.md). The
[delegation diagram](../architecture/system-design.md#multi-agent-delegation) is on the System
Design page.
