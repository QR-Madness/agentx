# Client Layer

The Tauri desktop application provides the user interface for AgentX.

## Technology Stack

- **Desktop Framework**: Tauri v2
- **Frontend**: React 19 with TypeScript
- **Build Tool**: Vite
- **Icons**: Lucide React
- **Styling**: CSS with CSS variables (theme system)

## Project Structure

```
client/
├── src/
│   ├── App.tsx                # Main app, wraps RootLayout
│   ├── layouts/               # Layout components
│   │   ├── RootLayout.tsx     # Top bar + page content area
│   │   ├── TopBar.tsx         # Logo, page nav, conversation tabs, toolbar
│   │   └── ConversationTabBar.tsx  # Browser-style conversation tabs
│   ├── pages/                 # Page components
│   │   ├── StartPage.tsx      # Landing page with agent greeting
│   │   ├── DashboardPage.tsx  # Health, memory stats, DB metrics
│   │   └── AgentXPage.tsx     # Main conversation workspace
│   ├── components/
│   │   ├── chat/              # Chat panel, message bubbles, input
│   │   ├── panels/            # Settings, Memory, Tools (drawer content)
│   │   └── modals/            # Translation, Prompt Library (modal content)
│   ├── contexts/              # React contexts
│   │   ├── ServerContext.tsx   # Multi-server state
│   │   ├── ConversationContext.tsx  # Tab management + message state
│   │   ├── AgentProfileContext.tsx  # Agent profile selection
│   │   ├── ThemeContext.tsx    # CSS variable theme system
│   │   └── ModalContext.tsx    # Stack-based modal/drawer management
│   ├── lib/
│   │   ├── api.ts             # Typed API client
│   │   ├── hooks.ts           # React data hooks
│   │   ├── messages.ts        # Discriminated union message types
│   │   └── theme.ts           # Theme definitions
│   └── main.tsx               # Entry point
├── src-tauri/                 # Rust/Tauri backend
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
└── vite.config.ts
```

## Layout Architecture

Three-page layout with horizontal navigation:

1. **Start** — Landing page with agent avatar and greeting
2. **Dashboard** — Health status, memory stats, DB storage metrics
3. **AgentX** — Main conversation workspace with browser-style tabs

### Navigation

- **TopBar** provides page nav pills (left), conversation tab bar (center), and toolbar icons (right)
- **Conversation tabs** are browser-style: add, close, switch, rename, reorder
- **Drawer panels** slide in from the right for Settings, Memory, and Tools (triggered from toolbar)
- **Modal dialogs** for Translation and Prompt Library (triggered from toolbar)

### State Management

- `ConversationContext` manages tabs and messages with localStorage persistence
- `AgentProfileContext` manages agent profile selection and settings
- `ServerContext` provides multi-server state
- `ThemeContext` applies CSS variable themes via `document.documentElement.style`
- `ModalContext` manages a stack of open modals/drawers

## Message Types

Conversation messages use a discriminated union (`lib/messages.ts`):

- `UserMessage` — user input with optional edit action
- `AssistantMessage` — agent response with markdown, thinking, metadata
- `ToolCallMessage` — tool invocation with status (pending/running/completed/failed)
- `ToolResultMessage` — tool output with success/fail and duration
- `MemoryInjectionMessage` — recalled facts with confidence, entities with type badges
- `SystemMessage` — subtle centered text
- `ErrorMessage` — red-tinted error block

## SSE Streaming

The chat stream (`lib/api.ts` `streamChat()`) handles these events:

- `start` — agent name, model, context window info
- `memory_context` — recalled facts, entities, relevant turns
- `chunk` — streaming token content
- `tool_call` — tool name, arguments, call ID
- `tool_result` — tool output, success/fail, duration
- `done` — total tokens, time, thinking content
- `close` — stream complete

## Development

- Dev server: `http://localhost:1420` (Vite)
- HMR enabled on port 1421 for fast iteration
- Tauri window wraps the Vite dev server

See [Development Setup](../development/setup.md) for more information.
