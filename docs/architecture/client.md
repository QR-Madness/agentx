# Client Layer

The Tauri desktop application provides the user interface for AgentX.

## Technology Stack

- **Desktop Framework**: Tauri v2
- **Frontend**: React 19 with TypeScript
- **Build Tool**: Vite
- **Styling**: CSS Modules (or your choice)

## Project Structure

```
client/
├── src/
│   ├── App.tsx              # Main app component
│   ├── components/          # React components
│   │   ├── TabBar.tsx       # Tab navigation
│   │   └── tabs/            # Tab components
│   └── main.tsx             # Entry point
├── src-tauri/               # Rust/Tauri backend
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
└── vite.config.ts
```

## Tab System

Four main tabs with persistent state:

1. **Dashboard** - Overview and stats
2. **Translation** - Interactive translation
3. **Chat** - AI conversations
4. **Tools** - Utilities and settings

All tabs remain mounted; visibility controlled via CSS.

## Development

- Dev server: `http://localhost:1420` (Vite)
- HMR enabled for fast iteration
- Tauri window wraps the Vite dev server

See [Development Setup](../development/setup.md) for more information.
