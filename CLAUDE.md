# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentX is a hybrid desktop application combining:
- **Backend**: Django REST API providing AI-powered language translation and detection services
- **Frontend**: Tauri desktop application with React/TypeScript UI and Vite build system
- **AI Features**: Multi-level language detection and translation using HuggingFace transformers (NLLB-200, M2M100)

## Architecture

### Two-Tier System

The project has a clean separation between API and client:

1. **API Layer** (`api/` directory)
   - Django 5.2.8 application running on port 12319
   - Main app: `agentx_ai` with endpoints for translation and language detection
   - SQLite database for settings and storage
   - Planned integrations: FAISS vector database, Neo4j for relationship analysis (see api/agentx_ai/tests.py:9-12)

2. **Client Layer** (`client/` directory)
   - Tauri v2 desktop app with Rust backend
   - React 19 with TypeScript for UI
   - Vite build system for fast development
   - Tab-based navigation: Dashboard, Translation, Chat, Tools
   - Communicates with Django API

**Note**: The `client-old/` directory contains the previous Electron implementation and can be ignored.

### Translation System Architecture

The translation system implements a two-level approach:

**Level I**: Fast language detection (~20 languages)
- Model: `eleldar/language-detection`
- Used for: Initial language detection with confidence scores
- Returns ISO 639-1 codes (e.g., "en", "fr")

**Level II**: Comprehensive translation (200+ languages)
- Model: `facebook/m2m100_418M` (currently) or `facebook/nllb-200-distilled-600M` (configured)
- Used for: Multi-language translation via NLLB-200 architecture
- Uses extended ISO 639 codes with script info (e.g., "eng_Latn", "zho_Hans")

The `LanguageLexicon` class bridges Level I and Level II by converting between ISO 639 code formats using the `python-iso639` library.

### Key Components

**API Kit System** (`api/agentx_ai/kit/`)
- `translation.py`: Contains `TranslationKit` and `LanguageLexicon` classes
- `conversation.py`: Placeholder for conversation management
- `lib/memory_graph.py`: Skeleton for graph-based memory system
- `lib/neo4j.py`: Neo4j integration utilities

**Client Tabs** (`client/src/components/tabs/`)
- Each tab is a separate React component (DashboardTab, TranslationTab, ChatTab, ToolsTab)
- Tab switching handled by `App.tsx` state management
- All tabs remain mounted to preserve state (visibility controlled via CSS)

## Development Commands

### Running the Application

Use Task (Taskfile.yaml) for all development operations:

```bash
# Start both API and client in development mode
task dev

# API only
task api:runserver          # Starts Django server on port 12319

# Client only (Tauri dev mode)
cd client && npm run tauri dev

# Install all dependencies
task install                # Runs: uv install && npm install
```

### Django API Commands

```bash
# Database operations
task api:migrate
task api:makemigrations

# Django shell
task api:shell

# Direct command (if needed)
cd api && python manage.py runserver --port 12319
```

### Tauri Client Commands

```bash
# Development (starts Vite dev server + Tauri window)
cd client && npm run tauri dev

# Build distributable packages
cd client && npm run tauri build

# Vite-only development (browser preview, no Tauri)
cd client && npm run dev          # Runs on localhost:1420
cd client && npm run build        # TypeScript check + Vite build
cd client && npm run preview      # Preview production build
```

### Testing

```bash
# Run all tests
task test

# Run specific test
python api/manage.py test agentx_ai.TranslationKitTest.test_translate_to_french

# Run specific test class
python api/manage.py test agentx_ai.TranslationKitTest
```

## Important Technical Details

### API Endpoints

Base URL: `http://localhost:12319/api/`

- `GET /api/index` - Health check
- `GET /api/language-detect` - Detect language from hardcoded test text
- `POST /api/translate` - Translate text to target language
  - Body: `{"text": "...", "target_language": "fr"}` (ISO 639-1 code)

### Translation Model Loading

The `TranslationKit` class in `api/agentx_ai/kit/translation.py`:
- Loads models at initialization (not lazy-loaded)
- Currently uses M2M100-418M model despite NLLB-200 being configured
- See lines 89-92 for the model override

### Tauri Configuration

- Main window config: `client/src-tauri/tauri.conf.json`
  - Window dimensions: 800x600
  - Dev URL: http://localhost:1420
  - Frontend build output: `../dist`
- Rust dependencies: `client/src-tauri/Cargo.toml`
  - tauri v2 with opener plugin
  - serde for serialization
- Vite config: `client/vite.config.ts`
  - Dev server port: 1420 (strict)
  - HMR port: 1421

### Client Architecture

**Tab Management Pattern:**
- All tabs are always mounted in the DOM
- Only one tab visible at a time (controlled by `display` CSS property)
- This preserves component state when switching between tabs
- State management done in `App.tsx` via `useState` hook

**File Structure:**
- `client/src/App.tsx` - Main app component with tab routing
- `client/src/components/TabBar.tsx` - Tab navigation UI
- `client/src/components/tabs/*.tsx` - Individual tab components
- `client/src-tauri/` - Rust/Tauri backend code

## Python Dependencies

Managed via `pyproject.toml` with uv:
- django>=5.2.8
- torch>=2.9.1 (for ML models)
- transformers>=4.57.1 (HuggingFace)
- sentencepiece>=0.2.1 (tokenization)
- python-iso639>=2025.11.16 (language code conversion)
- termcolor>=3.2.0 (terminal output)
- faiss-cpu>=1.13.0 (vector database - planned)

## Known Issues & TODOs

From api/agentx_ai/tests.py:
- FAISS integration for vector database
- Django ORM for AI settings storage
- Neo4j for structured information and relationship analysis
- Translation function returns placeholder "add this later" (translation.py:140)

## Migration Notes

The project recently migrated from Electron to Tauri:
- Old Electron code is in `client-old/` directory (can be ignored)
- New Tauri implementation is in `client/` directory
- Tauri provides smaller binary sizes and better security model
- Frontend stack remains React + TypeScript, but build system changed from Webpack to Vite
