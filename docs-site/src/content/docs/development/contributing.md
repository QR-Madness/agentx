# Contributing

## Development Workflow

1. Fork and clone the repository
2. Create a feature branch from `master`: `git checkout -b feat/my-feature`
3. Run `task setup` (first time) or `task install` (subsequent)
4. Make changes — `task dev` for hot reload
5. Run `task test:quick` (fast) or `task test` (full)
6. Run `task lint` and `task check:types`
7. Commit with a descriptive message
8. Submit a pull request

### Branch Naming

| Prefix | Use |
|--------|-----|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Code restructuring |
| `docs/` | Documentation changes |
| `test/` | Test additions/fixes |

### Commit Style

Use imperative mood: "Add memory recall layer" not "Added memory recall layer".

Prefix with scope when relevant: `fix(mcp): resolve env var expansion in headers`.

## Code Style

### Python

- **Linter/formatter**: ruff (`task lint:python`, `task format:python`)
- **Type checker**: pyright (`task check:types:python`)
- Follow existing patterns — look at neighboring code before writing new code
- Use type hints on all public function signatures
- Docstrings on public classes and non-obvious functions

### TypeScript

- Type check: `task check:types:client`
- Build check: `task check:build:client`

## Testing

### Running Tests

```bash
task test:quick     # Fast — skips model loading (HealthCheck, MCP tests)
task test           # Full — includes TranslationKit tests (slow first run)

# Specific test class or method:
uv run python api/manage.py test agentx_ai.tests.MCPClientTest -v2
uv run python api/manage.py test agentx_ai.tests_memory.MemoryIntegrationTest -v2
```

### Writing Tests

- Add tests to `tests.py` (core) or `tests_memory.py` (memory system)
- Use `@unittest.skipUnless` for tests that require Docker or API keys
- Tests should be independent — no shared mutable state between test methods
- Memory tests: skip gracefully when databases are unavailable

## Architecture Guidelines

### Lazy Initialization

Heavy subsystems use `@lazy_singleton` (see `utils/decorators.py`). Never import heavy modules at file scope — use deferred imports inside functions.

### Sync/Async Boundary

Django is synchronous. MCP and provider streaming use asyncio. Bridge with `call_tool_sync()` or `asyncio.run_coroutine_threadsafe()`. Only `agent_chat_stream` and `providers_health` are async views.

### Adding a New Provider

1. Create `providers/my_provider.py` implementing `ModelProvider` ABC
2. Add models to `providers/models.yaml` with `provider: my_provider`
3. Register in `providers/registry.py` loader
4. Add env vars to `.env.example`

### Adding a Reasoning Strategy

1. Create `reasoning/my_strategy.py` implementing `ReasoningStrategy` ABC
2. Add to orchestrator's strategy map in `orchestrator.py`
3. Add task type classification keywords if needed

### Adding a Memory Subsystem

1. Create module under `kit/agent_memory/memory/`
2. Wire into `AgentMemory` interface (`memory/interface.py`)
3. Add database schema if needed (`management/commands/init_memory_schemas.py`)
4. Add tests in `tests_memory.py`

## PR Checklist

- [ ] Tests pass (`task test:quick` at minimum)
- [ ] Linter passes (`task lint`)
- [ ] Type checker passes (`task check:types`)
- [ ] New public APIs have docstrings
- [ ] Documentation updated if user-facing behavior changed
- [ ] No hardcoded API keys or secrets
