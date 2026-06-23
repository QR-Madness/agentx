# Miscellaneous / Uncategorized Backlog

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

- [ ] Global Default Model (ultimate fallback model) not Configurable
- [ ] Store Consolidation costs
- [ ] Chat steaming affect is very disorientating: use animation smoothing avoid ripping the page scroll around
- [x] Generative Agent Avatar — shipped (`0.21.124`). AvatarPicker "Generate" tab → `POST /api/agent/avatar/generate`
      (OpenRouter `flux.2-klein-4b` default; app-level style prompt in Settings → Images + per-agent subject prompt;
      cost-tracked, source `image`). Stored as a blob in the personal **"Home"** workspace and referenced as
      `media:{ws}/{doc}` on `profile.avatar` (rendered via `AgentAvatar`). Built on the new **image transport**
      (blob serving + `store_media`, `0.21.123`).

### Multi-modal pipeline (next, builds on the image transport)

- [ ] **Agent image-generation tool** — an internal tool so an agent can generate an image mid-conversation
      (reuses `provider.generate_image` + `store_media`), emitting it as an exhibit/attachment.
- [ ] **Image-in-context (vision input)** — send images to vision-capable models (multimodal message content
      blocks); the model capabilities (`supports_vision`, `input_modalities`) are already surfaced via `list_agents`.
- [ ] **Conversation default workspace** — auto-attach the personal **"Home"** workspace to a conversation that
      lacks one when it needs to store media, with a `temp/` scratch namespace + UI to view/clean it.
- [ ] **"Shared" agent-store workspace** — a reserved cross-agent shared workspace (distinct from per-user "Home").
- [ ] **Audio analysis** — agent-side audio understanding (beyond the ambassador's TTS/STT).
- [ ] Fibonacci complexity planning scales (augment planning behaviour based on complexity)
- [ ] Disabled memory conversation prompt message banner - informs the model that memory is off for this conversation and the details are not persistent, and also that the conversation may contain confidential material.
- [ ] Nightly consolidation scheduler — persistent job scheduler (Django Q, Celery, or custom) with cron-like registration, restart survival, graceful shutdown
- [ ] Consolidation job logs endpoint (`GET /api/jobs/{id}/logs`)
- [ ] Real-time job progress (polling while running)
- [ ] Consolidation preview (`POST /api/memory/consolidate/preview`)
- [ ] Fact Transience (a confidence bias) — ranked on extraction for the predicted rate that the fact will be incorrect or irrelevant (e.g., "The user's home PC is slow" = high transience)
- [ ] Extraction example sets for LLM to see a large list of examples to compare from
- [x] GPU acceleration for translation models — shipped `[v0.21.6]`. Shared `kit/device.py`
      `resolve_device()` (`AGENTX_DEVICE`: auto/cpu/cuda/cuda:N); `translation.py` moves both NLLB-200
      + detection models `.to(device)` and moves tokenizer inputs onto it in both hot paths (they ran
      CPU-only before); `embeddings.py` passes `device=` to `SentenceTransformer`. Device surfaced at
      `GET /api/health` → `compute` + logged at load. Docs: Windows Setup + GPU Acceleration pages.
- [ ] Lazy model loading with progress indicator
- [ ] Multiple server support (user can log out of server, and into another one seamlessly)
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Offline mode with cached models
- [ ] Cross-encoder reranking model for retrieval quality
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)
- [ ] Blocking tool call approval (pause stream, user approves/rejects before execution) — the same
      pause/hold-run/resume subsystem would also enable the **blocking in-run Exhibits `choice`**
      round-trip (the user's click becomes the `tool_result` and resumes the same turn, vs. the
      shipped next-turn model). Build once, both benefit.
- [ ] Server authentication (single access key per server, session resume on reconnect)
- [ ] Mobile-responsive breakpoints and touch-friendly gestures
- [ ] Additional themes beyond cosmic (light theme, high contrast, etc.)
- [ ] Message injection into delegated tasks (agent interdiction tools)
- [ ] Custom window chrome — frameless Tauri window with our own title bar + window controls (minimize / maximize / close) and a drag region, styled to the cosmic theme. **Windows + Linux first; macOS later** (traffic-light insets + native fullscreen need separate handling). Touches `client/src-tauri/tauri.conf.json` (`decorations: false`) + a top-of-app titlebar component using the Tauri window API.
- [ ] macOS runner for the client release matrix — add a `macos-latest` leg to `.github/workflows/client-release.yml` (currently Windows + Linux only). Builds `.dmg`/`.app` (`tauri_bundles: dmg,app`); `client/src-tauri/tauri.macos.conf.json` already exists. Needs Apple Developer signing + notarization (certs/secrets) for distributable builds — without them the app is unsigned/Gatekeeper-blocked.
