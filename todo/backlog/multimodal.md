# Multi-Modal Fluency — media across agents, surfaces, and delegation

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

> Direction (product): agents must be *fluent* in many media types — and where a model
> can't handle a medium itself, agents should **autonomously find one that can**.
> Foundation shipped `[v0.21.237]` (content blocks, audio in/out, media exhibits) and
> `[v0.21.238]` (neutral speech seam ADR-11, capability probe, media-aware budgets);
> see Development-Notes → "Multi-modal content protocol".

Open:
- [ ] **Media-capable delegation routing** (the autonomy piece) — `delegate_to`/`delegate_start`
      carry a **text task only** today: attached images/audio can't ride a delegation at all, so a
      supervisor physically can't hand media to a specialist. Slice: (1) `media` param on the
      delegation tools (list of `MediaRef`s, validated like chat's `images[]`/`audio[]`, injected
      into the specialist's first turn via the existing converters); (2) the delegation roster block
      + tool descriptions annotate each member's modalities derived from its model's
      `input_modalities` via the shared probe (`providers/capabilities.py`) — e.g.
      `Analyst 👁 sees images · Scout 🎧 hears audio` — so the supervisor routes media
      autonomously; (3) capability mismatch degrades honestly (STT transcript for audio → text-only
      specialist, vision refs stripped with a note), reusing the chat gates.
- [ ] **Video input to models** — render-only today (deliberate). When wired: gate on
      `input_modalities` has `video` (Gemini-class via OpenRouter), same `MediaRef` seam, hard size
      caps, and *no* silent fallback (video has no cheap transcript analog — a can't-see model gets
      a clear notice, or routes to a capable agent per the item above).
- [ ] **Video streaming instead of blob-slurp** — `resolveMediaBlob` loads the whole file into
      memory before playback (fine for clips; wrong for 100MB mp4). A `<video>`-grade path wants
      authenticated **range requests**: short-lived signed URLs on the raw-blob route (or a
      token-in-query variant) + `Accept-Ranges` on `workspace_document_raw`. The client LRU
      (v0.21.238) is the stopgap.
- [ ] **`listen_audio` internal tool** — the `view_image` analog for stored audio docs (cut from
      the v0.21.237 slice): native injection for audio-capable models via the
      `_view_image_messages` pattern, STT-transcript tool result otherwise. Manifest marking (🎧)
      mirrors the image catalog.
- [ ] **`speech.*` config namespace** — the neutral seam (`kit/speech.py`) still reads the
      historical `ambassador.speech_model`/`voice`/`transcription_model` keys (ADR-11 boundary
      note). Migrate to `speech.*` with `ambassador.*` fallback + settings relabel, one slice,
      no behavior change.
- [ ] **Media in memory/consolidation** — extraction currently sees transcript text only (an
      audio turn's cached transcript rides the turn text; images contribute nothing). Someday:
      alt-text/caption extraction into facts, media provenance on entities ("the diagram in
      conversation X").

Related: [exhibits.md](exhibits.md) (gallery panel, richer layouts) ·
[open-platform.md](open-platform.md) (ACP/A2A bridge — the content-block vocabulary is the prep) ·
[agentic-organizations.md](agentic-organizations.md) (durable delegation threads the media routing
will ride on).
