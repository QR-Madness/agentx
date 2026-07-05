# Archived Research — Memory Extraction (write path)

> **Provenance:** external deep-research run, archived 2026-07-05. Its agents could **not** read
> this codebase (see the body's own Caveats) — current-state claims were inferred from the public
> docs. Integrated into [`Memory-Roadmap.md`](../../Memory-Roadmap.md) (§2.10 + §2.3/§2.7/§3.12
> enrichments) and [`todo/backlog/retrieval-extraction.md`](../backlog/retrieval-extraction.md);
> those docs own the work items — this file is the citation/literature record.
>
> **Verification corrections (code-checked at v0.21.152 — trust these over the body where they conflict):**
> - Extraction runs through the `providers/` `ModelProvider` abstraction → **LM Studio**
>   (OpenAI-compatible), not vLLM/SGLang/Ollama. So the constrained-decoding mechanism for AgentX is
>   **LM Studio structured outputs (`response_format` json_schema) plumbed through the provider
>   layer** — not server-side XGrammar/Outlines. `provider.complete()` already carries
>   `tools`/`tool_choice` (`providers/base.py`); `response_format` is the missing plumbing.
> - The prompt-only-formatting claim is **confirmed**: no `response_format`/schema/function-calling
>   anywhere in `extraction/service.py`; output is parsed heuristically (`_parse_combined_response`,
>   `_repair_truncated_json`).
> - Live consolidation uses the `combined` stage (`check_relevance_and_extract`), default
>   `lmstudio:nvidia/nemotron-3-nano`; the single-call `extraction` stage default is
>   `lmstudio:google/gemma-3-4b` — not Haiku as the recall survey's A.1 states.
> - **No NLI/entailment pass exists** (confirmed net-new) and **no write-time salience exists** —
>   fact salience is a constant 0.5 default (`models.py`; `learn_fact` takes no salience), so the
>   write-time importance recommendation is *stronger* than the body assumes.

---

# Memory Extraction for AgentX — A State-of-the-Art Survey and Integration Plan

## TL;DR
- **The two pain points are the same problem attacked from opposite ends: extract *fewer, atomic, schema-typed* propositions and *force* their format with constrained decoding.** For usefulness, adopt an extract → salience-score → novelty-gate (ADD/UPDATE/DELETE/NOOP) pipeline like Mem0's; for format consistency, replace prompt-only local-model extraction with a strict JSON Schema enforced by grammar-constrained decoding (XGrammar/Outlines) or a function-calling frontier model — AgentX's own backlog already flags "Claude Sonnet for Extraction" as the fix.
- **AgentX is architecturally ahead of most OSS memory layers** (bitemporal facts, consolidation-owns-writes, SUPERSEDES edges, salient core, a heuristic pre-filter that already drops ~75% of extraction calls) but its extraction service still runs on local models with prompt-only formatting, static hand-tuned confidence tiers (0.95/0.85/0.70/0.50) and a single 0.92 dedup point-estimate — exactly the knobs that make output inconsistent and noisy.
- **Highest-leverage moves, in order:** (1) constrained/structured decoding against a canonical fact+entity+edge schema; (2) atomic-proposition extraction (Dense X / propositional chunking) for the pgvector side + triple extraction (Zep/Graphiti-style) for the Neo4j side; (3) an NLI/self-consistency verification pass to kill hallucinated facts; (4) a golden-set `eval_recall` harness driven by LoCoMo/LongMemEval so every prompt/threshold change is measured, not guessed.

## Key Findings

**1. "Usefulness" is an explicit write-time decision, not a byproduct.** Every mature system scores or gates candidate memories before committing them. Generative Agents (Park et al., UIST 2023) ask the LLM to rate poignancy 1–10 ("1 is purely mundane… 10 is extremely poignant"), store it, and combine it with recency (exponential decay) and relevance (cosine) at retrieval. Mem0 (Chhikara et al., ECAI 2025, arXiv:2504.19413) runs a two-phase pipeline — an extraction phase that distills each turn (latest exchange + rolling summary + recent messages) into candidate facts, then an update phase where an LLM routing controller inspects the top-k similar existing memories and emits ADD/UPDATE/DELETE/NOOP. This is the single most-copied write-time pattern, and the reported payoff is large: on LOCOMO, Mem0 "achieves 26% relative improvements in the LLM-as-a-Judge metric over OpenAI, while Mem0 with graph memory achieves around 2% higher overall score," and it "attains a 91% lower p95 latency [1.44s vs 17.12s] and saves more than 90% token cost" versus full-context baselines.

**2. "Consistency" is solved by moving format enforcement out of the prompt.** Grammar-constrained decoding (Outlines, Guidance/llguidance, XGrammar) compiles a JSON Schema into a finite-state machine and masks invalid tokens to −∞ at each step, guaranteeing 100% schema adherence by construction. XGrammar (Dong et al., arXiv:2411.15100) enforces JSON-schema/regex/EBNF with under 40µs per-token overhead and "can achieve up to 100x speedup over existing solutions… near-zero overhead structure generation," and is now the default backend across vLLM, SGLang, TensorRT-LLM and MLC-LLM. This directly addresses AgentX's "produce it in a consistent format" pain point far more reliably than prompt engineering with a local model.

**3. The graph side and vector side want different extraction units.** For Neo4j, the state of the art is LLM-based triple/edge extraction with entity resolution and bitemporal stamping (Zep/Graphiti, GraphRAG, HippoRAG's OpenIE). For pgvector, the state of the art is atomic propositions — self-contained, minimal, single-fact sentences (Dense X Retrieval / "propositional chunking"), which measurably beat sentence- and passage-level retrieval.

**4. Verification is what stops hallucinated/low-value memories.** NLI entailment checks (DeBERTa-v3-large/MNLI, AlignScore, INFUSE, FactCC/SummaC) verify that each extracted claim is entailed by its source turn; self-consistency (sample-and-agree) and extract-then-verify passes further filter. This is cheap insurance against the "extracted a hallucination as a durable fact" failure.

**5. Evaluation must be downstream-anchored.** Fact-level precision/recall against a gold set is necessary but not sufficient; the true metric is downstream retrieval/QA accuracy on LoCoMo and LongMemEval. LLM-as-judge is the standard scorer but carries documented position, verbosity and self-preference/self-enhancement biases that must be mitigated (order-swapping, length control, per-claim atomic judging).

## Details

### A. AgentX current state (confirmed from repo docs; code files noted where inaccessible)

From `Memory-Roadmap.md`, `Decisions.md`, `todo/backlog/retrieval-extraction.md` and `todo/backlog/procedural.md` (the actual `extraction/service.py` and `consolidation/jobs.py` could **not** be fetched — GitHub raw/tree and the docs site at agentx.thejpnet.net are robots-blocked, so prompt/schema bodies are inferred from the roadmap's file-map and prose, not read verbatim):

- **Extraction runs on local models today.** The backlog explicitly lists "Claude Sonnet for Extraction — switch extraction from local models to Claude Sonnet for better structured-output adherence, nuance detection, and entity resolution." This confirms both that (a) extraction is currently local and (b) the maintainer already suspects structured-output adherence is weak.
- **Embeddings are 1024-dim bge-m3**, hardcoded across 4 Neo4j vector indexes + 3 pgvector tables. Similarity thresholds are "silently bge-m3-calibrated."
- **Confidence is a static hand-tuned ladder:** `explicit=0.95 / implied=0.85 / inferred=0.70 / uncertain=0.50`. The roadmap itself flags these as static and proposes fitting per-`extraction_model` isotonic calibration.
- **Dedup is a single point estimate:** `0.92` cosine gate for facts; a separate `procedural_dedupe_threshold` for procedures. §3.12 proposes conformal bands (auto-merge above / auto-pass below / LLM-adjudicate the middle).
- **A heuristic relevance pre-filter already cut ~75% of extraction calls.** §3.3 proposes adding a "surprise" score (embedding distance vs working-memory centroid + recall-hit-rate).
- **Durable writes belong to consolidation, not the hot path** (core invariant). Hot-path recall may *queue* discoveries but never mints facts inline. A `ConsolidationWorker` runs a **15-minute sweep** over unconsolidated turns; §3.1 proposes a debounced idle trigger to cut the 14-minute blindness.
- **Facts carry `temporal_context ∈ {current, past, future}`**, `confidence`, `source_turn_id`, `[:ABOUT]` edges to entities, and `[:SUPERSEDES]` edges. `check_contradictions` / `check_correction` functions exist in the contradiction pipeline; correction = mark superseded `past` + link + prioritize corrections in recall. §1.7 plans full bitemporal (`valid_from/valid_to` world time + `asserted_at/superseded_at` transaction time).
- **Entity resolution exists** via `dedupe_entities` (never merges Agent nodes — `agent_id` is durable identity), plus `_resolve_fact_entity_ids`/`link_facts_to_entities` repair paths.
- **Eval:** `eval_consolidation` is shipped (write-path); `eval_recall` (read-path golden set) is planned (§2.7).

**Where AgentX diverges from best practice.** (1) Format is enforced by prompt + a local model, not constrained decoding or function-calling — the direct cause of inconsistency. (2) There is no explicit LLM salience/importance score at write time (Generative-Agents style); usefulness is proxied only by the heuristic pre-filter and downstream salience/decay. (3) No atomic-proposition normalization step before embedding — facts are extracted as whole sentences, diluting pgvector recall. (4) No NLI/faithfulness verification pass, so a hallucinated local-model fact can be committed with confidence 0.70. (5) Thresholds (0.92, confidence ladder) are hand-set and un-calibrated. (6) `eval_recall` doesn't exist yet, so extraction-prompt changes are currently unmeasured.

### B. WHAT to extract — salience, novelty, and typing

- **LLM importance scoring (Generative Agents).** Prompt: "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. Memory: <…> Rating: <fill in>." Score is set at object creation; retrieval score = normalized(recency + importance + relevance), equal weights, recency decay γ≈0.995/hour. Reflections fire when summed importance exceeds a threshold (150 in the paper).
- **Mem0 extract-then-compare.** Step 1 fact extraction via `get_fact_retrieval_messages()` (user vs agent prompt variants); step 2 the memory-update decision prompt: "You are a smart memory manager… perform four operations: (1) add… (2) update… (3) delete… (4) no change… Compare newly retrieved facts with the existing memory." Returns strict JSON with `event ∈ {ADD, UPDATE, DELETE, NONE}`. The graph variant Mem0ᵍ extracts typed entity nodes + directed relation triplets (v_s, r, v_d), embeds entities, searches for similar nodes above a threshold, and runs LLM conflict-resolution — stored in Neo4j.
- **The novelty-gate insight (SAGE, 2025).** Mem0's weakness is that it makes one routing LLM call per candidate at every write step regardless of novelty — write cost scales with turns, not new information. A novelty gate (embedding distance to existing memory) that only invokes the LLM router on genuinely new content is the efficiency fix — and maps precisely onto AgentX's proposed surprise-gating (§3.3).
- **MemGPT/Letta core-memory decisions.** Self-editing memory: the agent itself calls `core_memory_append` / `core_memory_replace` / `archival_memory_insert` during its loop, with size-limited in-context "human" and "persona" blocks. Trade-off (per Letta/Vectorize): self-editing is adaptive but memory quality depends entirely on model judgment and every op costs inference tokens; passive extraction (Mem0) is consistent and token-efficient but less nuanced. A documented failure mode: the agent learns that calling the memory tool "makes the loop feel productive" and does it instead of answering.
- **LangMem memory types.** Three types — semantic (facts), episodic (past interactions, often as few-shot examples), procedural (behavior/prompt rules the agent rewrites). Two mechanisms: hot-path tools vs background ("subconscious") extraction that reflects after the fact for higher recall. Fact extraction prompt is deliberately simple: "Extract all personal facts about the user… Output one fact per line using the format 'FACT: <content>'… Only output FACT lines."
- **Durable vs ephemeral.** The consensus filter: extract stable preferences, identity facts, decisions, corrections, commitments, and skills; drop pleasantries, transient state, and chain-of-thought. AgentX already encodes "CoT/thinking is process, not result — never persisted" as an invariant.
- **Procedural/skill extraction.** LangMem's procedural memory and AgentX's own `distill_procedures` (nightly "sleep"-time distillation, strengthening a cosine-similar existing procedure instead of duplicating) are the reference designs; A-Mem's memory-evolution is the aggressive end (rewriting old notes).

### C. FORMAT consistency — schemas, canonicalization, temporal stamping

- **Constrained decoding.** Outlines compiles JSON Schema into an index for O(1) valid-token lookup; llguidance enforces arbitrary CFG at ~50µs/token; XGrammar is SOTA (<40µs/token, up to 100x faster than prior grammar engines) and the default across vLLM/SGLang/TensorRT-LLM/MLC-LLM. Practical caveats from JSONSchemaBench and 2026 practitioner guides: keep schemas shallow (4+ nesting levels raise errors), enums under ~50 values, put reasoning fields *before* answer fields (chain-of-thought models commit early otherwise), and note that constraints can still produce semantically wrong-but-valid JSON — so semantic validation and retries remain necessary. There is a documented "constraint tax": tool-calling can be suppressed under structured-output constraints on open-weight models.
- **Canonical KG extraction (GraphRAG).** The `GRAPH_EXTRACTION_PROMPT` extracts entities as `("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<description>)` and relationships with a `relationship_strength` integer 1–10. Four prompt sections: instructions, few-shot examples, real-data placeholder, and **gleanings** (multi-turn "did you miss any?" passes that raise recall without hurting precision). Auto-tuning generates domain-specific few-shot examples automatically.
- **Zep/Graphiti entity+edge extraction with bitemporal validity.** Each `add_episode` routes text (plus the last *n* messages for context) through an LLM to extract entities, then edges (facts) as node-edge-node triplets with the fact stored as an edge property; the same fact can be extracted between multiple entity pairs (hyper-edges). Entity resolution and edge dedup are constrained to edges between the same entity pair (shrinks search space, prevents wrong merges). Crucially, **every edge carries two time axes**: event time T (when true in the world) and ingestion time T′ (when learned), plus explicit validity intervals `(t_valid, t_invalid)`. A dedicated date-extraction prompt runs on both new and existing edges each episode; conflicts invalidate (not delete) stale edges. This is the direct blueprint for AgentX's planned §1.7 bitemporal migration and existing SUPERSEDES mechanism.
- **HippoRAG OpenIE.** Offline indexing uses an instruction-tuned LLM for schemaless OpenIE (NER then relation triples), building an open KG over the corpus; retrieval uses Personalized PageRank seeded on query entities — which is exactly AgentX's planned §3.6 PPR over ABOUT/RELATES_TO/SUPERSEDES.
- **Atomic propositions (Dense X Retrieval, Chen et al., EMNLP 2024, arXiv:2312.06648).** A "Propositionizer" (GPT-4 seed → fine-tuned Flan-T5-large) decomposes text into propositions that are (a) atomic/minimal, (b) self-contained (resolve coreference), (c) a distinct factoid. Result: 100–200 words fits ~10 propositions vs ~5 sentences vs ~2 passages. With unsupervised retrievers (DPR, ANCE) proposition units give a 17–25% relative Recall@5 improvement on EntityQuestions, with the biggest gains on rare entities; even with supervised retrievers the paper reports average Recall@5 improvements of 4.5%/3.2%/2.4%/4.2% for DPR/ANCE/TAS-B/GTR and downstream QA EM@100 gains of +5.9/+7.8/+5.8/+4.9/+5.9/+6.9 across six retrievers. Propositional chunking pairs especially well with a Postgres + pgvector hybrid (keyword + semantic) — precisely AgentX's stack.
- **Contradiction/invalidation at extraction time.** Mem0's DELETE on contradiction; Graphiti's temporal invalidation; AgentX's `check_contradictions`/`check_correction` + SUPERSEDES. Best practice (and AgentX's §3.10 plan): never last-writer-wins on shared channels; auto-supersede only with same resolved subject + compatible provenance, else flag for review.

### D. PIPELINE design and tuning

- **Single-pass vs multi-pass.** GraphRAG's gleaning and Generative Agents' reflection show multi-pass raises recall; the cost is more LLM calls. A-Mem issues two LLM calls per memory event (note construction; combined link-generation + evolution). Newer Mem0 shipped a "single-pass ADD-only" mode (one LLM call, no UPDATE/DELETE — memories accumulate) for latency.
- **Granularity/trigger.** Per-turn (Letta hot path, expensive), per-session/batched (Mem0 rolling summary + async), or idle/consolidation-batched (AgentX). AgentX's "durable writes belong to consolidation" + 15-min sweep + proposed idle trigger is a sound, cost-aware design — batching is where extraction *should* live.
- **Small fine-tuned vs frontier models.** The Dense X Propositionizer (Flan-T5-large distilled from GPT-4) and HippoRAG's use of Llama-3.3-70B / GPT-3.5 for OpenIE show distillation works for extraction. Trade-off: a 7B local model won't match GPT-4o/Claude on complex extraction, but constrained decoding closes the *format* gap (not the *judgment* gap). AgentX's pragmatic path: keep local models but add constrained decoding now; route hard/high-surprise turns to Sonnet via the existing fallback chain.
- **Few-shot exemplar selection and prompt tuning.** GraphRAG auto-tuning; LangMem prompt optimization; AgentX's own "Improved Extraction Prompts — few-shot examples, better schema definitions, domain-specific tuning" backlog item. Key: version the prompt so calibration/thresholds can be keyed to it (AgentX's §2.1 `pipeline_version` should be the shared invalidation key for both confidence calibration and conformal dedup bands).
- **Verification passes.** NLI entailment per claim (P_entail = P(entailment | premise=source_turn, hypothesis=claim); DeBERTa-v3-large/MNLI is the common backbone); AlignScore/INFUSE for long sources; self-consistency sampling. Cheap, and the missing layer in AgentX.
- **Dedup/merge at write time.** Mem0 vector-similarity + LLM router; Graphiti entity-pair-constrained edge dedup; AgentX 0.92 gate → proposed conformal bands.

### E. EVALUATION and tuning methodology

- **Metrics.** Fact-level precision/recall vs an annotated gold set; but the true metric is downstream QA/retrieval accuracy. Build recall@k / MRR per technique and fused (AgentX's planned `eval_recall`).
- **Benchmarks.** LoCoMo — 10 multi-session human-human dialogues, ~600 turns / ~16K tokens each, question types single-hop/multi-hop/temporal/open-domain/adversarial(abstention). LongMemEval (Wu et al., ICLR 2025, arXiv:2410.10813) — 500 curated questions; LongMemEval-S has chat histories of roughly 115,000 tokens spread over about 40 sessions (LongMemEval-M extends to ~500 sessions / ~1.5M tokens), covering five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. The paper reports that long-context LLMs show a 30–60% performance drop on LongMemEval-S and that state-of-the-art commercial systems such as GPT-4o only achieve 30–70% accuracy in a simpler setting. MSC (Multi-Session Chat), PerLTQA, MemoryAgentBench round out the set. These are the right corpora to drive AgentX's extraction tuning.
- **LLM-as-judge pitfalls.** Position bias (prefers first/slot-A — mitigate by order-swapping), verbosity bias (prefers longer — length control/regression), self-enhancement/self-preference bias (favors same-family outputs — hide identity, use a different judge family), sycophancy. Mitigations: per-criterion atomic judging, ensembles/majority vote, and reliability metrics (ICC, Krippendorff's α, Kendall's τ).
- **Ablation & drift.** Version prompts; A/B thresholds on a frozen corpus; monitor extraction-consistency drift over time via the persisted eval runs keyed by `pipeline_version`.

### F. Mapping techniques to AgentX's architecture

- **Neo4j (graph side):** Graphiti-style entity+edge triple extraction with entity-pair-constrained dedup; GraphRAG gleaning passes to raise entity recall; bitemporal `(t_valid, t_invalid)` on edges (AgentX §1.7); HippoRAG PPR at recall (AgentX §3.6); `dedupe_entities` canonicalization with alias tracking (already present). GraphRAG community summaries (Leiden) map to AgentX's §3.7 CommunitySummary nodes.
- **pgvector (vector side):** Dense X atomic propositions as the embedding unit — extract facts, then propositionalize before embedding; store the atomic proposition text-only with derived embeddings (matches AgentX invariant "embeddings are always derived, never canonical"). This is the single biggest recall win available on the existing stack.
- **Consolidation pipeline:** batched extraction (already the design); Mem0 ADD/UPDATE/DELETE/NOOP as the merge-op vocabulary; NLI verification pass before commit; conformal dedup bands (§3.12); generative replay for contested facts (§3.9). Keep the 15-min sweep + add idle trigger.
- **RecallLayer:** importance/salience as a scored signal (Generative Agents), score-decomposition logging (§2.2), PPR as a technique in the contextual bandit (§3.5).
- **Context Ledger:** the salient core already reads `ORDER BY salience`; a proper write-time importance score makes that core meaningfully ranked, and atomic propositions make the `shrink_memory_to_facts` degradation cleaner.

## Recommendations

**Stage 1 — Format consistency (do first; directly fixes pain point B).**
1. Define one canonical schema: `Fact { subject, predicate, object, proposition_text, confidence, temporal_context, valid_from?, valid_to?, source_turn_id, entities[] }` plus `Entity { name, type, aliases[] }` and `Edge { src, relation, dst, valid_at?, invalid_at? }`.
2. Enforce it with grammar-constrained decoding (XGrammar via vLLM/SGLang if self-hosting the local model; or function-calling with `strict: true`). Keep schemas shallow, enums small, reasoning-before-answer.
3. Add a Pydantic semantic-validation + one-retry wrapper (valid JSON ≠ correct JSON).
4. **Benchmark to change the decision:** if constrained local-model extraction still yields <90% schema+semantic validity or misses entities, route to Sonnet (existing fallback chain) for hard/high-surprise turns only.

**Stage 2 — Usefulness (pain point A).**
5. Add an explicit write-time salience score (Generative-Agents 1–10 prompt) as a field; feed it into the salient core ranking and decay half-life.
6. Adopt Mem0's extract-then-compare ADD/UPDATE/DELETE/NOOP as the consolidation merge vocabulary, gated by novelty (only invoke the LLM router when embedding distance to existing memory exceeds a band — reuse the §3.3 surprise score).
7. Propositionalize facts (Dense X) before embedding into pgvector.

**Stage 3 — Trust & calibration.**
8. Add an NLI entailment verification pass (DeBERTa-v3/MNLI) per candidate fact against its source turn; drop/downgrade non-entailed claims.
9. Replace the static confidence ladder and 0.92 gate with fitted calibration (isotonic) and conformal dedup bands, keyed to `pipeline_version`.

**Stage 4 — Measurement (do in parallel; unblocks everything).**
10. Ship `eval_recall` with a seeded corpus + (query → expected-fact-ids) golden set, and wire in LoCoMo + LongMemEval-S. Score recall@k/MRR per technique and downstream QA accuracy. Use a non-self judge with order-swapping + per-claim atomic judging. This is the guardrail that turns every threshold/prompt change from a guess into a measured decision.

**Thresholds that should change the plan:** if `eval_recall` shows atomic-proposition embedding lifts recall@k by a meaningful margin, make it default; if constrained local extraction reaches parity with Sonnet on schema+semantic validity and entity-resolution F1, don't pay for Sonnet; if NLI verification drops <2% of facts, downgrade it to a sampled audit rather than an every-fact gate.

## Follow-up questions
- What exact fields and prompt does the current `ExtractionService` emit today (needed to size the schema migration)? The file could not be read remotely.
- Is the local extraction model served via vLLM/SGLang/Ollama? That determines whether XGrammar/Outlines can be dropped in without a serving change.
- Do you already have any labeled gold facts, or would the golden set need to be built from scratch (LLM-bootstrapped then human-audited)?
- Should salience be user-configurable per channel (e.g., a project channel weights procedural/skill facts higher than a personal channel)?

## Caveats
- **Primary-source code was not directly readable.** `extraction/service.py` and `consolidation/jobs.py` could not be fetched (GitHub raw/tree and agentx.thejpnet.net/docs are robots-blocked). Current-state claims about AgentX derive from `Memory-Roadmap.md`, `Decisions.md`, and the backlog files, which describe the code via a file-map and prose rather than the literal prompt/schema text. Confirm the exact extraction prompt and JSON schema locally.
- **Some arXiv IDs in search results are from 2026 preprints** (SAGE, ElasticMem, several memory surveys) that corroborate mechanisms but are recent and not all peer-reviewed; treat their specific numbers as indicative.
- **Constrained decoding guarantees syntax, not truth** — it fixes format consistency but not hallucination; that's why Stage 3 (NLI verification) is separate and necessary.
- **Benchmark scores are not directly comparable across papers** (different LoCoMo question subsets, adversarial included/excluded); use them for relative ablation on your own corpus, not absolute claims.