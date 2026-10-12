# Archived Research — Torch-free CPU Inference (footprint)

> **Provenance:** external deep-research run, archived 2026-10-11. Its agents could **not** read
> this codebase — current-state claims were inferred from the prompt. Integrated into
> [phase-19 §19.2.6](../phases/phase-19-cloud-operation.md) (the footprint track) and
> [translation-web-search.md](../backlog/translation-web-search.md) (the source-language bug); those
> docs own the work items — this file is the citation record.
>
> **Verification corrections (code-checked at v0.21.272 — trust these over the body where they conflict):**
> - **Torch-free is achievable.** Only `kit/device.py` and `kit/translation.py` import torch
>   directly; `sentence-transformers` pulls it in for the embedder and reranker. The other models in
>   the local HF cache (`chatterbox`, `rtdetr`, `nomic-embed`, `m2m100`) are **not used by the API** —
>   speech goes through OpenRouter (`openai/whisper-1` + TTS).
> - **bge-m3 is dense-only** (`sentence-transformers`; no FlagEmbedding, no sparse or ColBERT heads),
>   so §5's caveat about EmbeddingGemma losing those outputs does not apply.
> - **Translation is a user-facing feature** — Translation panel, palette command, Settings →
>   Translation, the status strip. §5/§6's "remove NLLB if it only serves retrieval" does not apply;
>   removal would be a product decision, not an optimization.
> - **The 20-language detector never reaches translation — the real defect is worse.**
>   `translate_text` never sets NLLB's `src_lang` (the assignment is commented out), so the
>   tokenizer default `eng_Latn` encodes *every* input as English. Neither the UI (no source-language
>   picker) nor `/translate` (text + target only) supplies one. Filed in translation-web-search.md.
>   Moving NLLB to CTranslate2 (§2) forces explicit source handling anyway — fix it there.
> - **The image case is the CUDA one, and worse than the body's example.** `torch` resolves from the
>   default PyPI index — the CUDA build, with 15 `nvidia-*` packages in `uv.lock`.
>   `qrmadness/agentx-api:latest` reports **13.1 GB** in Docker Desktop (which counts compressed blobs
>   plus the unpacked snapshot — ~8.5 GB unpacked); its `uv sync` layer alone is **7.97 GB**. Installed,
>   `nvidia/` 4.3 GB + `torch/` 1.7 GB + `triton/` 0.59 GB = **6.6 GB of a 7.2 GB site-packages
>   (92%)**. Models are not baked in. **A CPU-only wheel recovers most of this with no code change** —
>   but `docker-compose.gpu.yml` runs the same image with an NVIDIA reservation, so it needs a GPU
>   variant tag.
> - **Cloud↔local embedding parity.** AgX deliberately locks local and cloud (OpenRouter
>   `baai/bge-m3`) to one vector space so a cluster can switch without re-embedding (phase-19, Ingest
>   flow). int8 local weakens that to ~0.999; **EmbeddingGemma breaks it outright** — the cloud path
>   would need an EmbeddingGemma provider too. The run could not have known the property exists.
> - **Licensing is not moot.** §5 dismisses the Gemma terms because "AgentX is personal and
>   non-production" — it is not any more: AgX is a launch candidate for sale under a licence
>   (Notion Project Pool). Shipping or auto-downloading EmbeddingGemma in a commercial product means
>   carrying the Gemma Terms of Use and prohibited-use policy through to users. fastText lid.176 is
>   CC BY-SA 3.0 — attribution required when redistributing the model file. Neither blocks; both
>   need a decision before slice 1 and slice 5.
> - The baseline RSS figures quoted in the body (1.44 / 0.04 / 2.43 / 0.60 / 0.68 GB) are AgX's own
>   measurements of 2026-10-11 — see phase-19's measured resource budget.

---

## Summary
- **Every model can run without PyTorch.** Use ONNX Runtime for the embedder, reranker and classifier, and CTranslate2 for NLLB. All of these runtimes had releases in the last two months.
- **bge-m3 at int8 is close to lossless.** Reported fp32-to-int8 dense cosine similarity is 0.999 (bge-m3-lite), and GGUF Q8_0 is 0.9995 against f16. Q4_K_M drops to 0.980.
- **Expected memory use falls from about 5.2 GB to roughly 1.0–1.9 GB.** The range depends on whether NLLB stays.
- **Two findings matter more than quantization:**
  1. Your language detector covers only 20 languages, against NLLB's 200.
  2. EmbeddingGemma-300M scores higher on multilingual retrieval than bge-m3 at about half the size, according to Google's own paper.

## 1. Runtimes: torch-free support and maintenance

| Runtime | What it covers here | Latest release (evidence) | Status |
|---|---|---|---|
| ONNX Runtime (Microsoft) | All encoder models (bge-m3, MiniLM, XLM-R classifier) | 1.31.0, 2026-10-09; monthly releases through 2026 | Active |
| fastembed (Qdrant) | ONNX wrapper with no torch; `TextCrossEncoder` uses `Xenova/ms-marco-MiniLM-L-6-v2` | 0.9.0, 2026-10-07; Python ≥3.10 | Active. I could not confirm bge-m3 is in its catalogue (the supported-models page did not load) |
| CTranslate2 (OpenNMT) | NLLB translation | 4.8.2, 2026-08-31; seven releases in 2026 | Active (this corrects my earlier note about a slowing cadence) |
| Optimum / optimum-onnx (Hugging Face) | One-off export to ONNX on a build machine | optimum-onnx 0.1.0, 2025-12-23; requires `transformers` | Young split-off package. Use it only at build time, never in the runtime image |
| llama.cpp / GGUF | bge-m3 dense embeddings in `--embedding` mode | Very active (no specific release checked) | Dense vectors only; the sparse and ColBERT heads are not supported |
| bge-m3-lite (PyPI) | bge-m3 dense, sparse and ColBERT on onnxruntime alone | 0.6.2, 2026-09-07; first release 2026-09-05 | Promising but only weeks old. Treat it as a reference implementation, not a dependency |

**Tokenizers:** none of these paths need `transformers` at runtime. ONNX paths use the `tokenizers` library with `tokenizer.json`, CTranslate2 uses `sentencepiece`, and bge-m3-lite bundles its own tokenizer. `sentence-transformers` always imports torch, so it has to go.

## 2. Per-model comparison

| Model | Torch-free path | fp32 | int8 | int4 | Quality evidence |
|---|---|---|---|---|---|
| **bge-m3** (568M) | ONNX Runtime int8; GGUF via llama.cpp; OpenVINO int8 | ~2.2 GB on disk; your RSS 1.44–2.3 GB | ~0.57–0.63 GB on disk if the embedding table is quantized (OpenVINO int8 570 MB; GGUF Q8_0 635 MB); **1.30 GB** if only Linear layers are quantized | GGUF Q4_K_M 438 MB | Q8_0 vs f16: mean cosine 0.99948, min 0.99928. Q4_K_M: mean 0.97964, min 0.96919. bge-m3-lite int8: dense cosine 0.999. OpenVINO int8: Pearson 0.9941 vs fp32 on a grounding score. **No published MIRACL/MMTEB delta for int8 bge-m3 was found** |
| **ms-marco-MiniLM-L-6-v2** (22M) | fastembed `TextCrossEncoder` or raw ONNX | 0.04 GB (your measurement) | ~0.02 GB (estimated) | Not worthwhile | No published int8 delta found. It is already tiny; the gain is dropping torch, not saving RAM |
| **nllb-200-distilled-600M** | CTranslate2 `int8` or `int8_float32` | your RSS 2.43 GB | ~0.6 GB of weights (arithmetic); RSS not published | Not supported in CTranslate2 | int8 CT2 on TICO-19: EN→ES spBLEU 53.31 / chrF++ 72.19; EN→FR 43.25 / 64.17; EN→ZH 24.9 / 33.87. **No same-model fp32-vs-int8 comparison was found.** A CTranslate2 maintainer says compute type had "little effect" on one float32 model |
| **eleldar/language-detection** (XLM-R base, ~278M) | ONNX Runtime | your RSS 0.60 GB | ~0.28 GB (estimated) | — | 99.6% test accuracy, **but only 20 languages** |
| → replacement: fastText lid.176 | `fasttext` library, no torch | 126 MB `.bin` / 917 kB `.ftz` | — | — | 176 languages; CC BY-SA 3.0 |
| → replacement: GlotLID v2 | `fasttext` library | size not stated | — | — | 1,626 well-supported languages at F1 0.996 on its own test set |

**Why "int8" can mean 1.3 GB or 0.57 GB for bge-m3:** about 45% of bge-m3's parameters sit in its 250k-token vocabulary embedding table (a lookup matrix that maps each token to a vector). Quantizing only the Linear (fully connected) layers left a 1,299 MB model, about 40% smaller. Exports that also quantize the embedding table reach about 570 MB. When you export with ONNX Runtime, include the `Gather` operator (the lookup op) in quantization, or start from an export that already does.

**Two different meanings of "int8":** weight quantization (above) changes the model. Vector quantization stores the *output* vectors as int8 or binary values in the index. They can be adopted independently.

## 3. Vector compatibility (fp32 → int8 bge-m3)
- **Closeness:** a reported 0.999 dense cosine between fp32 and the int8 backbone (bge-m3-lite), and 0.9995 for Q8_0 against f16. At Q4_K_M the minimum cosine is 0.969, which is enough to reorder close results.
- **Can old vectors stay?** At int8, mixing old fp32 vectors with new int8 query vectors will very likely work, because 0.999 cosine is well inside typical score gaps between results. However, **I found no published recall test of a mixed index**, so this is an inference, not a measured result.
- **Recommendation:** re-embed. It runs locally, so privacy is unaffected and the only cost is CPU time. Use a shadow index (a second index built in parallel, queried alongside the live one). Compare recall@10 and top-k overlap against the fp32 index on real queries, then switch.
- **At int4, re-embedding is required.** Never mix int4 vectors with fp32 ones.
- **If you change embedder** (for example to EmbeddingGemma), re-embedding is mandatory, since the vector spaces are unrelated.

## 4. Docker image size
- A published measurement put a CUDA torch image at **7.05 GB**. The ONNX Runtime equivalent on `python:3.9-slim` was **699 MB**. Inside the torch image, NVIDIA libraries took 2,961 MB, torch 1,644 MB and triton 439 MB, about 5.0 GB in total.
- The default `pip install torch` on Linux grew a virtualenv from about 0.6 GB to 5.7 GB (PyTorch forum, November 2024).
- I found no verified installed-size figure for the CPU-only wheel. Expect it to sit between the two cases above, still several hundred MB.
- **Expected saving for AgentX:** about 5 GB if the image currently pulls default CUDA wheels, or about 1–1.5 GB if it already uses CPU wheels (the second figure is an estimate).

## 5. Alternatives

### Smaller multilingual embedders (MTEB Multilingual v2, from the EmbeddingGemma paper)

| Model | Params | Mean (task) | Retrieval |
|---|---|---|---|
| Qwen3-Embedding-0.6B | ~600M | 64.34 | 64.65 |
| **EmbeddingGemma-300M** | 308M | 61.15 | **62.49** |
| bge-m3 (current) | 568M | 59.56 | 54.60 |
| gte-multilingual-base | ~305M | 58.24 | 56.50 |
| snowflake-arctic-embed-m-v2.0 | ~305M | 53.7 | 54.8 |
| multilingual-e5-base | 278M | 57.0 | 52.7 |
| multilingual-e5-small | 118M | 55.8 | 49.3 |
| granite-embedding-278m-multilingual | 278M | 53.7 | 52.2 |

- EmbeddingGemma keeps quality under quantization: 61.15 at bf16, 60.93 at int8 and 60.62 at int4 (mean task score).
- **Caveat:** these figures come from Google's own paper. Confirm them on your own memory data.
- EmbeddingGemma uses the Gemma license. Since AgentX is personal and non-production, this is unlikely to matter, but read the terms before any redistribution.
- You would lose bge-m3's sparse and ColBERT outputs. That only matters if you use them; you currently use dense only.

### Replacing NLLB with the user's own LLM API key
- **Quality:** in WMT25 (preliminary automatic ranking), frontier LLMs such as Gemini-2.5-Pro and GPT-4.1 led, Claude-4 sat in a second tier, and they beat commercial MT engines on high-resource pairs. The older evidence (Robinson et al., WMT 2023) is mixed: NLLB beat ChatGPT on 84% of 201 FLORES-200 languages, and GPT-4 beat NLLB on only 5 of 20 languages. NLLB is still the safer choice for low-resource languages.
- **Cost (2023 prices, now outdated):** translating the full FLORES-200 devtest set (1,012 sentences) cost $0.09 with NLLB, $0.35 with ChatGPT and $25.93 with GPT-4 5-shot. Current small models are far cheaper than GPT-4 was, but I did not verify current per-token prices.
- **Latency:** I found no CPU benchmark comparing the two. An API round-trip typically costs hundreds of milliseconds to seconds per call, while CTranslate2 int8 on CPU depends on your core count. Measure both on your hardware.
- **Privacy:** this sends memory text off the machine, which breaks the product promise. It can only be an explicit, per-user opt-in.
- **Better option:** if NLLB exists only so that cross-language memories match each other, drop it. bge-m3 and EmbeddingGemma are both multilingual, so cross-language retrieval works without translating first.

## 6. Recommended torch-free stack

| Role | Choice | Estimated RSS |
|---|---|---|
| Embedder | bge-m3 ONNX int8 with the embedding table quantized; or EmbeddingGemma-300M int8 after an A/B test on your data | ~0.7–0.9 GB (bge-m3) / ~0.4 GB (Gemma), estimated |
| Reranker | fastembed `TextCrossEncoder` (MiniLM-L-6-v2) | ~0.05 GB |
| Language ID | fastText lid.176 `.bin` (or GlotLID if you need more languages) | ~0.15 GB |
| Translation | Remove if it only serves retrieval; otherwise CTranslate2 NLLB `int8_float32` | 0 or ~0.8 GB, estimated |
| Runtime | onnxruntime + tokenizers + fasttext (+ ctranslate2, sentencepiece) | ~0.1–0.2 GB, estimated |
| **Total** | | **~1.0–1.9 GB** vs ~5.2 GB today (estimate; measure after migrating) |

**Migration order:**
1. Replace the language detector (no index impact).
2. Move the reranker (no index impact).
3. Move NLLB to CTranslate2, or remove it.
4. Move bge-m3 to ONNX int8 behind a shadow index, then re-embed.
5. Optionally A/B test EmbeddingGemma.

## 7. Risks
- **Unmeasured retrieval loss:** no MIRACL or MMTEB number exists for int8 bge-m3. Cosine 0.999 is strong evidence, but it is not recall. Run your own recall@10 check.
- **Export correctness:** community ONNX exports differ in pooling (CLS vs. mean), normalization and output heads. Compare outputs against FlagEmbedding or sentence-transformers fp32 on a fixed test set before switching.
- **Tokenizer parity:** the `tokenizers` output must exactly match the original for truncation at 8,192 tokens and for special tokens. Diff the token IDs.
- **CPU instruction sets:** int8 speedups need AVX2, AVX-512 VNNI or ARM dot-product instructions. On older CPUs or some Fly.io machine types, expect the memory savings without a speedup.
- **Immature packages:** bge-m3-lite (weeks old) and optimum-onnx (0.x) could change. Pin versions or vendor the code.
- **Language-detector change:** lid.176 and GlotLID return different label sets than the current model. Any downstream code that maps languages to NLLB codes must be updated.
- **Licensing:** lid.176 is CC BY-SA 3.0, which requires attribution. EmbeddingGemma uses the Gemma terms.

## Sources
- [bge-m3-lite on PyPI](https://pypi.org/project/bge-m3-lite/)
- [smarttasks/bge-m3-GGUF](https://huggingface.co/smarttasks/bge-m3-GGUF)
- [stellars/bge-m3-openvino-int8](https://huggingface.co/stellars/bge-m3-openvino-int8)
- [VladimirRH/bge-m3-quantized-int8](https://huggingface.co/VladimirRH/bge-m3-quantized-int8)
- [EmbeddingGemma paper (arXiv 2509.20354)](https://arxiv.org/pdf/2509.20354)
- [onnxruntime release history (piwheels)](https://www.piwheels.org/project/onnxruntime/)
- [ctranslate2 release history](https://simple-repository.app.cern.ch/project/ctranslate2)
- [fastembed on PyPI](https://pypi.org/project/fastembed) · [fastembed on GitHub](https://github.com/qdrant/fastembed)
- [optimum-onnx release history](https://simple-repository.app.cern.ch/project/optimum-onnx)
- [OpenNMT forum: NLLB-200 with CTranslate2](https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090)
- [OpenNMT forum: CTranslate2 quantization types](https://forum.opennmt.net/t/worse-performance-with-different-ctranslate2-quantization-types/5355)
- [eleldar/language-detection](https://huggingface.co/eleldar/language-detection)
- [fastText language identification](https://fasttext.cc/docs/en/language-identification)
- [GlotLID](https://github.com/AI-Natural-Language-Processing-Lab/GlotLID-language-identification-model)
- [Robinson et al., "ChatGPT MT: Competitive for High- (but not Low-) Resource Languages"](https://ar5iv.labs.arxiv.org/html/2309.07423)
- [Slator: WMT25 preliminary results](https://slator.com/wmt25-preliminary-results-gemini-2-5-pro-gpt-4-1-lead-ai-translation/)
- [Towards Data Science: reducing Docker image size for model serving](https://towardsdatascience.com/reducing-the-size-of-docker-images-serving-llm-models-b70ee66e5a76/)
- [PyTorch forum: install size](https://discuss.pytorch.org/t/pytorch-install-is-that-large-5-gbs/213629)
