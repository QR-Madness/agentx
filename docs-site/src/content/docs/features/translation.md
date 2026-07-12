# Translation

AgentX includes a built-in **translation** kit: detect a language and translate across **200+
languages**, running **locally** on your own machine — no external API, no per-word cost, and it
works offline once the models are downloaded.

## Using it

Open the **Translation** tool from the app, paste or type your text, and AgentX detects the
source language automatically and translates to the target you pick. It's a standalone utility —
useful on its own, and the same kit backs any translation an agent needs.

## How it works — two levels

Translation runs in two stages, trading a fast first pass for broad final coverage:

- **Level I — detection.** A small, fast model identifies the source language across ~20 common
  languages and returns an ISO 639-1 code (`fr`, `de`, `ja`, …) with a confidence score.
- **Level II — translation.** The detected language is bridged to an **NLLB-200** code, and the
  full model translates across 200+ languages.

A **LanguageLexicon** bridges the two — converting a Level I code like `fr` into the Level II code
NLLB expects (`fra_Latn`). NLLB codes pair an ISO 639-3 language with its script,
`{language}_{script}` — a handful for orientation:

| Language | NLLB-200 code |
|----------|---------------|
| English | `eng_Latn` |
| Chinese (Simplified) | `zho_Hans` |
| Japanese | `jpn_Jpan` |
| Arabic | `arb_Arab` |
| Hindi | `hin_Deva` |
| Russian | `rus_Cyrl` |

See the [translation pipeline](../architecture/system-design.md#translation-pipeline) on the
System Design page.

## Models

Two models power it, both pulled from HuggingFace on first use:

| Purpose | Model | Size |
|---------|-------|------|
| Language detection | `eleldar/language-detection` | ~50 MB |
| Translation | `facebook/nllb-200-distilled-600M` | ~600 MB |

They load **lazily** — nothing downloads until the first translation request, so server startup
stays fast — and you can pre-fetch them with `task models:download`. The programmatic surface
(detect, translate) is in the [API Reference](../api/endpoints.md#translation).

## Related

- [Configuration](../getting-started/configuration.md) — model download and cache location
- [Getting Started](../getting-started/installation.md) — first-run model setup
