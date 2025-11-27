# Translation System Architecture

## Overview

AgentX implements a sophisticated two-level translation system supporting 204 languages through the NLLB-200 (No Language Left Behind) model. The system is designed for flexibility, allowing both fast language detection and comprehensive multilingual translation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client (Tauri/React)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         TranslationTab Component                       │ │
│  │  - 204 NLLB-200 languages with search                 │ │
│  │  - Language filtering and selection                   │ │
│  │  - Text input/output panels                           │ │
│  └────────────────────────┬───────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────┘
                              │ HTTP POST /api/translate
                              │ { text, targetLanguage }
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Django API Backend                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              TranslationKit Class                      │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  Level I: Language Detection (~20 languages)    │ │ │
│  │  │  Model: eleldar/language-detection              │ │ │
│  │  │  Output: ISO 639-1 codes (en, fr, es, etc.)     │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  Level II: Translation (204 languages)          │ │ │
│  │  │  Model: facebook/m2m100_418M (temporary)        │ │ │
│  │  │  Target: facebook/nllb-200-distilled-600M       │ │ │
│  │  │  Output: Translated text                        │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  LanguageLexicon Class                          │ │ │
│  │  │  - Converts between ISO 639 formats             │ │ │
│  │  │  - Maps Level I ↔ Level II codes                │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Two-Level System

### Level I: Language Detection

**Purpose**: Fast, accurate language detection for common languages

**Model**: `eleldar/language-detection`

**Capabilities**:
- Detects ~20 common languages
- Returns ISO 639-1 codes (2-letter: `en`, `fr`, `es`, etc.)
- Provides confidence scores
- Fast inference time

**Supported Languages**:
```python
{
    "arabic": "ar", "bulgarian": "bg", "german": "de",
    "modern greek": "el", "english": "en", "spanish": "es",
    "french": "fr", "hindi": "hi", "italian": "it",
    "japanese": "ja", "dutch": "nl", "polish": "pl",
    "portuguese": "pt", "russian": "ru", "swahili": "sw",
    "thai": "th", "turkish": "tr", "urdu": "ur",
    "vietnamese": "vi", "chinese": "zh"
}
```

**Usage**:
```python
# api/agentx_ai/kit/translation.py
detected_language, confidence = translation_kit.detect_language_level_i(text)
# Returns: ("english", 98.5)
```

### Level II: Translation

**Purpose**: Comprehensive translation supporting 204 languages

**Current Model**: `facebook/m2m100_418M` (temporary - see Known Issues)

**Target Model**: `facebook/nllb-200-distilled-600M` (configured but not active)

**Language Code Format**: ISO 639-3 with script tag
- Format: `{language}_{Script}`
- Examples: `eng_Latn`, `fra_Latn`, `zho_Hans`, `arb_Arab`

**Capabilities**:
- 204 languages with diverse scripts
- Supports multiple variants of same language (e.g., `zho_Hans` vs `zho_Hant`)
- Direct translation without source language detection

**Usage**:
```python
# api/agentx_ai/kit/translation.py
translated = translation_kit.translate_text(
    text="Hello, world!",
    target_language="fra_Latn",  # French in Latin script
    target_language_level=2       # Use Level II format
)
# Returns: "Bonjour le monde!"
```

## Language Code Conversion

The `LanguageLexicon` class bridges Level I and Level II code formats using the `python-iso639` library.

**Conversion Flow**:
```
Level I (ISO 639-1)  →  ISO 639-3 Terminological  →  Level II (NLLB-200)
       "en"          →         "eng"               →    "eng_Latn"
       "fr"          →         "fra"               →    "fra_Latn"
       "zh"          →         "zho"               →    "zho_Hans" or "zho_Hant"
```

**Implementation**:
```python
# api/agentx_ai/kit/translation.py:60-71
def convert_level_i_detection_to_level_ii(self, level_i_detected_language_code: str):
    language = None
    try:
        language = iso639.Language.from_part2t(level_i_detected_language_code.lower())
    except:
        try:
            language = iso639.Language.from_part1(level_i_detected_language_code.lower())
        except:
            raise ValueError(f"Language {level_i_detected_language_code} not supported")

    terminological_language_code = language.part2t
    return self.nlb200_list_mapped_to_iso639_terminological[terminological_language_code]
```

## Complete Language List (204 Languages)

The full list is defined in:
- **Backend**: `api/agentx_ai/kit/translation.py:6-31` (`nlb200_list`)
- **Frontend**: `client/src/data/nllb200Languages.ts` (with human-readable names)

### Language Coverage by Script:

| Script | Count | Examples |
|--------|-------|----------|
| Latin (Latn) | 110+ | English, Spanish, French, German, Vietnamese |
| Arabic (Arab) | 20+ | Modern Standard Arabic, Egyptian Arabic, Persian |
| Cyrillic (Cyrl) | 12+ | Russian, Ukrainian, Bulgarian, Serbian |
| Devanagari (Deva) | 10+ | Hindi, Nepali, Marathi, Sanskrit |
| Han (Hans/Hant) | 2 | Simplified Chinese, Traditional Chinese |
| Bengali (Beng) | 3 | Bengali, Assamese, Santali |
| Other Scripts | 40+ | Greek, Hebrew, Thai, Georgian, Armenian, etc. |

### Notable Language Variants:

- **Chinese**: `zho_Hans` (Simplified), `zho_Hant` (Traditional), `yue_Hant` (Cantonese)
- **Arabic**: Multiple dialects (Egyptian, Moroccan, Iraqi, Levantine, Gulf, etc.)
- **Norwegian**: `nob_Latn` (Bokmål), `nno_Latn` (Nynorsk)
- **Acehnese**: `ace_Arab`, `ace_Latn` (same language, different scripts)

## API Integration

### Frontend → Backend Flow

1. **User Input**: User enters text and selects target language from searchable dropdown
2. **API Request**: Client sends POST to `/api/translate`
3. **Backend Processing**: Django view calls `TranslationKit.translate_text()`
4. **Model Inference**: HuggingFace transformer generates translation
5. **Response**: Translated text returned to client

### API Endpoint

**POST** `/api/translate`

**Request Body**:
```json
{
    "text": "Hello, how are you?",
    "targetLanguage": "fra_Latn"
}
```

**Response**:
```json
{
    "original": "Hello, how are you?",
    "translatedText": "Bonjour, comment allez-vous?"
}
```

**Field Descriptions**:
- `text` (required): Text to translate (any language)
- `targetLanguage` (required): NLLB-200 language code (Level II format)

**Implementation**: `api/agentx_ai/views.py:15-54`

## Client Implementation

### TranslationTab Component

**Location**: `client/src/components/tabs/TranslationTab.tsx`

**Key Features**:
- **No Source Language Selection**: Model auto-detects source language
- **Searchable Language Picker**: Filter 204 languages by name or code
- **Language Counter**: Shows available languages (filtered or total)
- **Real-time Search**: Instant filtering as user types

**State Management**:
```typescript
const [sourceText, setSourceText] = useState('');
const [translatedText, setTranslatedText] = useState('');
const [targetLanguage, setTargetLanguage] = useState('spa_Latn');
const [searchQuery, setSearchQuery] = useState('');
```

**Language Filtering**:
```typescript
const filteredLanguages = searchQuery.trim()
    ? nllb200Languages.filter(lang =>
        lang.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        lang.code.toLowerCase().includes(searchQuery.toLowerCase())
    )
    : nllb200Languages;
```

### Language Data Structure

**Location**: `client/src/data/nllb200Languages.ts`

**Interface**:
```typescript
export interface Language {
    code: string;      // NLLB-200 code: "eng_Latn"
    name: string;      // Human-readable: "English"
    script?: string;   // Script name: "Latin"
}
```

**Helper Functions**:
```typescript
// Get language name by code
getLanguageName("eng_Latn") // Returns: "English"

// Search languages
searchLanguages("French") // Returns: [{ code: "fra_Latn", name: "French", ... }]
```

## Model Loading & Performance

### Initialization

Models are loaded at Django startup when `TranslationKit` is instantiated:

```python
# api/agentx_ai/views.py:7
translation_kit = TranslationKit()
```

**Load Time**: ~5-10 seconds on first startup (downloads models if needed)

**Memory Usage**:
- Language Detection Model: ~500MB
- Translation Model (M2M100-418M): ~1.6GB
- **Total**: ~2.1GB RAM

### Model Storage

Models are cached by HuggingFace transformers in:
```
~/.cache/huggingface/hub/
```

Once downloaded, subsequent starts load from cache (much faster).

## Known Issues & Future Improvements

### Current Issues

1. **Model Mismatch** (`api/agentx_ai/kit/translation.py:89-91`):
   ```python
   # Currently using M2M100 instead of NLLB-200
   self.translation_tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
   self.translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
   ```

   **Should be**:
   ```python
   # Target configuration (already defined in class variables)
   self.level_ii_translation_model_name = "facebook/nllb-200-distilled-600M"
   ```

2. **Placeholder Implementation**: The original `translate_text` function returned `"add this later"` - now fixed but verify behavior

3. **No Source Language Auto-Detection Integration**: Level I detection exists but isn't used in translation flow

### Planned Integrations

From `api/agentx_ai/tests.py:9-12`:

- **FAISS** (`faiss-cpu>=1.13.0`): Vector database for semantic search
- **Neo4j** (`api/agentx_ai/lib/neo4j.py`): Graph-based relationship analysis
- **Django ORM**: AI settings storage and user preferences
- **Memory Graph** (`api/agentx_ai/lib/memory_graph.py`): Graph-based memory system

### Recommended Improvements

1. **Switch to NLLB-200 Model**: Use the configured model for better quality
2. **Auto-detect Source Language**: Integrate Level I detection before translation
3. **Translation History**: Store translations in Django ORM
4. **Batch Translation**: Support multiple texts in single request
5. **Caching**: Cache common translations to reduce inference time
6. **Language Auto-Detection UI**: Add "Detect Language" button in client
7. **Translation Confidence**: Return confidence scores from model
8. **Streaming**: Stream translation tokens for long texts

## Testing

### Run Translation Tests

```bash
# All translation tests
task test

# Specific test class
python api/manage.py test agentx_ai.TranslationKitTest

# Specific test method
python api/manage.py test agentx_ai.TranslationKitTest.test_translate_to_french
```

### Manual Testing

1. Start the development servers:
   ```bash
   task dev
   ```

2. Open the Tauri app (automatically opens on `task dev`)

3. Navigate to Translation tab

4. Test workflow:
   - Enter text in source panel
   - Search for a language (e.g., "Japanese")
   - Select target language (`jpn_Jpan`)
   - Click "Translate"
   - Verify translation appears

### Test Different Language Scripts

Try these to verify script support:

- **Latin**: `eng_Latn` → `fra_Latn` (English → French)
- **Cyrillic**: `eng_Latn` → `rus_Cyrl` (English → Russian)
- **Han**: `eng_Latn` → `zho_Hans` (English → Simplified Chinese)
- **Arabic**: `eng_Latn` → `arb_Arab` (English → Modern Standard Arabic)
- **Devanagari**: `eng_Latn` → `hin_Deva` (English → Hindi)

## Dependencies

### Python (Backend)

```toml
# pyproject.toml
torch = ">=2.9.1"                    # PyTorch for model inference
transformers = ">=4.57.1"            # HuggingFace transformers
sentencepiece = ">=0.2.1"            # Tokenization for NLLB models
python-iso639 = ">=2025.11.16"       # ISO 639 language code conversion
```

### TypeScript (Frontend)

```typescript
// client/src/data/nllb200Languages.ts
// Self-contained, no external dependencies
```

## File Reference

### Backend Files

| File | Purpose |
|------|---------|
| `api/agentx_ai/kit/translation.py` | Core translation logic, models, language lexicon |
| `api/agentx_ai/views.py` | API endpoints for translation and detection |
| `api/agentx_ai/urls.py` | URL routing for API endpoints |

### Frontend Files

| File | Purpose |
|------|---------|
| `client/src/components/tabs/TranslationTab.tsx` | Translation UI component |
| `client/src/data/nllb200Languages.ts` | Language definitions and helpers |
| `client/src/models/translation.ts` | API client for translation endpoint |
| `client/src/styles/TranslationTab.css` | Translation tab styling |

### Documentation Files

| File | Purpose |
|------|---------|
| `docs/translation-system.md` | This file - comprehensive translation docs |
| `docs/api-endpoints.md` | API endpoint reference |
| `CLAUDE.md` | Project overview and development guide |

---

**Last Updated**: 2025-11-25
**System Version**: AgentX v0.1.0 (Tauri migration)
