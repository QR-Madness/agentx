# AgentX Documentation

Welcome to the AgentX documentation! This directory contains comprehensive guides for understanding and working with the AgentX AI translation system.

## Documentation Index

### Core Documentation

1. **[Translation System Architecture](./translation-system.md)** 📖
   - Two-level translation system overview
   - NLLB-200 model integration
   - Language detection and conversion
   - Complete language list (204 languages)
   - Frontend/backend integration
   - Performance considerations
   - **Start here** to understand how translation works

2. **[API Endpoints Reference](./api-endpoints.md)** 🔌
   - Complete API endpoint documentation
   - Request/response examples
   - Error handling
   - CORS configuration
   - Client integration examples
   - **Use this** for API integration

### Project Overview

See **[CLAUDE.md](../CLAUDE.md)** in the root directory for:
- Project architecture
- Development setup
- Running the application
- File structure
- Migration notes (Electron → Tauri)

## Quick Start

### For Developers

1. **Understanding the System**:
   ```
   Read: docs/translation-system.md
   → Understand two-level architecture
   → Learn about NLLB-200 languages
   → See code examples
   ```

2. **Using the API**:
   ```
   Read: docs/api-endpoints.md
   → See all available endpoints
   → Copy request examples
   → Integrate with your app
   ```

3. **Development Setup**:
   ```
   Read: CLAUDE.md
   → Install dependencies
   → Run dev servers
   → Start building
   ```

### For API Users

**Translate text in 3 steps**:

1. Start the API server:
   ```bash
   task api:runserver
   ```

2. Send a translation request:
   ```bash
   curl -X POST http://localhost:12319/api/translate \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, world!",
       "targetLanguage": "fra_Latn"
     }'
   ```

3. Receive translation:
   ```json
   {
     "original": "Hello, world!",
     "translatedText": "Bonjour le monde!"
   }
   ```

See [api-endpoints.md](./api-endpoints.md) for complete API documentation.

## System Architecture

```
┌─────────────────────────────────────┐
│     Tauri Desktop App (Client)     │
│   ┌─────────────────────────────┐  │
│   │   TranslationTab (React)    │  │
│   │  - 204 NLLB-200 languages   │  │
│   │  - Searchable dropdown      │  │
│   │  - Real-time translation    │  │
│   └────────────┬────────────────┘  │
└────────────────┼────────────────────┘
                 │ HTTP POST
                 │ /api/translate
                 ▼
┌─────────────────────────────────────┐
│       Django API (Backend)          │
│   ┌─────────────────────────────┐  │
│   │    TranslationKit           │  │
│   │  - Language Detection       │  │
│   │  - NLLB-200 Translation     │  │
│   │  - ISO 639 Conversion       │  │
│   └─────────────────────────────┘  │
│   ┌─────────────────────────────┐  │
│   │  HuggingFace Transformers   │  │
│   │  - M2M100 Model (temp)      │  │
│   │  - NLLB-200 (target)        │  │
│   └─────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Framework**: Django 5.2.8
- **ML Framework**: PyTorch 2.9.1+
- **Models**: HuggingFace Transformers 4.57.1+
- **Language Detection**: `eleldar/language-detection` (~20 languages)
- **Translation**: `facebook/m2m100_418M` (temporary, targeting NLLB-200)
- **Language Codes**: `python-iso639` for ISO 639 conversion

### Frontend
- **Framework**: React 19 with TypeScript
- **Desktop**: Tauri v2 (Rust backend)
- **Build Tool**: Vite
- **Styling**: Custom CSS with CSS variables

### Database
- **Current**: SQLite (Django default)
- **Planned**:
  - FAISS for vector embeddings
  - Neo4j for relationship graphs

## Supported Languages

AgentX supports **204 languages** across multiple writing systems:

- **110+ Latin script languages**: English, Spanish, French, German, Vietnamese, etc.
- **20+ Arabic script languages**: Modern Standard Arabic, Egyptian Arabic, Persian, Urdu, etc.
- **12+ Cyrillic languages**: Russian, Ukrainian, Bulgarian, Serbian, etc.
- **10+ Devanagari languages**: Hindi, Nepali, Marathi, Sanskrit, etc.
- **40+ other scripts**: Greek, Hebrew, Thai, Chinese, Japanese, Korean, Georgian, Armenian, etc.

**Complete list**: See [translation-system.md](./translation-system.md#complete-language-list-204-languages)

## Language Code Format

AgentX uses NLLB-200 language codes with script tags:

**Format**: `{iso639-3}_{Script}`

**Examples**:
- `eng_Latn` - English (Latin script)
- `fra_Latn` - French (Latin script)
- `spa_Latn` - Spanish (Latin script)
- `arb_Arab` - Modern Standard Arabic (Arabic script)
- `zho_Hans` - Chinese Simplified (Simplified Han characters)
- `zho_Hant` - Chinese Traditional (Traditional Han characters)
- `jpn_Jpan` - Japanese (Japanese script)
- `rus_Cyrl` - Russian (Cyrillic script)
- `hin_Deva` - Hindi (Devanagari script)

**Why this format?**
- Supports language variants (e.g., Simplified vs Traditional Chinese)
- Distinguishes script usage (e.g., Acehnese in Arabic vs Latin)
- Required by NLLB-200 model tokenizer

## Key Features

### Translation Tab
- ✅ 204 languages supported
- ✅ Searchable language picker
- ✅ Auto-detect source language
- ✅ Real-time translation
- ✅ Copy to clipboard
- ✅ Character counter

### API
- ✅ RESTful JSON API
- ✅ CORS support for local development
- ✅ Detailed error messages
- ✅ No authentication required (development)

### Future Features
- ⏳ Translation history storage
- ⏳ Batch translation
- ⏳ Conversation context (chat)
- ⏳ FAISS vector search
- ⏳ Neo4j knowledge graph
- ⏳ User preferences storage

## Development Commands

### Start Everything
```bash
task dev                    # Start API + Client in dev mode
```

### Backend Only
```bash
task api:runserver          # Start Django on port 12319
task api:shell              # Open Django shell
task api:migrate            # Run database migrations
task test                   # Run all tests
```

### Frontend Only
```bash
cd client
npm run tauri dev           # Start Tauri app + Vite dev server
npm run dev                 # Vite dev server only (browser)
npm run build               # TypeScript check + build
```

### Testing Translation
```bash
# Test specific translation function
python api/manage.py test agentx_ai.TranslationKitTest.test_translate_to_french

# Test all translation tests
python api/manage.py test agentx_ai.TranslationKitTest

# Test language detection
python api/manage.py test agentx_ai.TranslationKitTest.test_detect_language
```

## File Structure

```
agentx-source/
├── docs/                          # 📚 Documentation (you are here!)
│   ├── README.md                  # This file
│   ├── translation-system.md      # Translation architecture
│   └── api-endpoints.md           # API reference
│
├── api/                           # 🐍 Django Backend
│   ├── agentx_ai/                 # Main app
│   │   ├── kit/                   # AI toolkit
│   │   │   ├── translation.py     # TranslationKit & LanguageLexicon
│   │   │   └── lib/               # Libraries (Neo4j, MemoryGraph)
│   │   ├── views.py               # API endpoints
│   │   ├── urls.py                # URL routing
│   │   └── tests.py               # Tests
│   ├── agentx_api/                # Django settings
│   └── manage.py                  # Django CLI
│
├── client/                        # ⚛️ Tauri/React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── tabs/
│   │   │       └── TranslationTab.tsx  # Translation UI
│   │   ├── data/
│   │   │   └── nllb200Languages.ts     # 204 language definitions
│   │   ├── models/
│   │   │   └── translation.ts          # API client
│   │   └── styles/
│   │       └── TranslationTab.css      # Styling
│   ├── src-tauri/                 # Rust/Tauri backend
│   └── package.json
│
├── CLAUDE.md                      # 🤖 Project overview for Claude Code
├── Taskfile.yaml                  # 📋 Task definitions
└── pyproject.toml                 # 📦 Python dependencies
```

## Common Issues & Solutions

### Issue: "Module not found" error
**Solution**: Install dependencies
```bash
task install                # Installs Python + npm dependencies
```

### Issue: Translation model not found
**Solution**: Models download on first run (~2GB). Wait for download to complete.
```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/hub/
```

### Issue: CORS errors in browser
**Solution**: Make sure you're accessing from allowed origins:
- `http://localhost:1420` (Vite dev server)
- `https://tauri.localhost` (Tauri app)

### Issue: Translation returns error
**Solution**: Check logs in Django console for detailed error messages.

### Issue: Language code not recognized
**Solution**: Use NLLB-200 format (`eng_Latn`, not `en`). See language list in docs.

## Contributing

### Adding New Languages
Languages are defined in the NLLB-200 model. To add support:
1. Verify language exists in `api/agentx_ai/kit/translation.py:6-31` (`nlb200_list`)
2. Add to `client/src/data/nllb200Languages.ts` with human-readable name
3. Test translation with new language code

### Extending the API
1. Add view function in `api/agentx_ai/views.py`
2. Add URL route in `api/agentx_ai/urls.py`
3. Add client function in `client/src/models/`
4. Document in `docs/api-endpoints.md`

### Improving Translation Quality
1. Switch from M2M100 to NLLB-200 model:
   ```python
   # In api/agentx_ai/kit/translation.py:89-91
   self.translation_tokenizer = AutoTokenizer.from_pretrained(
       self.level_ii_translation_model_name  # Use this instead
   )
   ```
2. Increase model size for better quality (600M → 1.3B → 3.3B)
3. Add context/conversation history

## Resources

### External Documentation
- [NLLB-200 Model](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [M2M100 Model](https://huggingface.co/facebook/m2m100_418M)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Django Documentation](https://docs.djangoproject.com/)
- [Tauri Documentation](https://tauri.app/v2/)
- [React Documentation](https://react.dev/)

### Related Files
- [CLAUDE.md](../CLAUDE.md) - Project overview and setup guide
- [Taskfile.yaml](../Taskfile.yaml) - Task definitions
- [pyproject.toml](../pyproject.toml) - Python dependencies

## Questions?

For questions about:
- **Translation system**: See [translation-system.md](./translation-system.md)
- **API usage**: See [api-endpoints.md](./api-endpoints.md)
- **Development setup**: See [CLAUDE.md](../CLAUDE.md)
- **Language support**: See the language list in [translation-system.md](./translation-system.md#complete-language-list-204-languages)

---

**Documentation Version**: 1.0
**Last Updated**: 2025-11-25
**AgentX Version**: 0.1.0 (Tauri Migration)
**Maintained by**: Claude Code
