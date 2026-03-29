import React, {useState} from 'react';
import {
  Languages,
  ArrowRight,
  Copy,
  Trash2,
  Loader2,
  Sparkles,
  Search
} from 'lucide-react';
import {postTranslation} from '../../models/translation';
import {nllb200Languages} from '../../data/nllb200Languages';
import '../../styles/TranslationTab.css';

export const TranslationTab: React.FC = () => {
    const [sourceText, setSourceText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [targetLanguage, setTargetLanguage] = useState('spa_Latn');
    const [isTranslating, setIsTranslating] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [copied, setCopied] = useState(false);

    const handleTranslate = async () => {
        if (!sourceText.trim()) return;

        setIsTranslating(true);
        try {
            const response = await postTranslation({
                text: sourceText,
                targetLanguage,
            });
            setTranslatedText(response.translatedText);
        } catch (error) {
            console.error('Translation failed:', error);
            setTranslatedText('Translation failed. Please try again.');
        } finally {
            setIsTranslating(false);
        }
    };

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(translatedText);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy to clipboard:', err);
        }
    };

    // Filter languages based on search query
    const filteredLanguages = searchQuery.trim()
        ? nllb200Languages.filter(lang =>
            lang.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            lang.code.toLowerCase().includes(searchQuery.toLowerCase())
        )
        : nllb200Languages;

    const selectedLanguage = nllb200Languages.find(l => l.code === targetLanguage);

    return (<div className="translation-tab">
        <div className="translation-header fade-in">
            <h1 className="page-title">
                <Languages className="page-icon-svg" />
                Translation
            </h1>
            <p className="page-subtitle">Translate text between 200+ languages instantly</p>
        </div>

        <div className="translation-container">
            {/* Language Selection */}
            <div className="language-bar card glass">
                <div className="language-from">
                    <span className="language-label">Auto Detect</span>
                </div>
                <ArrowRight size={20} className="language-arrow" />
                <div className="language-to">
                    <span className="language-label">{selectedLanguage?.name || 'Select language'}</span>
                </div>
            </div>

            {/* Main Panels */}
            <div className="translation-panels">
                {/* Source Panel */}
                <div className="translation-panel card">
                    <div className="panel-header">
                        <span className="panel-title">Source Text</span>
                        <span className="char-count">{sourceText.length} chars</span>
                    </div>
                    <textarea
                        value={sourceText}
                        onChange={(e) => setSourceText(e.target.value)}
                        placeholder="Enter text to translate..."
                        className="translation-textarea"
                    />
                </div>

                {/* Translate Button (between panels) */}
                <div className="translate-action">
                    <button
                        className="translate-button button-primary"
                        onClick={handleTranslate}
                        disabled={!sourceText.trim() || isTranslating}
                    >
                        {isTranslating ? (
                            <>
                                <Loader2 size={18} className="spin" />
                                Translating
                            </>
                        ) : (
                            <>
                                <Sparkles size={18} />
                                Translate
                            </>
                        )}
                    </button>
                </div>

                {/* Translation Panel */}
                <div className="translation-panel card">
                    <div className="panel-header">
                        <span className="panel-title">Translation</span>
                        {translatedText && (
                            <button className="button-ghost" onClick={handleCopy}>
                                <Copy size={14} />
                                {copied ? 'Copied!' : 'Copy'}
                            </button>
                        )}
                    </div>
                    <textarea
                        value={translatedText}
                        readOnly
                        placeholder="Translation will appear here..."
                        className="translation-textarea result"
                    />
                </div>
            </div>

            {/* Language Selector */}
            <div className="language-selector card">
                <div className="selector-header">
                    <h3 className="section-title">
                        <Languages size={18} className="section-title-icon" />
                        Target Language
                    </h3>
                    <div className="search-wrapper">
                        <Search size={16} className="search-icon" />
                        <input
                            type="text"
                            placeholder="Search languages..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="language-search"
                        />
                    </div>
                </div>
                <div className="language-grid">
                    {filteredLanguages.slice(0, 50).map(lang => (
                        <button
                            key={lang.code}
                            className={`language-chip ${targetLanguage === lang.code ? 'active' : ''}`}
                            onClick={() => setTargetLanguage(lang.code)}
                        >
                            {lang.name}
                        </button>
                    ))}
                    {filteredLanguages.length > 50 && (
                        <span className="more-languages">+{filteredLanguages.length - 50} more</span>
                    )}
                </div>
                <div className="language-count">
                    {filteredLanguages.length} languages available
                </div>
            </div>

            {/* Actions */}
            <div className="translation-actions">
                <button
                    className="button-secondary"
                    onClick={() => {
                        setSourceText('');
                        setTranslatedText('');
                    }}
                >
                    <Trash2 size={16} />
                    Clear All
                </button>
            </div>
        </div>
    </div>);
};
