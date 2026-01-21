import React, {useState} from 'react';
import {postTranslation} from '../../models/translation';
import {nllb200Languages} from '../../data/nllb200Languages';
import '../../styles/TranslationTab.css';

export const TranslationTab: React.FC = () => {
    const [sourceText, setSourceText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [targetLanguage, setTargetLanguage] = useState('spa_Latn');
    const [isTranslating, setIsTranslating] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

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

    // Filter languages based on search query
    const filteredLanguages = searchQuery.trim()
        ? nllb200Languages.filter(lang =>
            lang.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            lang.code.toLowerCase().includes(searchQuery.toLowerCase())
        )
        : nllb200Languages;

    return (<div className="translation-tab">
        <div className="translation-header fade-in">
            <h1 className="page-title">
                <span className="page-icon">🌐</span>
                Translation Tool
            </h1>
            <p className="page-subtitle">Translate text between multiple languages instantly</p>
        </div>

        <div className="translation-container">
            <div className="translation-controls card">
                <div className="language-selector">
                    <div className="language-select-wrapper">
                        <label htmlFor="target-language" className="language-label">
                            Translate to:
                        </label>
                        <input
                            type="text"
                            placeholder="Search languages... (e.g., 'French', 'Arabic', 'Chinese')"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="language-search"
                        />
                        <select
                            id="target-language"
                            value={targetLanguage}
                            onChange={(e) => setTargetLanguage(e.target.value)}
                            className="language-select"
                            size={8}
                        >
                            {filteredLanguages.map(lang => (
                                <option key={lang.code} value={lang.code}>
                                    {lang.name}
                                </option>
                            ))}
                        </select>
                        <div className="language-info">
                            {filteredLanguages.length} language{filteredLanguages.length !== 1 ? 's' : ''} available
                        </div>
                    </div>
                </div>
            </div>

            <div className="translation-panels">
                <div className="translation-panel card">
                    <div className="panel-header">
                        <span className="panel-title">Source Text</span>
                        <span className="char-count">{sourceText.length} characters</span>
                    </div>
                    <textarea
                        value={sourceText}
                        onChange={(e) => setSourceText(e.target.value)}
                        placeholder="Enter text to translate..."
                        className="translation-textarea"
                    />
                </div>

                <div className="translation-panel card">
                    <div className="panel-header">
                        <span className="panel-title">Translation</span>
                        {translatedText && (<button className="copy-button button-secondary"
                                                    onClick={async () => {
                                                        try {
                                                            await navigator.clipboard.writeText(translatedText);
                                                        } catch (err) {
                                                            console.error('Failed to copy to clipboard:', err);
                                                        }
                                                    }}>
                            📋 Copy
                        </button>)}
                    </div>
                    <textarea
                        value={translatedText}
                        readOnly
                        placeholder="Translation will appear here..."
                        className="translation-textarea"
                    />
                </div>
            </div>

            <div className="translation-actions">
                <button
                    className="translate-button button-primary"
                    onClick={handleTranslate}
                    disabled={!sourceText.trim() || isTranslating}
                >
                    {isTranslating ? '⏳ Translating...' : '✨ Translate'}
                </button>
                <button
                    className="clear-button button-secondary"
                    onClick={() => {
                        setSourceText('');
                        setTranslatedText('');
                    }}
                >
                    🗑️ Clear
                </button>
            </div>
        </div>
    </div>);
};
