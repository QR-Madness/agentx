import React, {useState} from 'react';
import {postTranslation} from '../../models/translation';
import '../../styles/TranslationTab.css';

export const TranslationTab: React.FC = () => {
    const [sourceText, setSourceText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [sourceLanguage, setSourceLanguage] = useState('--');
    const [targetLanguage, setTargetLanguage] = useState('es');
    const [isTranslating, setIsTranslating] = useState(false);

    const languages = [{code: '--', name: 'Auto-detect'}, {code: 'en', name: 'English'}, {
        code: 'es',
        name: 'Spanish'
    }, {code: 'fr', name: 'French'}, {code: 'de', name: 'German'}, {code: 'it', name: 'Italian'}, {
        code: 'pt',
        name: 'Portuguese'
    }, {code: 'ru', name: 'Russian'}, {code: 'ja', name: 'Japanese'}, {code: 'zh', name: 'Chinese'}, {
        code: 'ar',
        name: 'Arabic'
    },];

    const handleTranslate = async () => {
        if (!sourceText.trim()) return;

        setIsTranslating(true);
        try {
            const response = await postTranslation({
                text: sourceText, sourceLanguage, targetLanguage,
            });
            setTranslatedText(response.translatedText);
        } catch (error) {
            console.error('Translation failed:', error);
            setTranslatedText('Translation failed. Please try again.');
        } finally {
            setIsTranslating(false);
        }
    };

    const swapLanguages = () => {
        setSourceLanguage(targetLanguage);
        setTargetLanguage(sourceLanguage);
        setSourceText(translatedText);
        setTranslatedText(sourceText);
    };

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
                    <select
                        value={sourceLanguage}
                        onChange={(e) => setSourceLanguage(e.target.value)}
                        className="language-select"
                    >
                        {languages.map(lang => (<option key={lang.code} value={lang.code}>{lang.name}</option>))}
                    </select>

                    <button className="swap-button button-secondary" onClick={swapLanguages}>
                        <span>⇄</span>
                    </button>

                    <select
                        value={targetLanguage}
                        onChange={(e) => setTargetLanguage(e.target.value)}
                        className="language-select"
                    >
                        {/* Remove auto-detect language option from target language selector */}
                        {languages.map(lang => (lang.code != "--" ?
                            <option key={lang.code} value={lang.code}>{lang.name}</option> : null))}
                    </select>
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
                                                    onClick={() => navigator.clipboard.writeText(translatedText)}>
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
