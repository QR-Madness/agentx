import React, { useState } from 'react';
import styled from 'styled-components';

const ViewContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 24px;
`;

const ViewHeader = styled.div`
  margin-bottom: 24px;

  h2 {
    font-size: 24px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const TranslateContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
`;

const TranslateSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;

  label {
    font-size: 14px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textSecondary};
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  textarea {
    padding: 16px;
    background: ${({ theme }) => theme.colors.bgSecondary};
    border: 1px solid ${({ theme }) => theme.colors.borderColor};
    border-radius: 8px;
    color: ${({ theme }) => theme.colors.textPrimary};
    font-size: 15px;
    font-family: inherit;
    resize: vertical;

    &:focus {
      outline: none;
      border-color: ${({ theme }) => theme.colors.accentPrimary};
    }
  }
`;

const DetectedLanguage = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: ${({ theme }) => theme.colors.bgTertiary};
  border-radius: 6px;
  font-size: 14px;

  .label {
    color: ${({ theme }) => theme.colors.textMuted};
  }

  .value {
    color: ${({ theme }) => theme.colors.accentPrimary};
    font-weight: 600;
  }
`;

const TranslateControls = styled.div`
  display: flex;
  gap: 12px;
  align-items: center;
`;

const LangSelect = styled.select`
  flex: 1;
  padding: 12px 16px;
  background: ${({ theme }) => theme.colors.bgSecondary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.textPrimary};
  font-size: 15px;
  cursor: pointer;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.colors.accentPrimary};
  }
`;

const SwapButton = styled.button`
  padding: 12px 16px;
  background: ${({ theme }) => theme.colors.bgSecondary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 20px;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.bgTertiary};
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const TranslateButton = styled.button`
  padding: 12px 32px;
  background: ${({ theme }) => theme.colors.accentPrimary};
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.accentHover};
  }
`;

const languages = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
];

export const TranslationView: React.FC = () => {
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [sourceLang, setSourceLang] = useState('auto');
  const [targetLang, setTargetLang] = useState('en');
  const [detectedLang, setDetectedLang] = useState('—');

  const handleSourceChange = (text: string) => {
    setSourceText(text);
    if (text.trim()) {
      // Stub: Simulate language detection
      const randomLang = languages[Math.floor(Math.random() * languages.length)];
      setDetectedLang(randomLang.label);
    } else {
      setDetectedLang('—');
    }
  };

  const handleTranslate = () => {
    const text = sourceText.trim();
    if (!text) {
      setTargetText('');
      return;
    }

    // Stub: Simulate translation
    const sourceLangName =
      sourceLang === 'auto'
        ? detectedLang
        : languages.find((l) => l.value === sourceLang)?.label || sourceLang;
    const targetLangName =
      languages.find((l) => l.value === targetLang)?.label || targetLang;

    setTargetText(
      `[Translated from ${sourceLangName} to ${targetLangName}]\n\n` +
        `Translation functionality will be implemented here.\n` +
        `Original text: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`
    );
  };

  const handleSwap = () => {
    if (sourceLang !== 'auto') {
      setSourceLang(targetLang);
      setTargetLang(sourceLang);
      setSourceText(targetText);
      setTargetText(sourceText);
    }
  };

  return (
    <ViewContainer>
      <ViewHeader>
        <h2>Translation & Language Detection</h2>
      </ViewHeader>
      <TranslateContainer>
        <TranslateSection>
          <label htmlFor="sourceText">Source Text</label>
          <textarea
            id="sourceText"
            placeholder="Enter text to translate or detect language..."
            rows={8}
            value={sourceText}
            onChange={(e) => handleSourceChange(e.target.value)}
          />
          <DetectedLanguage>
            <span className="label">Detected:</span>
            <span className="value">{detectedLang}</span>
          </DetectedLanguage>
        </TranslateSection>

        <TranslateControls>
          <LangSelect value={sourceLang} onChange={(e) => setSourceLang(e.target.value)}>
            <option value="auto">Auto-detect</option>
            {languages.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </LangSelect>
          <SwapButton onClick={handleSwap}>⇄</SwapButton>
          <LangSelect value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
            {languages.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </LangSelect>
          <TranslateButton onClick={handleTranslate}>Translate</TranslateButton>
        </TranslateControls>

        <TranslateSection>
          <label htmlFor="targetText">Translation</label>
          <textarea
            id="targetText"
            placeholder="Translation will appear here..."
            rows={8}
            value={targetText}
            readOnly
          />
        </TranslateSection>
      </TranslateContainer>
    </ViewContainer>
  );
};
