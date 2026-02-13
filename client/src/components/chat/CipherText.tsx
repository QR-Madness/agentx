import React, { useState, useEffect, useRef } from 'react';
import './CipherText.css';

interface CipherTextProps {
  text: string;
  speed?: number;  // ms per character reveal
  scrambleIterations?: number;  // number of random chars before settling
  className?: string;
  onComplete?: () => void;
}

// Characters to use for the cipher scramble effect
const CIPHER_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%&*!?+=';

export const CipherText: React.FC<CipherTextProps> = ({
  text,
  speed = 30,
  scrambleIterations = 3,
  className = '',
  onComplete
}) => {
  const [revealedLength, setRevealedLength] = useState(0);
  const [scrambleChars, setScrambleChars] = useState<string[]>([]);
  const animationRef = useRef<number | null>(null);
  const iterationCountRef = useRef(0);

  useEffect(() => {
    if (revealedLength >= text.length) {
      onComplete?.();
      return;
    }

    const animate = () => {
      iterationCountRef.current++;
      
      if (iterationCountRef.current >= scrambleIterations) {
        // Reveal next character
        iterationCountRef.current = 0;
        setRevealedLength(prev => prev + 1);
        setScrambleChars([]);
      } else {
        // Show random character for unrevealed portion
        const nextChars: string[] = [];
        for (let i = revealedLength; i < Math.min(revealedLength + 3, text.length); i++) {
          if (text[i] === ' ' || text[i] === '\n') {
            nextChars.push(text[i]);
          } else {
            nextChars.push(CIPHER_CHARS[Math.floor(Math.random() * CIPHER_CHARS.length)]);
          }
        }
        setScrambleChars(nextChars);
      }
      
      animationRef.current = window.setTimeout(animate, speed);
    };

    animationRef.current = window.setTimeout(animate, speed);

    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [text, revealedLength, speed, scrambleIterations, onComplete]);

  // Reset when text changes
  useEffect(() => {
    setRevealedLength(0);
    setScrambleChars([]);
    iterationCountRef.current = 0;
  }, [text]);

  // Build display string
  const revealed = text.slice(0, revealedLength);
  const scrambled = scrambleChars.join('');

  return (
    <span className={`cipher-text ${className}`}>
      <span className="cipher-revealed">{revealed}</span>
      {scrambled && <span className="cipher-scramble">{scrambled}</span>}
    </span>
  );
};

export default CipherText;
