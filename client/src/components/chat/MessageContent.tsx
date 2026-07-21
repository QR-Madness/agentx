import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import { Copy, Check } from 'lucide-react';
import { stripThinkingTags } from '../../lib/messages';
import 'katex/dist/katex.min.css';
import './MessageContent.css';

interface MessageContentProps {
  content: string;
  className?: string;
}

/**
 * Convert LaTeX bracket delimiters to the dollar form `remark-math` understands.
 * Many models emit `\( ... \)` (inline) and `\[ ... \]` (display) instead of
 * `$ ... $` / `$$ ... $$`, which would otherwise render as plain text.
 *
 * Fenced code blocks and inline code spans are protected so we never rewrite
 * literal backslashes inside code samples.
 *
 * Also escapes currency-style `$` (a `$` directly before a digit, e.g. "$20")
 * to `\$` so remark-math doesn't treat it as a math delimiter — an unescaped
 * stray `$` collides with real `$...$` math and splits expressions, producing
 * KaTeX "Expected EOF, got '}'" errors.
 */
export function normalizeMathDelimiters(src: string): string {
  // Split on (and keep) fenced code blocks and inline code; odd-indexed
  // segments are the captured code spans and are left untouched.
  const segments = src.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg;
      return seg
        .replace(/\\\[([\s\S]+?)\\\]/g, (_m, body) => `$$${body.trim()}$$`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => `$${body.trim()}$`)
        // Escape currency $ (not already escaped, not part of $$).
        .replace(/(?<![\\$])\$(?=\d)/g, '\\$');
    })
    .join('');
}

const MessageContentImpl: React.FC<MessageContentProps> = ({ content, className = '' }) => {
  const [copiedCode, setCopiedCode] = React.useState<string | null>(null);
  // Think-tag strip is the render-level BACKSTOP: persistence, restore
  // mapping, and the live buffer all strip upstream, but content mapped
  // before those fixes survives in cached tabs — and rehype-raw turns a raw
  // <think> into an unknown DOM element (React error, observed live).
  const normalized = React.useMemo(
    () => normalizeMathDelimiters(stripThinkingTags(content, true)),
    [content],
  );

  const handleCopyCode = async (code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className={`message-content-markdown ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        // rehype-raw must run first so raw HTML (e.g. `<br/>` inside table
        // cells — the only way GFM tables express line breaks) is parsed into
        // the tree while the math-* element nodes from remark-math survive;
        // rehype-katex then renders those, highlight runs last on code.
        //
        // KaTeX runs non-strict and never throws: a malformed expression (or a
        // stray `$` from currency colliding with delimiters) renders its source
        // in a muted error colour instead of blanking the whole message red.
        rehypePlugins={[
          rehypeRaw,
          [rehypeKatex, { throwOnError: false, strict: false, errorColor: 'var(--text-error, #f87171)' }],
          rehypeHighlight,
        ]}
        components={{
          // Custom code block rendering with copy button
          pre({ children, ...props }) {
            const codeElement = React.Children.toArray(children).find(
              (child): child is React.ReactElement => 
                React.isValidElement(child) && child.type === 'code'
            );
            
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const codeContent = (codeElement?.props as any)?.children?.toString() || '';
            
            return (
              <div className="code-block-wrapper">
                <button 
                  className="code-copy-btn"
                  onClick={() => handleCopyCode(codeContent)}
                  title="Copy code"
                >
                  {copiedCode === codeContent ? (
                    <Check size={14} />
                  ) : (
                    <Copy size={14} />
                  )}
                </button>
                <pre {...props}>{children}</pre>
              </div>
            );
          },
          // Style inline code
          code({ className, children, ...props }) {
            const isInline = !className;
            return (
              <code 
                className={`${className || ''} ${isInline ? 'inline-code' : ''}`} 
                {...props}
              >
                {children}
              </code>
            );
          },
          // External links open in new tab
          a({ href, children, ...props }) {
            const isExternal = href?.startsWith('http');
            return (
              <a 
                href={href} 
                target={isExternal ? '_blank' : undefined}
                rel={isExternal ? 'noopener noreferrer' : undefined}
                {...props}
              >
                {children}
              </a>
            );
          },
          // Style tables
          table({ children, ...props }) {
            return (
              <div className="table-wrapper">
                <table {...props}>{children}</table>
              </div>
            );
          },
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
};

// Memoized: ReactMarkdown re-parses the AST on every render, which is
// expensive for any non-trivial message. The component only depends on
// `content` + `className` props (internal copy state is local), so shallow
// equality is exactly what we want.
export const MessageContent = React.memo(MessageContentImpl);

export default MessageContent;
