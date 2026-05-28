import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import { Copy, Check } from 'lucide-react';
import 'katex/dist/katex.min.css';
import './MessageContent.css';

interface MessageContentProps {
  content: string;
  className?: string;
}

const MessageContentImpl: React.FC<MessageContentProps> = ({ content, className = '' }) => {
  const [copiedCode, setCopiedCode] = React.useState<string | null>(null);

  const handleCopyCode = async (code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className={`message-content-markdown ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
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
        {content}
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
