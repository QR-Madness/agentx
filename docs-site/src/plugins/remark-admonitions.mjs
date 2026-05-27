// Transforms MkDocs-style admonitions into styled <aside> blocks, e.g.
//
//   !!! warning "Change default passwords"
//       Update passwords in your `.env` file before deploying.
//
// CommonMark parses the marker line + its single indented line as one paragraph (lazy
// continuation). We split the marker off the first text node and keep the remaining inline
// children (so inline code/links in the body still render), wrapping them in an <aside>.
// Styling lives entirely in the token layer (src/styles/global.css). Source markdown is untouched.

// Astro's built-in remark-smartypants runs before this plugin and rewrites straight quotes
// to curly ones, so the optional title accepts both `"…"` and `“…”`.
const MARKER_RE = /^!!!\s+([A-Za-z][\w-]*)(?:\s+["“]([^"”]*)["”])?[ \t]*(?:\r?\n([\s\S]*))?$/;

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/** If the paragraph opens with an admonition marker, return its parts; otherwise null. */
function matchAdmonition(paragraph) {
  const first = paragraph.children?.[0];
  if (!first || first.type !== 'text') return null;

  const match = first.value.match(MARKER_RE);
  if (!match) return null;

  const type = match[1].toLowerCase();
  const title = match[2] !== undefined && match[2] !== '' ? match[2] : capitalize(type);
  const remainder = (match[3] ?? '').replace(/^[ \t]+/, '');

  const bodyChildren = [];
  if (remainder) bodyChildren.push({ type: 'text', value: remainder });
  for (let i = 1; i < paragraph.children.length; i++) bodyChildren.push(paragraph.children[i]);

  return { type, title, bodyChildren };
}

/** Recursively expand admonition paragraphs into raw-HTML-wrapped asides. */
function transform(node) {
  if (!node || !Array.isArray(node.children)) return;

  const result = [];
  for (const child of node.children) {
    const adm = child.type === 'paragraph' ? matchAdmonition(child) : null;
    if (adm) {
      result.push({
        type: 'html',
        value:
          `<aside class="ax-admon" data-kind="${escapeHtml(adm.type)}">` +
          `<p class="ax-admon-title">${escapeHtml(adm.title)}</p>`,
      });
      if (adm.bodyChildren.length) {
        result.push({ type: 'paragraph', children: adm.bodyChildren });
      }
      result.push({ type: 'html', value: '</aside>' });
    } else {
      transform(child);
      result.push(child);
    }
  }
  node.children = result;
}

export function remarkAdmonitions() {
  return (tree) => transform(tree);
}
