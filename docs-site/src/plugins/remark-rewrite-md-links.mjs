// Rewrites internal Markdown links (e.g. `../api/endpoints.md#agent`, `chat.md`) to the
// site's `/docs/...` routes, preserving any `#anchor`. MkDocs resolved `*.md` links
// automatically; Astro does not, so we do it here — keeping the source markdown untouched.
//
// Dependency-free: walks the mdast manually rather than importing unist-util-visit.

import path from 'node:path';

const CONTENT_SEGMENT = 'content/docs';

/** Path of the current file relative to the content/docs base, e.g. "getting-started/x.md". */
function docRelativePath(filePath) {
  if (!filePath) return 'index.md';
  const norm = filePath.split(path.sep).join('/');
  const marker = `${CONTENT_SEGMENT}/`;
  const idx = norm.lastIndexOf(marker);
  if (idx === -1) return 'index.md';
  return norm.slice(idx + marker.length);
}

/** True for relative links that point at a `.md` file (the only ones we rewrite). */
function isInternalMdLink(url) {
  if (!url) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return false; // http:, https:, mailto:, etc.
  if (url.startsWith('//')) return false; // protocol-relative
  if (url.startsWith('#')) return false; // in-page anchor
  if (url.startsWith('/')) return false; // already a site-absolute path
  return /\.md(#.*)?$/i.test(url);
}

/** Resolve a relative `.md` target (against the current dir) to a `/docs/...` route. */
function toDocsRoute(currentDir, target) {
  let resolved = path.posix.normalize(path.posix.join(currentDir, target));
  resolved = resolved.replace(/\.md$/i, '');
  resolved = resolved.replace(/^\.\//, '').replace(/^\/+/, '');
  if (resolved === 'index' || resolved === '.' || resolved === '') resolved = '';
  else resolved = resolved.replace(/\/index$/, '');
  return resolved ? `/docs/${resolved}` : '/docs';
}

function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'link' || node.type === 'definition') visit(node);
  if (Array.isArray(node.children)) {
    for (const child of node.children) walk(child, visit);
  }
}

export function remarkRewriteMdLinks() {
  return (tree, file) => {
    const rel = docRelativePath(file?.path ?? file?.history?.[file.history.length - 1]);
    const dir = path.posix.dirname(rel);
    const currentDir = dir === '.' ? '' : dir;

    walk(tree, (node) => {
      if (!isInternalMdLink(node.url)) return;
      const hashIdx = node.url.indexOf('#');
      const pathPart = hashIdx === -1 ? node.url : node.url.slice(0, hashIdx);
      const hash = hashIdx === -1 ? '' : node.url.slice(hashIdx);
      node.url = toDocsRoute(currentDir, pathPart) + hash;
    });
  };
}
