// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import mermaid from 'astro-mermaid';
import tailwindcss from '@tailwindcss/vite';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

import { remarkAdmonitions } from './src/plugins/remark-admonitions.mjs';
import { remarkRewriteMdLinks } from './src/plugins/remark-rewrite-md-links.mjs';

// Cosmic-dark mermaid theme — ported from the design handoff (docs.html). Pinned
// via theme 'base' + explicit themeVariables; `autoTheme: false` because the site
// is dark-only (no data-theme switching).
const mermaidThemeVariables = {
  background: '#0a0d14',
  primaryColor: '#181c27',
  primaryTextColor: '#e6e8ee',
  primaryBorderColor: '#343a4c',
  secondaryColor: '#11141c',
  tertiaryColor: '#0b0d12',
  lineColor: '#5a6478',
  textColor: '#e6e8ee',
  clusterBkg: 'rgba(99,102,241,0.06)',
  clusterBorder: '#343a4c',
  edgeLabelBackground: '#0a0d14',
  fontFamily: 'Geist Mono, ui-monospace, monospace',
  fontSize: '13px',
  actorBkg: '#181c27',
  actorBorder: '#6366f1',
  actorTextColor: '#e6e8ee',
  actorLineColor: '#343a4c',
  signalColor: '#8b94a5',
  signalTextColor: '#e6e8ee',
  labelBoxBkgColor: '#181c27',
  labelBoxBorderColor: '#343a4c',
  labelTextColor: '#e6e8ee',
  loopTextColor: '#c084fc',
  noteBkgColor: 'rgba(34,211,238,0.10)',
  noteTextColor: '#22d3ee',
  noteBorderColor: '#22d3ee',
  activationBkgColor: '#222633',
  activationBorderColor: '#6366f1',
  sequenceNumberColor: '#07080c',
};

// https://astro.build/config
export default defineConfig({
  site: 'https://agentx.thejpnet.net',

  // `astro-mermaid` must precede `mdx` so it intercepts ```mermaid fences
  // before they reach Shiki, then renders them client-side.
  integrations: [
    mermaid({
      theme: 'base',
      autoTheme: false,
      mermaidConfig: {
        themeVariables: mermaidThemeVariables,
        flowchart: { curve: 'basis' },
      },
    }),
    mdx(),
  ],

  vite: {
    plugins: [tailwindcss()],
  },

  markdown: {
    // Order matters: rewrite links first, then transform admonitions.
    remarkPlugins: [remarkRewriteMdLinks, remarkAdmonitions],
    // Append an empty hover anchor to each heading. The `#` glyph is drawn
    // presentationally via CSS (`a.anchor::after` in global.css) rather than as a
    // real text node — otherwise it leaks into Astro's collected heading text and
    // shows up appended to every right-rail TOC entry. IDs are added by rehype-slug
    // (idempotent) before autolink runs.
    rehypePlugins: [
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: 'append',
          properties: { className: ['anchor'], ariaHidden: true, tabIndex: -1 },
          content: [],
        },
      ],
    ],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
