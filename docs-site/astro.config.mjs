// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import mermaid from 'astro-mermaid';
import tailwindcss from '@tailwindcss/vite';

import { remarkAdmonitions } from './src/plugins/remark-admonitions.mjs';
import { remarkRewriteMdLinks } from './src/plugins/remark-rewrite-md-links.mjs';

// https://astro.build/config
export default defineConfig({
  site: 'https://agentx.thejpnet.net',

  // `astro-mermaid` must precede `mdx` so it intercepts ```mermaid fences
  // before they reach Shiki, then renders them client-side.
  integrations: [
    mermaid({ theme: 'dark' }),
    mdx(),
  ],

  vite: {
    plugins: [tailwindcss()],
  },

  markdown: {
    // Order matters: rewrite links first, then transform admonitions.
    remarkPlugins: [remarkRewriteMdLinks, remarkAdmonitions],
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
