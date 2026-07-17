# AgentX Docs Site — architecture & design handoff

This is the **AgentX project hub**: an Astro static site with a bare landing page at `/` and
the documentation under `/docs`. It replaced a MkDocs/Material site. This document is written for
the **design pass** — the structure exists; the visuals are deliberately minimal and waiting for you.

> **Status:** the **docs** design pass is **done** — the "claude design" handoff (cosmic-dark + indigo,
> Geist typefaces, `.ax-*` convention) is implemented across the token layer, prose, and docs chrome.
> The **landing page** (`src/pages/index.astro`) is still a blank canvas awaiting its own pass.
> The architecture still supports **rapid overhauls from a single token file** — read "Design system" first.

---

## TL;DR

- **Every color, font, radius, and key dimension lives in one file:** [`src/styles/global.css`](src/styles/global.css).
  Reskinning the whole site = editing the `@theme` block there. Don't hardcode visual values anywhere else.
- **Markdown content is sacred** — `src/content/docs/**/*.md` was migrated as-is. Style it via the
  `.ax-prose` rules in `global.css`; do **not** edit content files to achieve a look.
- **Landing page (`src/pages/index.astro`) is a blank canvas** — intentionally bare. Build the real hero/hub here.
- **Dark-only.** No theme toggle. Fonts are **Geist / Geist Mono**, loaded via `<link>` in `BaseLayout.astro`.
- **Class convention:** chrome and prose use the design's `.ax-*` prefix (e.g. `.ax-prose`, `.ax-admon`,
  `.ax-toc`, `.ax-docs-sidebar`). Future handoffs in the same convention drop in cleanly.
- **Search is Pagefind** — indexed at build (`bun run build` runs `pagefind --site dist`); live under
  `preview`/prod, inert in `astro dev`.
- After editing any remark/rehype plugin, **`rm -rf .astro dist` before rebuilding** (Astro caches parsed content).

---

## Tech stack

| Concern | Choice |
|---|---|
| Framework | **Astro 6** (static output) |
| Styling | **Tailwind CSS v4** via `@tailwindcss/vite` (CSS-first `@theme`, no `tailwind.config.js`) |
| Markdown | `.md` content collection; **MDX** integration available for future interactive pages |
| Code highlighting | **Shiki** (`github-dark` theme), built into Astro |
| Heading anchors | **rehype-slug** + **rehype-autolink-headings** (hover `#` on h2–h4) |
| Diagrams | **astro-mermaid** (client-side, cosmic-dark `themeVariables` in `astro.config.mjs`) |
| Search | **Pagefind** (static index built into `dist/pagefind/`; ⌘K modal in `Search.astro`) |
| Package manager | **bun** |
| Deploy | **Vercel** (static), custom domain `agentx.thejpnet.net` |

Commands (from `docs-site/`): `bun install` · `bun run dev` (→ http://localhost:4321) ·
`bun run build` (→ `dist/`) · `bun run preview`. Also exposed as `task docs:serve|build|preview|deploy`.

---

## Directory map

```
docs-site/
├─ astro.config.mjs          Integrations (mermaid→mdx), Shiki, remark+rehype wiring, mermaid theme, site URL
├─ vercel.json               Link headers (api-catalog/service-desc/service-doc/llms.txt) + .md content type
├─ middleware.ts             ★ Vercel Routing Middleware: `Accept: text/markdown` → the page's .md twin (ADR-13)
├─ scripts/
│  ├─ distill-hero-run.mjs        Distills a real agent run into config/hero-run.json for the hero console
│  └─ gen-markdown-manifest.mjs   Generates src/generated/markdown-manifest.ts (twin token estimates)
├─ src/
│  ├─ styles/global.css      ★ THE TOKEN LAYER + .ax-prose / .ax-admon / .ax-toc / docs @media
│  ├─ config/
│  │  ├─ site.ts             Site metadata (name, description, repo, docsBasePath, copyright)
│  │  ├─ landing.ts          Landing copy/data (FEATURES, PILLARS, HERO_STATS…) + product version
│  │  └─ nav.ts              ★ Docs sidebar structure + helpers (slugToHref, groupForSlug, editUrlForSlug…)
│  ├─ content.config.ts      `docs` collection: glob loader over content/docs
│  ├─ content/docs/          ★ 43 markdown files — the actual docs content
│  ├─ content/homepage.md    The home page's Markdown twin (authored for agents; served at /index.md)
│  ├─ lib/
│  │  └─ markdown-negotiation.ts  Pure Accept q-value parsing + page→twin mapping (middleware's brain)
│  ├─ generated/
│  │  └─ markdown-manifest.ts     GENERATED — per-twin token estimates for `x-markdown-tokens`
│  ├─ layouts/
│  │  ├─ BaseLayout.astro    <html> shell, <head>, global.css + Geist <link> — used by every page
│  │  └─ DocsLayout.astro    Docs shell: mobile top bar + drawer, breadcrumb, (Sidebar | article | TOC), article footer
│  ├─ components/
│  │  ├─ icons/GitHubGlyph.astro   GitHub mark (currentColor)
│  │  └─ layout/
│  │     ├─ Sidebar.astro          Brand + GitHub pill + Search + nav (sticky/drawer)
│  │     ├─ SidebarGroup.astro     Mono-uppercase group label + links
│  │     ├─ SidebarLink.astro      One link; color dot (subsystems) + active pill
│  │     ├─ Search.astro           ⌘K box + Pagefind-backed modal
│  │     └─ TableOfContents.astro  Right rail "On this page" (scroll-spy) + source/issue links
│  ├─ pages/
│  │  ├─ index.astro         ★ The landing page at / (hero, why, subsystems, system map)
│  │  ├─ index.md.ts         Serves content/homepage.md at /index.md (the home page's twin)
│  │  ├─ docs/[...slug].astro Renders docs under /docs; computes last-edited (git) + edit URL
│  │  ├─ docs/[...slug].md.ts Raw-Markdown twin of every docs page at /docs/<slug>.md
│  │  ├─ llms.txt.ts         Agent-oriented index of the docs (llmstxt.org), linking the .md twins
│  │  ├─ sitemap.xml.ts      Sitemap built from the nav tree
│  │  └─ 404.astro           Custom not-found
│  └─ plugins/
│     ├─ remark-rewrite-md-links.mjs  Internal *.md links → /docs/... routes
│     └─ remark-admonitions.mjs       MkDocs `!!! type "title"` → <aside class="ax-admon" data-kind="…">
└─ public/                   favicon.svg · favicon.ico · AgentX-Logo-v3-badge.png · (pagefind/ at build)
```

---

## Design system (read this carefully)

### The token layer — `src/styles/global.css`

Tailwind v4 reads design tokens from a CSS `@theme { … }` block. **Defining a variable there both
sets the value and generates the matching utility class.** Current tokens (cosmic-dark + indigo, from
the "claude design" handoff):

| Token group | Tokens | Notes |
|---|---|---|
| Surfaces | `--color-bg` `#07080c` (deepest) · `--color-bg-2` `#0b0d12` (page) · `--color-surface` `#11141c` · `--color-surface-2` `#181c27` · `--color-surface-3` `#222633` · `--color-code-bg` `#0a0d14` | `bg-*`/`text-*` utilities |
| Borders | `--color-border` `#262b38` · `--color-border-2` `#343a4c` | `border-*` |
| Text | `--color-text` `#e6e8ee` · `--color-text-strong` `#ffffff` · `--color-text-muted` `#8b94a5` · `--color-text-dim` `#5a6478` | body / headings / secondary / faint |
| Accent | `--color-accent` `#6366f1` · `--color-accent-hover` `#818cf8` · `--color-accent-subtle` `#1e1b4b` · `--color-accent-fg` `#fff` | brand, links, active states |
| Status | `--color-ok` `#10b981` · `--color-warning` `#f59e0b` · `--color-danger` `#ef4444` | — |
| Subsystem accents | `--c-agent` `--c-reasoning` `--c-drafting` `--c-mcp` `--c-providers` `--c-prompts` `--c-memory` `--c-translation` | dots/tags only; **plain vars** (no utilities), used via `var(--c-*)` — see nav.ts Features color dots & admonition `note` accent |
| Type | `--font-sans` / `--font-display` (Geist) · `--font-mono` (Geist Mono) | `font-sans` / `font-display` / `font-mono` |
| Shape | `--radius-card` `0.75rem` · `--radius-sm` `0.375rem` · `--radius-pill` `999px` | `rounded-card` / `-sm` / `-pill` |
| Layout | `--layout-sidebar-width` `17rem` · `--layout-toc-width` `14rem` · `--layout-docs-max` `90rem` · `--layout-article-max` `47.5rem` · `--layout-content-max` `72rem` | *(no utilities — see below)* |

**Two mechanics to know:**

1. **`--color-*`, `--font-*`, `--radius-*` are Tailwind "namespaces"** → they auto-generate
   `bg-*`/`text-*`/`border-*`, `font-*`, `rounded-*`. `--layout-*` and `--c-*` are **not** namespaces,
   so they generate no utilities; reference them as arbitrary CSS-var utilities
   (`w-(--layout-sidebar-width)`) or via `var(--c-agent)` in scoped component styles.
2. The token names produce a few **awkward utilities** (`bg-bg`, `text-text`). If that bothers you,
   rename the tokens — just update the consuming components too. Grep is your friend; usages are shallow.

Below the `@theme` block, `global.css` also contains the **`.prose` rules** (styling for rendered
markdown), **`.admonition` rules**, and **`.mermaid` rules** — all written in plain CSS that reads
`var(--token)`. This is where most of the "docs look" lives. Restyle headings, tables, code blocks,
callouts, etc. here.

### Fonts

`--font-sans`/`--font-display` are **Geist** and `--font-mono` is **Geist Mono**, loaded via a Google
Fonts `<link>` in `BaseLayout.astro` (with `preconnect`). To swap typefaces: change the `<link>` and
the three font tokens — one place each.

### Layout & components

- **`DocsLayout.astro`** owns the docs page frame: a centered `--layout-docs-max` flex row of
  `Sidebar | main(article.ax-prose) | TableOfContents` (no top header/footer — brand lives in the
  sidebar). The sidebar is sticky full-height; the TOC hides ≤1024px; ≤720px a sticky mobile top bar
  + hamburger drives a slide-in sidebar drawer (vanilla `<script>` in the layout). Breadcrumb sits
  above the article; an edit-link + last-edited line below it.
- Components are intentionally **small and single-purpose**. Sidebar is split Sidebar → SidebarGroup →
  SidebarLink; swap markup/classes without touching nav data.
- **Routing/breadcrumb helpers** live in `nav.ts` (`slugToHref`, `labelForSlug`, `groupForSlug`,
  `isGroup`, `editUrlForSlug`, `sourceUrlForSlug`) — reuse them; don't hand-roll URL logic.

### Special content rendering

- **Code blocks:** Shiki emits `<pre class="astro-code github-dark">…`. Its theme paints token
  colors; we override only the *background* to `--color-code-bg` (`!important`, because Shiki sets an
  inline bg). To change the syntax theme entirely, edit `markdown.shikiConfig.theme` in
  `astro.config.mjs` (single or dual themes supported).
- **Admonitions:** rendered by `remark-admonitions.mjs` as `<aside class="ax-admon" data-kind="{note|tip|info|warning|danger}">`
  with a `<p class="ax-admon-title">`. Per-kind accent + tint are set per `[data-kind]` in `global.css`
  (`note`→`--c-agent`, `tip`→`--color-ok`, `warning`/`danger`→status). Add a kind by adding a `[data-kind="…"]` rule.
- **Mermaid:** `astro-mermaid` renders ` ```mermaid ` fences client-side. Themed via `theme: 'base'`
  + a full cosmic-dark `themeVariables` object in `astro.config.mjs` (`autoTheme: false`, since the
  site is dark-only). Container styled via `.ax-prose .mermaid`.
- **Heading anchors:** `rehype-autolink-headings` appends a hover `#` link (`a.anchor`) to h2–h4
  (h1's is hidden in CSS). Styled in the `.ax-prose` heading rules.

---

## Content & routing

- Docs are a **content collection** (`content.config.ts`) loaded by glob from `src/content/docs`.
  Files are plain `.md`, **no frontmatter** — page titles come from the nav label (`labelForSlug`)
  or the first H1.
- `pages/docs/[...slug].astro` statically generates one route per entry. The glob `id` is the file
  path minus extension (`features/chat` → `/docs/features/chat`); the root `index` maps to `/docs`.
- The **sidebar is data, not markup**: edit `src/config/nav.ts` to add/reorder/group pages. Slugs are
  docs-relative; `slugToHref` prefixes `/docs`.

---

## Rules of the road (please keep these invariants)

1. **No hardcoded colors / spacing / typography outside `global.css`.** Components use token-backed
   utilities (`bg-surface`, `text-accent`, `rounded-card`) or `var(--token)`. This is what makes a
   one-file redesign possible — preserve it.
2. **Don't edit content for styling.** `src/content/docs/**` is migrated as-is and may be re-synced
   from `../docs`. Achieve looks via `.ax-prose`/components, not by touching markdown.
3. **Keep components small and composable.** Prefer adding/splitting a component over growing one.
4. **MDX is available** if you need a bespoke interactive page, but the docs themselves should stay `.md`.

---

## Gotchas that will cost you time

- **Astro caches parsed content in `.astro/`.** Editing a remark/rehype plugin does **not** re-run it
  on unchanged markdown — stale HTML persists across builds. After plugin changes: `rm -rf .astro dist`.
- **`remark-smartypants` runs before user remark plugins** (Astro built-in) and rewrites straight
  quotes `"` → curly `“ ”`. Any markdown transform that matches quotes must accept both (see
  `remark-admonitions.mjs`). It also affects how titles/strings appear in output.
- **No `curl` in some sandboxes** — for smoke-testing the preview server use Node's built-in `fetch`.
- Tailwind v4 has **no config file**; don't look for `tailwind.config.js`. Everything is `@theme` + CSS.
- **Pagefind index only exists after a build.** `bun run build` runs `pagefind --site dist`; the ⌘K
  modal is live under `bun run preview`/prod but inert in `astro dev` (shows a graceful notice).

---

## What's intentionally unfinished (good first targets)

- **Landing page** (`pages/index.astro`): still a bare centered title + button. This is the hub hero —
  the next design pass owns it. The design handoff's `landing.jsx` / `system-map.jsx` (in the zip)
  are the reference. The site is meant to grow into a project hub (docs are one large module).
- **Theme**: dark-only. If a light theme is wanted, introduce theming (e.g. tokens under
  `:root` / `[data-theme="light"]`) plus a toggle — the token layer is structured to localize this.

Brand reference: logo/banner assets at repo root (`AgentX-Logo-v3-banner.png`, `…-badge.png`); badge
also in `public/`. Primary accent is indigo (`#6366f1`); direction is "glassbox / cosmic dark", Geist type.
