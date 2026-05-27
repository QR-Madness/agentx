# AgentX Docs Site — architecture & design handoff

This is the **AgentX project hub**: an Astro static site with a bare landing page at `/` and
the documentation under `/docs`. It replaced a MkDocs/Material site. This document is written for
the **design pass** — the structure exists; the visuals are deliberately minimal and waiting for you.

> **Your job (design pass):** make it beautiful. The architecture was built so you can do
> **rapid, large-scale visual overhauls from a single token file** without touching content,
> routing, or logic. Read "Design system" and "Rules of the road" first.

---

## TL;DR for the design pass

- **Every color, font, radius, and key dimension lives in one file:** [`src/styles/global.css`](src/styles/global.css).
  Reskinning the whole site = editing the `@theme` block there. Don't hardcode visual values anywhere else.
- **Markdown content is sacred** — `src/content/docs/**/*.md` was migrated as-is. Style it via the
  `.prose` rules in `global.css`; do **not** edit content files to achieve a look.
- **Landing page (`src/pages/index.astro`) is a blank canvas** — intentionally bare. Build the real hero/hub here.
- **Dark-only right now.** No theme toggle. Fonts (`Inter`, `JetBrains Mono`) are *named* in tokens
  but **not actually loaded** — they fall back to system fonts until you add the webfonts.
- After editing any remark/rehype plugin, **`rm -rf .astro dist` before rebuilding** (Astro caches parsed content).

---

## Tech stack

| Concern | Choice |
|---|---|
| Framework | **Astro 6** (static output) |
| Styling | **Tailwind CSS v4** via `@tailwindcss/vite` (CSS-first `@theme`, no `tailwind.config.js`) |
| Markdown | `.md` content collection; **MDX** integration available for future interactive pages |
| Code highlighting | **Shiki** (`github-dark` theme), built into Astro |
| Diagrams | **astro-mermaid** (client-side render, dark theme) |
| Package manager | **bun** |
| Deploy | **Vercel** (static), custom domain `agentx.thejpnet.net` |

Commands (from `docs-site/`): `bun install` · `bun run dev` (→ http://localhost:4321) ·
`bun run build` (→ `dist/`) · `bun run preview`. Also exposed as `task docs:serve|build|preview|deploy`.

---

## Directory map

```
docs-site/
├─ astro.config.mjs          Integrations (mermaid→mdx), Shiki theme, remark plugin wiring, site URL
├─ src/
│  ├─ styles/global.css      ★ THE TOKEN LAYER — all design decisions live here
│  ├─ config/
│  │  ├─ site.ts             Site metadata (name, description, repo, docsBasePath, copyright)
│  │  └─ nav.ts              ★ Docs sidebar structure (ported from old mkdocs.yml nav) + helpers
│  ├─ content.config.ts      `docs` collection: glob loader over content/docs
│  ├─ content/docs/          ★ 27 migrated markdown files (untouched) — the actual docs content
│  ├─ layouts/
│  │  ├─ BaseLayout.astro    <html> shell, <head>, imports global.css — used by every page
│  │  └─ DocsLayout.astro    Docs chrome: Header + MobileNav + (Sidebar | article.prose | TOC) + Footer
│  ├─ components/layout/
│  │  ├─ Header.astro        Sticky top bar (brand + Docs/GitHub links)
│  │  ├─ Sidebar.astro       Renders nav.ts → groups + links
│  │  ├─ SidebarGroup.astro  A labeled section of sidebar links
│  │  ├─ SidebarLink.astro   One link; resolves /docs/<slug>; active-state highlight
│  │  ├─ TableOfContents.astro  Right-rail "On this page" from heading data (depth 2–3)
│  │  ├─ MobileNav.astro     <details> disclosure wrapping Sidebar for < lg screens
│  │  └─ Footer.astro        Copyright + repo link
│  ├─ pages/
│  │  ├─ index.astro         ★ Bare landing at /  (your hero/hub goes here)
│  │  ├─ docs/[...slug].astro Renders every docs entry under /docs (id "index" → /docs)
│  │  └─ 404.astro           Custom not-found
│  └─ plugins/
│     ├─ remark-rewrite-md-links.mjs  Internal *.md links → /docs/... routes
│     └─ remark-admonitions.mjs       MkDocs `!!! type "title"` → <aside class="admonition …">
└─ public/favicon.svg
```

---

## Design system (read this carefully)

### The token layer — `src/styles/global.css`

Tailwind v4 reads design tokens from a CSS `@theme { … }` block. **Defining a variable there both
sets the value and generates the matching utility class.** Current tokens (neutral indigo-on-dark
defaults — change freely):

| Token | Value | Generated utilities | Used for |
|---|---|---|---|
| `--color-bg` | `#0b0d12` | `bg-bg`, `text-bg`… | page background |
| `--color-surface` | `#12151c` | `bg-surface`… | cards, table stripes |
| `--color-surface-2` | `#1a1e27` | `bg-surface-2`… | hover, inline-code bg, table headers |
| `--color-border` | `#262b36` | `border-border`… | all hairlines |
| `--color-code-bg` | `#0e1117` | `bg-code-bg`… | code block background (Shiki override) |
| `--color-text` | `#e6e8ee` | `text-text`… | body text |
| `--color-text-muted` | `#9aa3b2` | `text-text-muted`… | secondary text, nav idle |
| `--color-accent` | `#6366f1` (indigo-500) | `bg-accent`, `text-accent`… | brand, links, active states |
| `--color-accent-hover` | `#818cf8` | `bg-accent-hover`… | hover |
| `--color-accent-subtle` | `#1e1b4b` | `bg-accent-subtle`… | active sidebar pill |
| `--color-accent-fg` | `#ffffff` | `text-accent-fg`… | text on an accent fill |
| `--color-note` / `--color-warning` / `--color-danger` | accent / `#f59e0b` / `#ef4444` | — | admonition accents |
| `--font-sans` / `--font-mono` | Inter… / JetBrains Mono… | `font-sans` / `font-mono` | typography (see fonts note) |
| `--radius-card` / `--radius-sm` | `0.75rem` / `0.375rem` | `rounded-card` / `rounded-sm` | shape |
| `--layout-sidebar-width` / `--layout-toc-width` / `--layout-content-max` | `17rem` / `14rem` / `52rem` | *(none — see below)* | layout dimensions |

**Two mechanics to know:**

1. **`--color-*`, `--font-*`, `--radius-*` are Tailwind "namespaces"** → they auto-generate
   `bg-*`/`text-*`/`border-*`, `font-*`, `rounded-*`. `--layout-*` is **not** a namespace, so it
   generates no utilities; those are referenced as arbitrary CSS-var utilities, e.g.
   `w-(--layout-sidebar-width)`, `max-w-(--layout-content-max)`. Use this same pattern if you add
   non-namespaced tokens.
2. The token names produce a few **awkward utilities** (`bg-bg`, `text-text`). If that bothers you,
   rename the tokens (e.g. `--color-base`, `--color-fg`) — just update the consuming components too.
   Grep is your friend; usages are shallow.

Below the `@theme` block, `global.css` also contains the **`.prose` rules** (styling for rendered
markdown), **`.admonition` rules**, and **`.mermaid` rules** — all written in plain CSS that reads
`var(--token)`. This is where most of the "docs look" lives. Restyle headings, tables, code blocks,
callouts, etc. here.

### Fonts (action needed)

`--font-sans`/`--font-mono` reference **Inter** and **JetBrains Mono**, but **no webfont is loaded** —
they currently fall back to `system-ui` / `ui-monospace`. To actually use them, add the fonts
(e.g. `@fontsource-variable/inter`, imported once in `BaseLayout` or `global.css`) — or pick your
own and update the two tokens. One place to change.

### Layout & components

- **`DocsLayout.astro`** owns the docs page frame: a centered `max-w-[88rem]` flex row of
  `Sidebar | main(article.prose) | TableOfContents`, with a sticky `Header` and a `Footer`. Sidebar
  hides below `lg`; TOC hides below `xl`; `MobileNav` (a `<details>`) covers small screens.
- Components are intentionally **small and single-purpose** so you can restructure freely. Sidebar
  is split Sidebar → SidebarGroup → SidebarLink; swap the markup/classes without touching nav data.
- **Active states / routing helpers** live in `nav.ts` (`slugToHref`, `labelForSlug`, `isGroup`) —
  reuse them; don't hand-roll URL logic.

### Special content rendering

- **Code blocks:** Shiki emits `<pre class="astro-code github-dark">…`. Its theme paints token
  colors; we override only the *background* to `--color-code-bg` (`!important`, because Shiki sets an
  inline bg). To change the syntax theme entirely, edit `markdown.shikiConfig.theme` in
  `astro.config.mjs` (single or dual themes supported).
- **Admonitions:** rendered as `<aside class="admonition admonition-{note|warning|danger|…}">` with a
  `<p class="admonition-title">`. Per-type accent is driven by `--admonition-accent` set per class in
  `global.css`. Add new types by adding a `.admonition-<type>` rule.
- **Mermaid:** `astro-mermaid` turns ` ```mermaid ` fences into `<pre class="mermaid">` rendered
  client-side with `theme: 'dark'`. To match diagram colors to the brand, pass `mermaidConfig`
  (theme variables) to the integration in `astro.config.mjs`. Style the container via `.prose pre.mermaid`.

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
   from `../docs`. Achieve looks via `.prose`/components, not by touching markdown.
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

---

## What's intentionally unfinished (good first targets)

- **Landing page** (`pages/index.astro`): currently a centered title + button. This is the hub hero —
  design it fully. The site is meant to grow into a project hub (docs are one large module).
- **Theme**: dark-only. If a light theme is wanted, introduce theming (e.g. tokens under
  `:root` / `[data-theme="light"]`) plus a toggle — the token layer is structured to make this a
  localized change.
- **Webfonts**: load Inter / JetBrains Mono (or your picks) — see Fonts note.
- **Header/Footer/TOC/Sidebar** styling is functional but plain — all open for visual treatment.

Brand reference: logo/banner assets live at repo root (`AgentX-Logo-v3-banner.png`, `…-badge.png`);
primary accent is indigo (`#6366f1`); aesthetic direction is "glassbox / cosmic dark."
