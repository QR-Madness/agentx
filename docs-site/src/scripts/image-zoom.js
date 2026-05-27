/* ============================================================================
 * image-zoom — fluid pan + zoom overlay for diagrams and images.
 *
 * Plain JS · zero deps. Drop this file in and add `data-zoomable` (or class
 * `.ax-zoomable`) to any element; Mermaid diagrams (`.mermaid`) are picked up
 * automatically.
 *
 * UX:
 *   - click          → open overlay with content fit to viewport
 *   - drag           → pan
 *   - wheel / pinch  → cursor-anchored zoom (smooth, inertia-free)
 *   - double-click   → toggle 1x ⇄ 2x at click point
 *   - + / - / 0      → zoom in / out / reset
 *   - esc            → close (also background click)
 *
 * In React/Astro: just include `<script src="image-zoom.js"></script>` once
 * per page that has diagrams or images. The script attaches on DOMContentLoaded
 * and works with content rendered later (it observes the DOM).
 *
 * Vendored from the "claude design" handoff. Local changes:
 *   - SELECTOR also matches `.ax-prose img` (content images become zoomable).
 *   - Zoom is realized via the content's rendered width/height (vector-crisp for
 *     SVG diagrams) and the stage transform pans only — a CSS `scale()` on a
 *     compositor layer rasterizes once and upsamples, which looked blurry.
 * ========================================================================== */
(function () {
  "use strict";

  // ── Tunables ─────────────────────────────────────────────────────────
  const MIN_SCALE = 0.4;
  const MAX_SCALE = 12;
  const WHEEL_SENSITIVITY = 0.0018;     // wheel-pixels → scale-delta-log
  const DOUBLE_CLICK_ZOOM = 2.4;        // double-click zoom multiplier
  const CURSOR_HINT_HIDE_MS = 2200;
  const SELECTOR = ".mermaid, .ax-prose img, .ax-zoomable, [data-zoomable]";

  // ── Styles (injected once) ───────────────────────────────────────────
  function injectStyles() {
    if (document.getElementById("__ax-image-zoom-styles")) return;
    const css = `
      .ax-zoom-affordance {
        cursor: zoom-in;
        position: relative;
        transition: outline-color .15s ease;
        outline: 1px solid transparent;
        outline-offset: -1px;
      }
      .ax-zoom-affordance:hover {
        outline-color: color-mix(in oklab, var(--color-accent, #6366f1) 35%, transparent);
      }
      .ax-zoom-hint {
        position: absolute;
        top: 10px; right: 10px;
        padding: 4px 8px;
        font-family: var(--font-mono, ui-monospace, monospace);
        font-size: 10.5px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--color-text-dim, #5a6478);
        background: color-mix(in oklab, var(--color-bg, #0a0d14) 70%, transparent);
        border: 1px solid var(--color-border, #262b38);
        border-radius: 999px;
        backdrop-filter: blur(6px);
        opacity: 0;
        transform: translateY(-2px);
        transition: opacity .15s ease, transform .15s ease;
        pointer-events: none;
        z-index: 2;
      }
      .ax-zoom-affordance:hover .ax-zoom-hint {
        opacity: 1;
        transform: translateY(0);
      }

      /* Overlay */
      .ax-zoom-overlay {
        position: fixed; inset: 0;
        z-index: 9999;
        background: color-mix(in oklab, var(--color-bg, #07080c) 92%, transparent);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        display: flex;
        align-items: center; justify-content: center;
        opacity: 0;
        transition: opacity .18s ease;
        overscroll-behavior: contain;
        touch-action: none;
      }
      .ax-zoom-overlay[data-open="true"] { opacity: 1; }

      /* dotted grid background, very subtle */
      .ax-zoom-overlay::before {
        content: "";
        position: absolute; inset: 0;
        pointer-events: none;
        background-image:
          radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: radial-gradient(70% 70% at 50% 50%, black, transparent 90%);
        -webkit-mask-image: radial-gradient(70% 70% at 50% 50%, black, transparent 90%);
      }

      .ax-zoom-stage {
        position: absolute;
        left: 50%; top: 50%;
        transform-origin: 0 0;
        will-change: transform;
        user-select: none;
        -webkit-user-select: none;
        cursor: grab;
      }
      .ax-zoom-stage.dragging { cursor: grabbing; }
      .ax-zoom-stage > * {
        display: block;
        max-width: none !important;
        max-height: none !important;
      }
      .ax-zoom-stage svg {
        background: transparent;
      }
      .ax-zoom-stage img {
        display: block;
      }
      /* mermaid clones use a code-bg pane to read against the overlay */
      .ax-zoom-stage[data-kind="mermaid"] {
        background: var(--color-code-bg, #0a0d14);
        border: 1px solid var(--color-border, #262b38);
        border-radius: 12px;
        padding: 32px;
      }

      /* Top bar */
      .ax-zoom-topbar {
        position: absolute;
        top: 16px; left: 0; right: 0;
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 20px;
        font-family: var(--font-mono, ui-monospace, monospace);
        font-size: 11.5px;
        color: var(--color-text-muted, #8b94a5);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        pointer-events: none;
      }
      .ax-zoom-topbar > * { pointer-events: auto; }

      .ax-zoom-label {
        padding: 6px 12px;
        background: color-mix(in oklab, var(--color-surface, #11141c) 80%, transparent);
        border: 1px solid var(--color-border, #262b38);
        border-radius: 999px;
        color: var(--color-text, #e6e8ee);
        display: inline-flex; align-items: center; gap: 8px;
      }
      .ax-zoom-label .dot {
        width: 6px; height: 6px; border-radius: 999px;
        background: var(--c-memory, #22d3ee);
        box-shadow: 0 0 6px var(--c-memory, #22d3ee);
      }

      .ax-zoom-close {
        appearance: none;
        background: color-mix(in oklab, var(--color-surface, #11141c) 80%, transparent);
        border: 1px solid var(--color-border, #262b38);
        color: var(--color-text, #e6e8ee);
        width: 32px; height: 32px;
        border-radius: 999px;
        display: inline-flex; align-items: center; justify-content: center;
        cursor: pointer;
        font-size: 14px;
        line-height: 1;
        transition: background .15s ease, color .15s ease, border-color .15s ease;
      }
      .ax-zoom-close:hover {
        color: var(--color-text-strong, #fff);
        border-color: var(--color-border-2, #343a4c);
      }

      /* Bottom toolbar */
      .ax-zoom-toolbar {
        position: absolute;
        bottom: 24px;
        left: 50%; transform: translateX(-50%);
        display: inline-flex; align-items: center; gap: 4px;
        padding: 6px;
        background: color-mix(in oklab, var(--color-surface, #11141c) 85%, transparent);
        border: 1px solid var(--color-border, #262b38);
        border-radius: 999px;
        font-family: var(--font-mono, ui-monospace, monospace);
        font-size: 11px;
        letter-spacing: 0.08em;
        color: var(--color-text-muted, #8b94a5);
        backdrop-filter: blur(8px);
      }
      .ax-zoom-btn {
        appearance: none;
        background: transparent;
        border: 0;
        color: var(--color-text-muted, #8b94a5);
        width: 30px; height: 30px;
        border-radius: 999px;
        cursor: pointer;
        display: inline-flex; align-items: center; justify-content: center;
        transition: background .15s ease, color .15s ease;
      }
      .ax-zoom-btn:hover {
        background: var(--color-surface-2, #181c27);
        color: var(--color-text-strong, #fff);
      }
      .ax-zoom-scale {
        min-width: 56px;
        text-align: center;
        padding: 0 10px;
        color: var(--color-text, #e6e8ee);
        border-left: 1px solid var(--color-border, #262b38);
        border-right: 1px solid var(--color-border, #262b38);
      }
      .ax-zoom-key {
        padding: 0 10px;
        opacity: 0.7;
      }
      .ax-zoom-key kbd {
        font-family: inherit;
        font-size: 10px;
        padding: 1px 5px;
        margin: 0 2px;
        background: var(--color-surface-2, #181c27);
        border: 1px solid var(--color-border, #262b38);
        border-radius: 3px;
        color: var(--color-text, #e6e8ee);
      }
    `;
    const style = document.createElement("style");
    style.id = "__ax-image-zoom-styles";
    style.textContent = css;
    document.head.appendChild(style);
  }

  // ── Pick affordance label per element kind ───────────────────────────
  function affordanceLabelFor(el) {
    if (el.classList.contains("mermaid")) return "diagram · click to zoom";
    if (el.tagName === "IMG")             return "click to zoom";
    return "click to zoom";
  }

  // Add hover affordance + hint chip to a candidate element.
  function decorate(el) {
    if (el.__axZoomDecorated) return;
    el.__axZoomDecorated = true;
    el.classList.add("ax-zoom-affordance");
    const hint = document.createElement("span");
    hint.className = "ax-zoom-hint";
    hint.textContent = affordanceLabelFor(el);
    // Mermaid containers are usually `display:block; text-align:center`; we
    // want the chip positioned relative to the box.
    if (getComputedStyle(el).position === "static") {
      el.style.position = "relative";
    }
    el.appendChild(hint);

    el.addEventListener("click", function (ev) {
      // Mermaid SVGs may have internal interactive elements; we still capture.
      if (ev.target.closest("a[href]")) return; // honor explicit links
      ev.preventDefault();
      openOverlay(el);
    });
  }

  // ── Find/track candidates ────────────────────────────────────────────
  function scan(root) {
    root.querySelectorAll(SELECTOR).forEach(decorate);
  }

  // Observe DOM for late-rendered Mermaid SVGs.
  function observe() {
    const mo = new MutationObserver(muts => {
      for (const m of muts) {
        m.addedNodes.forEach(n => {
          if (n.nodeType !== 1) return;
          if (n.matches && n.matches(SELECTOR)) decorate(n);
          if (n.querySelectorAll)              scan(n);
        });
      }
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  // ── Overlay machinery ────────────────────────────────────────────────
  let activeOverlay = null;

  function openOverlay(srcEl) {
    if (activeOverlay) return;

    const overlay = document.createElement("div");
    overlay.className = "ax-zoom-overlay";

    // top bar with label + close
    const topbar = document.createElement("div");
    topbar.className = "ax-zoom-topbar";
    const label = document.createElement("span");
    label.className = "ax-zoom-label";
    label.innerHTML = `<span class="dot"></span><span>${
      srcEl.classList.contains("mermaid") ? "Diagram" : "Image"
    }</span>`;
    const closeBtn = document.createElement("button");
    closeBtn.className = "ax-zoom-close";
    closeBtn.setAttribute("aria-label", "Close");
    closeBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;
    topbar.appendChild(label);
    topbar.appendChild(closeBtn);

    // stage = the cloned content, transformed
    const stage = document.createElement("div");
    stage.className = "ax-zoom-stage";

    // Clone src content. For mermaid, prefer the rendered SVG. `baseW`/`baseH`
    // capture the intrinsic size; zoom is applied by scaling the content's
    // rendered width/height (crisp for vectors), not via a CSS transform scale.
    const isMermaid = srcEl.classList.contains("mermaid");
    const svg = isMermaid ? srcEl.querySelector("svg") : null;
    let stageContent;
    let baseW;
    let baseH;
    if (svg) {
      stage.setAttribute("data-kind", "mermaid");
      stageContent = svg.cloneNode(true);
      const bbox = svg.getBoundingClientRect();
      baseW = bbox.width;
      baseH = bbox.height;
      // Drive size via CSS width/height; keep aspect through the viewBox so the
      // vectors re-rasterize crisply at any scale.
      if (!stageContent.getAttribute("viewBox") && baseW && baseH) {
        stageContent.setAttribute("viewBox", `0 0 ${baseW} ${baseH}`);
      }
      stageContent.removeAttribute("width");
      stageContent.removeAttribute("height");
      stageContent.style.maxWidth = "none";
      stageContent.style.maxHeight = "none";
    } else if (srcEl.tagName === "IMG") {
      stageContent = srcEl.cloneNode(true);
      const r = srcEl.getBoundingClientRect();
      baseW = srcEl.naturalWidth || r.width;
      baseH = srcEl.naturalHeight || r.height;
    } else {
      stageContent = srcEl.cloneNode(true);
      const r = srcEl.getBoundingClientRect();
      baseW = r.width;
      baseH = r.height;
    }
    // strip the hint from any cloned host
    stageContent.querySelectorAll?.(".ax-zoom-hint").forEach(n => n.remove());
    stage.appendChild(stageContent);

    // Bottom toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "ax-zoom-toolbar";
    toolbar.innerHTML = `
      <button class="ax-zoom-btn" data-act="out" aria-label="Zoom out"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3 8h10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
      <div class="ax-zoom-scale">100%</div>
      <button class="ax-zoom-btn" data-act="in" aria-label="Zoom in"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M8 3v10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
      <button class="ax-zoom-btn" data-act="reset" aria-label="Reset"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M4 7V4h3M12 9v3H9M4 4l3.5 3.5M12 12L8.5 8.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
      <div class="ax-zoom-key">
        <kbd>scroll</kbd> zoom · <kbd>drag</kbd> pan · <kbd>esc</kbd> close
      </div>
    `;

    overlay.appendChild(topbar);
    overlay.appendChild(stage);
    overlay.appendChild(toolbar);
    document.body.appendChild(overlay);

    // Lock background scroll while overlay is open
    const prevBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // State + transform
    const state = { scale: 1, tx: 0, ty: 0, fitScale: 1 };
    const scaleReadout = toolbar.querySelector(".ax-zoom-scale");

    function applyTransform() {
      // Scale through the content's rendered box (vector-crisp), pan via transform.
      stageContent.style.width = baseW * state.scale + "px";
      stageContent.style.height = baseH * state.scale + "px";
      stage.style.transform = `translate(${state.tx}px, ${state.ty}px)`;
      scaleReadout.textContent = Math.round((state.scale / state.fitScale) * 100) + "%";
    }

    // Center the (scaled) content in the viewport.
    function center() {
      state.tx = -(baseW * state.scale) / 2;
      state.ty = -(baseH * state.scale) / 2;
    }

    // Fit the intrinsic content size to the viewport.
    function fitToViewport() {
      const padding = 80;
      const vw = window.innerWidth - padding * 2;
      const vh = window.innerHeight - padding * 2;
      const fit = Math.min(vw / baseW, vh / baseH, 1.6); // never auto-zoom past 1.6×
      state.scale = fit;
      state.fitScale = fit;
      center();
      applyTransform();
    }

    // Cursor-anchored zoom. cx, cy in viewport coordinates.
    function zoomAt(cx, cy, factor) {
      const next = Math.min(MAX_SCALE * state.fitScale, Math.max(MIN_SCALE * state.fitScale, state.scale * factor));
      if (next === state.scale) return;
      // stage anchor at viewport center: (innerWidth/2 + tx, innerHeight/2 + ty)
      const sx = window.innerWidth  / 2 + state.tx;
      const sy = window.innerHeight / 2 + state.ty;
      // point in stage-local coords before zoom:
      const px = (cx - sx) / state.scale;
      const py = (cy - sy) / state.scale;
      // recompute tx/ty so (px,py) lands on (cx,cy) under next scale
      state.tx = cx - window.innerWidth  / 2 - px * next;
      state.ty = cy - window.innerHeight / 2 - py * next;
      state.scale = next;
      applyTransform();
    }

    // ── Pointer pan + pinch ────────────────────────────────────────────
    const pointers = new Map();
    let pinchStart = null;

    stage.addEventListener("pointerdown", e => {
      stage.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      stage.classList.add("dragging");
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchStart = {
          dist: Math.hypot(a.x - b.x, a.y - b.y),
          mid:  { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
          scale: state.scale,
        };
      }
    });

    stage.addEventListener("pointermove", e => {
      if (!pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId);
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.size === 1) {
        state.tx += dx; state.ty += dy;
        applyTransform();
      } else if (pointers.size === 2 && pinchStart) {
        const [a, b] = [...pointers.values()];
        const dist  = Math.hypot(a.x - b.x, a.y - b.y);
        const mid   = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        const factor = (dist / pinchStart.dist) * (pinchStart.scale / state.scale);
        zoomAt(mid.x, mid.y, factor);
      }
    });

    function endPointer(e) {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinchStart = null;
      if (pointers.size === 0) stage.classList.remove("dragging");
    }
    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);
    stage.addEventListener("pointerleave", endPointer);

    // ── Wheel zoom (cursor anchored) ──────────────────────────────────
    overlay.addEventListener("wheel", e => {
      e.preventDefault();
      // pinch trackpad → ctrlKey on wheel events; scale aggressively
      const sensitivity = e.ctrlKey ? 0.005 : WHEEL_SENSITIVITY;
      const factor = Math.exp(-e.deltaY * sensitivity);
      zoomAt(e.clientX, e.clientY, factor);
    }, { passive: false });

    // ── Double click → toggle zoom ────────────────────────────────────
    stage.addEventListener("dblclick", e => {
      e.preventDefault();
      const target = (state.scale > state.fitScale * 1.4)
        ? state.fitScale
        : state.fitScale * DOUBLE_CLICK_ZOOM;
      zoomAt(e.clientX, e.clientY, target / state.scale);
    });

    // ── Toolbar buttons ───────────────────────────────────────────────
    toolbar.addEventListener("click", e => {
      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      const cx = window.innerWidth / 2;
      const cy = window.innerHeight / 2;
      if (btn.dataset.act === "in")    zoomAt(cx, cy, 1.4);
      if (btn.dataset.act === "out")   zoomAt(cx, cy, 1 / 1.4);
      if (btn.dataset.act === "reset") {
        state.scale = state.fitScale;
        center();
        applyTransform();
      }
    });

    // ── Keyboard ──────────────────────────────────────────────────────
    function onKey(e) {
      if (e.key === "Escape") close();
      else if (e.key === "+" || e.key === "=") zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1.4);
      else if (e.key === "-" || e.key === "_") zoomAt(window.innerWidth / 2, window.innerHeight / 2, 1 / 1.4);
      else if (e.key === "0") {
        state.scale = state.fitScale;
        center();
        applyTransform();
      }
    }
    window.addEventListener("keydown", onKey);

    // ── Background click closes ───────────────────────────────────────
    overlay.addEventListener("pointerdown", e => {
      // only when the bare overlay is clicked (not stage/toolbar/topbar)
      if (e.target === overlay) close();
    });
    closeBtn.addEventListener("click", close);

    function close() {
      window.removeEventListener("keydown", onKey);
      overlay.removeAttribute("data-open");
      overlay.addEventListener("transitionend", () => {
        overlay.remove();
        document.body.style.overflow = prevBodyOverflow;
        activeOverlay = null;
      }, { once: true });
    }

    activeOverlay = overlay;

    // Defer one frame so layout settles, fit, then fade in.
    requestAnimationFrame(() => {
      fitToViewport();
      requestAnimationFrame(() => {
        overlay.setAttribute("data-open", "true");
      });
    });

    // Re-fit on resize
    const onResize = () => {
      // Preserve relative zoom: keep state.scale / state.fitScale ratio
      const ratio = state.scale / state.fitScale;
      const padding = 80;
      const vw = window.innerWidth  - padding * 2;
      const vh = window.innerHeight - padding * 2;
      state.fitScale = Math.min(vw / baseW, vh / baseH, 1.6);
      state.scale    = state.fitScale * ratio;
      center();
      applyTransform();
    };
    window.addEventListener("resize", onResize);
    // tear down resize listener when overlay closes
    overlay.__cleanup = () => window.removeEventListener("resize", onResize);
  }

  // Bootstrap
  function init() {
    injectStyles();
    scan(document);
    observe();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Expose a manual hook for things outside MutationObserver reach
  window.axImageZoom = {
    decorate,
    scan,
    open: openOverlay,
  };
})();
