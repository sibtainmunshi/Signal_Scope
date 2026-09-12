# Frontend QA fixes

Applied 12 September 2026 by Claude Code, against the findings in
`docs/CLAUDE_QA_REVIEW.md`. Frontend only. Backend, model, dependencies,
`app/frontend/scripts/smoke.mjs`, README and handoff files were not touched, and
nothing was committed or pushed. The shared server on `http://127.0.0.1:8002`
was never restarted or stopped.

Files changed (the four I own) plus the build output they produce:

| Path | Change |
|---|---|
| `app/frontend/src/main.tsx` | Logic and accessibility fixes (+67 / -21 across the three source files) |
| `app/frontend/src/style.css` | Contrast, type scale, tap targets, skip-link and focus styles |
| `app/frontend/index.html` | One line: favicon link |
| `app/frontend/public/favicon.svg` | **New** (506 B) — Vite copies it to `dist/` |

## Build output — `dist/` is git-tracked

`npm run build` (`tsc -b && vite build`) succeeded with no type errors, 1,578
modules. Because `app/frontend/dist/` is tracked, the build changed tracked
files. **Left uncommitted for Codex to review and commit:**

```
 D app/frontend/dist/assets/index-Bs6f0GON.js     (superseded)
 D app/frontend/dist/assets/index-CeeHV6Kk.css    (superseded)
 M app/frontend/dist/index.html                   (favicon link + new asset hashes)
?? app/frontend/dist/assets/index-CSqkczBm.js     (new, 248,555 B)
?? app/frontend/dist/assets/index-BCLcdUCP.css    (new, 17,751 B)
?? app/frontend/dist/favicon.svg                  (new)
?? app/frontend/public/                           (new source directory)
```

The running server picked the rebuild up without a restart — Starlette's
`StaticFiles` reads from disk per request. Verified: the HTML served from
`127.0.0.1:8002` references `index-CSqkczBm.js` / `index-BCLcdUCP.css`, matching
what is on disk, and `/favicon.svg` returns HTTP 200 `image/svg+xml`.

## What was fixed

Numbering follows `docs/CLAUDE_QA_REVIEW.md`.

**2 — Rejected replacement no longer discards the previous analysis.**
`selectFile` now validates type and size *before* clearing any state
(`main.tsx:97-104`). Previously `setError(''); setAnalysis(null)` ran first, so
picking an unsupported file destroyed a completed result.

**3 — Tabs now follow the ARIA tabs pattern.** All three tabs carry
`id`/`aria-controls`; the panel has a stable `id="evidence-panel"`,
`aria-labelledby` tracking the selected tab, and `tabIndex=0`. Added roving
tabindex and `ArrowLeft`/`ArrowRight`/`Home`/`End` handling
(`main.tsx:139-150`, `main.tsx:196-197`). `aria-controls` points at an element
that always exists, so no reference dangles.

**4 and 5 — Contrast and type scale.** Reworked the muted palette and raised the
type scale throughout `style.css`: micro-labels 7-9px → 10-11px, body copy
10-12px → 12-13px. Every previously failing colour was darkened toward the same
sage/green family so the visual design is unchanged in character. The disabled
Analyze button was also improved (2.09 → 3.63) even though WCAG exempts disabled
controls. The overlay pill was made fully opaque (`#16241f`) so its legibility no
longer depends on the image behind it.

**6 — Remove clears the error.** The X button now calls a `clearImage` helper
that also resets the error and the announcement (`main.tsx:106-108`).

**8 — Skip link and `aria-current`.** Added a real skip link as the first
focusable element, targeting `<main id="main-content" tabIndex={-1}>`, and
`aria-current="page"` on the active navigation button
(`main.tsx:158`, `:177-179`, `:187`).

**9 — Focused live announcement.** Removed `aria-live="polite"` from the whole
results panel. A single visually-hidden `role="status"` element now carries one
concise sentence (`main.tsx:185`), so a tab switch no longer re-announces the
entire analysis.

**10 — Same-file re-selection (defensive).** `openFilePicker` clears
`input.value` before opening the dialog, and a rejected file clears it too
(`main.tsx:88-95`). **This remains an unreproduced finding** — it was reasoned
from source and standard browser behaviour, never reproduced through the
harness, and the change is defensive hardening, not a confirmed bug fix.

**11 — Tap targets.** "Change image" 57x12 → 63x30; option checkboxes 14 → 17px
inside 36px-high labels; the overlay toggle, remove button and download button
are all at least 30x30.

**12 — favicon.** Added `public/favicon.svg` (a mark matching the sidebar logo)
and linked it, removing the 404 that fired on every page load.

## Left for Codex (backend / model / docs — not mine)

Findings **1** (EXIF orientation vs reported dimensions), **7** (zero-byte
upload message), **13** (multi-frame WebP), **14** (degenerate inputs), **15**
(`GET /api/predict` 404 vs 405), and **16-18** (documentation) all live in
`app/backend/`, `src/signalscope/`, README or `docs/API_CONTRACT.md`.

Finding 1 is worth flagging: it surfaces in the UI at `main.tsx:190`, which
renders `image_width x image_height` straight from the API. No frontend change
was made — once the backend reports EXIF-oriented dimensions, the UI is correct
automatically.

## Verification

Two harnesses in `tmp/claude_qa/`, both read-only against the running server:
`fix_regression.mjs` (function + regression) and `contrast_recheck.mjs`
(steady-state accessibility audit). Raw output: `fix_results.json`,
`contrast_recheck.json`, `screenshots-after/`, `exported-analysis-after.json`.
Both were re-run against the **final** build after the last CSS change.

### Fix assertions — all passed

| Assertion | Result |
|---|---|
| Rejected replacement preserves the verdict | `"Likely real"` before and after; file details unchanged |
| Rejected replacement still shows the error | "Choose a JPEG, PNG or WebP image." |
| Rejected file does not stay in the input | `input.value === ""` |
| Remove clears the error | `error === null`, panel back to empty state |
| Picker opens with a cleared input | `value === ""` at `filechooser`; same file re-selected successfully |

### Accessibility

- **Tabs:** `aria-controls="evidence-panel"` on all three; panel
  `id="evidence-panel"`, `aria-labelledby="tab-evidence"`, `tabIndex=0`; roving
  tabindex `0 / -1 / -1`. ArrowRight cycles Evidence → Stability → Metadata →
  Evidence; ArrowLeft, Home and End all verified; a single Tab leaves the
  tablist and lands on the panel.
- **Skip link:** first tab stop, `href="#main-content"`, offscreen at
  `top:-63px` until focused; Enter moves focus to `#main-content`.
- **`aria-current`:** `"page"` on the active item only.
- **Live region:** `.results-panel` `aria-live` is now `null`; one
  `role="status"` region announced
  *"Analysis complete. Likely real. AI-generated score 26.3%, decision threshold 45.5%."*
- **Steady-state audit** (reduced motion, settled, pointer moved off all
  controls) across 8 page/viewport combinations — desktop analyze, Stability
  tab, Metadata tab, report, about; mobile analyze, report, about:

  **0 contrast failures, 0 text below 10px, 0 tap targets below 24x24.**

### One measurement caveat, stated plainly

The default-motion harness still reports `.analyze-button` at 4.03:1. That is a
sampling artifact, not a real failure: the button has `transition:background .2s`
and the audit samples it while the background interpolates from the disabled to
the enabled colour as analysis finishes. Measured at rest with transitions
disabled, it is **9.74:1** (`rgb(220,246,229)` on `rgb(27,66,57)`) — recorded in
`contrast_recheck.json` under `probes`. The earlier `.overlay-toggle` reading of
1.05:1 was likewise an artifact of the first auditor skipping backgrounds below
0.95 alpha; `contrast_recheck.mjs` composites translucent layers properly and
measures it at 14.71:1. No CSS change was made on account of the 4.03 reading.

### Regression — nothing broken

- Upload → analyze → verdict works; `/api/predict` 200.
- JSON export: 337,879 bytes, valid, `filename` + `prediction` + `explanation` +
  `robustness` + `metadata` all intact.
- All five error paths still render correct, readable text with `role="alert"`
  (415 unsupported format, 400 undecodable, 413 megapixels, 413 narrow canvas,
  client-side 10 MiB).
- Model report: no stale "remain unevaluated" text, no "Pending" cells,
  checkpoint SHA-256 and both reserved AUCs (0.5651 / 0.6430) render.
- No horizontal overflow at 390px on any of six views.
- **0 console errors** on clean load (the favicon 404 is gone).
- **Codex's unmodified `smoke.mjs` still passes** against the new build:
  `ok: true`, explanation present, 6 transformations, no JavaScript errors, no
  mobile overflow. Run with `SIGNALSCOPE_URL` / `SIGNALSCOPE_FIXTURE` /
  `SIGNALSCOPE_QA_DIR=tmp/claude_qa/smoke-check` so no existing artifact was
  overwritten.
- **CSS integrity:** the rewritten stylesheet was diffed selector-by-selector
  against the committed version — **0 selectors removed**, 8 added
  (`.skip-link`, `.skip-link:focus-visible`, `main:focus`,
  `[role=tabpanel]:focus-visible`, `.nav-item .nav-text`, and three `:hover`
  rules).

Screenshots before and after are in `tmp/claude_qa/screenshots/` and
`tmp/claude_qa/screenshots-after/`.

## Notes for integration

- Codex began editing `app/backend/main.py`, `src/signalscope/inference.py`,
  `src/signalscope/evidence.py`, `tests/` and `docs/API_CONTRACT.md` while this
  work was in progress. The uvicorn process was not restarted, so those Python
  changes were not live during these checks; all evidence above is frontend-only
  and unaffected.
- Synthetic fixtures in `tmp/claude_qa/fixtures/` test input handling only.
  Nothing here is evidence about detection accuracy.
- `docs/PROGRESS.md`, `docs/SUBMISSION_CHECKLIST.md` and `docs/AGENT_HANDOFF.md`
  were deliberately left untouched for Codex, per the bounded scope.
