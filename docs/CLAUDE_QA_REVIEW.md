# Independent UI / API QA review

Bounded parallel review by Claude Code, 12 September 2026, against the shared
running instance on `http://127.0.0.1:8002` (server left running; never restarted).
Application code, dependencies, model, manifest and existing reports were not
modified. No training, reserved-test evaluation, dataset download or git operation
was run. Findings are reported for Codex to integrate; nothing was fixed here.

Scope covered: desktop (1440x1050) and 390 px mobile flows (upload, image
replacement, tabs, JSON export, Model report), keyboard navigation, focus
visibility, labels and error messages, generated corrupt/unsupported/EXIF
fixtures, and README / `docs/API_CONTRACT.md` against actual behaviour.

**Fixtures are synthetic and test input handling only. Nothing in this document
is evidence about detection accuracy, and the scores quoted below must not be
used as accuracy or benchmark evidence.**

Reproduction artifacts: `tmp/claude_qa/` (`make_fixtures.py`, `api_probe.py`,
`ui_qa.mjs`, `fixtures/`, `api_results.json`, `ui_results.json`, `screenshots/`).

## Summary

| # | Severity | Finding |
|---|---|---|
| 1 | Medium | Reported image dimensions ignore EXIF orientation; UI and exported JSON contradict the analysed and displayed image |
| 2 | Medium | A rejected replacement image silently discards the previous completed analysis |
| 3 | Medium | Tab pattern incomplete: no `aria-controls`/`aria-labelledby`, and arrow keys do not move tab selection |
| 4 | Medium | Widespread text contrast below WCAG 2.1 AA (worst measured 1.58:1) |
| 5 | Medium | 7-9 px type used throughout, including at 390 px |
| 6 | Low | Error message persists after the image is removed |
| 7 | Low | Zero-byte upload returns "Upload an image first." |
| 8 | Low | No skip link; navigation never sets `aria-current` |
| 9 | Low | Entire results panel is a single `aria-live="polite"` region |
| 10 | Low | "Change image" cannot re-select the same file (reasoned from source, not harness-reproduced) |
| 11 | Low | Mobile "Change image" tap target is 57x12 px |
| 12 | Low | `/favicon.ico` returns 404 on every page load |
| 13 | Info | Multi-frame WebP accepted and silently scored as frame 1 while animated GIF is rejected |
| 14 | Info | Degenerate inputs (flat colour, 1x1) produce confident verdicts with no low-information guard |
| 15 | Info | `GET /api/predict` returns 404 rather than 405 |
| 16 | Doc | README "EXIF-oriented" wording vs reported dimensions (see finding 1) |
| 17 | Doc | EXIF allow-list is undocumented |
| 18 | Doc | Validation errors return a list-valued `detail` the UI cannot render |

---

## 1. Reported image dimensions ignore EXIF orientation — Medium

**Expected.** The dimensions shown next to the image and stored in the exported
JSON describe the image that is displayed and analysed.

**Actual.** They describe the *stored* pixel buffer before EXIF orientation is
applied. Browsers render EXIF-oriented, and the detector analyses EXIF-oriented,
so for any rotated photo the number shown disagrees with both.

**Reproduction.** Upload `tmp/claude_qa/fixtures/exif_orient_6_portrait.jpg`
(stored 120x240, `Orientation=6`, displays as 240x120 landscape).

- UI "Your image" detail line shows `120 x 240 px`, while the preview beside it
  is visibly landscape.
- `POST /api/predict` returns `image_width: 120, image_height: 240`.
- The same mismatch occurs for `exif_orient_6.jpg` and `exif_orient_8.jpg`
  (API `[240, 120]`, oriented `[120, 240]`).

**Cause.** `Prediction` is built from the untransposed image at
`src/signalscope/inference.py:129`, whereas `Detector.tensor()` applies
`ImageOps.exif_transpose` at `src/signalscope/inference.py:70`. The value is
rendered at `app/frontend/src/main.tsx:135` and exported by
`app/frontend/src/main.tsx:101`.

**Not affected.** The 20-megapixel geometry gate in `app/backend/main.py:133`
also uses untransposed dimensions, but `native_canvas_size` depends only on the
width x height product, so the gate is orientation-independent. The explanation
region is computed after transposition (`src/signalscope/evidence.py:75`) and is
consistent. Only the reported dimensions are wrong.

**Impact.** The exported JSON is a submission artifact; it records geometry that
disagrees with the image actually scored.

**Evidence.** `tmp/claude_qa/api_results.json`, section "DIMENSION REPORTING".

## 2. Rejected replacement image discards the previous analysis — Medium

**Expected.** Choosing an unsupported file as a replacement shows an error and
leaves the existing valid image and its completed analysis intact.

**Actual.** The completed analysis is destroyed before the new file is validated.
The user is left with the old image still in the preview, its old file details,
an error message, and an empty Analysis panel.

**Reproduction.**
1. Upload `fixtures/valid_plain.png` and run Analyze; a verdict appears.
2. Click "Change image" and choose `fixtures/unsupported.gif`.

Observed: error "Choose a JPEG, PNG or WebP image."; preview still shows
`valid_plain.png`; file details still read "6 KB / Ready for analysis"; the
Analysis panel has reverted to the empty state "A clearer picture starts here."
The previous result is unrecoverable without re-running.

**Cause.** `selectFile` clears state before validating:
`setError(''); setAnalysis(null)` at `app/frontend/src/main.tsx:77` runs ahead of
the type and size checks at `app/frontend/src/main.tsx:78-79`.

**Evidence.** `tmp/claude_qa/screenshots/desktop-rejected-replacement.png`;
`ui_results.json` -> `imageReplacement.afterRejectedReplacement`.

## 3. Tab pattern is not keyboard/AT complete — Medium

**Expected.** A `role="tablist"` implements the ARIA authoring-practices pattern:
tabs reference their panel, the panel references its tab, and Left/Right arrows
move selection.

**Actual.** Measured wiring: every `role="tab"` has `aria-controls: null` and no
`id`; the `role="tabpanel"` has no `id`, no `aria-labelledby`, and `tabIndex=-1`.
Pressing ArrowRight with the "Evidence" tab focused leaves selection and focus on
"Evidence".

**Reproduction.** Run an analysis, focus the "Evidence" tab, press ArrowRight.
Expected selection "Stability"; actual selection "Evidence".

**Working.** Tab + Enter does switch tabs (verified: selection became
"Stability"), and `aria-selected` is maintained correctly, so the tabs are
operable by keyboard — this is a conformance and efficiency gap, not a block.

**File.** `app/frontend/src/main.tsx:145-146`.

**Evidence.** `ui_results.json` -> `tabs.wiring`, `tabs.arrowKey`, `tabs.afterEnter`.

## 4. Text contrast below WCAG 2.1 AA — Medium

Seventeen distinct rendered text styles fall below the 4.5:1 normal-text minimum,
measured in-browser against each element's effective background. Worst cases:

| Ratio | Size | Sample | Class |
|---:|---:|---|---|
| 1.58 | 8 px | `/` footer separator | `footer i` |
| 2.48 | 8 px | "SignalScope / See the signal." | `footer` |
| 2.53 | 9 px | "Up to 10 MiB · Images stay local" | `.upload-button small` |
| 2.65 | 8 px | "JPEG, PNG, WEBP" | `.panel-caption` |
| 2.70 | 11 px | "Add an image to explore its visual score…" | `.empty-state p` |
| 2.74 | 8 px | "Processed on this machine. Uploads are not saved." | `.privacy-note` |
| 3.04 | 8 px | "ANALYSIS OPTIONS" | `.option-heading` |
| 3.13 | 11 px | "Workspace" | `.breadcrumb` |
| 3.74 | 12 px | "Explore whether an image is synthetic…" | `.page-heading p` |

**Deliberately excluded as a non-finding:** the disabled Analyze button measures
2.09:1, but WCAG 1.4.3 exempts disabled controls.

**File.** `app/frontend/src/style.css` (single-line minified; selectors above).
**Evidence.** `ui_results.json` -> `textContrastDesktop` (full list of 17).

## 5. 7-9 px type throughout, including mobile — Medium

At 390 px, 23 rendered text nodes are below 10 px, including 7 px footer text
("SignalScope / See the signal.", "Local inference · Reproducible evidence"),
8 px "JPEG, PNG, WEBP", "ANALYSIS OPTIONS", "VISUAL ASSESSMENT", the score-track
labels and the calibration note, and 9 px file name, file size and dimensions.
The 650 px breakpoint raises several sizes but leaves these.

**Evidence.** `ui_results.json` -> `mobileTextContrast`;
`tmp/claude_qa/screenshots/mobile-result.png`.

## 6. Error message persists after the image is removed — Low

**Reproduction.** Select `fixtures/unsupported.gif` (error appears), then click
the X "Remove image".
**Expected.** Clearing the selection clears the error.
**Actual.** "Choose a JPEG, PNG or WebP image." remains displayed with no image
selected and the Analyze button disabled.
**Cause.** The remove handler at `app/frontend/src/main.tsx:133` resets `file`,
`analysis` and `fileInput.value` but never calls `setError('')`.
**Evidence.** `ui_results.json` -> `imageReplacement.afterRemove`.

## 7. Zero-byte upload returns "Upload an image first." — Low

**Reproduction.** Upload `fixtures/corrupt_empty.png` (0 bytes). Reachable from
the UI: the client gate checks only MIME type and the 10 MiB ceiling.
**Expected.** A message indicating the file is empty or unreadable, consistent
with the other decode failures.
**Actual.** HTTP 400 "Upload an image first." — instructing the user to do the
thing they just did.
**File.** `app/backend/main.py:151-152`.

## 8. No skip link; navigation never sets `aria-current` — Low

`document.skipLink` is false — keyboard and screen-reader users tab through the
brand link and three navigation buttons on every page. The active navigation
button is styled with `.nav-item active` but exposes `aria-current: null`, so the
current page is conveyed visually only.
**Files.** `app/frontend/src/main.tsx:114-118`, `:124`.

## 9. Entire results panel is one live region — Low

`aria-live="polite"` is set on the whole `.results-panel`
(`app/frontend/src/main.tsx:141`), which contains the verdict card, tablist and
tab panel. Every re-render — including a tab switch — queues the entire block for
announcement rather than just the changed result.

## 10. "Change image" cannot re-select the same file — Low

**Status: reasoned from source and standard browser behaviour. NOT reproduced
through the automated harness**, because Playwright's `setInputFiles` sets the
file list programmatically and fires `change` regardless, bypassing the native
picker. Flagged for manual confirmation rather than asserted as observed.

Mechanism: the "Change image" button and the upload button both call
`fileInput.current?.click()` (`app/frontend/src/main.tsx:135` and `:133`) without
resetting `value`. Only the Remove button resets it. Re-picking the identical
path in the native dialog therefore leaves `value` unchanged, fires no `change`
event, and nothing happens. Harness state confirms the input retains the previous
path (`inputValue: "C:\\fakepath\\valid_plain.png"`, and `"unsupported.gif"`
after a rejected pick).

## 11. Mobile "Change image" tap target is 57x12 px — Low

Below the WCAG 2.5.8 (AA) 24x24 CSS px minimum, measured at 390 px.

**Deliberately excluded as a non-finding:** the two analysis-option checkboxes
render 14x14 px, but each is wrapped in a `<label>` measuring 481x31 px, so the
effective activation target passes.

**Evidence.** `ui_results.json` -> `mobile390.smallTapTargets`.

## 12. `/favicon.ico` returns 404 on every page load — Low

`app/frontend/dist/` contains only `assets/` and `index.html`, and no favicon is
declared, so every load logs a console error. Confirmed: `curl` -> 404, and the
first entry in the captured console log is this 404.

## 13. Multi-frame WebP accepted and silently scored as frame 1 — Info

`fixtures/animated.webp` (3 frames) returns HTTP 200 and is scored with no
indication that only the first frame was analysed, while `unsupported_animated.gif`
is rejected with 415. The gate at `app/backend/main.py:122` is format-based, so
animated WebP passes. Consider either documenting or surfacing this.

## 14. Degenerate inputs give confident verdicts with no low-information guard — Info

**This is an input-handling observation on synthetic fixtures. It is not evidence
about model accuracy and must not be cited as such.**

- `fixtures/exactly_20mp.png`, a single flat colour at 5000x4000, returns
  `ai_generated` with score 0.777 and `review_recommended: false`.
- `fixtures/one_by_one.png` (1x1) is accepted, upscaled to 128x128 and scored
  `real` (0.174).

The `min(image.size) < 64` limitation string at `src/signalscope/inference.py:124`
covers tiny inputs, but a featureless large image gets a confident label with no
caveat. Whether that warrants a guard is Codex's call.

## 15. `GET /api/predict` returns 404 rather than 405 — Info

`StaticFiles` is mounted at `/` (`app/backend/main.py:160`) and absorbs unmatched
GETs, so a wrong-method call to a documented endpoint reports "Not Found".
`POST /api/health` correctly returns 405.

## 16-18. Documentation vs behaviour

Everything else checked in README and `docs/API_CONTRACT.md` matched (see
"Verified correct" below). Three gaps:

**16.** README:63 states "The image is EXIF-oriented and converted to RGB". True
of the analysis path, but the dimensions reported to the user and to the exported
JSON are pre-orientation (finding 1). The wording implies otherwise.

**17.** `metadata_evidence` only ever surfaces `Make`, `Model`, `Software`,
`DateTime`, `DateTimeOriginal` (`src/signalscope/evidence.py:30`); every other
EXIF tag, including `Orientation`, is dropped. Neither README nor API_CONTRACT
documents this allow-list. Verified: `exif_orient_*.jpg` fixtures report
`exif_present: true` with `fields: {}`. Combined with finding 1, a user has no way
to see why the dimensions look transposed.

**18.** `docs/API_CONTRACT.md` describes the success payload but not error shapes.
FastAPI validation failures return `detail` as a *list*
(`{"detail":[{"type":"missing","loc":["body","image"],...}]}`), and the UI handler
at `app/frontend/src/main.tsx:92` renders only string details, falling back to
"Analysis could not be completed." Only reachable through direct API misuse, not
through the UI.

---

## Verified correct (no action needed)

These were tested and passed; recording them so the same ground is not re-covered.

**Responsive.** No horizontal overflow at 390 px on any view: analyze-empty,
analyze-result, Stability tab, Metadata tab, Model report, How it works
(`scrollWidth == innerWidth == 390` in all six).

**JSON export.** Downloads as `signalscope-analysis.json`, 337,878 bytes, parses
cleanly, contains `filename` plus the full `prediction`/`explanation`/
`robustness`/`metadata` payload. The immediate `URL.revokeObjectURL` after
`.click()` (`app/frontend/src/main.tsx:104`) did **not** truncate the download in
Chrome.

**JavaScript health.** Zero uncaught exceptions across the whole desktop and
mobile run. The eight captured console entries are the expected HTTP error
responses from the deliberate error cases, plus the favicon 404 (finding 12).

**Focus visibility.** Every interactive element shows a 2px solid `#289582`
outline at 4px offset. Non-text contrast against the effective backdrop measured
3.06-4.72:1, meeting the WCAG 1.4.11 3:1 minimum everywhere; the active sidebar
item is the marginal case at 3.06:1. Keyboard order matches visual order, all
nine interactive elements are reachable, and the visually-hidden file input is
clip-based rather than `display:none`, so it stays focusable.

**Labels.** File input has `aria-label="Choose image"`; icon-only buttons
("Remove image", "Download analysis JSON") are labelled; both option checkboxes
use wrapping `<label>`s; `role="alert"` is correctly applied to the error region
and `role="status"` to the readiness notice; `lang="en"`, one `<h1>`, and
`aside`/`nav`/`header`/`main`/`footer` landmarks are present.

**Format gating.** JPEG, PNG, WebP accepted. GIF, BMP, TIFF and animated GIF
rejected with 415 "Supported image formats are JPEG, PNG and WebP." Content
sniffing correctly beats the extension in both directions: BMP renamed `.png`
-> 415; PNG renamed `.jpg` -> 200.

**Corrupt input.** Truncated JPEG, PNG magic + garbage, plain text renamed `.jpg`,
and SVG renamed `.png` all return 400 "The upload could not be decoded as a
supported image." (Zero-byte is the exception — finding 7.)

**Limits, exactly as documented.** >10 MiB -> 413; >20 MP -> 413; exactly
20,000,000 px accepted. The narrow-canvas guard rejects 20000x10 and 13000x10
(21.3 MP canvas) and accepts 12000x10 (19.7 MP canvas) in 0.11 s — the
pre-allocation rejection described in API_CONTRACT:30-31 works, with no memory or
latency blow-up at the boundary.

**Colour modes.** Grayscale (L), palette (P), RGBA with alpha and CMYK JPEG all
decode and score without error.

**Model report page.** Matches `/api/model` exactly; no "Pending" cells; no stale
"remain unevaluated" text; the rendered checkpoint SHA-256 matches the loaded
model; reserved as-distributed (0.5651) and format-matched (0.6430) means, all
four per-generator rows, external development, CIFAKE author test and the
validation confusion matrix all render correctly at both widths.

**README results table.** Spot-checked against live `/api/model`: CIFAKE
validation, GenImage validation, external development (both protocols), reserved
GLIDE/DALLE (both protocols) and the CIFAKE author test all match, including
AUC, accuracy, FPR, the 13-29% / 22-48% recall ranges and both confusion
matrices. All README and API_CONTRACT relative links resolve. The documented
prediction and health field lists match the live responses exactly. README's
port 8000 matches `scripts/run.py:11`. README's `predict(image_path)` helper
exists at `model/predict.py:19`.

## Method and limits

Desktop 1440x1050 and mobile 390x844 in installed headless Chrome via
`playwright-core`, driving the already-running server. 33 generated fixtures
covering valid formats, colour modes, EXIF orientations 1/3/6/8, corrupt and
undecodable files, unsupported formats, extension/content mismatches and the
size and aspect-ratio boundaries.

Measurement caveats: the API probe and browser pass were run sequentially, not
concurrently, because the detector is serialized behind a single lock and
overlapping runs would distort latency. Timings come from a shared instance that
Codex may also have been using, so treat them as indicative. Finding 10 is
explicitly reasoned from source rather than harness-reproduced. Contrast and
tap-target measurements are computed from rendered `getComputedStyle` values
against each element's effective background.

Per the user's bounded scope, only this file and `tmp/claude_qa/` were written;
`docs/PROGRESS.md`, `docs/SUBMISSION_CHECKLIST.md` and `docs/AGENT_HANDOFF.md`
were deliberately left for Codex rather than updated here.
