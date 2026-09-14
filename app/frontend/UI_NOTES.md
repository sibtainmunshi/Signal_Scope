# Frontend handoff - 14 September 2026

The UI keeps the existing React/Vite app and API requests. All changes for this
polish are inside `app/frontend/`, including the tracked production bundle.

- Midnight navy navigation, blue accents, responsive layouts, local font fallbacks,
  and reduced-motion support. No new runtime dependencies or remote assets.
- The image inspector offers Original, Influence and a draggable comparison.
  Arrow keys, Home and End operate the comparison. Both layers contain the whole
  image; the backend overlay is not recoloured, cropped or replaced.
- Results show the binary label, returned-class confidence, AI score and fixed
  threshold. Review recommendations also cover reasons other than proximity to
  the threshold. Per-image limitations remain visible on request, and open by
  default when review is recommended.
- Evidence renders actual statements, localisation support and masking score
  changes. The overlay is model influence, not verified defect segmentation.
- Stability shows score changes in percentage points, threshold markers and
  verdict changes. Metadata uses the returned C2PA marker status and keeps it
  separate from the visual assessment.

## Verification

With the existing local backend running and serving `dist/`:

```sh
npm run build
npm run test:ui
```

The UI test uses installed Chrome via the existing `playwright-core` dependency.
It runs a real prediction on the repository's published animal-image example,
then uses explicitly marked response fixtures for uncommon UI states. Screenshots,
the real response, exported JSON and the check summary go to ignored `.qa/`.
Optional overrides: `SIGNALSCOPE_URL`, `SIGNALSCOPE_FIXTURE`, `SIGNALSCOPE_QA_DIR`.

Verified: production build; real v0.4.0 prediction with both optional checks;
comparison pointer/keyboard controls; evidence keyboard navigation; exact JSON
export; drag/drop, file replacement and rejection; all three pages at desktop,
320px, 390px and 768px; reduced motion; optional results; localisation states;
all C2PA statuses; structured errors and unavailable model. No browser JS errors.

## 15 September - compact layout revision

Reduced header, upload, option and card spacing. Image and assessment occupy the
top row; the evidence tabs now span a dedicated section below, with localisation
and explanation statements in adjacent desktop columns. Re-analysis controls
become a compact row after a result. A dark assessment surface strengthens the
score hierarchy. Report and About layouts also use less empty padding.

At the same 1440px desktop width and 1050px viewport height, the published animal
fixture's full result screenshot decreased from 1745px to 1302px in height (25%).
All returned statements, limitations and confidence fields remain available.
Production build and the existing real-inference/UI regression suite passed;
desktop and mobile screenshots were visually reviewed.

## Data boundary

At verification, `/api/model` returned null validation metrics, confusion matrix
and public reserved results for `mixed_clip_mlp_v2_release`. The report explicitly
shows those measurements as not supplied rather than substituting another
checkpoint's numbers. It renders populated fields when the existing service
supplies them. No backend or evaluation artifacts were changed.
