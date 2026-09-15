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

## Data boundary

At verification, `/api/model` returned null validation metrics, confusion matrix
and public reserved results for `mixed_clip_mlp_v2_release`. The report explicitly
shows those measurements as not supplied rather than substituting another
checkpoint's numbers. It renders populated fields when the existing service
supplies them. No backend or evaluation artifacts were changed.

## 15 September refinement

Retains the existing two-column Analyze layout, light verdict cards and original
heatmap controls. Reduces hero/upload/empty-state spacing, raises body and evidence
text sizes, and gives the navigation a local mountain background inspired by the
user's reference. Short viewports do not trap tall image controls in a sticky panel.
The About introduction is more compact and covers export, batch CLI and all built
PS-2 modules, with unimplemented B/E and limited C2PA support explicitly described.

The report now has a separate **Published release evidence** explorer, not a
replacement API response. `src/release-evidence.json` contains exact copies of the
four evaluation records in `report/experiments/mixed_clip_mlp_v2/results.json`,
the transformation/active-defence records in
`report/experiments/robustness_v040/metrics.json`, and the stated audit fields from
`report/explanation_audit/mixed_clip_mlp_v2_release/summary.json`. Its identity comes
from `report/releases/v0.4.0/freeze.json`. Sources link to immutable commit 4252d85.
The section appears only when API readiness, model version, checkpoint SHA-256 and
threshold all match. Null API metrics remain null; redundant empty cards are
omitted when the matching published evidence is available. Public image holdouts
are labelled as such, not as an organizer or generator-disjoint test. Reported
gate failure, degradation weaknesses and explanation limitations stay visible.

The browser regression checks compare the snapshot to original release records,
exercise all benchmark selectors/matrices and the study, and check that a mismatch
in any identity field hides the archive. Existing upload, live inference, heatmap,
keyboard, metadata, stability, export, errors and responsive checks are retained.

### Sidebar asset

- Saved asset: `app/frontend/src/assets/sidebar-mountains.png`.
- Generated with the built-in imagegen tool; no runtime external-image requests.
- Decorative only; it is never used as model input or evidence.
- Prompt: "Create a premium app sidebar BACKGROUND ASSET only, no UI, no text, no
  logos. Portrait 1024x1536. Near-black midnight navy (#0b111b) background. Fine,
  delicate slate-blue wireframe contour lines form a realistic layered alpine
  mountain range across the lower middle of the composition, dramatic left peak
  and smaller right peaks, subtle atmospheric depth. Mountains occupy vertical
  40%-76%, fading seamlessly into dark background above and below. Top 35% is clean
  near-black navy for navigation text placed later in CSS, bottom 20% clean dark
  for status card. Elegant scientific terrain visualization, restrained luminosity,
  thin detailed topographic ridges, like a luxury dark analytics product.
  Monochromatic blue-gray linework, no stars, no bright sky, no gradients of purple,
  no text or interfaces. Output the background image itself."
