# QA integration review - 12 September 2026

Claude's completed frontend changes were reviewed and tested together with the
updated backend on a separate CPU server (8004). The shared 8002 server was not
stopped for integration. Both original Claude reports and their artifacts remain.

## Backend fixes

- EXIF orientations 5-8 now swap reported dimensions, matching the displayed and
  scored image. Regression cases compare API scores with explicitly oriented inputs.
- Empty files return a specific 400 error. Animated WebP/PNG are rejected with
  415 instead of silently scoring a frame. GET /api/predict returns 405 with Allow.
- Nearly flat images keep the binary label and score but recommend review and
  explain the lack of spatial information. This heuristic is not an accuracy claim.
- API documentation specifies oriented dimensions, animation policy, error
  shapes and the EXIF allowlist. Generic rendering of uncommon 422 validation
  errors remains; the documented normal upload paths have specific messages.

## Frontend integration

Reviewed Claude's completed source and matching dist assets. Independently reran
its regression and settled-state contrast harnesses against the latest backend,
writing separate artifacts under tmp/review. No failed regression assertions,
no clean-load JavaScript/console errors, and no mobile horizontal overflow across
all six checked views. No settled-state text-contrast failures, text below 10px,
or targets below 24px in the harness's eight page/viewport combinations. This is
an automated sample, not an accessibility certification. Mobile result screenshot
also visually checked.

Rejected replacements preserve the completed result; Remove clears stale errors.
Arrow keys/Home/End select and focus tabs; panel IDs and labels resolve. The
skip link, active navigation, concise status announcement and favicon work.
The same-file-picker fix is defensive; native dialog behavior remains a manual
check, as Claude reported. No dependency changes were needed.

31 Python tests pass. Integration smoke uses actual released-model CPU inference
with an explanation and six score measurements. Reference animal score is
0.86916852, unchanged. Ruff and whitespace checks pass. An independent TypeScript/Vite build reproduces the reviewed asset hashes.
Full measured integration record: report/reproducibility/v0.2.0_qa_integration.json.

## Explanation supplement and model decision

Default app attribution remains unchanged. A new optional fixed target supports
a separate 40-image sanity check; details are in EXPLANATION_AUDIT.md. The earlier
audit is preserved. Model-development probes and their calibrated recall tradeoff
are recorded in POST_RELEASE_EXPERIMENTS.md; none replaced the frozen model.

These fixes improve usability and the quality of the evidence. They do not
improve the published unseen-generator scores or complete human explanation review.
