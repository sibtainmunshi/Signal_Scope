# SignalScope handoff - 12 September 2026

This replaces accumulated stale handoff entries. Read README.md and
SUBMISSION_CHECKLIST.md for current evidence and outstanding deliverables.

## Workspace and authorization

- Actual folder: `C:/Users/Sibtainhaidar/OneDrive/Desktop/signal_scope` (renamed by user).
- Public repo: https://github.com/sibtainmunshi/Signal_Scope.git, branch main.
- User authorized implementation, tests, commits, pushes and publishing the model.
- Deadline: 15 September 2026, 17:00 IST; aim to finish at 14:00 IST.
- No selection or broad accuracy guarantee. Local laptop only; RTX 4050 6 GB.
- About 31 GiB free at this review. Raw data, caches and checkpoints stay out of Git.

## Final model and protected evaluation

`mixed_resnet18_native_v1_calibrated`, ResNet-18, up to five native 128 px crops,
mean logits, temperature 1.649664402, threshold 0.4553663730621338.
SHA-256: `eab7d9d82c8250bd0dfbbb7beb4d8e271dca4ca61ebbcd777819b89c53e6b303`.
Weights: 44,782,411 bytes. v0.2.0 manifest names the same frozen checkpoint.

Training: 8,000 CIFAKE plus 6,239 audited GenImage BigGAN/SD1.5 images. Native v1
beat the earlier models on format-matched development AUC (~0.653); VQDM and
Midjourney additions failed the predeclared replacement rule. Preserve negative
experiments. No further candidate selection against the final results.

Freeze commit cda2234 precedes final evaluation commit 4d3f8a5. Both the original
weights and final measurement files remain unchanged. CIFAKE test: AUC 0.99257688,
accuracy 95.125%, real FPR 1.92%. Reserved GLIDE x3/DALLE: mean AUC 0.5651065 as
distributed, 0.643012 matched, 0.6402025 matched-native. Most unseen AI images are
missed at the fixed threshold. The three GLIDE configurations are one family;
reserved comparisons share 500 real photos (4,500 unique images total).

**Do not rerun final evaluation or tune against those labels.**
`python scripts/verify_frozen_results.py` recomputes metrics from the archived
scores without inference or data downloads. It verifies arithmetic, not image
labels or the full training provenance. Development bootstrap CIs do not account
for model-selection uncertainty or all unseen generator families.

## Review and packaging fixes

- Reviewed Claude's implementation and recomputed metrics from saved predictions;
  all final and development numbers checked matched. Original 20 tests passed.
- API/UI now show final reserved results separately from development validation.
- Extremely narrow inputs are rejected before excessive upscaling allocates memory.
- Native integration tests use the release checkpoint, not an unpublished parent.
- 23 tests and Ruff pass; real CPU upload and desktop/mobile Chrome smoke pass.
- One-page PDF corrected for tiny-image upscaling, final confusion matrix, sample
  counts, limited causal interpretation and pending human review; visually checked.
- Archived final score CSVs (1.6 MB) are in report/final/predictions; no image inputs.
- Updated README and compact current checklist supersede tmp/release README draft.

## Release execution

`gh` is absent but **not a blocker**. Existing Git credential-manager credentials
can authenticate GitHub REST requests; never print credentials or put them in files.
Publish the release/tag at the reviewed code commit, upload the exact checkpoint,
verify its public unauthenticated URL, then push the manifest on main. Check for an
existing release/asset first. User need not upload the model manually.

Run a fresh public clone with a new CPU virtual environment. Do not delete or use
the GPU development environment as the evaluator environment. Record duration,
package-cache status, checksum, actual API prediction, explanation and static UI.

## Outstanding human evidence

The 40-image automated explanation audit is complete; the two-reviewer form is
still blank. Do not invent human reviews. Semantic artifact correctness and
localization against organizer annotations are unverified.

The PDF explicitly restricts imagery of identifiable people. General public
benchmarks contain incidental people despite category exclusions; an external
LAION audit image contains a political poster. These were not identity-targeting
tasks, but do not claim the raw benchmark is person-free. Public demo material
must use reviewed animals/objects only. Do not publish the full private audit sheet.

Finish the 3-5 minute actual-inference video, publish an accessible link, and check
all submission links. Prepared animal images in tmp/demo/selected are GenImage
validation examples, not new accuracy evidence. A demonstrated failure must remain.
