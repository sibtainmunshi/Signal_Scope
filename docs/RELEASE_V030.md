# v0.3.0 preparation - prepared, measured, NOT released

> **Outcome, 13 September 2026 (user decision): v0.3.0 is not activated and v0.2.0 is
> submitted.** Everything below is preserved as the record of what was built and
> measured. Steps 2-5 of "Finish in this order" were deliberately not carried out.
> Do not read this page as a live plan, and do not activate the prepared manifest
> without a new explicit decision.
>
> Why: the candidate ranks better (development mean AUC 0.771/0.789 against
> 0.647/0.653) but failed its predeclared real-photo false-positive gate three times,
> at 22.0% against an allowed 17.8%. Activating it inside the remaining time also
> required publishing a 608 MB asset, re-timing the clean-setup gate that v0.2.0
> currently passes at 204.81 s with a 44.8 MB download, re-verifying the app flow,
> re-running the explanation audit and re-recording the demo video. That put rows we
> already pass at risk to improve one metric, on a model that flags roughly one real
> web photograph in four and a half as AI.
>
> This is a deadline decision, not a measurement claim. The candidate's advantage is
> real and published; see [POST_RELEASE_EXPERIMENTS.md](POST_RELEASE_EXPERIMENTS.md).

13 September 2026. This is the current release status, supplementing
[CLAUDE_NEXT.md](CLAUDE_NEXT.md). The active `model/manifest.json` remains v0.2.0.
No v0.3.0 release assets have been published and no reserved images were scored
during this packaging work. Claude owns `src/signalscope/evidence.py` and
`tests/test_explanation*`; Codex has left those files alone.

## Decision and operating point

Prepare `mixed_clip_l14_balanced_v1`, not the COCO variant. Original/matched
development mean AUC is 0.771/0.789 versus v0.2.0's 0.647/0.653. The declared gate
**failed**: original LDM/LAION real FPR 22.0% versus 12.8%, allowed increase 5 points.
Keep that failure, threshold-policy retry and equal-FPR comparison published in
POST_RELEASE_EXPERIMENTS.md. AUC is ranking performance, not classification accuracy.
No fourth variant or further threshold search is planned.

The prepared manifest is `model/releases/v0.3.0.json`. Threshold is
0.8214277320372911; temperature is 0.9469072146104929. The release-specific head
preserves the trained parameters and binds the verified exported tower by digest.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `mixed_clip_l14_balanced_v1_release/head.pt` | 7,949 | `51e1e07d9b0fed13895b7f7a7fe7b207133cae68eca585582fb3c9c05606613f` |
| `clip_vitl14_visual/visual_fp16.ts` | 608,352,029 | `12403c44d349dee827e7d0fe016c2f255ec51ccf2750d5b4260ab9c878bf7989` |

Both paths are under `model/checkpoints/`. The parent balanced head and all v0.2.0
weights/reports are unchanged. `report/releases/v0.3.0/preparation.json` records
provenance and the failed gate. Asset URLs in the prepared manifest are planned
destinations, **not yet downloadable release assets**.

## Completed by Codex

- `scripts/package_clip_release.py` created the verified, separate release package.
  It refuses to overwrite an existing package; do not rerun it to regenerate bytes.
- Downloader now verifies both artifacts, bounds downloads, and refuses unsafe or
  duplicate paths and mismatched existing files. Setup supports `--manifest` and
  installs torch without torchvision for CLIP. ResNet's torchvision import is lazy.
- Packaged CPU scoring and backend prediction with all seven stability variants
  succeeded while imports of `clip`, `torchvision`, `ftfy` and `regex` were blocked.
  See `minimal_runtime.json` in the release report folder; explanation was excluded.
  This checks inference dependencies, not fresh-install timing or model accuracy.
- Twelve focused installer/CLIP tests and the full 46-test suite passed; Ruff passed
  on the changed Python files. Setup with existing local assets and
  `--skip-install --manifest model/releases/v0.3.0.json` passed digest and CPU load
  verification. Full regression: 46 passed in 27.49 seconds, three dependency
  deprecation warnings; active default still v0.2.0, CLIP tests included.
- API model identity supports CLIP architecture, tower digest and a disclosed
  second-use public-reserve status. It does not relabel old metrics as new results.
- `scripts/evaluate_clip_release.py` is written with active-manifest and committed
  freeze checks, artifact/source verification, resumable scoring and no fitting.
  No reserved inference has run. It will report pooled and per-generator metrics,
  and COCO real FPR with Wilson intervals against unchanged v0.2.0.
- `scripts/build_clip_report.py` creates a separate, clearly labelled one-page
  draft in `report/releases/v0.3.0/`; actual PDF rendered and visually checked.
  Update pending explanation/robustness text and rebuild after integration.

Prepared package commands for this checkout (artifacts already exist locally):

```shell
python scripts/setup.py --skip-install --manifest model/releases/v0.3.0.json
python scripts/run.py --manifest model/releases/v0.3.0.json
```

The second command is for integration testing; explanation support must land
before describing the complete CLIP app flow as verified. Public setup without
local artifacts must wait for asset publication. Measured CLIP CPU scoring is
about 0.38 s/image; the 608.36 MB download needs a new timed setup check.

## Finish in this order

*Status: step 1's code is complete; steps 2-5 were not carried out, by the decision
recorded at the top of this page.*

1. **Code done, audit not run.** CLIP explanation is committed (input-gradient
   attribution, `_clip_patch_map` in `src/signalscope/evidence.py`) and
   `model/explanation_audit.py` now accepts either backbone, so the 40-image audit is
   no longer blocked on a ResNet-only assumption. The audit itself has **not** been
   executed against the CLIP head - it needs the local development archive and the
   608 MB tower. Run it only if the candidate is revived; actual people must perform
   the human usefulness review. Never invent artifact annotations.
2. Verify packaged CLIP through CPU CLI/API and desktop/mobile app, including
   explanation, robustness and large-photo handling. Rerun the minimal-runtime
   import check through the full app. Measure new-model degradation/flip behavior;
   old ResNet results are not CLIP evidence. Update tests that assume a ResNet default.
3. Publish the exact head and tower, verify unauthenticated download digests and
   timed clean CPU setup, update the manifest status and activate the verified
   v0.3.0 manifest. Preserve v0.2.0's checkpoint, tag, assets and archived reports.
4. Freeze the active artifacts and commit the freeze before scoring:

   ```shell
   .venv/Scripts/python scripts/evaluate_clip_release.py --freeze-only --device cpu
   # Commit report/releases/v0.3.0/freeze.json before the next command.
   .venv/Scripts/python scripts/evaluate_clip_release.py --run --device cpu
   ```

   Device is part of the freeze; decide it before freezing. If choosing CUDA,
   first inspect GPU/process use and keep one GPU job at a time. GLIDE/DALLE is a
   disclosed second use of 4,500 public reserved images, not a fresh blind test.
   COCO's 354 reserved real photos are a first-use real-FPR check, not AI accuracy.
   Publish the results regardless of outcome; do not retune afterward.
5. Replace the README's default model details/module evidence, finish the one-page
   report with final overall/unseen metrics and confusion matrices, and update the
   3-5 minute demo to show the actual submitted version. Check public links.
   User submits the public repository link by 15 September 17:00 IST.

The supplied statement's mandatory core is a trained binary model, confidence,
honest splits, metrics and runnable prediction interface. A-G are optional.
Organizers' hidden data/baseline/results have not been supplied; public evaluations
must never be substituted for their official score. Module A correctness and
localisation against annotated defects remain unverified. B/E and C2PA are absent.

The user's 11 ChatGPT images and 18 phone photos are still absent from `tmp/user_eval`.
They were not retested. If restored, use only as a diagnostic/veto check, never for
training or threshold fitting; keep per-image outputs private under `tmp/`.
