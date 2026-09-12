# Independent review of the Claude handoff

Reviewed 12 September 2026 against commit d8b7558, source code, nine-page problem
statement, checkpoint payload, saved prediction files and rendered model report.

## Verdict

Substantial implemented and measured work; preserve the selected model. The
saved evidence supports the reported experiments and final statistics. It does
not establish strong real-world detection, perfect explanations or a selection
probability. Final unseen-generator performance and explanation usefulness remain
material limitations under the problem statement's scoring rubric.

## Claims checked

| Claim | Result |
|---|---|
| Native multicrop model and calibration implemented | Confirmed in preprocessing/inference/training and checkpoint payload |
| Correct weights and fixed operating point | 44,782,411 bytes; SHA-256 eab7d9d82c8250bd0dfbbb7beb4d8e271dca4ca61ebbcd777819b89c53e6b303; threshold 0.4553663730621338; temperature 1.649664402 |
| Freeze precedes final evaluation | Commit history records cda2234 before 4d3f8a5; inference/training code unchanged between freeze and d8b7558. This is a provenance check, not proof of all past local activity |
| Final statistics | Recomputed AUC/F1/accuracy/FPR/TPR/confusion matrices from saved scores for 20,000 CIFAKE and all three 4,500-image reserved protocols; all match |
| Development statistics | Recomputed as-distributed and matched scores for the calibrated native model; all match |
| Twenty tests pass | Reproduced before fixes |
| Forty-image explanation audit | Code and per-image/aggregate outputs present; deletion comparisons and limited randomization checks implemented. Human usefulness review is blank |
| Public model upload requires the user | Incorrect blocker: GitHub CLI is absent, but existing Git credentials support the authorized REST upload |

## Corrections made

1. Final evaluation status was stale in both API and UI. Display frozen public
   reserved results, sample counts, per-generator recall/FPR and CIFAKE test;
   retain checkpoint-hash matching so another model cannot inherit these results.
2. Very narrow images could allocate an enormous canvas during minimum-size
   upscaling despite passing the original pixel limit. Reject before allocation.
3. Native integration tests used an unpublished uncalibrated parent checkpoint.
   They now exercise the actual release checkpoint; added regression checks for
   final-result identity and extreme aspect ratios. 23 tests pass after changes.
4. Fix lint findings; rebuild and smoke-test the actual CPU UI on desktop/mobile.
5. Correct README/report claims about resizing, sample counts, final confusion
   matrix and causal interpretation. Larger report text remains one page and was
   visually inspected. Replace contradictory appended handoff/checklist entries.
6. Archive measured final scores (1.6 MB) with a Python-standard-library verifier.
   This verifies statistics without image downloads, not original label correctness.
7. Produce public model-linked explanation samples from three reviewed validation
   animal images, including a strongly incorrect prediction. Do not publish the
   full private contact sheet containing incidental people/political imagery.

## Interpretation limits

- Matching crop, resize and JPEG together is a useful stress test; an AUC drop
  does not prove that the entire gain was caused by file format or remove all
  earlier compression traces.
- Development paired bootstrap and one seed replicate support the observed
  comparison. They do not incorporate candidate-selection uncertainty or prove
  transfer to future generator families.
- The reserved 4,500 images contain 4,000 generated images plus 500 shared real
  photos. Per-generator comparisons reuse those reals; GLIDE x3 and DALLE are two
  generator families. Reported external accuracy/F1 are means over pairs.
- Validation FPR <=5% does not transfer: reserved real FPR is 10.4% as distributed
  and 14.6% matched. Most generated images are missed at the fixed threshold.
- No artifact annotations were supplied. A Grad-CAM/masking audit cannot establish
  correctness or localization of specific visible defects. Two-person review remains.
- The supplied PDF restricts sourcing identifiable-person imagery. Allowed public
  benchmark releases nevertheless contain incidental people; the raw dataset must
  not be described as person-free. Public examples were individually reviewed;
  no identity-targeting or political-claim functionality is implemented.

The frozen model bytes and original final measurement records were not changed.
Release/download verification and the independent CPU installation record are
tracked in report/reproducibility. The recorded demo and human review are separate
submission evidence; neither is implied by passing automated tests.
