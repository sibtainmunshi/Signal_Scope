# Post-release development probes

These probes resumed at the user's request after the v0.2.0 freeze. They use development/validation/calibration data only. The original final test is not reused for model selection. v0.2.0 remains the released model.

## Equal-weight score combinations

Three fixed combinations were declared before computing the comparisons. None passed the required +0.02 mean AUC in both processing protocols together with the false-positive constraint. The seed blend reached 0.670 / 0.671, but its matched gain was 0.018; the cutoff was not lowered after seeing it. [Protocol and results](../report/experiments/ensemble_probe_results.json).

## Reduce low-resolution CIFAKE training weight

The candidate uses 1,000 CIFAKE + 6,239 GenImage images for 20 epochs (2,280 optimizer updates), versus 8,000 + 6,239 for 10 epochs (2,230 updates). Architecture, seed, JPEG balancing and validation selection remain the same. Best checkpoint selected using the existing internal validation objective. The extra validation looks and approximate update match limit causal interpretation.

| Development metric | Released v0.2.0 | CIFAKE-1k candidate |
|---|---:|---:|
| As distributed: mean AUC | 0.647 | 0.682 |
| As distributed, guided: AUC | 0.630 | 0.608 |
| As distributed, guided: AI recall | 10.8% | 2.6% |
| As distributed, guided: real FPR | 4.0% | 0.2% |
| As distributed, ldm_200: AUC | 0.665 | 0.756 |
| As distributed, ldm_200: AI recall | 37.0% | 27.0% |
| As distributed, ldm_200: real FPR | 12.8% | 5.0% |
| Format matched: mean AUC | 0.653 | 0.674 |
| Format matched, guided: AUC | 0.611 | 0.588 |
| Format matched, guided: AI recall | 23.6% | 5.0% |
| Format matched, guided: real FPR | 11.8% | 2.8% |
| Format matched, ldm_200: AUC | 0.695 | 0.760 |
| Format matched, ldm_200: AI recall | 41.2% | 27.2% |
| Format matched, ldm_200: real FPR | 16.0% | 5.6% |

**Decision:** the candidate passes the predeclared ranking/validation gate and was calibrated with the same procedure. It is not deployed: gains concentrate in LDM, guided performance falls, and calibration reduces AI recall. This is an operating-point judgment after calibration, not a claim that it failed the initial gate. Lower real-photo FPR is a useful tradeoff and is reported alongside the missed images.

CIFAKE full validation AUC is 0.974 (release 0.993); GenImage is 0.953 (release 0.956). Candidate temperature 1.691 and threshold 0.801 are fit without external data. GenImage validation recall falls from 85.0% to 66.7%. No reserved/test data were scored; no candidate replacement or fresh unseen-test claim was made.

[Declared protocol](../report/experiments/cifake_reweight_protocol.json) | [Complete measured results](../report/experiments/cifake_reweight_results.json).

The accumulated experiments reuse the same two development domains, one related to a training family. Improvements on them can overfit model selection. A future replacement requires a new independent evaluation design and matching explanation/deployment checks.
