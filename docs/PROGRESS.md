# Development log

## 11 September 2026

- User approved local-only development and supplied the public Signal_Scope repo.
- Verified RTX 4050 6 GB, isolated Python 3.13 environment, CUDA computation.
- Audited 120,000 CIFAKE images. Excluded 378 author-train images overlapping the author test split by exact decoded pixels. Prepared grouped train/validation/calibration partitions.
- Trained baseline and robustness-augmented ResNet-18 models on 20,000 images; preserved actual configs, histories, checkpoints and validation reports.
- Built the upload app, CPU API/CLI, returned-class Grad-CAM, measured masking, EXIF inspection, per-upload stability checks and JSON export.
- Eight metric/integration tests passed, plus actual CPU headless Chrome desktop/mobile checks.
- Robust augmentation improved resize/blur performance; full clean validation fell slightly from 97.82% to 97.09% accuracy.
- Acquired/audited the external diffusion release. Fixed development/final generator roles before predictions. External development exposes a major domain-shift failure (released model mean AUC 0.5534), recorded without concealing false positives.
- Added versioned model manifest, integrity-checked download, pinned runtime, prebuilt UI and cross-platform CPU setup/start scripts.
- Initial commit pushed to the user's public repository. Working-app release packaging underway.

Next: test the preprocessing mismatch, run the frozen-feature candidate, and address external performance. Final test data remain unscored. Calibration, explanation audit, final report/video and fresh-clone verification are outstanding.

## 12 September 2026

- Verified public v0.1.0 prerelease asset is uploaded (44,778,635 bytes).
- Resumed after the prior automatic approval usage-limit interruption.
- Frozen CLIP ViT-B/32 experiment completed: 93.78% CIFAKE validation accuracy, external development mean AUC 0.5372. Not selected.
- Added requirement-by-requirement submission checklist, including explicit incomplete explanation audit, diverse training data, final evaluation, report/video and clean-clone verification.
- Next model step: independent diverse higher-resolution training data, keeping the external benchmark evaluation-only.

- Fresh public clone at ef4e6b5 with independent CPU-only venv completed setup in 257.2 seconds. Model downloaded publicly and SHA-256 matched; real API upload, explanation, six stability variants and prebuilt frontend passed. No training data or CUDA dependencies were used.
- Selective GenImage acquisition is underway using pinned archive byte ranges; 7,808 train-only images selected, approx 1.4 GB originals. Class mapping accounts for 42 categories absent from one real-image source.
- Mixed-source CNN and frozen-feature candidate scripts are prepared; no training begins until the duplicate audit completes.

- GenImage acquisition and audit completed: 7,808 downloaded, five protected exact overlaps excluded; 7,803 retained (6,239 train, 783 val, 781 calibration). Cache and manifests verified.
- Saved model-independent handoff and CLAUDE.md entry point. Mixed-data training is the next task; no new model result is claimed yet.

- Data audit before mixed training found a class-correlated format shortcut: GenImage real images are non-square JPEG (quality ~96) while every generated image is a square PNG (BigGAN 128 px, SD1.5 512 px). The external release has the same pattern (ImageNet/LAION JPEG vs 256 px PNG). Added a declared `matched` external protocol (identical centre crop, 224 px resize and JPEG q90 for both labels) to measure how much of any gain survives without format/aspect cues.
- `mixed_resnet18_v1` (planned CNN; 8k CIFAKE + 6,239 GenImage, 160 px squash, JPEG on every training image): GenImage val AUC 0.945 (BigGAN 0.918, SD1.5 0.966), CIFAKE val 0.995. External dev mean AUC 0.627 as distributed (guided 0.652, LDM 0.603) but 0.574 matched (guided 0.541, LDM 0.606). Real-image FPR at 0.5 fell to 8.4%/12.4% (released: 92.2%/76.4%), with low AI recall (22%/25%). Raw-file vs cache preprocessing agreement exact (max difference 0.0).
- `mixed_clip_b32_v1` (frozen CLIP + our balanced logistic head, C=10): GenImage val AUC 0.985, CIFAKE 0.975; external dev 0.708 as distributed but 0.542 matched (LDM 0.668 -> 0.462). Its external gain largely depends on format cues; not selected.
- Released model under matched protocol: 0.540. No candidate so far justifies replacing the release.
- Built shortcut-controlled v2 cache (`scripts/prepare_genimage_v2.py`): aspect-preserving centre crop (`pil_center_crop_v1`), BigGAN real photos resolution-matched at 128 px, symmetric/native JPEG balancing for training only; validation/calibration use exact inference preprocessing. Detector, Grad-CAM overlay (dims unanalysed borders) and evaluators support the new geometry. Added mixed-source calibration script, ECE metric and geometry tests.
- `mixed_resnet18_v2` (v2 cache, centre crop, final JPEG probability 0.5): GenImage val AUC 0.904 (BigGAN 0.853, SD1.5 0.948), CIFAKE val 0.995; external dev 0.550 as distributed and 0.549 matched (guided 0.539, LDM 0.559). With format cues removed the two protocols agree, but the remaining transferable signal is near chance: 160 px resizing appears to discard generator traces. Not selected.
- Backend model report and UI text now derive training source, validation and both external protocols from the loaded checkpoint (hash-checked reports only). 16 tests pass; frontend rebuilt with type-check; CPU smoke test of the centre-crop explanation mapping passed.
- Decision so far: keep the released detector. Next experiment: native-resolution crops without resizing, with format balancing, judged by both protocols and real-image FPR.
- `mixed_resnet18_native_v1` (`model/train_native.py`; random 128 px native crops, no resizing; every generated GenImage training image JPEG-compressed with qualities drawn from the real photos; five-crop mean-logit inference): GenImage val AUC 0.956 (BigGAN 0.946, SD1.5 0.969), CIFAKE val 0.993. External dev mean AUC 0.647 as distributed, 0.653 matched, 0.649 matched_native (new format-only control declared before scoring). First candidate whose gain survives format matching. Paired bootstrap (2,000 samples) difference versus the released model: +0.094 [0.063, 0.125] as distributed, +0.113 [0.083, 0.142] matched. By contrast the CLIP head's matched gain is +0.002 [-0.031, 0.035].
- Seed replicate (`..._seed2027`): as-distributed 0.670 versus 0.647, so run-to-run variation is of order 0.02. The seed-2026 run remains the pre-declared primary candidate.
- Calibrated copy `mixed_resnet18_native_v1_calibrated`: temperature 1.65 (calibration splits, equal domain weight), threshold 0.455 (strictest per-domain validation threshold at FPR <= 5%). Validation operating point: GenImage accuracy 89.8%, FPR 4.9%, TPR 85.0%; CIFAKE accuracy 95.2%, FPR 2.2%, TPR 92.5%. ECE improved on CIFAKE (0.027 -> 0.009) but worsened on GenImage (0.047 -> 0.061): one temperature does not calibrate both domains.
- Multi-crop stitched Grad-CAM and masking diagnostic for native models; `Detector.score_images/image_logits` (verified identical to the old path on CPU); paired bootstrap script; explanation-audit script written. 20 tests pass.
- Seed-2027 replicate complete: 0.670 / 0.667 / 0.659 (as distributed / matched / matched_native); matched gain versus release +0.126 [0.096, 0.156]. The improvement replicates.
- Calibrated checkpoint evaluated: external AUC unchanged; at threshold 0.455 real-image FPR 4.0%/12.8% as distributed and 11.8%/16.0% matched (ImageNet/LAION), but external AI recall only 11-43%. Conservative under domain shift; report plainly.
- Explanation audit (40 fixed development images, 27 correct): top-attribution windows reduced the returned-class score more than random windows in 72.5% (mean fill, p=0.002) and 67.5% (blur fill, p=0.014) of images, with small effects; JPEG q70 map Spearman median 0.95; weight-randomization Spearman median 0.52, so maps are only partly weight-dependent. Details in `docs/EXPLANATION_AUDIT.md`.
- Robustness (`benchmark_robustness.py`, now multi-crop aware, new `--dataset genimage`) for the calibrated native model at threshold 0.455. GenImage val AUC: original 0.956, JPEG q70 0.953, q30 0.906, half resolution 0.896, mild blur 0.937; bounded six-transformation search flipped 19.8% of initially correct predictions; real FPR rises from 4.9% to 10.6-14.9% under JPEG q90, half resolution and blur. CIFAKE val: original AUC 0.994 but half resolution 0.726 and flip rate 42.6% (released model 6.3%): the native model loses the release's resize robustness on tiny upscaled images. Released model on GenImage val: AUC 0.603, FPR 88%.
- Generator-diversity experiment: GenImage VQDM (2/class; 3,988 acquired, 447 MB; 2 protected exact overlaps excluded; 3,129 train / 441 val / 416 calibration; no overlap with the existing subset) and Midjourney (1/class; downloading). ADM (guided-diffusion family of the external dev set) and GLIDE (reserved) excluded by design.
- Selection rule declared BEFORE these results: among native_v1, native+VQDM and native+VQDM+Midjourney, adopt a data-extended model only if its format-matched external mean AUC exceeds native_v1 (0.653) by more than 0.02 (the seed-to-seed variation observed), with real-image FPR not worse; otherwise keep native_v1. Same seed, epochs and selection objective for all.
- Midjourney extension audited: 1,994 acquired (1,346,040,064 bytes), no exclusions; 1,579 train (804 real / 775 generated), 206 val, 209 calibration. Training copies of the 1024 px generated images are downscaled to a 512 px short side.
- `mixed_resnet18_native_vqdm_v1` (native_v1 recipe + 3,129 VQDM training images; best epoch 3 by the unchanged internal objective): GenImage val AUC 0.936 (FPR 14.9% at 0.5), VQDM val 0.881 at the selected epoch. External dev 0.609 as distributed, 0.575 matched, 0.595 matched_native (native_v1: 0.647/0.653/0.649), with LDM/LAION real FPR rising to 35%. Adding VQDM made unseen-generator performance worse; rejected under the pre-declared rule. Negative result preserved.
- Release asset for the native calibrated checkpoint prepared locally (44,782,411 bytes, SHA-256 eab7d9d8...b303); `gh` CLI is unavailable, so the GitHub release upload needs the user. App, CLI and tests now take the default checkpoint from `model/manifest.json`.
