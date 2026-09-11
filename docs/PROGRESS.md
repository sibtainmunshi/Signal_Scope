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
