# Demo video script (target 4 minutes; allowed 3–5)

The video is the primary evidence (PDF section 7.4). Record real, live inference;
never simulate a result. Numbers marked `<report>` must be read from the final
`report/model_report.pdf` after model/threshold freeze, not from this draft.

## Before recording

- Start the app: `python scripts/run.py`, open http://127.0.0.1:8000 (CPU is fine).
- Screen recorder: OBS, or Windows Game Bar (`Win+Alt+R`); 1080p, microphone on.
- Close unrelated tabs and notifications; no credentials or personal files on screen.
- Demo images (reviewed, no identifiable people, outside training):
  - one real photo you take yourself of an object or product (for example a mug);
  - two GenImage validation images (one real, one Stable Diffusion) chosen from
    `data/manifests/genimage.csv` (split `val`), checked by eye for people;
  - one known failure case (an AI image the model scores as real), to show honesty.
  Do not use external LAION images: the local audit found identifiable people there.
- Prepared locally in `tmp/demo/selected/` (reviewed by eye: animals only), from the
  GenImage validation split with native-v1-calibrated scores:
  | File prefix | Source, label | cache_index | AI score |
  |---|---|---|---|
  | `1_sd_fake_detected` | SD1.5, generated | 5910 | 0.869 |
  | `2_sd_fake_detected` | SD1.5, generated | 6150 | 0.911 |
  | `3_real_correct` | ImageNet photo (BigGAN archive), real | 21 | 0.006 |
  | `4_real_correct` | ImageNet photo (BigGAN archive), real | 48 | 0.016 |
  | `5_failure_fake_missed` | BigGAN, generated | 2287 | 0.016 |
  | `6_biggan_fake_detected` | BigGAN, generated | 2134 | 0.996 |
  Re-check scores in the app if the final model changes. Credit GenImage (CC BY-NC-SA 4.0) in the video description.

## Timeline

| Time | Screen | Say (short, plain) |
|---|---|---|
| 0:00–0:25 | Title, app home | "SignalScope estimates whether an image is likely AI-generated and shows the evidence behind that estimate. The hard part is new generators the model never saw, and not flagging real photos." |
| 0:25–1:15 | Upload own real photo, then the Stable Diffusion image | Read the verdict wording ("likely"), the AI score and the threshold. "The threshold was chosen on validation data to keep real-photo false positives near 5% on our validation sources." |
| 1:15–2:00 | Evidence tab | "The heat map shows regions that influenced this model, stitched from five native-resolution crops; dimmed areas were not analysed. Masking the top region changes the score by the stated amount. It does not prove a visible defect, and we never claim one." Then upload the failure case: "Here the model is wrong; this is why the output is a likelihood, not a verdict." |
| 2:00–2:35 | Stability tab, Metadata tab | "The same image re-compressed, resized and blurred: here is how the score moves." "EXIF is shown separately; C2PA is not checked, and metadata never changes the score." |
| 2:35–3:30 | Model report page, then `report/model_report.pdf` | "On generators we never trained on, AUC improved from 0.55 for our first CIFAKE-only model to about 0.65. We found that real photos in public datasets are JPEGs while generated images are PNGs; a detector can cheat on that. So we also score every image format-matched: a CLIP model that looked best dropped to chance, while ours held at about 0.65." Final reserved GLIDE/DALLE AUC: `<report>`. |
| 3:30–4:10 | README and terminal | Show `python scripts/setup.py`, `python scripts/run.py`, and `model/predict.py --image <file>` printing JSON. "Weights download from the GitHub release with a SHA-256 check; no dataset or GPU is needed." |
| 4:10–4:30 | Report limitations section | "Limits: unseen-generator accuracy is still modest, heavy resizing hurts, and calibration does not transfer to every source. Organizer hidden-test results were not available to us." |

## After recording

- Watch once: check that every number spoken matches the final report.
- Upload as unlisted (YouTube or Drive with link access), then put the link in the
  README "Demo video" line and the submission form.
