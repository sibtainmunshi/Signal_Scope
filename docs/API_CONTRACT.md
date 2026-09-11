# Application API contract

The model CLI and API use the same Detector and preprocessing. The positive class
is always AI-generated. API calls do not retrain the model or update thresholds.

`GET /api/health`: readiness, model version, device, and actual model status. A
missing checkpoint is reported as unavailable; no placeholder prediction occurs.

`POST /api/predict`: multipart `image` field; optional `explain` and `robustness`
boolean form fields. Returns `prediction`, optional `explanation`, optional
`robustness`, and `metadata`. No upload is persisted by default. Initial limits:
10 MiB encoded upload; 20 million decoded pixels; JPEG, PNG, and WebP images.

Prediction fields: `label` (`real` or `ai_generated`), `ai_score` (0-1), `threshold`,
`confidence` (score assigned to returned class, not estimated benchmark accuracy),
`calibrated`, `review_recommended`, `inference_ms`, `model_version`,
`checkpoint_sha256`, `image_width`, `image_height`, `limitations`.

Model confidence is a signal, not proof of origin. `review_recommended` is an
additional product hint and never removes an image from binary evaluation.

Explanation fields describe model influence and measured interventions. A
heatmap must never be presented as ground-truth artifact segmentation.
Metadata evidence is separate from, and never overrides, the image-only score.
