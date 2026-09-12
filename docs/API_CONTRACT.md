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

`GET /api/model` includes `external_reserved`, `external_reserved_matched` and
`cifake_test` when saved results match the loaded checkpoint SHA-256. These public
reserved results are distinct from the unavailable organizer hidden test.

Native crop preprocessing also limits the upscaled canvas to 20 million pixels.
Extremely narrow images that exceed this limit return HTTP 413 before allocation.

Image dimensions in the prediction describe the EXIF-oriented image, matching
what is displayed and scored. The visual score's preprocessing is unchanged.
Only single-frame uploads are supported; animated PNG/WebP return HTTP 415.
Empty files return HTTP 400 with an explicit empty-file message. GET /api/predict
returns HTTP 405 and `Allow: POST`.

Errors normally have a string `detail`; FastAPI request-validation errors (422)
may instead contain a list of objects with `loc`, `msg` and `type`. Clients should
handle both. EXIF fields are limited to Make, Model, Software, DateTime and
DateTimeOriginal; GPS and serial numbers are not exposed.

Nearly flat images (maximum channel standard deviation below 1 pixel level on a
64px RGB thumbnail) retain their binary label and score but recommend review with
a low-information limitation. This is a usability heuristic, not evidence of
accuracy or generation, and does not change benchmark predictions or thresholds.
