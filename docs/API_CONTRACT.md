# Application API contract

The model CLI and API use the same Detector and preprocessing. The positive class
is always AI-generated. API calls do not retrain the model or update thresholds.

Release preparation (13 September): the active default remains v0.2.0. The prepared
v0.3.0 CLIP manifest is separate at `model/releases/v0.3.0.json`; it fixes threshold
0.8214277320372911 and temperature 0.9469072146104929. Its setup/run commands and
remaining integration checks are in [RELEASE_V030.md](RELEASE_V030.md).
`GET /api/model` reports the loaded architecture and `visual_sha256` (null for
ResNet). CLIP uses the official 224px centre-crop transform, rather than ResNet's
native crops. A CLIP public-reserve result, once recorded, will explicitly disclose
the split's previous use for v0.2.0. Metrics are never transferred between checkpoint
hashes; absent results remain absent.

`GET /api/health`: readiness, model version, device, and actual model status. A
missing checkpoint is reported as unavailable; no placeholder prediction occurs.

`POST /api/predict`: multipart `image` field; optional `explain` and `robustness`
boolean form fields. Returns `prediction`, optional `explanation`, optional
`robustness`, and `metadata`. No upload is persisted by default. Limits:
25 MiB encoded upload; 40 million decoded pixels; JPEG, PNG, and WebP images.

Prediction fields: `label` (`real` or `ai_generated`), `ai_score` (0-1), `threshold`,
`confidence` (score assigned to returned class, not estimated benchmark accuracy),
`calibrated`, `review_recommended`, `inference_ms`, `model_version`,
`checkpoint_sha256`, `image_width`, `image_height`, `limitations`.

Model confidence is a signal, not proof of origin. `review_recommended` is an
additional product hint and never removes an image from binary evaluation.

Explanation fields describe model influence and measured interventions. A
heatmap must never be presented as ground-truth artifact segmentation.

`explanation.localisation` reports whether the highlighted region measurably carries
the verdict: `supported` is true only when masking that region moves the returned
class's own score by at least one percentage point and moves it further than equally
sized corner patches. `returned_class_drop` and `comparison_drop` carry the measured
values behind that decision. When `supported` is false the statements say the verdict
is not localised and the overlay is presented as model influence only. Clients must
not describe an unsupported overlay as the reason for a verdict. The rule is fixed in
code, not chosen per image, and the CLIP backbone frequently returns false because its
evidence is distributed: masking any single region typically moves the score by less
than one percentage point.
Metadata evidence is separate from, and never overrides, the image-only score.

`GET /api/model` includes `external_reserved`, `external_reserved_matched` and
`cifake_test` when saved results match the loaded checkpoint SHA-256. These public
reserved results are distinct from the unavailable organizer hidden test.

The v0.2.0 native crop preprocessing limits a canvas requiring upscaling to 20 million pixels.
Native images such as 6000 x 4000 phone photos retain their original resolution.
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

`metadata.c2pa_status` is one of `no_marker_found`, `marker_found_unverified` or
`not_checked` (only when the caller did not supply raw upload bytes; the API always
supplies them). This is a bounded ASCII substring scan for known C2PA (Content
Credentials) identifiers, never a JUMBF box parser or a manifest/signature validator.
`marker_found_unverified` means the scan found a plausible marker, not that a manifest
was decoded, validated or trusted; clients must not present it as verified provenance.

Nearly flat images (maximum channel standard deviation below 1 pixel level on a
64px RGB thumbnail) retain their binary label and score but recommend review with
a low-information limitation. This is a usability heuristic, not evidence of
accuracy or generation, and does not change benchmark predictions or thresholds.

Batch CLI: `python model/predict.py --images-dir photos --device cpu` recursively
processes JPEG/PNG/WebP files in path order with one loaded detector. Alternatively,
`--image-list inputs.txt` reads one path per line, relative to the list file.
Both write JSONL: one record with `image` and prediction fields, or `image` and
`error`. Bad inputs do not stop subsequent images. Exit codes: 0 all succeeded,
1 one or more image failures, 2 invalid inputs/setup. Single-image behavior remains.

Robustness includes `simulated_screenshot`: synthetic 90% display resizing capped
at 1080px, simple window borders and PNG encoding. This is a controlled simulation,
not a real OS/device capture or a photographed screen; its measured performance
must not be described as universal screenshot robustness.
