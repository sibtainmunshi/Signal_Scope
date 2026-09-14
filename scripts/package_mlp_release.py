"""Package mixed_clip_mlp_v2_candidate as the v0.4.0 release checkpoint.

Pins the exported CLIP image tower's digest into the head (the linear v0.3.0 release
already verified this exact tower; reused unchanged here, not re-exported), and writes
model/releases/v0.4.0.json. Does not touch model/manifest.json or any v0.2.0/v0.3.0
artifact; activation is a separate, explicit step.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOWER = ROOT / "model/checkpoints/clip_vitl14_visual/visual_fp16.ts"
TOWER_URL = "https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.3.0/signalscope-clip-vitl14-visual-fp16.ts"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source = ROOT / "model/checkpoints/mixed_clip_mlp_v2_candidate/head.pt"
    destination = ROOT / "model/checkpoints/mixed_clip_mlp_v2_release/head.pt"
    manifest_path = ROOT / "model/releases/v0.4.0.json"
    if destination.exists() or manifest_path.exists():
        raise SystemExit("Prepared release exists; verify it instead of overwriting.")
    if not TOWER.is_file():
        raise SystemExit(f"Tower missing: {TOWER}")
    tower_sha = digest(TOWER)
    if tower_sha != "12403c44d349dee827e7d0fe016c2f255ec51ccf2750d5b4260ab9c878bf7989":
        raise ValueError("Local tower does not match the verified v0.3.0 tower; do not proceed.")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["visual_path"] = "model/checkpoints/clip_vitl14_visual/visual_fp16.ts"
    payload["visual_sha256"] = tower_sha
    payload["config"] = payload["config"] | {
        "run": "mixed_clip_mlp_v2_release",
        "parent_run": "mixed_clip_mlp_v2_candidate",
        "training_description": {
            "dataset": "CIFAKE + GenImage BigGAN/SD1.5 + AI Detect Arena Benchmark + CommunityForensics-Eval",
            "training_source": "CIFAKE (8,000) + GenImage (6,239) + AIDA train (1,183, 17 current generators) "
                               "+ CommunityForensics train (2,122, ~20 generators)",
            "source_resolution": "32 px CIFAKE; 128-512 px GenImage; native-resolution AIDA/CommunityForensics",
            "transparency_note": ("Trained to favour current-generator (2025-2026) accuracy over the prior "
                                  "release's 2021-2023-vintage development benchmark; both are measured and "
                                  "disclosed. See README and docs/POST_RELEASE_EXPERIMENTS.md."),
        },
        "release_decision": (
            "Both AIDA-mixture attempts (linear 40% weight, MLP 60% weight across two "
            "independent fresh-generator sources) failed the declared advancement gate "
            "on external-dev regression. Deployed anyway by explicit user decision: the "
            "AIDA and CommunityForensics holdouts (never used for fitting) show 85.2%/"
            "71.8% accuracy and 7.3%/2.2% real-photo FPR on genuinely 2025-2026 "
            "generators, versus the previous release's measured ~55-60% accuracy and "
            "66-68% FPR on the same benchmarks. The 2021-2023-vintage development check "
            "regresses from 0.771/0.789 to 0.637/0.658 mean AUC as a direct result."),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    head_sha = digest(destination)

    manifest = {
        "schema_version": 2, "release": "v0.4.0", "model_version": "mixed_clip_mlp_v2_release",
        "path": "model/checkpoints/mixed_clip_mlp_v2_release/head.pt",
        "url": "https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.4.0/signalscope-clip-mlp-v2-head.pt",
        "bytes": destination.stat().st_size, "sha256": head_sha,
        "architecture": "clip_vitl14_mlp", "preprocessing": "clip_center_crop_v1", "image_size": 224,
        "threshold": payload["threshold"], "temperature": payload["temperature"], "calibrated": True,
        "hidden_units": payload["hidden_units"], "dropout": payload["dropout"],
        "artifacts": [{"name": "frozen_generic_clip_image_tower",
                      "path": "model/checkpoints/clip_vitl14_visual/visual_fp16.ts",
                      "url": TOWER_URL, "bytes": TOWER.stat().st_size, "sha256": tower_sha}],
        "training_data": "Our head: CIFAKE 8000 + GenImage BigGAN/SD1.5 6239 + AIDA train 1183 + "
                         "CommunityForensics train 2122; original and matched views. Generic CLIP "
                         "tower is pretrained, not ours.",
        "status": ("Active v0.4.0. Deliberately favours current-generator accuracy over the prior "
                  "release's 2021-2023-vintage development AUC; both are disclosed. See README "
                  "and docs/POST_RELEASE_EXPERIMENTS.md for the full measured tradeoff."),
        "license_note": "Generic CLIP image tower attribution applies. GenImage/CommunityForensics-derived "
                        "head is a noncommercial research artifact (CommunityForensics-Eval is CC BY-NC-SA 4.0).",
        "parent_checkpoint_sha256": digest(source),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"head_sha256": head_sha, "head_bytes": destination.stat().st_size,
                      "tower_sha256": tower_sha, "manifest": str(manifest_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
