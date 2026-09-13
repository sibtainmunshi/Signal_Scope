"""Prepare v0.3.0 artifacts without switching the active manifest or scoring reserves."""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from signalscope.clipmodel import DEFAULT_VISUAL


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parent = ROOT / "model/checkpoints/mixed_clip_l14_balanced_v1/head.pt"
    source_report = ROOT / "report/experiments/clip_l14_development_v1/results.json"
    recorded = next(r for r in json.loads(source_report.read_text())["candidates"] if r["run"] == "mixed_clip_l14_balanced_v1")
    if digest(parent) != recorded["checkpoint_sha256"]:
        raise ValueError("Candidate head changed since development evaluation")
    visual = ROOT / DEFAULT_VISUAL
    visual_sha = digest(visual)
    if visual.stat().st_size != 608352029 or visual_sha != "12403c44d349dee827e7d0fe016c2f255ec51ccf2750d5b4260ab9c878bf7989":
        raise ValueError("Exported image tower differs from verified artifact")
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    if payload["threshold"] != recorded["threshold"] or payload["temperature"] != recorded["temperature"]:
        raise ValueError("Candidate operating point differs from recorded development experiment")
    run = "mixed_clip_l14_balanced_v1_release"
    destination = ROOT / "model/checkpoints" / run / "head.pt"
    manifest_path = ROOT / "model/releases/v0.3.0.json"
    if destination.exists() or manifest_path.exists():
        raise SystemExit("Prepared release exists; verify it instead of overwriting.")
    payload["visual_path"] = DEFAULT_VISUAL
    payload["visual_sha256"] = visual_sha
    payload["config"] = payload["config"] | {"run": run, "parent_run": recorded["run"],
        "parent_checkpoint_sha256": recorded["checkpoint_sha256"],
        "release_decision": "Prioritize measured AUC improvement; declared FPR-equivalence gate FAILED and remains disclosed"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    base_url = "https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.3.0/"
    manifest = {"schema_version": 2, "release": "v0.3.0", "model_version": run,
                "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
                "url": base_url + "signalscope-clip-l14-balanced-head.pt", "bytes": destination.stat().st_size,
                "sha256": digest(destination), "architecture": "clip_vitl14_linear", "preprocessing": "clip_center_crop_v1",
                "image_size": 224, "threshold": payload["threshold"], "temperature": payload["temperature"], "calibrated": True,
                "artifacts": [{"name": "frozen_generic_clip_image_tower", "path": DEFAULT_VISUAL,
                               "url": base_url + "signalscope-clip-vitl14-visual-fp16.ts",
                               "bytes": visual.stat().st_size, "sha256": visual_sha}],
                "training_data": "Our head: CIFAKE8000 + GenImageBigGAN/SD1.5 6239; original and matched views,90% GenImage weight. Generic CLIP tower is pretrained, not ours.",
                "status": "Prepared candidate release; activation requires completed explanation/integration verification. Development AUC0.771/0.789; gate FAILED due to22.0% LAION real FPR.",
                "license_note": "Generic CLIP image tower attribution applies. GenImage-derived head is a noncommercial research artifact.",
                "parent_checkpoint_sha256": recorded["checkpoint_sha256"]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    output = ROOT / "report/releases/v0.3.0"
    output.mkdir(parents=True, exist_ok=True)
    decision = {"prepared_utc": datetime.now(UTC).isoformat(), "manifest_sha256": digest(manifest_path),
                "checkpoint_sha256": manifest["sha256"], "visual_sha256": visual_sha,
                "operating_point": {"threshold": manifest["threshold"], "temperature": manifest["temperature"]},
                "parent_development_report_sha256": digest(source_report), "development_results": recorded["external"],
                "gate_checks": recorded["gate_checks"], "gate_passed": False,
                "decision": "User-directed submission preparation prioritizing AUC despite failed FPR-equivalence gate; no fourth model variant or threshold tuning",
                "evaluation_limits": "Known development results only; actual TorchScript CPU parity measured separately; no new reserved result yet",
                "active_manifest_changed": False, "published": False}
    (output / "preparation.json").write_text(json.dumps(decision, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "threshold": manifest["threshold"],
                      "head_sha256": manifest["sha256"], "total_bytes": manifest["bytes"]+visual.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
