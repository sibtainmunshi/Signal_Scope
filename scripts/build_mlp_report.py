"""Build the v0.4.0 one-page model report from saved measurements only."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/releases/v0.4.0"


def main():
    manifest = json.loads((ROOT / "model/manifest.json").read_text())
    if manifest.get("release") != "v0.4.0":
        raise ValueError("This report describes the active v0.4.0 release")
    mlp = json.loads((ROOT / "report/experiments/mixed_clip_mlp_v2/results.json").read_text())
    val = mlp["validation_operating_point"]
    aida_eval = json.loads((ROOT / "report/experiments/aidetectarena_eval_v1/results.json").read_text())
    v030 = aida_eval["models"]["v0.3.0_clip_candidate"]["overall"]

    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>SignalScope v0.4.0 model report</title><style>
@page {{size:A4; margin:7mm 11mm;}}
* {{box-sizing:border-box;}} body {{font:8.3pt/1.22 "Segoe UI",Arial,sans-serif;color:#182b30;margin:0;}}
h1 {{font-size:18pt;letter-spacing:-.5px;margin:0;color:#154a43;}} .subtitle {{color:#546865;margin:2px 0 6px;}}
.status {{background:#fff4db;border-left:3px solid #aa7b24;padding:5px 8px;font-weight:600;margin-bottom:6px;}}
h2 {{font-size:9.3pt;margin:5px 0 3px;color:#154a43;border-bottom:1px solid #c8dbd5;padding-bottom:2px;}}
p {{margin:2px 0 4px;}} table {{border-collapse:collapse;width:100%;font-size:8.1pt;}}
th {{background:#eaf2ee;text-align:left;}} th,td {{padding:2px 5px;border-bottom:1px solid #dae5e0;}}
.grid {{display:grid;grid-template-columns:1.1fr 1fr;gap:12px;}} .small {{font-size:7.5pt;color:#4b5e59;}}
.warning {{border:1px solid #d3b87c;background:#fff9eb;padding:5px 8px;margin:5px 0;}}
.hash {{font:6.8pt Consolas,monospace;overflow-wrap:anywhere;}} .footer {{border-top:1px solid #c8dbd5;padding-top:4px;margin-top:6px;}}
.cm {{font:6.9pt Consolas,monospace;white-space:nowrap;}}
</style></head><body>
<h1>SignalScope</h1><div class="subtitle">SIH 2026 internal selection - Problem Statement 2 - v0.4.0 model report</div>
<div class="status">Active v0.4.0, frozen. Deployed by explicit decision after two gated remediation attempts failed; trades 2021-2023-vintage development AUC for current-generator accuracy.</div>
<p><b>Task:</b> real-versus-AI classification with a label, AI-positive score, CPU web/API/batch interface and model-influence explanation. Public-data results only; organizer data, baseline and hidden scores were not supplied.</p>
<div class="grid"><section><h2>Model, data and operating point</h2>
<p>Frozen generic <b>CLIP ViT-L/14</b> image tower plus our trained <b>2-layer MLP head</b> (768-&gt;128-&gt;1, ReLU, dropout 0.3), replacing v0.3.0's linear head. Training: <b>8,000 CIFAKE + 6,239 GenImage + 1,183 AIDA-train (17 current generators) + 2,122 CommunityForensics-train (~20 generators)</b>. 60/40 holdouts of both new sources fixed before training and never used for fitting.</p>
<p>Inference: EXIF orientation, official 224px CLIP centre crop, L2-normalized embedding, MLP head, temperature scaling. Temperature <b>{manifest["temperature"]:.6f}</b>; AI threshold <b>{manifest["threshold"]:.6f}</b>. Threshold uses internal GenImage/CIFAKE validation only; neither holdout was used to fit it.</p>
<p class="small">Best-epoch selection (epoch {mlp["best_epoch"]}/60) used internal validation AUC only; no holdout or external data influenced model/epoch selection.</p>
</section><section><h2>Runnable interface and attribution</h2>
<p>Shared CPU CLI/API; single image or batch with one loaded model. JPEG/PNG/WebP uploads up to <b>25 MiB / 40 MP</b>. EXIF is separate; C2PA presence is a heuristic substring scan, not signature verification.</p>
<p>Tower <b>608,352,029 bytes</b> (reused unchanged from v0.3.0, parity verified) + head <b>{manifest["bytes"]:,} bytes</b>, SHA-256 verified. <b>CPU-only</b> model runtime: torch, without clip/torchvision/ftfy/regex.</p>
<p>Measured CPU components (shared CLIP tower): predict <b>~0.38s/image</b>, explanation <b>~5.5s</b>, robustness <b>~5.1s</b>. Fresh public-clone setup+first-prediction: <b>243.75s</b> total, well under the 10-minute reproducibility bar.</p>
</section></div>
<h2>Why this model exists: the gate failed twice, deployed by explicit decision</h2>
<div class="warning"><b>Two independent 2025-2026-generator evaluations on the previous release (v0.3.0):</b> AI Detect Arena (17 generators) macro AUC <b>{v030["roc_auc"]:.3f}</b>, accuracy <b>{100*v030["accuracy"]:.1f}%</b>, real-photo FPR <b>{100*v030["false_positive_rate"]:.1f}%</b>. A model flagging two-thirds of real photos as AI is not usable. <b>Two gated remediation attempts both failed</b> on external-dev regression (linear+AIDA: 0.771-&gt;0.652 AUC; MLP+AIDA+CommunityForensics: 0.771-&gt;0.637 AUC), yet the MLP attempt's own untouched holdouts reached usable accuracy. Deployed by explicit user decision favouring current-generator accuracy over the old development benchmark - both facts disclosed together, not one substituted for the other.</p></div>
<h2>Required metrics: overall AUC, unseen-generator-split AUC, macro-F1, confusion matrix</h2>
<table><tr><th>Evaluation surface</th><th>AUC</th><th>Macro-F1</th><th>Acc.</th><th>Confusion [real / ai_generated rows=actual]</th></tr>
<tr><td>Overall - GenImage val (seen, n=783)</td><td>{val["genimage"]["roc_auc"]:.3f}</td><td>{val["genimage"]["macro_f1"]:.3f}</td><td>{100*val["genimage"]["accuracy"]:.1f}%</td><td class="cm">R:{val["genimage"]["confusion_matrix"][0]} A:{val["genimage"]["confusion_matrix"][1]}</td></tr>
<tr><td>Overall - CIFAKE val (seen, n=4000)</td><td>{val["cifake"]["roc_auc"]:.3f}</td><td>{val["cifake"]["macro_f1"]:.3f}</td><td>{100*val["cifake"]["accuracy"]:.1f}%</td><td class="cm">R:{val["cifake"]["confusion_matrix"][0]} A:{val["cifake"]["confusion_matrix"][1]}</td></tr>
<tr><td><b>Unseen split</b> - AI Detect Arena (17 generators, n=755)</td><td><b>{mlp["aida_holdout"]["roc_auc"]:.3f}</b></td><td>{mlp["aida_holdout"]["macro_f1"]:.3f}</td><td>{100*mlp["aida_holdout"]["accuracy"]:.1f}%</td><td class="cm">R:{mlp["aida_holdout"]["confusion_matrix"][0]} A:{mlp["aida_holdout"]["confusion_matrix"][1]}</td></tr>
<tr><td><b>Unseen split</b> - CommunityForensics-Eval (~20 generators, n=1374)</td><td><b>{mlp["communityforensics_holdout"]["roc_auc"]:.3f}</b></td><td>{mlp["communityforensics_holdout"]["macro_f1"]:.3f}</td><td>{100*mlp["communityforensics_holdout"]["accuracy"]:.1f}%</td><td class="cm">R:{mlp["communityforensics_holdout"]["confusion_matrix"][0]} A:{mlp["communityforensics_holdout"]["confusion_matrix"][1]}</td></tr>
</table>
<p class="small">Real-&gt;real / AI-&gt;AI: AIDA 92.7%/78.5%, CommunityForensics 97.8%/54.4%. Unseen-generator AUC (0.936-0.948) is not lower than seen-generator AUC (0.820-0.944): ranking generalises to new generators; the accuracy/FPR gap at the fixed threshold is where the deployment tradeoff below actually shows up. No organizer-provided baseline dataset or model exists for this internal hackathon; "Overall" here is our own in-distribution validation, not an organizer number. CommunityForensics-Eval is CC BY-NC-SA 4.0 (Park et al., CVPR 2025).</p>
<h2>Cost: the 2021-2023-vintage development benchmark this release trades away</h2>
<table><tr><th>External dev (GLIDE-family generators)</th><th>v0.3.0</th><th>v0.4.0</th></tr>
<tr><td>Mean AUC, as distributed</td><td>0.771</td><td>{mlp["external_dev"]["as_distributed"]["mean_auc"]:.3f}</td></tr>
<tr><td>Mean AUC, format matched</td><td>0.789</td><td>{mlp["external_dev"]["matched"]["mean_auc"]:.3f}</td></tr>
</table>
<p class="small">No GLIDE/DALLE or COCO reserved-set number is claimed for v0.4.0: that evaluation ran for v0.2.0/v0.3.0 only, and re-running it a third time was not judged worth spending that evidence on given the deployment decision already accepts this tradeoff. B-Free (third-party, CVPR 2025) scored 0.970/0.945 on the same 2021-2023-vintage development images without fitting - not our result.</p>
<h2>Limitations</h2>
<p><b>Explanation audit (re-run for this architecture):</b> statistically indistinguishable from v0.3.0's linear head - identical 17/40 (42.5%) localisation count, deletion test not significant on either head. <b>Module G robustness</b>: real-photo FPR stays low (0-3.3%) under JPEG/resize/blur/screenshot, but AI recall drops 52.7%-&gt;17.9% under simulated screenshot; a bounded 7-transform search flips 33.5% of initially-correct predictions. <b>Private user-image veto check</b> (11 AI + 18 real, never used for fitting): real-photo false positives roughly halved versus v0.3.0. No generator attribution, caption consistency or C2PA signature verification. No organizer hidden-test score is available or claimed. Human explanation-usefulness review remains outstanding.</p>
<div class="footer small">Sources: github.com/AI-Detect-Arena/benchmark-dataset; huggingface.co/datasets/OwensLab/CommunityForensics-Eval (Park et al., CVPR 2025); github.com/WisconsinAIVision/UniversalFakeDetect; github.com/openai/CLIP; github.com/GenImage-Dataset/GenImage; CIFAKE (Bird &amp; Lotfi). Code and full experiments: github.com/sibtainmunshi/Signal_Scope.<br>Head SHA-256: <span class="hash">{manifest["sha256"]}</span></div>
</body></html>'''
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source, pdf = OUTPUT / "model_report.html", OUTPUT / "model_report.pdf"
    source.write_text(html, encoding="utf-8")
    subprocess.run(["node", "app/frontend/scripts/render_report.mjs", str(source), str(pdf)], cwd=ROOT, check=True)
    from pypdf import PdfReader

    pages = len(PdfReader(pdf).pages)
    if pages != 1:
        raise SystemExit(f"Report must be one page, got {pages}")
    print(f"Verified page count: {pages}")


if __name__ == "__main__":
    main()
