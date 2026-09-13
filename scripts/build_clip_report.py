"""Build the v0.3.0 development report draft; never consume in-progress reserves."""
import json
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/releases/v0.3.0"


def main():
    manifest = json.loads((ROOT / "model/manifest.json").read_text())
    if manifest.get("release") != "v0.3.0":
        raise ValueError("This report describes the active v0.3.0 release")
    data = json.loads((ROOT / "report/experiments/clip_l14_development_v1/results.json").read_text())
    candidate = next(r for r in data["candidates"] if r["run"] == "mixed_clip_l14_balanced_v1")
    reserved_path = ROOT / "report/releases/v0.3.0/reserved_summary.json"
    reserved = json.loads(reserved_path.read_text()) if reserved_path.exists() else None
    state = ("Active v0.3.0 - frozen at b8d8d93; reserved evaluation complete, public asset publication pending"
             if reserved else "Active v0.3.0 - frozen at b8d8d93; final metrics and publication pending")
    rows = []
    def add(name, metrics):
        if metrics is None:
            rows.append(f'<tr><td>{escape(name)}</td><td colspan="5">Pending; no score claimed</td></tr>')
        else:
            rows.append(f'<tr><td>{escape(name)}</td><td>{metrics["roc_auc"]:.3f}</td><td>{metrics["macro_f1"]:.3f}</td>'
                        f'<td>{100*metrics["accuracy"]:.1f}%</td><td>{100*metrics["false_positive_rate"]:.1f}%</td>'
                        f'<td>{100*metrics["true_positive_rate"]:.1f}%</td></tr>')
    for domain, label in (("genimage", "GenImage validation (783)"), ("cifake", "CIFAKE validation subset (4,000)")):
        add(label, candidate["validation_operating_point"][domain])
    def averaged(report):
        return {key: sum(m[key] for m in report["per_generator"])/len(report["per_generator"])
                for key in ("roc_auc", "macro_f1", "accuracy", "false_positive_rate", "true_positive_rate")}
    for fmt, label in (("as_distributed", "External dev - original"), ("matched", "External dev - format matched")):
        add(label, averaged(candidate["external"][fmt]))
    if reserved:
        ext = reserved["external_reserved_second_use"]
        coco = reserved["coco_reserved_real_first_use"]
        ad, mt = ext["as_distributed"], ext["matched"]
        coco_cand_ad = coco["candidate"]["as_distributed"]
        coco_cand_mt = coco["candidate"]["matched"]
        coco_old_ad = coco["v0.2.0"]["as_distributed"]
        coco_old_mt = coco["v0.2.0"]["matched"]
        dalle_auc = next(g for g in ad["per_generator"] if g["generator"]=="dalle")["roc_auc"]
        glide_aucs = [g["roc_auc"] for g in ad["per_generator"] if g["generator"]!="dalle"]
        reserved_html = (f'<p><b>GLIDE/DALLE (4,500 images, disclosed second use):</b> macro AUC '
            f'<b>0.565 to {ad["macro_generator_roc_auc"]:.3f}</b> as distributed, '
            f'<b>0.643 to {mt["macro_generator_roc_auc"]:.3f}</b> matched. DALL-E {dalle_auc:.3f}, GLIDE '
            f'{min(glide_aucs):.3f}-{max(glide_aucs):.3f} (as distributed). '
            f'Accuracy/macro-F1 (as distributed): {100*ad["overall"]["accuracy"]:.1f}% / {ad["overall"]["macro_f1"]:.3f}. '
            f'<b>COCO reals (354, first use):</b> real FPR {100*coco_old_ad["false_positive_rate"]:.2f}% to '
            f'<b>{100*coco_cand_ad["false_positive_rate"]:.2f}%</b> as distributed, '
            f'{100*coco_old_mt["false_positive_rate"]:.2f}% to <b>{100*coco_cand_mt["false_positive_rate"]:.2f}%</b> matched. '
            f'The 22.0% development-set FPR finding does <b>not</b> reproduce on COCO - domain-specific to LAION-style '
            f'imagery, not universal. No organizer score claimed; AUC is ranking, not accuracy. 2021-2023 vintage '
            f'generators; organizer\'s set may differ. Full record: report/releases/v0.3.0/reserved_summary.json.</p>')
    else:
        reserved_html = ('<p><b>[PENDING - reserved evaluation in progress, will be filled before submission]</b></p>'
                        '<p class="small">Disclosed second use of the public GLIDE/DALLE reserve; COCO\'s 354 reserved reals '
                        'are first use. No retuning after freeze.</p>')
    cm = candidate["validation_operating_point"]["genimage"]["confusion_matrix"]
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>SignalScope v0.3.0 model report</title><style>
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
</style></head><body>
<h1>SignalScope</h1><div class="subtitle">SIH 2026 internal selection - Problem Statement 2 - v0.3.0 model report</div>
<div class="status">{state}</div>
<p><b>Task:</b> real-versus-AI classification with a label, AI-positive score, CPU web/API/batch interface and model-influence explanation. Public-data results only; organizer data, baseline and hidden scores were not supplied.</p>
<div class="grid"><section><h2>Model, data and operating point</h2>
<p>Frozen generic <b>CLIP ViT-L/14</b> image tower plus our trained logistic head. Training: <b>8,000 CIFAKE + 6,239 GenImage BigGAN/SD1.5</b> images; original and identical crop/JPEG views, 90% GenImage loss weight. C=10 selected on internal validation. The pretrained tower is credited to OpenAI; we trained the head.</p>
<p>Inference: EXIF orientation, official 224px CLIP centre crop, L2-normalized embedding, linear logit and temperature scaling. Temperature <b>{manifest['temperature']:.6f}</b>; AI threshold <b>{manifest['threshold']:.6f}</b>. Temperature uses separate calibration data; threshold uses internal validation, not external labels.</p>
<p class="small">Calibration: 9,969 CIFAKE + 781 GenImage; validation sizes below. Exact/perceptual overlap screening and grouped splits are heuristic. Backbone pretraining overlap is unknown. GenImage-derived head: noncommercial research use.</p>
</section><section><h2>Runnable interface and attribution</h2>
<p>Shared CPU CLI/API; single image or batch with one loaded model. JPEG/PNG/WebP uploads up to <b>25 MiB / 40 MP</b>. EXIF is separate and does not alter predictions; C2PA presence is a heuristic substring scan, not JUMBF parsing or signature verification.</p>
<p>Tower <b>608,352,029 bytes</b> + head <b>7,949 bytes</b>, SHA-256 verified. <b>CPU-only</b> model runtime: torch, without clip/torchvision/ftfy/regex; ordinary app dependencies remain required. Public assets and clean-download setup timing are pending.</p>
<p>Measured CPU components: predict <b>~0.38s/image</b> (ResNet ~64ms), explanation <b>~5.5s</b>, robustness <b>~5.1s</b>. Not combined-request guarantees. Input-gradient influence map uses a 14px patch grid plus masking diagnostic.</p>
</section></div>
<h2>Measured development performance at the stated operating point</h2>
<table><tr><th>Evaluation</th><th>AUC</th><th>Macro-F1</th><th>Accuracy</th><th>Real FPR</th><th>AI recall</th></tr>{''.join(rows)}</table>
<p class="small">External rows average two generator/source pairs, 2,000 unique dev images per format. AUC is ranking, not accuracy. Training-encoder metrics; CPU/export checks found zero label disagreements on measured samples with score drift up to 0.0097. GenImage validation confusion matrix (rows actual real/AI, columns predicted real/AI): <b>[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]</b>.</p>
<div class="warning"><b>Baseline and FAILED gate:</b> v0.2.0 to v0.3.0 mean AUC <b>0.647 to 0.771</b> original, <b>0.653 to 0.789</b> matched; guided matched <b>0.611 to 0.808</b>, LDM matched <b>0.695 to 0.769</b>. But <b>as_distributed_ldm_200_fpr FAILED</b>: real FPR <b>22.0% vs 12.8%</b> (allowed +5 points). Original, stricter-threshold and COCO attempts all failed this check. Selected model accepts this tradeoff. Guided recall at a diagnostic 5% FPR budget: <b>13.0% to 42.6%</b>; externally fitted comparison thresholds are not deployed.</div>
<h2>Final reserved evaluation (disclosed second use)</h2>
{reserved_html}
<div class="grid"><section><h2>User diagnostic and reference</h2>
<p>Private 11 ChatGPT + 18 phone images, v0.2.0 to CLIP: as-uploaded AI detections <b>0/11 to 9/11</b>, real flags <b>8/18 to 9/18</b>; matched detections <b>2/11 to 9/11</b>, real flags <b>2/18 to 4/18</b>. Small-sample CLIP AUC 0.732/0.788. <b>No accuracy percentage claim from 29 images</b>; no fitting, per-image detail private.</p>
<p><b>B-Free reference:</b> 0.970 original / 0.945 matched development AUC on the same images. Third-party weights, not our trained result or deployed component.</p>
</section><section><h2>Limitations and submission checks</h2>
<p><b>Weaker explanation localisation:</b> only <b>17/40</b> audit images satisfy the masking-support rule; the other 23 are disclosed as not localised. Deletion tests p=0.29/0.82. No verified visual-defect or AI-object localisation; human usefulness review pending.</p>
<p>Real-photo false positives remain high; unknown generators/calibration can fail and centre crops omit borders. New-model degradation benchmarks and updated demo pending; screenshot simulation is not device capture. No generator attribution, caption consistency or C2PA signature verification.</p>
</section></div>
<div class="footer small">Sources: github.com/WisconsinAIVision/UniversalFakeDetect (method/evaluation); github.com/openai/CLIP (backbone); github.com/GenImage-Dataset/GenImage; CIFAKE (Bird &amp; Lotfi). Code and full experiments: github.com/sibtainmunshi/Signal_Scope.<br>Head SHA-256: <span class="hash">{manifest['sha256']}</span></div>
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
