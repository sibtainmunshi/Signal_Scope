"""Build the v0.3.0 one-page report from recorded data, preserving v0.2.0 files."""
import json
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/releases/v0.3.0"


def main():
    manifest = json.loads((ROOT / "model/releases/v0.3.0.json").read_text())
    data = json.loads((ROOT / "report/experiments/clip_l14_development_v1/results.json").read_text())
    candidate = next(r for r in data["candidates"] if r["run"] == "mixed_clip_l14_balanced_v1")
    active = json.loads((ROOT / "model/manifest.json").read_text())
    reserved_path = OUTPUT / "reserved_summary.json"
    reserved = json.loads(reserved_path.read_text()) if reserved_path.exists() else None
    is_active = active.get("sha256") == manifest["sha256"]
    state = "Active release" if is_active else "Prepared release draft - activation and final checks pending"
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
    for fmt, label in (("as_distributed", "GLIDE/DALLE - second use"), ("matched", "GLIDE/DALLE matched - second use")):
        add(label, averaged(reserved["external_reserved_second_use"][fmt]) if reserved else None)
    cm = candidate["validation_operating_point"]["genimage"]["confusion_matrix"]
    cocotext = "354 COCO reserved real photographs remain unscored; a first-use false-positive check follows freeze. It measures real-photo FPR only."
    if reserved:
        metrics = reserved["coco_reserved_real_first_use"]["candidate"]["as_distributed"]
        lo, hi = metrics["wilson_ci95"]
        cocotext = (f'COCO first-use reserved real check: {metrics["false_positives"]}/{metrics["count"]} false positives '
                    f'({100*metrics["false_positive_rate"]:.1f}%, Wilson 95% interval {100*lo:.1f}-{100*hi:.1f}%). Real-photo FPR only.')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>SignalScope v0.3.0 model report</title><style>
@page {{size:A4; margin:11mm 12mm;}}
* {{box-sizing:border-box;}} body {{font:9pt/1.35 "Segoe UI",Arial,sans-serif;color:#182b30;margin:0;}}
h1 {{font-size:21pt;letter-spacing:-.5px;margin:0;color:#154a43;}} .subtitle {{color:#546865;margin:3px 0 9px;}}
.status {{background:#fff4db;border-left:3px solid #aa7b24;padding:7px 9px;font-weight:600;margin-bottom:9px;}}
h2 {{font-size:10pt;margin:10px 0 4px;color:#154a43;border-bottom:1px solid #c8dbd5;padding-bottom:3px;}}
p {{margin:3px 0 6px;}} table {{border-collapse:collapse;width:100%;font-size:8.6pt;}}
th {{background:#eaf2ee;text-align:left;}} th,td {{padding:4px 5px;border-bottom:1px solid #dae5e0;}}
.grid {{display:grid;grid-template-columns:1.1fr 1fr;gap:15px;}} .small {{font-size:8pt;color:#4b5e59;}}
.warning {{border:1px solid #d3b87c;background:#fff9eb;padding:7px 9px;margin:9px 0;}}
.hash {{font:7.2pt Consolas,monospace;overflow-wrap:anywhere;}} .footer {{border-top:1px solid #c8dbd5;padding-top:6px;margin-top:10px;}}
</style></head><body>
<h1>SignalScope</h1><div class="subtitle">SIH 2026 internal selection - Problem Statement 2 - v0.3.0 model report</div>
<div class="status">{state}</div>
<p>Real versus AI-generated image classification with an AI-positive score, local CPU inference, batch JSONL and a web interface. Results below are self-evaluated public-data measurements. Organizer hidden scores and qualifying rank are unknown.</p>
<div class="grid"><section><h2>Model, data and operating point</h2>
<p>Frozen generic <b>CLIP ViT-L/14</b> image tower plus our trained logistic head. Training: <b>8,000 CIFAKE + 6,239 GenImage BigGAN/SD1.5</b> images; original and identical crop/JPEG views, 90% GenImage loss weight. C=10 selected on internal validation. The pretrained tower is credited to OpenAI; we trained the head.</p>
<p>Inference: EXIF orientation, official 224px CLIP centre crop, L2-normalized embedding, linear logit and temperature scaling. Temperature <b>{manifest['temperature']:.6f}</b>; AI threshold <b>{manifest['threshold']:.6f}</b>. Temperature uses separate calibration data; threshold uses internal validation, not external labels.</p>
<p class="small">Calibration: 9,969 CIFAKE + 781 GenImage; validation sizes below. Exact/perceptual overlap screening and grouped splits are heuristic. Backbone pretraining overlap is unknown. GenImage-derived head: noncommercial research use.</p>
</section><section><h2>Runnable interface and attribution</h2>
<p>Shared CPU CLI/API; single image or batch with one loaded model. JPEG/PNG/WebP uploads up to <b>25 MiB / 40 MP</b>. EXIF is separate and does not alter predictions; C2PA is not checked.</p>
<p>Packaged image tower + head: <b>608.36 MB</b>, verified by SHA-256. CLIP inference requires torch and the application dependencies, with no CLIP/torchvision/ftfy/regex install. Measured CPU scoring: approximately <b>0.38 seconds/image</b>; fresh-download setup timing is pending.</p>
<p>CLIP explanation integration and its 40-image audit are pending. The planned input-gradient saliency and masking check describe model influence, not verified visual defects. Human usefulness review remains pending.</p>
</section></div>
<h2>Measured development performance at the stated operating point</h2>
<table><tr><th>Evaluation</th><th>AUC</th><th>Macro-F1</th><th>Accuracy</th><th>Real FPR</th><th>AI recall</th></tr>{''.join(rows)}</table>
<p class="small">External rows average generator/source pairs (2,000 unique dev images per format). Development numbers use the training encoder. CPU dtype/export checks found zero label disagreements on measured samples, score drift up to 0.0097; parity evidence, not another accuracy claim. Organizer data and baseline were not supplied.</p>
<div class="warning"><b>Published acceptance failure:</b> balanced L/14 passed 11 of 12 internal checks, but as-distributed LDM/LAION real FPR was <b>22.0%</b>, versus v0.2.0's <b>12.8%</b> (allowed increase: 5 points). The gate <b>failed</b>. Release preparation prioritizes mean AUC improvement <b>0.647 to 0.771</b> original and <b>0.653 to 0.789</b> matched, accepting this disclosed tradeoff. Threshold retry and COCO augmentation did not fix the failed check.</div>
<div class="grid"><section><h2>Confusion matrix and evaluation integrity</h2>
<p>GenImage validation; rows actual real/AI, columns predicted real/AI: <b>[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]</b>.</p>
<p>{cocotext}</p><p>GLIDE/DALLE was already evaluated for v0.2.0. Any new result is a disclosed second use after freeze, never described as a fresh blind test. Old release scores and weights remain unchanged.</p>
</section><section><h2>Limitations and submission checks</h2>
<p>False alarms and missed unseen AI images remain material. Scores are not guarantees of origin or event truth. Per-domain calibration may not transfer; centre cropping omits image borders. Private user images are unavailable and were not retested.</p>
<p>The robustness interface includes JPEG, resizing, blur and a clearly labelled screenshot simulation. <b>New-model robustness measurements are pending</b>; the old ResNet's results are not attributed to CLIP. Generator attribution and caption consistency are not implemented.</p>
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
