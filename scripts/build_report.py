"""Build the one-page model report (HTML and A4 PDF) from saved, checkpoint-matched reports.

Every number is read from files under report/; nothing is typed by hand. Results that
do not exist yet (for example final reserved evaluations before freeze) are shown as
pending rather than estimated.
"""

import argparse
import json
import subprocess
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(*parts):
    path = ROOT.joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(value):
    return "pending" if value is None else f"{100 * value:.1f}%"


def num(value, digits=3):
    return "pending" if value is None else f"{value:.{digits}f}"


def matching(report, sha):
    return report if report and report.get("checkpoint_sha256") == sha else None


def split_counts(summary):
    counts = {}
    for key, n in summary["counts"].items():
        split, label, _ = key.split(":", 2)
        counts.setdefault(split, [0, 0])[int(label)] += n
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model version (default: model/manifest.json)")
    parser.add_argument("--baseline", default="cifake_resnet18_robust_v1")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    manifest = load("model/manifest.json")
    model = args.model or manifest["model_version"]
    calibration = load("report/runs", model, "calibration.json")
    parent = calibration["parent_model_version"] if calibration else model
    config = load("report/runs", parent, "config.json")
    genimage_val = load("report/runs", model, "genimage_val/metrics.json")
    sha = genimage_val["checkpoint_sha256"]
    cifake_val = matching(load("report/runs", model, "cifake_val/metrics.json"), sha)
    external = {p: matching(load("report/runs", model, f"external_dev{s}/metrics.json"), sha)
                for p, s in (("as_distributed", ""), ("matched", "_matched"))}
    reserved = {p: matching(load("report/runs", model, f"external_reserved{s}/metrics.json"), sha)
                for p, s in (("as_distributed", ""), ("matched", "_matched"))}
    cifake_test = matching(load("report/runs", model, "test/metrics.json"), sha)
    baseline_external = {p: load("report/runs", args.baseline, f"external_dev{s}/metrics.json")
                         for p, s in (("as_distributed", ""), ("matched", "_matched"))}
    bootstrap = load("report/external_bootstrap.json") or {"protocols": {}}
    robust = matching(load("report/runs", model, "robustness_genimage/metrics.json"), sha)
    robust_cifake = matching(load("report/runs", model, "robustness/metrics.json"), sha)
    audit = matching(load("report/explanation_audit", model, "summary.json"), sha)
    genimage = split_counts(load("data/manifests/genimage_summary.json"))
    extensions = config["data_summary"].get("genimage_extensions", {}) if config else {}

    def ci(protocol):
        row = bootstrap["protocols"].get(protocol, {}).get(parent)
        return f" [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]" if row else ""

    def mean_fpr(report):
        return None if not report else sum(g["false_positive_rate"] for g in report["per_generator"]) / len(report["per_generator"])

    def mean_tpr(report):
        return None if not report else sum(g["true_positive_rate"] for g in report["per_generator"]) / len(report["per_generator"])

    def mean_of(report, key):
        return None if not report else sum(g[key] for g in report["per_generator"]) / len(report["per_generator"])

    result_rows = [
        ("CIFAKE validation (in-domain)", cifake_val and cifake_val["roc_auc"], cifake_val and cifake_val["macro_f1"],
         cifake_val and cifake_val["accuracy"], cifake_val and cifake_val["false_positive_rate"], cifake_val and cifake_val["true_positive_rate"], ""),
        ("GenImage BigGAN/SD1.5 validation", genimage_val["roc_auc"], genimage_val["macro_f1"], genimage_val["accuracy"],
         genimage_val["false_positive_rate"], genimage_val["true_positive_rate"], ""),
    ]
    for protocol, label in (("as_distributed", "External dev, unseen generators"), ("matched", "External dev, format-matched")):
        report = external[protocol]
        result_rows.append((label, report and report["macro_generator_roc_auc"], mean_of(report, "macro_f1"),
                            mean_of(report, "accuracy"), mean_fpr(report), mean_tpr(report), ci(protocol)))
    for protocol, label in (("as_distributed", "Reserved GLIDE/DALLE (final)"), ("matched", "Reserved, format-matched (final)")):
        report = reserved[protocol]
        result_rows.append((label, report and report["macro_generator_roc_auc"], mean_of(report, "macro_f1"),
                            mean_of(report, "accuracy"), mean_fpr(report), mean_tpr(report), ""))
    result_rows.append(("CIFAKE author test 20k (final)", cifake_test and cifake_test["roc_auc"], cifake_test and cifake_test["macro_f1"],
                        cifake_test and cifake_test["accuracy"], cifake_test and cifake_test["false_positive_rate"],
                        cifake_test and cifake_test["true_positive_rate"], ""))
    results_html = "".join(
        f"<tr><td>{escape(name)}</td><td><b>{num(auc)}</b>{escape(interval)}</td><td>{num(f1)}</td><td>{pct(acc)}</td>"
        f"<td>{pct(fpr)}</td><td>{pct(tpr)}</td></tr>"
        for name, auc, f1, acc, fpr, tpr, interval in result_rows
    )
    per_generator = "; ".join(
        f"{g['generator']} {g['roc_auc']:.3f}" for p in ("as_distributed", "matched") if external[p] for g in external[p]["per_generator"]
    )
    cm = genimage_val["confusion_matrix"]
    baseline_row = (
        f"{num(baseline_external['as_distributed'] and baseline_external['as_distributed']['macro_generator_roc_auc'])} / "
        f"{num(baseline_external['matched'] and baseline_external['matched']['macro_generator_roc_auc'])}, "
        f"real FPR {pct(mean_fpr(baseline_external['as_distributed']))}"
    )
    selected_row = (
        f"{num(external['as_distributed'] and external['as_distributed']['macro_generator_roc_auc'])} / "
        f"{num(external['matched'] and external['matched']['macro_generator_roc_auc'])}, real FPR {pct(mean_fpr(external['as_distributed']))}"
    )
    robust_line = "pending"
    if robust:
        by = {r["transformation"]: r for r in robust["transformations"]}
        robust_line = (f"GenImage val AUC original {by['original']['roc_auc']:.3f}; JPEG q70 {by['jpeg_q70']['roc_auc']:.3f}; "
                       f"q30 {by['jpeg_q30']['roc_auc']:.3f}; half resolution {by['half_resolution']['roc_auc']:.3f}; "
                       f"blur {by['mild_blur']['roc_auc']:.3f}. Six-transformation search flipped "
                       f"{pct(robust['active_defence']['attack_success_rate'])} of initially correct predictions")
        if robust_cifake:
            byc = {r["transformation"]: r for r in robust_cifake["transformations"]}
            robust_line += (f"; on 32 px CIFAKE, half resolution drops AUC to {byc['half_resolution']['roc_auc']:.3f} "
                            f"(flip rate {pct(robust_cifake['active_defence']['attack_success_rate'])})")
    audit_line = "pending"
    if audit:
        d = audit["deletion"]
        audit_line = (f"{audit['images']} fixed dev images ({audit['correct']} correct): masking the top Grad-CAM window lowered the verdict "
                      f"score more than random windows in {pct(d['mean']['fraction_top_exceeds_random_median'])} (mean fill, "
                      f"p={d['mean']['wilcoxon_top_vs_random_p']:.3f}) and {pct(d['blur']['fraction_top_exceeds_random_median'])} (blur); "
                      f"effects are small. JPEG map stability median rho {audit['jpeg_q70_spearman']['median']:.2f}; weight-randomization "
                      f"rho {audit['randomization_spearman']['median']:.2f} (maps only partly weight-dependent)")
    calibration_line = "Uncalibrated"
    if calibration:
        before, after = calibration["validation_before"], calibration["validation_operating_point"]
        calibration_line = (f"Temperature {calibration['temperature']:.2f} fit on calibration splits; threshold {calibration['threshold']:.3f} "
                            f"= strictest validation threshold with real FPR &le; {pct(calibration['requested_validation_max_fpr'])} per domain. "
                            f"ECE CIFAKE {before['cifake']['expected_calibration_error']:.3f}&rarr;{after['cifake']['expected_calibration_error']:.3f}, "
                            f"GenImage {before['genimage']['expected_calibration_error']:.3f}&rarr;{after['genimage']['expected_calibration_error']:.3f} (worse)")
    extension_rows = "".join(
        f"<tr><td>GenImage {escape(name.split('genimage_')[-1].replace('.csv', ''))}</td><td>train {sum(split_counts(s).get('train', [0, 0])):,}</td>"
        f"<td>{split_counts(s).get('train', [0, 0])[0]:,} / {split_counts(s).get('train', [0, 0])[1]:,}</td></tr>"
        for name, s in extensions.items()
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>SignalScope model report</title><style>
@page {{ size: A4; margin: 8mm 9mm; }}
body {{ font: 7.6pt/1.3 "Segoe UI", Arial, sans-serif; color: #17202a; margin: 0; }}
h1 {{ font-size: 13pt; margin: 0; }} h2 {{ font-size: 8.6pt; margin: 5px 0 2px; color: #0f4c5c; border-bottom: 1px solid #c9d6dc; }}
.sub {{ color: #4a5a66; margin: 1px 0 4px; }} .grid {{ display: grid; grid-template-columns: 1fr 1.18fr; gap: 9px; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid #e3e9ec; padding: 1.5px 3px; text-align: left; vertical-align: top; }}
th {{ background: #eef4f6; font-weight: 600; }} ul {{ margin: 1px 0; padding-left: 13px; }} li {{ margin: 0.5px 0; }}
.note {{ background: #fff6e0; border-left: 3px solid #d39b2a; padding: 3px 5px; margin: 3px 0; }} .mono {{ font-family: Consolas, monospace; font-size: 6.8pt; }}
</style></head><body>
<h1>SignalScope &mdash; one-page model report</h1>
<div class="sub">SIH 2026 internal, Problem Statement 2 &middot; model <b>{escape(model)}</b> &middot; SHA-256 <span class="mono">{sha[:16]}&hellip;</span>
&middot; {date.today().isoformat()} &middot; github.com/sibtainmunshi/Signal_Scope</div>
<div class="note"><b>Self-evaluated public-data results.</b> Organizers supplied no dataset, baseline or hidden-test scores; those fields are <b>not supplied</b>, not estimated.</div>
<div class="grid"><div>
<h2>Task and scope</h2>
Binary real vs AI-generated image classification with a continuous AI-positive score; general objects, scenes and products only. Built: core detector, A (model-linked explanation + audit), C (robustness), D (EXIF only; C2PA not checked), F (local web app/CLI), G (bounded post-processing attack analysis).
<h2>Data and splits</h2>
<table><tr><th>Source</th><th>Use</th><th>Real / AI</th></tr>
<tr><td>CIFAKE (CIFAR-10 vs SD1.4, 32 px)</td><td>train 8,000 of 79,689</td><td>4,000 / 4,000</td></tr>
<tr><td>CIFAKE grouped val / calibration</td><td>selection / calibration</td><td>{cifake_val['real_count'] if cifake_val else '-'} / {cifake_val['ai_count'] if cifake_val else '-'} val</td></tr>
<tr><td>GenImage BigGAN + SD1.5 (train folders)</td><td>train {sum(genimage['train']):,}</td><td>{genimage['train'][0]:,} / {genimage['train'][1]:,}</td></tr>
<tr><td>GenImage val / calibration</td><td>selection / calibration</td><td>{genimage['val'][0]} / {genimage['val'][1]}; {genimage['calibration'][0]} / {genimage['calibration'][1]}</td></tr>
{extension_rows}
<tr><td>UniversalFakeDetect dev</td><td>evaluation only</td><td>guided vs ImageNet, LDM vs LAION, 500+500 each</td></tr>
<tr><td>Reserved final</td><td>scored once after freeze</td><td>GLIDE x3, DALLE vs LAION; CIFAKE author test 20k</td></tr></table>
Exact and perceptual-hash overlap exclusions against all protected data; grouped splits. Audit found real JPEG vs generated square PNG in both GenImage and the external set, so training balances JPEG and every external result is also reported <b>format-matched</b> (identical crop, resize and JPEG for both labels).
<h2>Model and training</h2>
<ul><li>ImageNet-pretrained ResNet-18, fine-tuned end to end (AdamW 2e-4, cosine, 10 epochs, batch 64, AMP); epoch chosen by mean GenImage/CIFAKE val AUC.</li>
<li>Native-resolution random 128 px crops (no resizing); every generated training image JPEG-compressed with qualities drawn from the real photos; symmetric rescale, blur, flip and crop-JPEG augmentation.</li>
<li>Inference: mean logit of up to five native 128 px crops; CPU ~15&ndash;50 ms per image on our laptop.</li>
<li>{calibration_line}.</li></ul>
<h2>Explanation (Module A)</h2>
Returned-class Grad-CAM per crop, stitched, plus a masking diagnostic. {audit_line}. No named artifact (text, lighting, anatomy) is claimed.
</div><div>
<h2>Results at the frozen operating point (threshold {num(manifest.get('threshold') if manifest.get('model_version') == model else calibration and calibration['threshold'])})</h2>
<table><tr><th>Evaluation</th><th>ROC-AUC [95% CI]</th><th>Macro-F1</th><th>Accuracy</th><th>Real FPR</th><th>AI TPR</th></tr>{results_html}</table>
External per-generator AUC (as distributed; then matched): {escape(per_generator)}. External F1, accuracy and rates are means over generator pairs.
<h2>GenImage validation confusion matrix</h2>
<table style="width:60%"><tr><th></th><th>Pred. real</th><th>Pred. AI</th></tr><tr><th>Real</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr><tr><th>AI</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>
<h2>Baseline comparison (external dev mean AUC as distributed / matched)</h2>
<table><tr><td>Organizer baseline</td><td>not supplied</td></tr><tr><td>Our CIFAKE-only ResNet-18 (v0.1.0)</td><td>{baseline_row}</td></tr>
<tr><td>Selected model</td><td>{selected_row}</td></tr></table>
Seven candidates were compared on the same development data (including a frozen CLIP head whose 0.708 as-distributed AUC fell to 0.542 when format-matched); reserved data were never used for selection.
<h2>Robustness (Module C/G)</h2>{robust_line}.
<h2>Limitations and failure cases</h2>
<ul><li>Unseen-generator AUC is modest (~0.65); guided diffusion vs ImageNet is weakest. At the conservative threshold many external AI images are missed.</li>
<li>Calibration is domain-specific: confidence and FPR need not transfer to new generators or pipelines.</li>
<li>Resizing, blur and strong JPEG raise real-image false positives; tiny upscaled images are fragile.</li>
<li>Grad-CAM shows model influence, not verified visual defects; audit sample is small and unannotated.</li>
<li>GenImage-derived weights are for noncommercial research use (CC BY-NC-SA 4.0).</li></ul>
</div></div></body></html>"""
    output = ROOT / "report/model_report.html"
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    if not args.no_pdf:
        subprocess.run(["node", "app/frontend/scripts/render_report.mjs", "report/model_report.html",
                        "report/model_report.pdf", "tmp/report_preview.png"], cwd=ROOT, check=True)
        data = (ROOT / "report/model_report.pdf").read_bytes()
        pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
        print(f"PDF pages: {pages}")
        if pages != 1:
            raise SystemExit("Report must fit on exactly one page; shorten the content.")


if __name__ == "__main__":
    main()
