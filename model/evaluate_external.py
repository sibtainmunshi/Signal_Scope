"""Evaluate matched external domains without training or threshold fitting."""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT

PAIRS={"guided":"imagenet","ldm_200":"laion","glide_100_27":"laion","glide_50_27":"laion","glide_100_10":"laion","dalle":"laion"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--split",choices=["dev","reserved"],default="dev")
    parser.add_argument("--final-test",action="store_true")
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    if args.split == "reserved" and not args.final_test:
        parser.error("Reserved generators require --final-test after model and threshold freeze.")
    detector=Detector(args.checkpoint)
    with (ROOT/"data/manifests/external.csv").open(encoding="utf-8",newline="") as stream:
        records=[r for r in csv.DictReader(stream) if r["role"]==args.split and not r["exclusion"]]
    domains=["guided","ldm_200"] if args.split=="dev" else ["glide_100_27","glide_50_27","glide_100_10","dalle"]
    used=set(domains)|{PAIRS[d] for d in domains}
    records=[r for r in records if r["domain"] in used]
    with zipfile.ZipFile(ROOT/"data/downloads/universalfakedetect_diffusion.zip") as archive:
        for start in range(0,len(records),args.batch_size):
            batch=records[start:start+args.batch_size]
            inputs=[]
            for row in batch:
                with Image.open(io.BytesIO(archive.read(row["path"]))) as image:
                    inputs.append(detector.tensor(image))
            with torch.inference_mode():
                scores=detector.score_tensor(torch.cat(inputs)).cpu().tolist()
            for row,score in zip(batch,scores,strict=True):
                row["ai_score"]=score
    metrics=[]
    for domain in domains:
        selected=[r for r in records if r["domain"] in {domain,PAIRS[domain]}]
        row={"generator":domain,"real_source":PAIRS[domain],**binary_metrics([int(r["label"]) for r in selected],[r["ai_score"] for r in selected],detector.threshold)}
        metrics.append(row)
        print(json.dumps(row),flush=True)
    report={"source":"UniversalFakeDetect CVPR 2023 diffusion release","split":args.split,
            "evaluation_kind":"external_development" if args.split=="dev" else "frozen_model_public_reserved_generators",
            "model_version":detector.model_version,"checkpoint_sha256":detector.checkpoint_hash,
            "preprocessing":("Diagnostic native32 bicubic resize then normal 96px preprocessing" if detector.model_version.endswith("_native32_probe") else "Same full-image preprocessing as app/CLI; no external-domain adaptation."),
            "threshold":detector.threshold,"unique_images_evaluated":len(records),
            "macro_generator_roc_auc":float(np.mean([m["roc_auc"] for m in metrics])),"per_generator":metrics,
            "organizer_hidden_result":None,
            "limitations":["Domains are unseen in CIFAKE fine-tuning, not necessarily absent from backbone pretraining.","Development domains may guide model selection and are not a blind final test.","Shared LAION real images are reused for per-generator comparisons; no inflated pooled sample count is reported."]}
    output=ROOT/"report/runs"/detector.model_version/("external_"+args.split)
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    pred=ROOT/"report/predictions"/(detector.model_version+"_external_"+args.split+".csv")
    pred.parent.mkdir(parents=True,exist_ok=True)
    with pred.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("Macro generator ROC-AUC:",report["macro_generator_roc_auc"],flush=True)

if __name__ == "__main__":
    main()
