"""Paired degradation benchmark and bounded post-processing failure search."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import matplotlib
import numpy as np
from PIL import Image, ImageOps

from signalscope.dataset import CifakeDataset
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT
from signalscope.robustness import SCREENSHOT_PROTOCOL, TRANSFORMS, transform_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dataset(name, limit):
    """Validation images only; GenImage uses the original files through the app's preprocessing."""
    if name == "cifake":
        dataset = CifakeDataset(ROOT/"data/processed/cifake", "val", limit, 2026)
        ids = [int(i) for i in dataset.indices]
        description = "fixed 2000-image CIFAKE validation subset" if limit == 2000 else "CIFAKE validation subset"
        return ids, np.array([int(dataset.labels[i]) for i in ids]), [Image.fromarray(np.array(dataset.images[i])) for i in ids], description
    with (ROOT/"data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted((r for r in csv.DictReader(stream) if r["split"] == "val" and not r["exclusion"]),
                      key=lambda r: int(r["cache_index"]))
    images = []
    for row in rows:
        with Image.open(ROOT/row["path"]) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    return ([int(r["cache_index"]) for r in rows], np.array([int(r["label"]) for r in rows]), images,
            "full GenImage BigGAN/SD1.5 validation split (original files)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--dataset",choices=["cifake","genimage"],default="cifake")
    parser.add_argument("--limit",type=int,default=2000)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="New report directory; existing results are never overwritten")
    args=parser.parse_args()
    detector=Detector(args.checkpoint, args.device)
    output = ROOT / args.output if args.output else ROOT/"report/runs"/detector.model_version/("robustness" if args.dataset=="cifake" else "robustness_genimage")
    if (output / "metrics.json").exists():
        raise SystemExit("Results already exist; choose a new --output directory.")
    ids,labels,original,description=load_dataset(args.dataset,args.limit)
    all_scores=[]
    rows=[]
    for name in TRANSFORMS:
        scores=[]
        for start in range(0,len(original),args.batch_size):
            scores.extend(detector.score_images([transform_image(im,name) for im in original[start:start+args.batch_size]]))
        values=np.array(scores)
        all_scores.append(values)
        metrics=binary_metrics(labels,values,detector.threshold)
        rows.append({"transformation":name,**metrics})
        print(json.dumps({"transformation":name,"accuracy":metrics["accuracy"],"auc":metrics["roc_auc"],"fpr":metrics["false_positive_rate"]}),flush=True)
    predictions=np.array(all_scores)>=detector.threshold
    originally_correct=predictions[0]==labels
    failures=np.any(predictions[1:]!=labels[None,:],axis=0)&originally_correct
    defence={"threat_model":f"bounded grid of {len(TRANSFORMS)-1} common post-processing transformations",
             "initially_correct":int(originally_correct.sum()),"successfully_flipped":int(failures.sum()),
             "attack_success_rate":float(failures.sum()/originally_correct.sum()) if originally_correct.any() else None,
             "search_budget_per_image":len(TRANSFORMS)-1,
             "limitation":"This does not establish robustness to arbitrary adversarial attacks."}
    report={"dataset":args.dataset,"split":description,"seed":2026,"model_version":detector.model_version,
            "checkpoint_sha256":detector.checkpoint_hash,"threshold":detector.threshold,
            "preprocessing":detector.preprocessing,"paired_image_ids":ids,"transformations":rows,"active_defence":defence,
            "screenshot_protocol": SCREENSHOT_PROTOCOL}
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,4.2),layout="constrained")
    x=np.arange(len(rows))
    ax.plot(x,[r["accuracy"] for r in rows],"o-",label="Accuracy",color="#237d67")
    ax.plot(x,[r["roc_auc"] for r in rows],"s-",label="ROC-AUC",color="#a9884e")
    ax.set(xticks=x,xticklabels=[r["transformation"].replace("_"," ") for r in rows],ylim=(0,1.03),ylabel="Metric",
           title=f"Paired {'CIFAKE' if args.dataset=='cifake' else 'GenImage'} validation degradation benchmark")
    ax.tick_params(axis="x",rotation=25,labelsize=8)
    ax.grid(axis="y",alpha=.2)
    ax.legend()
    fig.savefig(output/"degradation.png",dpi=160)
    plt.close(fig)
    print(json.dumps(defence),flush=True)


if __name__ == "__main__":
    main()
