"""Paired degradation benchmark and bounded post-processing failure search."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import matplotlib
import numpy as np
import torch
from PIL import Image

from signalscope.dataset import CifakeDataset
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT
from signalscope.robustness import TRANSFORMS, transform_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--limit",type=int,default=2000)
    parser.add_argument("--batch-size",type=int,default=128)
    args=parser.parse_args()
    detector=Detector(args.checkpoint)
    dataset=CifakeDataset(ROOT/"data/processed/cifake","val",args.limit,2026)
    labels=np.array([int(dataset.labels[i]) for i in dataset.indices])
    original=[Image.fromarray(np.array(dataset.images[i])) for i in dataset.indices]
    all_scores=[]
    rows=[]
    for name in TRANSFORMS:
        scores=[]
        for start in range(0,len(original),args.batch_size):
            images=[transform_image(im,name) for im in original[start:start+args.batch_size]]
            inputs=torch.cat([detector.tensor(im) for im in images])
            with torch.inference_mode():
                scores.extend(detector.score_tensor(inputs).cpu().numpy().tolist())
        values=np.array(scores)
        all_scores.append(values)
        metrics=binary_metrics(labels,values,detector.threshold)
        row={"transformation":name,**metrics}
        rows.append(row)
        print(json.dumps({"transformation":name,"accuracy":metrics["accuracy"],"auc":metrics["roc_auc"],"fpr":metrics["false_positive_rate"]}),flush=True)
    predictions=np.array(all_scores)>=detector.threshold
    originally_correct=predictions[0]==labels
    failures=np.any(predictions[1:]!=labels[None,:],axis=0)&originally_correct
    defence={"threat_model":"bounded grid of six common post-processing transformations",
             "initially_correct":int(originally_correct.sum()),"successfully_flipped":int(failures.sum()),
             "attack_success_rate":float(failures.sum()/originally_correct.sum()) if originally_correct.any() else None,
             "search_budget_per_image":len(TRANSFORMS)-1,
             "limitation":"This does not establish robustness to arbitrary adversarial attacks."}
    report={"dataset":"CIFAKE","split":"fixed 2000-image validation subset" if args.limit==2000 else "validation subset",
            "seed":2026,"model_version":detector.model_version,"checkpoint_sha256":detector.checkpoint_hash,
            "threshold":detector.threshold,"paired_image_indices":dataset.indices.tolist(),
            "transformations":rows,"active_defence":defence}
    output=ROOT/"report/runs"/detector.model_version/"robustness"
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,4.2),layout="constrained")
    x=np.arange(len(rows))
    ax.plot(x,[r["accuracy"] for r in rows],"o-",label="Accuracy",color="#237d67")
    ax.plot(x,[r["roc_auc"] for r in rows],"s-",label="ROC-AUC",color="#a9884e")
    ax.set(xticks=x,xticklabels=[r["transformation"].replace("_"," ") for r in rows],ylim=(0,1.03),ylabel="Metric",title="Paired CIFAKE validation degradation benchmark")
    ax.tick_params(axis="x",rotation=25,labelsize=8)
    ax.grid(axis="y",alpha=.2)
    ax.legend()
    fig.savefig(output/"degradation.png",dpi=160)
    plt.close(fig)
    print(json.dumps(defence),flush=True)


if __name__ == "__main__":
    main()
