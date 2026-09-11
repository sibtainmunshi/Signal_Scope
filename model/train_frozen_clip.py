"""Train our own logistic head on frozen CLIP features; external dev is evaluation only."""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT


class ImageFeaturesDataset(Dataset):
    def __init__(self, data, preprocess):
        self.data, self.preprocess = data, preprocess
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        row = int(self.data.indices[index])
        return self.preprocess(Image.fromarray(np.array(self.data.images[row]))), int(self.data.labels[row])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="cifake_clip_b32_v1")
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(ROOT/".cache/clip"))
    model.eval().requires_grad_(False)
    report = ROOT/"report/runs"/args.run
    report.mkdir(parents=True, exist_ok=True)
    cache = ROOT/"data/processed/clip_b32"
    cache.mkdir(parents=True, exist_ok=True)
    def encode(images):
        with torch.inference_mode():
            features = model.encode_image(images.to(device)).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return features.cpu().numpy()
    def features_for(split, limit):
        data = CifakeDataset(ROOT/"data/processed/cifake", split, limit, 2026)
        digest = hashlib.sha256(data.indices.tobytes()).hexdigest()[:12]
        path = cache/f"{split}_{digest}.npz"
        if path.exists():
            saved = np.load(path)
            print(f"Reusing {len(saved['labels'])} {split} features", flush=True)
            return saved["features"], saved["labels"]
        all_features, labels = [], []
        started = time.monotonic()
        loader = DataLoader(ImageFeaturesDataset(data, preprocess), batch_size=args.batch_size, shuffle=False, num_workers=0)
        for i,(images,y) in enumerate(loader):
            all_features.append(encode(images))
            labels.append(y.numpy())
            if (i+1)%50 == 0:
                print(f"{split}: {min((i+1)*args.batch_size,len(data))}/{len(data)} in {time.monotonic()-started:.1f}s", flush=True)
        x, y = np.concatenate(all_features), np.concatenate(labels)
        np.savez(path, features=x, labels=y, indices=data.indices)
        return x,y
    train_x,train_y = features_for("train", args.train_limit)
    val_x,val_y = features_for("val", args.val_limit)
    candidates=[]
    best=None
    for regularization in (.1, 1., 10.):
        classifier=LogisticRegression(C=regularization,max_iter=2000,random_state=2026)
        classifier.fit(train_x,train_y)
        scores=classifier.predict_proba(val_x)[:,1]
        metrics=binary_metrics(val_y,scores,.5)
        candidates.append({"C":regularization,"validation":metrics})
        print(json.dumps(candidates[-1]),flush=True)
        if best is None or metrics["roc_auc"] > best[0]:
            best=(metrics["roc_auc"],classifier,regularization,metrics)
    _,classifier,regularization,validation=best
    config={"run":args.run,"architecture":"frozen CLIP ViT-B/32 + our logistic-regression head",
            "backbone_source":"https://github.com/openai/CLIP","clip_code_commit":"d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
            "preprocessing":str(preprocess),"embedding_l2_normalized":True,"C":regularization,
            "train_count":len(train_y),"validation_count":len(val_y),"seed":2026,
            "head_selection":"highest CIFAKE validation ROC-AUC among C=0.1,1,10",
            "training_data":"CIFAKE grouped training partition only; no external labels used for fitting",
            "created_utc":datetime.now(UTC).isoformat(),"threshold":.5,"calibrated":False}
    checkpoint=ROOT/"model/checkpoints"/args.run/"head.pt"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"weight":torch.from_numpy(classifier.coef_.astype(np.float32)),
                "bias":torch.from_numpy(classifier.intercept_.astype(np.float32)),"config":config},checkpoint)
    config["head_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (report/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    (report/"selection.json").write_text(json.dumps(candidates,indent=2)+"\n",encoding="utf-8")
    (report/"val").mkdir(exist_ok=True)
    (report/"val/metrics.json").write_text(json.dumps(validation|{"dataset":"CIFAKE","split":"val","model_version":args.run},indent=2)+"\n",encoding="utf-8")
    with (ROOT/"data/manifests/external.csv").open(encoding="utf-8",newline="") as stream:
        records=[r for r in csv.DictReader(stream) if r["role"]=="dev" and not r["exclusion"]]
    external_cache=cache/"external_dev.npz"
    if external_cache.exists():
        external_features=np.load(external_cache)["features"]
    else:
        collected=[]
        with zipfile.ZipFile(ROOT/"data/downloads/universalfakedetect_diffusion.zip") as archive:
            for start in range(0,len(records),args.batch_size):
                tensors=[]
                for row in records[start:start+args.batch_size]:
                    with Image.open(io.BytesIO(archive.read(row["path"]))) as image:
                        tensors.append(preprocess(image))
                collected.append(encode(torch.stack(tensors)))
        external_features=np.concatenate(collected)
        np.savez(external_cache,features=external_features,paths=np.array([r["path"] for r in records]))
    scores=classifier.predict_proba(external_features)[:,1]
    external=[]
    for generator,real in [("guided","imagenet"),("ldm_200","laion")]:
        selected=np.array([r["domain"] in {generator,real} for r in records])
        labels=np.array([int(r["label"]) for r in records])[selected]
        metrics=binary_metrics(labels,scores[selected],.5)
        external.append({"generator":generator,"real_source":real,**metrics})
        print(json.dumps(external[-1]),flush=True)
    external_report={"source":"UniversalFakeDetect CVPR 2023 diffusion release","split":"dev",
                     "evaluation_kind":"external_development","model_version":args.run,
                     "unique_images_evaluated":len(records),"macro_generator_roc_auc":float(np.mean([m["roc_auc"] for m in external])),
                     "per_generator":external,"organizer_hidden_result":None,
                     "limitation":"External development is not a blind final test; CLIP pretraining overlap is unknown. Experimental head is not integrated into the released app."}
    (report/"external_dev").mkdir(exist_ok=True)
    (report/"external_dev/metrics.json").write_text(json.dumps(external_report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"selected_C":regularization,"validation_auc":validation["roc_auc"],"external_macro_auc":external_report["macro_generator_roc_auc"]}),flush=True)

if __name__ == "__main__":
    main()
