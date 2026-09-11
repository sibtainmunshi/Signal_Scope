"""Audit the public diffusion release and enforce predeclared development/final roles."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from signalscope.paths import ROOT


def key(name):
    return hashlib.sha256(("signalscope-2026:"+name).encode()).hexdigest()

def main():
    with (ROOT/"data/manifests/cifake.csv").open(encoding="utf-8", newline="") as stream:
        cifake_hashes = {r["pixel_sha256"] for r in csv.DictReader(stream)}
    records = []
    with zipfile.ZipFile(ROOT/"data/downloads/universalfakedetect_diffusion.zip") as archive:
        domains = defaultdict(list)
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 4 and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                domains[parts[1]].append(name)
        for domain, names in sorted(domains.items()):
            for i, name in enumerate(sorted(names, key=key)):
                if domain in {"laion", "imagenet"}:
                    role = "dev" if i < 500 else "reserved"
                elif domain in {"guided", "ldm_200"}:
                    role = "dev" if i < 500 else "unused"
                elif domain.startswith("glide_") or domain == "dalle":
                    role = "reserved"
                else:
                    role = "unused"
                data = archive.read(name)
                with Image.open(io.BytesIO(data)) as image:
                    image = image.convert("RGB")
                    # CIFAKE's decoded-pixel hash is SHA256 of RGB bytes.
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    width, height = image.size
                records.append({"path":name,"domain":domain,"label":0 if domain in {"laion","imagenet"} else 1,
                                "role":role,"pixel_sha256":pixel_hash,"sha256":hashlib.sha256(data).hexdigest(),
                                "width":width,"height":height,"exclusion":""})
            print(f"Audited {domain}: {len(names)} images", flush=True)
    groups = defaultdict(list)
    for r in records:
        groups[r["pixel_sha256"]].append(r)
    for digest, group in groups.items():
        labels = {r["label"] for r in group}
        roles = {r["role"] for r in group}
        for r in group:
            if digest in cifake_hashes:
                r["exclusion"] = "exact_pixel_overlap_with_cifake"
            elif len(labels)>1:
                r["exclusion"] = "conflicting_labels"
            elif "dev" in roles and "reserved" in roles:
                r["exclusion"] = "exact_pixel_overlap_between_dev_and_reserved"
        # Within a domain, retain one copy per decoded pixel group.
        seen = set()
        for r in sorted(group, key=lambda row: row["path"]):
            if r["domain"] in seen and not r["exclusion"]:
                r["exclusion"] = "within_domain_exact_duplicate"
            seen.add(r["domain"])
    output=ROOT/"data/manifests/external.csv"
    with output.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary={"images":len(records),"manifest_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
             "retained_counts":dict(Counter(r["domain"]+":"+r["role"] for r in records if not r["exclusion"])),
             "exclusions":dict(Counter(r["exclusion"] for r in records if r["exclusion"])),
             "duplicate_groups":sum(len(g)>1 for g in groups.values()),
             "audit_limit":"Exact decoded pixels only; near duplicates and unknown backbone pretraining overlap are not ruled out.",
             "protocol":"docs/EXTERNAL_EVALUATION.md"}
    (output.parent/"external_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__":
    main()
