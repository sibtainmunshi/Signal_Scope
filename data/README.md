# Dataset provenance

The team was not supplied a dataset or baseline. We source public data and label
all of our metrics as self-evaluated. No organizer hidden score is claimed.

Start with the authors' [CIFAKE release](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images):
real CIFAR-10 images and synthetic Stable Diffusion 1.4 images. The publisher
declares an MIT license. Cite both Krizhevsky & Hinton (2009) and Bird & Lotfi
(2024), "CIFAKE: Image Classification and Explainable Identification of
AI-Generated Synthetic Images," IEEE Access.

`python scripts/download_cifake.py` downloads and safely extracts the archive,
then writes its actual checksum and acquisition metadata. Data bytes stay out of
Git. Dataset counts/splits are recorded by the preparation step. The source test
partition is reserved, not used for model selection. CIFAKE is a narrow benchmark
and alone cannot establish generalisation to unseen generators or modern images.

Extra high-resolution datasets will be added only with traceable source,
generator metadata, split/duplicate checks, and the problem statement's scope
exclusions. No face-swap or identifiable-person collection is part of this project.


## Bounded GenImage training acquisition

Original source: [GenImage authors](https://github.com/GenImage-Dataset/GenImage),
Zhu et al., NeurIPS 2023. Data access uses the explicitly identified third-party
[HF mirror jzousz/GenImage](https://huggingface.co/datasets/jzousz/GenImage), pinned
to revision `71c983e6262684bc2c6b6af99582e8f568c259a5`. Mirror provenance is
not claimed as author-hosted, and this is not the entire original benchmark.

The [authors' license](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
is CC BY-NC-SA 4.0 with additional noncommercial terms. Training images remain
outside Git. Any candidate trained on these images is treated as a noncommercial
research artifact; do not describe it as an unrestricted commercial model.

Selection uses **original train folders only**, BigGAN and Stable Diffusion 1.5,
two images per ImageNet category per class where real/generated counterparts
exist. Stable Diffusion's nature folder has 958 categories, so we use the
canonical 1000-class ImageNet mapping derived from BigGAN's complete directory;
we do not renumber missing categories. Person-centered indices 981?983 are
excluded. This yields 7,808 selected images before overlap exclusions. Category
filtering does not establish that every background is free of people. Demo
assets require their own review; no identity inference is performed.

The source is distributed as multipart ZIPs. We download only the central
directories and selected compressed entry byte-ranges, verify local filenames,
uncompressed sizes and CRC-32, decode each image and record SHA-256 hashes. The
full archives are not downloaded. Source URLs, revision, index hashes and actual
counts are recorded under data/manifests. Image bytes and per-image JSONL/CSV
records remain local. Re-run the deterministic scripts to regenerate them.

Before training, selected images are checked against all acquired CIFAKE and
external exact pixel hashes, plus a conservative external 64-bit DCT perceptual
hash comparison (distance <=4). Selected-set duplicate groups are split together
80/10/10 into train/validation/calibration. The perceptual check is a heuristic,
not exhaustive proof of non-overlap; false matches may be excluded conservatively.

GenImage/BigGAN and SD1.5 remain distinct from the reserved GLIDE/DALLE generators.
External benchmark images remain evaluation-only. LDM development is related
to Stable Diffusion's model family; it must not be described as a wholly unrelated
held-out generator family once SD1.5 is added to training. Backbone pretraining
exposure remains unknown.
