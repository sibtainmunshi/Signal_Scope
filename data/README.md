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

