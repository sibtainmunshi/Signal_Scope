# External evaluation protocol (fixed before predictions)

Source: [UniversalFakeDetect authors, CVPR 2023](https://github.com/WisconsinAIVision/UniversalFakeDetect).
Only the diffusion release is acquired (917,979,875 bytes), not face-swap datasets.
All images remain outside Git; public demos use manually reviewed objects/animals/scenes.
Source code licensing does not establish the license of every dataset image.

Development uses 500 LDM-200 fake images paired with 500 LAION real images and
500 guided-diffusion images paired with 500 ImageNet real images. Sort filenames
within each domain by SHA-256 of `signalscope-2026:` plus the archive filename,
then take the first 500. Authors' real/fake domain pairings are preserved.
These scores can guide engineering and are explicitly **development results**.
No image from this release is used for classifier gradient updates.

Reserve GLIDE (all three configurations) and DALLE synthetic domains for a final
frozen-model evaluation. Their real comparison set uses the *remaining* LAION
images. Do not evaluate these domains during model selection. They are unseen
relative to CIFAKE fine-tuning; unknown exposure in ImageNet-pretrained backbones
must be acknowledged. Generator names follow the source release, not current
commercial model versions. Count duplicates and exclude cross-role pixel overlaps
before scoring. Report per-domain AUC and macro average; sharing a real set across
domains must not be concealed by pooling it multiple times as unique samples.

Do not tune the threshold against external final labels. Use the persisted
development threshold. Record preprocessing, checkpoint hash, counts and data
hashes. Do not describe a public self-evaluation as the organizer's hidden test.
