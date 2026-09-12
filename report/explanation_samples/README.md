# Explanation samples

Three deliberately chosen GenImage validation examples: a real photograph, a detected generated image, and a confidently missed generated image. These are demonstrations, not a random accuracy estimate. All were outside classifier training and visually inspected for public presentation. No human usefulness rating is claimed.

Source: [GenImage authors](https://github.com/GenImage-Dataset/GenImage), Zhu et al., NeurIPS 2023, BigGAN/Stable Diffusion 1.5 subsets. Dataset terms: CC BY-NC-SA 4.0 and noncommercial research use. Original archive paths, exact source hashes, labels, unchanged model identity, and actual API results are in [samples.json](samples.json). Overlays are SignalScope Grad-CAM visualizations.

| Example | Dataset label | Model output | Source / overlay |
|---|---|---|---|
| real_photo | real | real, AI score 0.006 | [image](real_photo.jpeg) / [overlay](real_photo_overlay.png) |
| generated_detected | AI | ai_generated, AI score 0.869 | [image](generated_detected.png) / [overlay](generated_detected_overlay.png) |
| generated_missed | AI | real, AI score 0.016 | [image](generated_missed.png) / [overlay](generated_missed_overlay.png) |

The missed BigGAN image is especially important: a low AI score can still be wrong. Grad-CAM identifies model attribution, not an annotated visual defect. The 40-image automated diagnostic is separate, and its two-person review remains pending.
