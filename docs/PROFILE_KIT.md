# Profile kit: LinkedIn, GitHub, Medium

Drafts built only from numbers already in README.md. Replace anything in [brackets].
Do not add claims the repo cannot back up (no "99% accurate", no "wins").

## 1. One-line story (use everywhere)

I built SignalScope, an AI-image detector that runs on a normal laptop, and I published
exactly where it fails, not just where it works.

## 2. GitHub

Do now (about 30 minutes):
- Pin 4-6 repos. Put Signal_Scope first.
- Make a profile README: create a repo named exactly `sibtainmunshi` with a README.md.
  Keep it under 15 lines: who you are, 2-3 projects, how to reach you.
- Repo "About" box for Signal_Scope: "Real vs AI image detector. CLIP ViT-L/14 + trained head.
  Local CPU app, API, honest benchmarks." Topics: `ai-image-detection`, `deepfake`,
  `clip`, `pytorch`, `fastapi`, `machine-learning`.
- Archive or hide unfinished junk repos. A half-empty repo hurts more than no repo.
- Add a short LICENSE if missing.

Weak spot to fix: the README opens with a long paragraph. Put "At a glance" and the demo
link first, move the disclosure paragraph below it (keep it; the honesty is your edge).

## 3. LinkedIn

Headline (pick one, edit the bracket):
- "[Role/Student] | Building AI-image detection with honest evaluation | PyTorch, CLIP, FastAPI"

About section (draft):
- I build machine-learning tools and I report what they get wrong.
- Recent project: SignalScope, a real-vs-AI image detector for SIH 2026.
  - Frozen CLIP ViT-L/14 image model plus a trained classifier head.
  - Runs on CPU only. Fresh clone to first prediction in about 4 minutes.
  - On two unseen-generator benchmarks: ROC-AUC 0.948 and 0.936.
  - It also got worse on an older benchmark (AUC 0.771 -> 0.637). I published that too.
- Looking for: [internships / ML engineer roles / research projects].
- Contact: [email]

Featured section: add the GitHub repo, the demo video, and your best Medium article.

Experience entry for the project:
- Title: "SignalScope - AI image detection (SIH 2026)"
- 3 bullets: what you built, one measured result, one failure you disclosed.

Skills (keep 10-15, pin top 3): PyTorch, Machine Learning, Computer Vision, Python, FastAPI.

## 4. Medium: article plan

Write 3, in this order. Each 6-8 minutes. Lead with the story, not the code.

1. "My AI-image detector scored 0.99 AUC, then failed on new generators"
   - Show the CIFAKE result vs unseen generators. Lesson: benchmarks lie.
2. "I shipped a model that got worse on the old benchmark, on purpose"
   - The v0.3.0 -> v0.4.0 decision. Why 2025 generators mattered more.
3. "How to run an AI-image detector on a laptop with no GPU"
   - Setup steps, CPU timing (243.75 s), what the app shows.

Rules: link the repo in every article; cross-post a short summary to LinkedIn; do not
hide the limitation ("research detector, not proof of image origin").

## 5. Order of work (one weekend)

1. GitHub: pins, About box, profile README.
2. LinkedIn: headline, About, Featured.
3. Medium: publish article 1 only. Link it from LinkedIn and GitHub.
4. Only then write articles 2 and 3.
