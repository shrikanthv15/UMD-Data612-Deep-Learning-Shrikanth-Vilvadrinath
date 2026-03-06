# Task P1-05: Write Proposal Draft
## Status: NEXT UP

---

## What This Task Produces
A complete 2-3 page LaTeX proposal document ready for Overleaf compilation and submission.
Output: `Team Project/proposal/proposal.tex` + compiled `proposal.pdf`

---

## What the Assignment Requires (from Project.MD)
> "Prepare a 2-3 page proposal document, present background, significance and statement
> of the problem, how to obtain the training data, the metric for validation and
> verification and the DL framework to implement the transformer or diffusion model."

The proposal must cover these 5 things:
1. **Background** — What are diffusion models? What is anomaly detection?
2. **Significance** — Why does this matter? (manufacturing quality control)
3. **Statement of the problem** — What exactly are we solving?
4. **How to obtain training data** — MVTec AD dataset, how to get it, what it contains
5. **Metric for validation and verification** — AUROC (image + pixel), PRO
6. **DL framework** — PyTorch, with specific libraries

---

## Proposal Structure (6 Sections, 2.5 Pages Target)

### Section 1: Introduction (0.4 pages)
**What to write:**
- Manufacturing quality control problem: human visual inspection is slow, expensive, inconsistent
- Deep learning for automated visual inspection
- Our approach: diffusion model learns "normality" from defect-free images, then detects
  anomalies via reconstruction error
- Key innovation: Diffusion Transformer (DiT) backbone — combines transformer architecture
  with diffusion framework; no published DiT-based anomaly detection on MVTec exists
- Target: image-level AUROC > 95%

### Section 2: Background and Related Work (0.5 pages)
**What to write:**
- Diffusion models: DDPM (Ho et al. 2020) — forward noise process, reverse denoising
- Diffusion for anomaly detection: AnoDDPM (Wyatt 2022), DDAD (Mousakhan 2023)
- Diffusion Transformers: DiT (Peebles & Xie 2023) replaces UNet with Vision Transformer
- Key insight: models trained only on normal data cannot reconstruct anomalies well
- Improvement over prior work: DDIM fast sampling, dual-level scoring (pixel + feature),
  DiT backbone for better global context

### Section 3: Problem Statement (0.3 pages)
**What to write:**
- Given: a set of defect-free training images from a manufacturing process
- Goal: detect and localize defects in new test images without any labeled anomaly data
- Formulation: unsupervised anomaly detection — train on normal only, detect deviations
- Sub-problems: (1) image-level detection (is this item defective?), (2) pixel-level
  localization (where is the defect?)

### Section 4: Dataset and Data Acquisition (0.4 pages)
**What to write:**
- MVTec Anomaly Detection dataset (Bergmann et al. 2019)
- 15 categories: 5 textures + 10 objects
- 5,354 high-resolution images total
- Training: only defect-free ("good") images (~200-400 per category)
- Testing: mix of good + various defect types, with pixel-level ground truth masks
- How to obtain: free academic download from mvtec.com (registration form)
- Preprocessing: resize to 128x128 (or 256x256), normalize to [-1,1], augmentation
  (flip, rotation, color jitter)
- Already downloaded and available

### Section 5: Methodology and Evaluation Plan (0.5 pages)
**What to write:**

Architecture:
- DiT backbone: patchify input (4x4 patches), transformer blocks with AdaLN
  time conditioning, unpatchify to reconstruct
- Cosine noise schedule, T=1000 training steps
- DDIM sampling at inference (50 steps for 20x speedup)
- Partial noise: add T_partial steps of noise to test image, then denoise back
- Anomaly map: dual-level — SSIM (pixel) + ResNet-18 feature distance (semantic)

Evaluation metrics:
- Image-level AUROC: scalar anomaly score per image, threshold-independent
- Pixel-level AUROC: per-pixel anomaly score vs ground truth mask
- PRO (Per-Region Overlap): official MVTec localization metric
- Report per-category AND mean across categories

Baselines:
- Classical: PCA reconstruction, Autoencoder
- SOTA: PatchCore, Reverse Distillation (via anomalib library)

Ablation studies:
- DiT vs UNet backbone
- T_partial sweep (100-500)
- SSIM vs L2 anomaly map
- With/without feature-level scoring

### Section 6: Framework and Reproducibility (0.3 pages)
**What to write:**
- PyTorch 2.x (primary framework)
- Key libraries: anomalib, pytorch_msssim, lpips, torchvision
- Compute: Google Colab Free (T4 GPU) + Kaggle Free (P100 GPU)
- All code open-source, seeds fixed (42) for reproducibility
- Reference implementations: facebookresearch/DiT, lucidrains/denoising-diffusion-pytorch
  (both cited)

### References
- Ho et al. 2020 — DDPM
- Peebles & Xie 2023 — DiT
- Mousakhan et al. 2023 — DDAD
- Song et al. 2021 — DDIM
- Bergmann et al. 2019 — MVTec AD
- Wyatt et al. 2022 — AnoDDPM
- Roth et al. 2022 — PatchCore

---

## How to Execute This Task (One Prompt)

Give Claude this instruction:
> "Write the full proposal as a LaTeX document. Follow the structure in
> `docs/tasks/P1-05_PROPOSAL_DRAFT.md`. Use the architecture from `docs/plan/PLAN.md`.
> Save to `Team Project/proposal/proposal.tex`. Keep it to 2.5 pages.
> Use the same LaTeX style as Assignment 2 (12pt, a4paper, proper bibliography)."

Claude will:
1. Read the task file + PLAN.md for architecture details
2. Write the complete LaTeX proposal
3. Save to `proposal/proposal.tex`
4. The team compiles on Overleaf and submits

---

## Pre-Requisites
- [x] MVTec dataset downloaded (P0-01 DONE)
- [x] Architecture finalized (PLAN.md)
- [x] All 18 critique items addressed
- [ ] Team reviews this task file and confirms structure

---

## After This Task
- Compile on Overleaf -> proposal.pdf
- Submit proposal
- Move to Phase 2: core implementation (next task: P2-01 through P2-07)
