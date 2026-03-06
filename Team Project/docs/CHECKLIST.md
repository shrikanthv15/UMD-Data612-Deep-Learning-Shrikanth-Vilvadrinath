# Project Pre-Planning Checklist
## Anomaly Detection via Diffusion Reconstruction — Group 6

**Created:** 2026-03-05
**Purpose:** Answer all fundamental questions BEFORE writing the proposal, so the
proposal reflects reality and the implementation does not deviate from what was promised.

---

## 1. THE PROJECT IN ONE PARAGRAPH

We train a Diffusion Transformer (DiT) exclusively on NORMAL images from an industrial
inspection dataset (MVTec AD). At inference, we corrupt a test image with partial noise,
then run DDIM reverse sampling (50 steps) to reconstruct it. Normal images reconstruct
cleanly; anomalous images do NOT. We compute a dual-level anomaly map: pixel-level
(SSIM) + feature-level (ResNet-18), and measure detection quality with AUROC and PRO.

No anomaly labels are used during training. This is a fully unsupervised anomaly detector.

**Status: Plan revised per Remarks.MD critique. Architecture upgraded from vanilla
DDPM/UNet to DiT/DDIM/dual-scoring. See `docs/plan/PLAN.md` for full execution plan.**

---

## 2. DATA CHECKLIST

### 2.1 Dataset: MVTec Anomaly Detection (MVTec AD)

| Item | Detail |
|------|--------|
| What it is | 15 categories of industrial objects/textures. Each category has: ~200-400 normal training images + test images with labeled defect masks |
| Total size | ~5 GB compressed, ~4.9 GB extracted |
| Image format | PNG, varying resolution (~700x700 to 1024x1024 pixels) |
| Categories | Bottle, cable, capsule, carpet, grid, hazelnut, leather, metal nut, pill, screw, tile, toothbrush, transistor, wood, zipper |
| License | Academic/non-commercial research use (free) |
| Source URL | https://www.mvtec.com/company/research/datasets/mvtec-ad |

### 2.2 How to Get the Data

**Step 1:** Go to https://www.mvtec.com/company/research/datasets/mvtec-ad
**Step 2:** Fill out a short academic use form (name, institution, purpose) — takes 5 minutes.
**Step 3:** Receive download link via email (usually within minutes to a few hours).
**Step 4:** Download `mvtec_anomaly_detection.tar.xz` (~5 GB).
**Step 5:** Extract to `data/mvtec/` in your project directory.

**Difficulty: EASY** — Form is straightforward, no IRB approval, no credentialing.
**Time to obtain: 1-4 hours** (mostly waiting for email response + download time).
**STATUS: DONE** — Dataset downloaded on 2026-03-05.

### 2.3 Alternative: Faster Data Option
If the MVTec form is slow, use the **MVTec Screws** subset (~100MB) available on
Kaggle directly (search "MVTec anomaly detection"). Good for prototyping before
the full dataset arrives.

Also available: **VisA (Visual Anomaly)** dataset from Amazon — downloadable directly
from GitHub without any form. 12 categories, similar structure.

### 2.4 Preprocessing Pipeline

| Step | What happens | Difficulty |
|------|-------------|------------|
| Extract TAR | `tar -xf mvtec_anomaly_detection.tar.xz` | Trivial |
| Folder parsing | Walk `train/good/` for normal images; `test/*/` for test images | Easy |
| Resize | Resize all images to 256x256 or 128x128 (uniform input) | Easy |
| Normalize | Pixel values [0,1] or [-1,1] (diffusion standard) | Easy |
| DataLoader | PyTorch Dataset class, train on `train/good/` only | Moderate |
| Ground truth masks | Load `ground_truth/` PNG masks for test AUROC evaluation | Easy |

**Estimated preprocessing time (coding): 3-4 hours for one person.**
The folder structure is very clean — this is one of the most well-organized ML datasets.

**Time to understand the data: 1-2 hours.** The dataset paper (Bergmann et al. 2019)
is 8 pages and explains everything. The folder structure is self-explanatory.

---

## 3. WHAT WE ARE CODING

### 3.1 The Full Pipeline (Revised -- DiT + DDIM + Dual Scoring)

```
[1] Data Pipeline
    data/mvtec/ -> PyTorch Dataset -> DataLoader (normal images only)
    Augmentations: flip, rotation(+/-10), color jitter

[2] DiT Backbone (Diffusion Transformer)
    - Patchify 128x128 -> 4x4 patches -> 1024 tokens
    - 12 transformer blocks with AdaLN time conditioning
    - Unpatchify -> predicted noise epsilon_theta(x_t, t)

[3] Diffusion Process
    - Forward: cosine noise schedule, T=1000
    - Reverse: DDIM sampling (50 deterministic steps, 20x faster than DDPM)
    - Loss: MSE between predicted noise and actual noise

[4] Training
    - Train ONLY on normal images (train/good/), per-category
    - AMP (mixed precision) for 2x speedup on free Colab T4
    - ~100 epochs on 128x128 images

[5] Inference / Anomaly Scoring (Dual-Level)
    - Partial noise: x_0 -> x_{T_partial} (T_partial ~ 250)
    - DDIM reverse: x_{T_partial} -> x_0_hat (50 steps)
    - Pixel map: 1 - SSIM(x_0, x_0_hat)
    - Feature map: L2(ResNet18(x_0), ResNet18(x_0_hat)) at layers 1,2,3
    - Combined: alpha * pixel + (1-alpha) * feature

[6] Evaluation
    - Image-level AUROC (target > 95%)
    - Pixel-level AUROC (target > 90%)
    - PRO via official MVTec eval scripts (target > 85%)
    - Baselines: PCA, AE, PatchCore, Reverse Distillation (anomalib)

[7] Visualization
    - 6-panel: Original | Noised | Reconstruction | Pixel Map | Feature Map | GT
    - Per-category AUROC bar chart, ROC curves, training loss
```

### 3.2 Key Concepts to Understand

| Concept | Difficulty |
|---------|-----------|
| Forward process math (cosine schedule) | Moderate |
| Reverse process / DDIM sampling | Moderate-Hard |
| DiT backbone (patchify + AdaLN + transformer blocks) | Moderate |
| SSIM-based anomaly maps | Easy |
| Feature extraction (ResNet-18 intermediate layers) | Easy |
| Loss function (noise prediction MSE) | Easy |
| The reconstruction-as-anomaly-score insight | Easy |

### 3.3 Existing Implementations We Adapt (and Cite)

| Resource | What it provides |
|----------|-----------------|
| Ho et al. 2020 (DDPM) | Foundational diffusion math |
| Peebles & Xie 2023 (DiT) | Our backbone architecture (facebookresearch/DiT) |
| Mousakhan et al. 2023 (DDAD) | Conditioned denoising + feature comparison (99.8% AUROC) |
| Song et al. 2021 (DDIM) | Fast deterministic sampling |
| lucidrains/denoising-diffusion-pytorch | Clean diffusion training loop reference |
| anomalib (Intel) | PatchCore + Reverse Distillation baselines |

See `docs/plan/PLAN.md` for the full execution plan.

---

## 4. TIME ESTIMATE (4 Students, Realistic)

| Phase | Estimated Time | Who |
|-------|---------------|-----|
| Data download + extraction | 4-8 hours (mostly waiting) | 1 person |
| Data pipeline (Dataset class, DataLoader) | 4-6 hours coding | 1 person |
| Understanding DDPM math + paper reading | 10-15 hours | All 4, can split |
| UNet implementation (or adapt from reference) | 8-12 hours | 2 people |
| Training loop + noise schedule | 4-6 hours | 1 person |
| Inference + anomaly scoring | 4-6 hours | 1 person |
| Evaluation (AUROC, visualization) | 4-6 hours | 1 person |
| Baseline (Autoencoder comparison) | 4-6 hours | 1 person |
| Report writing | 6-10 hours | All 4 |
| Presentation prep | 4-6 hours | All 4 |
| **TOTAL** | **~55-80 hours** | 4 people = ~15-20 hrs each |

**This is 4-6 weeks of part-time work for 4 people. Feasible for a semester project.**

### Training Time Reality Check

| Hardware | Image size | Epochs | Estimated training time |
|----------|------------|--------|------------------------|
| CPU only | 128x128 | 50 | 8-12 hours |
| CPU only | 256x256 | 100 | 30-50 hours (NOT feasible) |
| Google Colab Free (T4) | 256x256 | 100 | 4-6 hours |
| Google Colab Pro (A100) | 256x256 | 100 | 1-2 hours |
| Local GPU (RTX 3060+) | 256x256 | 100 | 2-3 hours |

**Recommendation: Use Google Colab Pro T4/A100. This is the biggest practical constraint.**
Budget for Colab compute units or use the free tier with 128x128 images.

---

## 5. AUTOMATION WITH CLAUDE — FEASIBILITY ASSESSMENT

### 5.1 What Claude Can Automate End-to-End

| Task | Claude Can Do? | Confidence |
|------|---------------|------------|
| Write the full data pipeline (PyTorch Dataset, DataLoader, transforms) | Yes, fully | High |
| Adapt a reference UNet implementation for our noise schedule | Yes, with review | High |
| Write the DDPM training loop (noise sampling, MSE loss, scheduler) | Yes | High |
| Write the inference/anomaly scoring script | Yes | High |
| Write the AUROC evaluation + matplotlib plots | Yes | High |
| Write the proposal document (LaTeX or markdown) | Yes | High |
| Debug shape mismatches, CUDA errors, training instability | Yes, iteratively | Medium |
| Tune noise schedule / T value for best AUROC | Needs experiments | Medium |
| Pick the best partial-T value for reconstruction | Requires runs | Medium |

### 5.2 The Automation Workflow (How We Use Claude)

```
Step 1: Claude writes data pipeline -> we test on 10 images -> Claude fixes errors
Step 2: Claude writes UNet + DDPM class -> we run a 5-epoch smoke test
Step 3: Claude writes training loop -> we run full training on Colab
Step 4: Claude writes inference + anomaly scorer -> we inspect anomaly maps visually
Step 5: Claude writes evaluation (AUROC) -> we run on test set
Step 6: Claude writes visualization (side-by-side plots) -> we review output
Step 7: Claude writes the report based on our results
```

### 5.3 What Claude CANNOT Automate (Human Work Required)

| Task | Why human needed |
|------|-----------------|
| Actually running training on Colab | Requires your Google account + GPU session |
| Getting anomaly maps to look visually good | Needs visual inspection + iteration |
| Tuning T_partial (the noise steps at inference) | Trial and error with GPU runs |
| Team coordination and division of work | People problem, not code problem |
| Presentation delivery | Must be done by humans in front of professor |
| Verifying results make sense per category | Domain knowledge + visual judgment |

---

## 6. WHAT WILL CHANGE BETWEEN PROPOSAL AND IMPLEMENTATION

This is the honest reality check the team needs before committing:

| Proposal Says | Likely Reality |
|---------------|---------------|
| 256x256 input images | May need to drop to 128x128 if Colab memory limits hit |
| 100 training epochs | May converge faster at 50-70; may need more for some categories |
| Evaluate all 15 MVTec categories | Likely focus on 3-5 categories due to time constraints |
| AUROC > 85% target | Achievable for texture categories (carpet, leather); harder for object categories |
| Pixel-level anomaly maps | May only report image-level AUROC if time is tight |
| Pure from-scratch DDPM | May adapt lucidrains/denoising-diffusion-pytorch to save 10-15 hours |

### The Honest Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Training takes too long on free Colab | Medium | Use 128x128; or split categories across 4 Colab accounts |
| UNet implementation has bugs | Medium | Start from a tested reference implementation |
| AUROC is low for some categories | Medium | Focus presentation on categories where it works well |
| Group doesn't understand diffusion math | Low-Medium | Assign 2 people to paper reading in week 1 |
| MVTec form email is slow | Low | Use VisA or Kaggle subset in the meantime |
| Team coordination issues | Medium | Define deliverables per person this week |

---

## 7. VALIDATION AND SUCCESS CRITERIA

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Image-level AUROC | > 0.85 | > 0.75 |
| Pixel-level AUROC (if done) | > 0.80 | > 0.70 |
| Visual anomaly maps | Clearly highlight defect regions | At least show some signal |
| Baseline comparison | Beat AE reconstruction by > 2% AUROC | Match AE baseline |
| Categories evaluated | All 15 | At least 5 |

**AUROC of 0.85+ is achievable.** AnoDDPM reports 0.88+ on several MVTec categories.
We should expect similar with a well-implemented DDPM.

---

## 8. DECISION: GO / NO-GO

| Criteria | Assessment |
|----------|-----------|
| Data accessible? | YES — downloaded (DONE) |
| Computationally feasible for students? | YES — Colab Free + Kaggle Free (zero cost) |
| Coding complexity manageable for 4 people? | YES — 10-15 hrs each |
| Unique enough to stand out? | YES — DiT backbone is novel for anomaly detection |
| Claude-automatable enough to be efficient? | YES — ~70% of code can be Claude-generated |
| Proposal writable without full implementation? | YES — design is clear enough |
| All tools free and open-source? | YES — verified, zero budget |

**DECISION: GO. Proceeded to planning. Plan complete. Next: proposal draft.**

See `docs/plan/PLAN.md` for full execution plan.
See `docs/tasks/P1-05_PROPOSAL_DRAFT.md` for next task details.

---

## 9. IMMEDIATE NEXT STEPS (This Week)

- [ ] Fill out MVTec download form (1 person — today)
- [ ] Download and extract dataset (1 person — this week)
- [ ] Read AnoDDPM paper (Wyatt et al. 2022) — all 4 people (this week)
- [ ] Read Ho et al. 2020 DDPM paper sections 1-3 (foundational math) (all 4)
- [ ] Write proposal draft (split: 2 people write, 2 people review) (this week)
- [ ] Set up shared Google Colab notebook (1 person — this week)
- [ ] Define team roles formally (see `TEAM_ROLES.md` to be created)
