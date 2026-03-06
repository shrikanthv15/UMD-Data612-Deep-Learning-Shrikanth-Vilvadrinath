# MASTER EXECUTION PLAN
## DiT-Based Anomaly Detection via Diffusion Reconstruction
### DATA-MSML 612 Group 6

**Created:** 2026-03-05
**Last Updated:** 2026-03-05
**Source:** Built from Remarks.MD critique + CHECKLIST.md pre-planning
**Principle:** Every tool, dataset, and library in this plan is FREE and open-source.

---

## 0. PROJECT SUMMARY

**One-liner:** We train a Diffusion Transformer (DiT) on normal industrial images,
then use its reconstruction error to detect manufacturing defects — turning a
generative model into an unsupervised anomaly detector.

**What makes this unique:**
1. DiT backbone (Transformer + Diffusion combined) — no published DiT-based anomaly
   detection on MVTec exists
2. Non-obvious use of diffusion: reconstruction-based detection, not image generation
3. Dual-level anomaly scoring: pixel (SSIM) + feature (ResNet-18) comparison
4. DDIM fast sampling: 50 steps instead of 1000

**Target metrics:**
- Image-level AUROC: > 95%
- Pixel-level AUROC: > 90%
- PRO (Per-Region Overlap): > 85%

---

## 1. ARCHITECTURE (Final — Incorporates All Remarks.MD Critiques)

```
=== THE COMPLETE PIPELINE ===

[STAGE 1] DATA PIPELINE
  MVTec AD (15 categories, free download)
  -> Per-category train/test split (use MVTec's own splits)
  -> Resize to 256x256 (or 128x128 if GPU memory is tight)
  -> Normalize to [-1, 1]
  -> Augmentations: RandomHorizontalFlip, RandomRotation(10),
     ColorJitter(brightness=0.1, contrast=0.1)
  -> PyTorch DataLoader (train on train/good/ ONLY)

[STAGE 2] DIFFUSION TRANSFORMER (DiT) BACKBONE
  Input: noised image x_t + timestep t
  -> Patchify: split image into 4x4 patches (for 256x256: 64x64=4096 patches,
     for 128x128: 32x32=1024 patches)
  -> Linear patch embedding (patch_dim -> hidden_dim=384)
  -> Positional encoding (sinusoidal 2D)
  -> N transformer blocks (N=12 for DiT-S, adjustable):
       Each block: LayerNorm -> MultiHeadSelfAttention -> LayerNorm -> MLP
       Time conditioning via AdaLN (Adaptive Layer Norm):
         scale, shift = MLP(time_embedding)
         output = scale * LayerNorm(x) + shift
  -> Unpatchify: reconstruct to image shape
  -> Output: predicted noise epsilon_theta(x_t, t)

  WHY DiT OVER UNET:
  - Combines both project tracks (Transformer architecture for Diffusion model)
  - DiT scales better than UNet (transformer scaling laws)
  - Architecturally novel for anomaly detection
  - Self-attention captures global structure (UNet relies on local convolutions)

[STAGE 3] DIFFUSION PROCESS
  Forward (training):
    - Sample t ~ Uniform(1, T)
    - Sample noise epsilon ~ N(0, I)
    - Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    - Loss = MSE(epsilon, epsilon_theta(x_t, t))

  Noise schedule: COSINE (Nichol & Dhariwal, 2021) — settled, not debatable
  T = 1000 (training), but inference uses DDIM with 50 steps

  Reverse (inference — DDIM, NOT full DDPM):
    - Given test image x_0, corrupt to x_{T_partial} (T_partial ~ 250)
    - Run DDIM reverse: x_{T_partial} -> x_{T_partial-k} -> ... -> x_0_hat
    - Only 50 DDIM steps needed (not 1000 DDPM steps)
    - Result: x_0_hat (the reconstruction)

[STAGE 4] ANOMALY SCORING (DUAL-LEVEL)

  Pixel-Level Score (SSIM):
    - anomaly_map_pixel = 1 - SSIM(x_0, x_0_hat, window_size=11)
    - SSIM is perceptually superior to L2 distance
    - Library: pytorch_msssim

  Feature-Level Score (ResNet-18):
    - Extract features from layers 1, 2, 3 of pretrained ResNet-18
    - features_original = ResNet18(x_0) at each layer
    - features_reconstructed = ResNet18(x_0_hat) at each layer
    - anomaly_map_feature = sum of L2 distances at each layer
    - Upsample all feature maps to input resolution, average

  Combined Score:
    - anomaly_map = alpha * anomaly_map_pixel + (1-alpha) * anomaly_map_feature
    - alpha tuned per category (or fixed at 0.5 as default)
    - Image-level score = max(anomaly_map) or mean(top-k pixels)

[STAGE 5] EVALUATION

  Metrics (all three required):
    - Image-level AUROC: scalar score per image, binary label (normal/anomaly)
    - Pixel-level AUROC: per-pixel score, ground truth mask from MVTec
    - PRO (Per-Region Overlap): official MVTec metric, uses their eval scripts

  Protocol:
    - One model trained per category (standard MVTec protocol)
    - Report per-category AND mean across all categories
    - Output anomaly maps as .tiff for MVTec official evaluation scripts

[STAGE 6] BASELINES (via anomalib — free, pip install)

  Classical:
    - PCA reconstruction (our implementation, ~5 lines)
    - Autoencoder reconstruction (our implementation, ~50 lines)

  SOTA (via anomalib):
    - PatchCore (Roth et al. 2022) — strongest non-generative method, 99.1% AUROC
    - Reverse Distillation (Deng et al. 2022) — knowledge distillation approach

  anomalib handles data loading, training, and evaluation.
  Each baseline takes ~10-20 lines of config code and runs in minutes.

[STAGE 7] ABLATION STUDIES (minimum 4)

  A1. DiT vs UNet backbone (same diffusion framework, swap backbone only)
  A2. T_partial sweep: 100, 200, 300, 400, 500 — plot AUROC vs T_partial
  A3. SSIM vs L2 vs LPIPS for pixel-level anomaly map
  A4. With vs without feature-level comparison
  A5. DDPM (1000 steps) vs DDIM (50 steps) — quality and speed tradeoff
  A6. Cosine vs linear noise schedule (optional, cosine expected to win)

[STAGE 8] VISUALIZATIONS (for report and presentation)

  V1. 6-panel grid: Original | Noised | Reconstruction | Pixel Map |
      Feature Map | Ground Truth Mask
  V2. Per-category AUROC bar chart (all 15 categories)
  V3. ROC curves overlaid (DiT vs baselines)
  V4. Training loss curve
  V5. T_partial sensitivity plot (AUROC vs T_partial)
  V6. DDIM step count vs quality plot
  V7. Comparison table: our method vs baselines vs published SOTA
```

---

## 2. TECH STACK (All Free)

| Layer | Tool | Version | License |
|-------|------|---------|---------|
| Language | Python | 3.10+ | PSF |
| DL Framework | PyTorch | 2.x | BSD |
| Diffusion reference | lucidrains/denoising-diffusion-pytorch | latest | MIT |
| DiT reference | facebookresearch/DiT | latest | CC-BY-NC 4.0 |
| Baseline library | anomalib | 1.x | Apache 2.0 |
| Perceptual metrics | pytorch_msssim, lpips | latest | MIT |
| Feature extractor | torchvision (ResNet-18 pretrained) | latest | BSD |
| Image processing | Pillow, scikit-image | latest | MIT/BSD |
| Evaluation | scikit-learn (AUROC), MVTec eval scripts | latest | BSD/custom |
| Plotting | matplotlib, seaborn | latest | BSD |
| Compute | Google Colab Free + Kaggle Free | N/A | Free tier |
| Report | LaTeX via Overleaf Free | N/A | Free tier |

---

## 3. EXECUTION PHASES

### PHASE 0 — Environment & Data (Days 1-3)
**Goal:** Everyone can load data and run a forward pass.

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P0-01 | Download MVTec AD dataset (~5GB) | HUMAN | Files in `data/mvtec/` |
| P0-02 | Create requirements.txt (all pip deps) | AGENT | `pip install -r` works |
| P0-03 | Write PyTorch Dataset class for MVTec | AGENT | `dataset.py` loads images, prints shapes |
| P0-04 | Set up shared Google Drive for checkpoints | HUMAN | Drive link shared |
| P0-05 | Set up project GitHub repo | HUMAN | All can push |

### PHASE 1 — Research & Proposal (Days 3-7)
**Goal:** All 4 understand the math. Proposal document ready.

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P1-01 | Read Ho et al. 2020 (DDPM) sections 1-3 | HUMAN | Can explain forward/reverse process |
| P1-02 | Read DiT paper (Peebles & Xie 2023) sec 1-3 | HUMAN | Understand patchify + AdaLN |
| P1-03 | Read DDAD paper (Mousakhan 2023) sec 1-3 | HUMAN | Understand conditioned denoising |
| P1-04 | Read DDIM paper (Song 2021) sec 4 | HUMAN | Understand deterministic sampling |
| P1-05 | Write proposal (LaTeX, 2.5 pages) | AGENT | `proposal/proposal.tex` complete |
| P1-06 | Compile + submit proposal PDF | HUMAN | PDF submitted to course portal |

### PHASE 2 — Core Implementation (Days 7-21)
**Goal:** DiT model trains on one MVTec category and produces reconstructions.

**Week 1 of coding:**

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P2-01 | Implement cosine noise schedule | AGENT | `diffusion.py` -- `cosine_beta_schedule()` unit tested |
| P2-02 | Implement forward process q(x_t|x_0) | AGENT | `diffusion.py` -- `q_sample()` verified visually |
| P2-03 | Implement DiT backbone (from facebookresearch/DiT) | AGENT | `dit.py` -- forward: (B,3,128,128) -> (B,3,128,128) |
| P2-04 | Implement DDIM reverse sampling | AGENT | `diffusion.py` -- `ddim_sample()` generates images |
| P2-05 | Data augmentations (flip, rotate, jitter) | AGENT | `dataset.py` -- visual check on 10 images |
| P2-06 | Training loop with AMP | AGENT | `train.py` -- runs 5 epochs without error |

**Week 2 of coding:**

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P2-07 | Smoke test: 5 epochs on hazelnut | HUMAN | Loss curve looks reasonable (Colab/Kaggle) |
| P2-08 | Reconstruction inference (partial noise + DDIM) | AGENT | `inference.py` -- normal images reconstruct cleanly |
| P2-09 | SSIM anomaly map | AGENT | `scoring.py` -- anomaly map highlights defects |
| P2-10 | ResNet-18 feature anomaly map | AGENT | `scoring.py` -- feature distance higher on anomalies |
| P2-11 | Combined scoring + AUROC evaluation | AGENT | `evaluate.py` -- prints AUROC for hazelnut |
| P2-12 | 6-panel visualization script | AGENT | `visualize.py` -- saves grid to `output/figures/` |
| P2-13 | Full training on hazelnut (100 epochs) | HUMAN | Checkpoint saved (Colab/Kaggle) |
| P2-14 | Evaluate hazelnut end-to-end | HUMAN | AUROC computed, anomaly maps look good |

### PHASE 3 — Full Evaluation & Baselines (Days 21-28)
**Goal:** All 15 categories evaluated. Baselines run. Ablations complete.

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P3-01 | Train all 15 categories (split across 4 accounts) | HUMAN | 15 checkpoints, all have AUROC |
| P3-02 | Run PatchCore baseline via anomalib | AGENT | Per-category AUROC table logged |
| P3-03 | Run Reverse Distillation via anomalib | AGENT | Per-category AUROC table logged |
| P3-04 | Run PCA + AE baselines | AGENT | Per-category AUROC table logged |
| P3-05 | Ablation: UNet vs DiT | AGENT | AUROC comparison table |
| P3-06 | Ablation: T_partial sweep | HUMAN | Sensitivity plot saved |
| P3-07 | Ablation: SSIM vs L2 vs LPIPS | AGENT | Comparison table |
| P3-08 | Ablation: With/without features | AGENT | AUROC delta table |
| P3-09 | Pixel-level AUROC all categories | HUMAN | Per-category pixel AUROC |
| P3-10 | .tiff export + MVTec PRO eval | AGENT | PRO scores from official scripts |

### PHASE 4 — Report & Presentation (Days 28-35)
**Goal:** Final report and slides ready.

| ID | Task | Type | Done When |
|----|------|------|-----------|
| P4-01 | Write final report (LaTeX) | AGENT | `report/report.tex` complete |
| P4-02 | Finalize all figures (V1-V7) | AGENT | `output/figures/` complete |
| P4-03 | Compile report PDF | HUMAN | `report/report.pdf` submitted |
| P4-04 | Create presentation slides (10-15) | AGENT | Slides reviewed |
| P4-05 | Presentation rehearsal (2 dry runs) | HUMAN | Timed under 15 min |
| P4-06 | Code cleanup + README | AGENT | README.md in project root |

---

## 4. COMPUTE STRATEGY (Zero Cost)

### Available Free GPU Resources

| Platform | GPU | Session Limit | Weekly Hours | Account |
|----------|-----|--------------|-------------|---------|
| Google Colab Free | T4 (16GB) | ~12 hours | ~30 hrs | 1 per person = 4 accounts |
| Kaggle Notebooks | P100 (16GB) | ~9 hours | 30 hrs | 1 per person = 4 accounts |
| **Total available** | — | — | **~240 hrs/week** | Across 4 people |

### Estimated Compute Needs

| Task | Resolution | GPU Hours | Platform |
|------|-----------|-----------|----------|
| DiT training (1 category, 100 epochs) | 128x128 | ~1.5-2 hrs | Colab T4 |
| DiT training (all 15 categories) | 128x128 | ~25-30 hrs | Split across 4 accounts |
| UNet training for ablation (1 category) | 128x128 | ~1 hr | Colab T4 |
| DDIM inference (1 category test set) | 128x128 | ~10 min | Colab T4 |
| Full evaluation (all 15 categories) | 128x128 | ~2.5 hrs | Colab T4 |
| anomalib baselines (PatchCore, RD) | 256x256 | ~2 hrs total | Kaggle P100 |
| Ablation experiments | 128x128 | ~8 hrs total | Split |
| **TOTAL** | — | **~45-50 hrs** | Fits in free tier easily |

### Resolution Decision
- **Primary:** 128x128 — fits in free Colab T4 VRAM, fast training
- **Stretch:** 256x256 — if Kaggle P100 has headroom, run best category at 256x256
- This is NOT a compromise: many papers report 128x128 results for MVTec

### Training Parallelization Strategy
```
Account 1 trains: bottle, cable, capsule, carpet (4 categories)
Account 2 trains: grid, hazelnut, leather, metal_nut (4 categories)
Account 3 trains: pill, screw, tile, toothbrush (4 categories)
Account 4 trains: transistor, wood, zipper (3 categories)
                  + UNet ablation on hazelnut
```

Each person runs their categories on their own Colab/Kaggle account.
Checkpoints saved to shared Google Drive.
Total wall-clock time: ~2 days (parallelized across 4 people).

---

## 5. KEY ARCHITECTURAL DECISIONS (Settled)

These are NO LONGER open questions (closed by Remarks.MD critique):

| Question (from old KB-008) | Decision | Rationale |
|---------------------------|----------|-----------|
| Cosine or linear noise schedule? | **Cosine** | Objectively better (Nichol & Dhariwal 2021) |
| 128x128 or 256x256? | **128x128 primary, 256x256 stretch** | Free Colab constraint |
| One model per category or unified? | **Per category** | Standard MVTec protocol |
| SSIM or L2 for anomaly map? | **SSIM primary, L2 as ablation** | SSIM is perceptually superior |
| UNet or DiT backbone? | **DiT primary, UNet as ablation** | Uniqueness differentiator |
| DDPM or DDIM sampling? | **DDIM (50 steps) primary, DDPM as ablation** | 20x faster inference |
| What value of T_partial? | **Sweep 100-500, report best** | This IS an ablation study |
| Feature-level comparison? | **Yes, ResNet-18 layers 1-3** | 10%+ AUROC boost per DDAD |

---

## 6. RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Free Colab GPU unavailable (quota hit) | Medium | High | Use Kaggle P100 as backup; 4 accounts |
| DiT too large for T4 VRAM at 128x128 | Low | High | Reduce to DiT-Tiny (6 blocks, 192 dim) |
| AUROC below 90% despite best efforts | Medium | Medium | Feature-level scoring is the safety net; anomalib baselines still tell a story |
| MVTec form email delayed > 3 days | Low | Medium | Use VisA dataset (GitHub, instant download) |
| One team member drops the ball | Medium | High | Cross-assign critical tasks; weekly check-ins |
| DiT training unstable (loss diverges) | Low-Med | Medium | Start with UNet (known stable), swap to DiT after UNet works |
| Report/presentation deadline crunch | Medium | High | Start report skeleton in Phase 2, not Phase 4 |

### Critical Fallback Path
If DiT proves too difficult or unstable:
1. Fall back to UNet backbone (still a strong project)
2. Keep everything else: DDIM, dual-level scoring, cosine schedule, anomalib baselines
3. Present DiT as "attempted but UNet selected for stability" — shows you tried
4. UNet-based project with all other modern techniques still scores 85-90/100

---

## 7. FILE STRUCTURE (Final)

```
Team Project/
  Project.MD                    <- Assignment spec (read-only)
  Remarks.MD                    <- Architecture critique (reference)
  docs/
    PROJECT_LOG.md              <- Decision log (append-only)
    CHECKLIST.md                <- Pre-planning answers
    KNOWLEDGE_BASE.md           <- Running reference
    TEAM_ROLES.md               <- Role assignments
    plan/
      PLAN.md                   <- THIS FILE — master execution plan
    tracking/
      TASKS.md                  <- Task list: ID, owner, status, deadline
      STATUS.md                 <- Weekly status snapshots
  code/
    requirements.txt            <- pip dependencies
    src/
      dataset.py                <- MVTec PyTorch Dataset + augmentations
      dit.py                    <- Diffusion Transformer backbone
      unet.py                   <- UNet backbone (for ablation comparison)
      diffusion.py              <- Forward process, cosine schedule, DDIM sampling
      scoring.py                <- SSIM pixel map + ResNet feature map + combined
      train.py                  <- Training loop with AMP
      inference.py              <- Partial noise + DDIM reconstruction
      evaluate.py               <- AUROC, pixel-AUROC, PRO computation
      visualize.py              <- All visualization functions
      run_baselines.py          <- anomalib PatchCore + Reverse Distillation
    notebooks/
      01_data_exploration.ipynb <- EDA, sample visualizations
      02_training.ipynb         <- Colab training notebook
      03_evaluation.ipynb       <- Results analysis notebook
  data/
    mvtec/                      <- Dataset (gitignored, ~5GB)
      bottle/
      cable/
      ... (15 categories)
    .gitkeep
  output/
    checkpoints/                <- Model weights (gitignored)
    figures/                    <- All plots for report
    predictions/                <- .tiff anomaly maps for MVTec eval
    results/                    <- CSV/JSON per-category metrics
  proposal/
    proposal.tex                <- LaTeX source
    proposal.pdf                <- Compiled (submit this)
  report/
    report.tex                  <- Final report LaTeX
    report.pdf                  <- Compiled (submit this)
  presentation/
    slides.pptx                 <- Or Google Slides link
```

---

## 8. PROPOSAL OUTLINE (2-3 Pages)

```
1. INTRODUCTION (0.5 page)
   - Anomaly detection in manufacturing: the problem
   - Why diffusion models: learned distribution of normality
   - Our twist: reconstruction error as anomaly signal
   - Contribution: first DiT-based anomaly detection on MVTec

2. BACKGROUND AND SIGNIFICANCE (0.5 page)
   - DDPM basics (Ho et al. 2020)
   - Diffusion Transformers (Peebles & Xie 2023)
   - Prior work: AnoDDPM, DDAD, DiffusionAD
   - Gap: no DiT backbone for anomaly detection

3. DATASET AND DATA PLAN (0.5 page)
   - MVTec AD: 15 categories, 5354 images, free academic download
   - Per-category training on normal images only
   - Preprocessing: resize, normalize, augment
   - Test set: mix of normal + anomalous with ground truth masks

4. METHODOLOGY (0.5 page)
   - DiT backbone architecture
   - Cosine schedule, DDIM sampling
   - Dual-level anomaly scoring (SSIM + ResNet features)
   - Baselines: PCA, AE, PatchCore, Reverse Distillation

5. EVALUATION PLAN (0.25 page)
   - Image AUROC, Pixel AUROC, PRO
   - Ablation studies: DiT vs UNet, T_partial sweep, SSIM vs L2
   - Per-category breakdown

6. FRAMEWORK AND REPRODUCIBILITY (0.25 page)
   - PyTorch, anomalib, free compute (Colab/Kaggle)
   - All code on GitHub, seeds fixed for reproducibility

7. REFERENCES
```

---

## 9. WHAT WILL CHANGE (Honest Pre-Mortem)

| Proposal Will Say | Reality May Be | Acceptable? |
|-------------------|---------------|-------------|
| DiT backbone | May fall back to UNet if unstable | Yes — still a modern project |
| All 15 categories | May focus on 10 if compute is tight | Yes — report the ones you have |
| 256x256 resolution | Likely 128x128 on free Colab | Yes — many papers use 128x128 |
| AUROC > 95% | May achieve 88-93% with DiT (first attempt) | Yes — novelty matters more than raw numbers |
| 4 ablation studies | May get 3 if time runs out | Yes — 3 is the minimum |
| PRO metric | May skip if .tiff export is tricky | Acceptable — AUROC is the standard metric |

The proposal should be written to accommodate these realities without over-promising.

---

## 10. SUCCESS CRITERIA — WHAT "FULL MARKS" LOOKS LIKE

| Criterion | Threshold for Full Marks |
|-----------|--------------------------|
| Code runs end-to-end without errors | Mandatory |
| Unique architecture (DiT or at minimum modern conditioned denoising) | Mandatory |
| AUROC > 90% on at least 5 categories | Strong |
| Comparison against PatchCore (SOTA baseline) | Mandatory |
| At least 3 ablation studies with tables | Mandatory |
| Anomaly map visualizations (side-by-side grids) | Mandatory |
| Per-category results table | Mandatory |
| Clean report with figures, tables, references | Mandatory |
| Presentation: clear, timed, all 4 present | Mandatory |
| Code is documented, reproducible, has README | Mandatory |

---

*This plan supersedes CHECKLIST.md for execution. CHECKLIST.md remains as the
pre-planning reference. All changes to this plan must be logged in PROJECT_LOG.md.*
