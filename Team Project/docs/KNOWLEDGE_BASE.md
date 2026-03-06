# Knowledge Base — DiT-Based Anomaly Detection via Diffusion Reconstruction
## DATA-MSML 612 Group 6

**Purpose:** Running technical reference. Updated as we learn, build, and iterate.

---

## KB-001 — Core Papers (Required Reading)

| Priority | Paper | Year | Why |
|----------|-------|------|-----|
| MUST READ | Denoising Diffusion Probabilistic Models (Ho et al.) | 2020 | Foundational DDPM. Forward/reverse process, loss function. |
| MUST READ | Diffusion Transformers (DiT) (Peebles & Xie) | 2023 | Our backbone. Patchify + AdaLN + transformer blocks. |
| MUST READ | DDAD (Mousakhan et al.) arXiv:2305.15956 | 2023 | 99.8% AUROC on MVTec. Conditioned denoising + feature comparison. |
| MUST READ | MVTec AD dataset (Bergmann et al.) | 2019 | Our dataset paper. |
| SHOULD READ | DDIM (Song et al.) arXiv:2010.02502 | 2021 | Fast deterministic sampling. 50 steps instead of 1000. |
| SHOULD READ | AnoDDPM (Wyatt et al.) arXiv:2205.11616 | 2022 | Earlier diffusion anomaly detection. Simplex noise approach. |
| SHOULD READ | Improved DDPM (Nichol & Dhariwal) | 2021 | Cosine noise schedule. |
| REFERENCE | PatchCore (Roth et al.) | 2022 | SOTA non-generative baseline. 99.1% AUROC. |
| REFERENCE | anomalib docs | 2022 | Intel's library for running baselines. |

---

## KB-002 — Core Math

### Forward Process (Adding Noise)
```
q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Where alpha_bar_t = product of (1 - beta_s) for s=1..t
beta_t = cosine schedule (Nichol & Dhariwal 2021)
```

### Reverse Process (Denoising — What DiT Learns)
```
The DiT epsilon_theta(x_t, t) predicts the noise added at step t.
Loss = E[|| epsilon - epsilon_theta(x_t, t) ||^2]  (simple MSE on noise)
```

### DDIM Sampling (Fast Inference)
```
Instead of 1000 DDPM steps, DDIM uses 50 deterministic steps.
Same quality, 20x faster. Deterministic (no stochastic sampling).
```

### Anomaly Detection via Reconstruction
```
1. Take test image x_0
2. Corrupt to x_{T_partial} (T_partial ~ 250, tuned via ablation)
3. DDIM reverse: x_{T_partial} -> x_0_hat (50 steps)
4. Pixel anomaly map = 1 - SSIM(x_0, x_0_hat)
5. Feature anomaly map = L2(ResNet18(x_0), ResNet18(x_0_hat)) at layers 1,2,3
6. Combined = alpha * pixel_map + (1-alpha) * feature_map
7. Image score = max(combined_map) or mean(top-k)
```

---

## KB-003 — Implementation References (All Free, Open Source)

| Resource | License | Use |
|----------|---------|-----|
| facebookresearch/DiT (GitHub) | CC-BY-NC 4.0 | DiT backbone architecture reference |
| lucidrains/denoising-diffusion-pytorch (GitHub) | MIT | Diffusion training loop, noise schedule |
| anomalib (Intel, pip install) | Apache 2.0 | PatchCore + Reverse Distillation baselines |
| pytorch_msssim (pip install) | MIT | SSIM anomaly maps |
| lpips (pip install) | BSD | Learned perceptual similarity (ablation) |
| torchvision ResNet-18 pretrained | BSD | Feature-level anomaly maps |
| MVTec AD eval scripts | In data/ folder | Official AUROC + PRO computation |

---

## KB-004 — Project File Structure

```
Team Project/
  Project.MD                         <- Assignment spec (read-only)
  docs/
    PROJECT_LOG.md                   <- Decision log
    CHECKLIST.md                     <- Pre-planning (reference)
    KNOWLEDGE_BASE.md                <- This file
    TEAM_ROLES.md                    <- Role assignments
    plan/
      PLAN.md                        <- Master execution plan
    tracking/
      TASKS.md                       <- Task tracker
      STATUS.md                      <- Completion log
    tasks/
      P1-05_PROPOSAL_DRAFT.md        <- Next task: detailed how-to
  code/
    requirements.txt
    src/
      dataset.py, dit.py, unet.py, diffusion.py,
      scoring.py, train.py, inference.py, evaluate.py,
      visualize.py, run_baselines.py
    notebooks/
      01_data_exploration.ipynb
      02_training.ipynb
      03_evaluation.ipynb
  data/
    mvtec/                           <- Dataset (downloaded, gitignored)
    mvtec_ad_evaluation/             <- Official eval scripts (already have)
  output/
    checkpoints/                     <- Model weights (gitignored)
    figures/                         <- Plots for report
    predictions/                     <- .tiff anomaly maps
    results/                         <- CSV/JSON metrics
  proposal/
    proposal.tex + proposal.pdf
  report/
    report.tex + report.pdf
  presentation/
    slides
```

---

## KB-005 — Metrics

| Metric | What it measures | Target | Why not accuracy? |
|--------|-----------------|--------|-------------------|
| Image-level AUROC | Scalar anomaly score per image | > 95% | Class-imbalanced test set |
| Pixel-level AUROC | Per-pixel anomaly score vs GT mask | > 90% | Measures localization quality |
| PRO (Per-Region Overlap) | Official MVTec localization metric | > 85% | Penalizes false positives in large regions |

All three reported per-category AND as mean across categories.

---

## KB-006 — Baselines

| Baseline | Type | Expected AUROC | Effort |
|----------|------|---------------|--------|
| PCA reconstruction | Classical | ~65-70% | 5 lines (sklearn) |
| Autoencoder | Classical DL | ~75-80% | ~50 lines |
| PatchCore (anomalib) | SOTA non-generative | ~99.1% | ~10 lines config |
| Reverse Distillation (anomalib) | SOTA knowledge distillation | ~97% | ~10 lines config |
| **Our DiT method** | **Novel** | **Target: 95%+** | Full implementation |

---

## KB-007 — Settled Decisions (No Longer Open)

| Question | Answer | Rationale |
|----------|--------|-----------|
| Noise schedule | Cosine | Objectively better than linear (Nichol & Dhariwal 2021) |
| Resolution | 128x128 primary | Free Colab T4 compatible |
| Per-category or unified | Per-category | Standard MVTec evaluation protocol |
| Anomaly map metric | SSIM primary | Perceptually superior to L2 |
| Backbone | DiT primary | Uniqueness; UNet as ablation only |
| Sampling | DDIM (50 steps) | 20x faster than DDPM (1000 steps) |
| T_partial | Sweep 100-500 | This is an ablation study |
| Feature comparison | ResNet-18 layers 1,2,3 | 10%+ AUROC boost per DDAD |
| Budget | Zero | All free tools verified |

---

## KB-008 — Open Questions (Implementation-Time Only)

- [ ] What value of T_partial gives best AUROC? (ablation study P3-06)
- [ ] DiT-S or DiT-Tiny? (depends on Colab T4 VRAM at 128x128)
- [ ] Alpha weighting for pixel vs feature map fusion? (default 0.5, tune if needed)
- [ ] Gaussian noise or simplex noise? (start with Gaussian, try simplex if time)

---

## Update Log

| Date | Entry |
|------|-------|
| 2026-03-05 | Initial KB created |
| 2026-03-05 | Revised: upgraded architecture (DiT, DDIM, dual-level scoring), updated papers, settled open questions, added anomalib baselines |
