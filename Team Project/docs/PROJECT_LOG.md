# Project Log — DATA-MSML 612 Group 6
## Anomaly Detection via Diffusion Reconstruction

**Course:** DATA-MSML 612 — Deep Learning, University of Maryland
**Group 6 Members:** Akshay Kamkhalia, Kevin Rathod, Shrikanth Vilvadrinath, Sivram Sahu
**Track:** Diffusion Model
**Due Date:** TBD (Proposal due first)

---

## Log Entry 005 — Proposal Draft Written
**Date:** 2026-03-05
**Status:** P1-05 DONE — proposal ready for Overleaf compilation

### What Was Created
`Team Project/proposal/proposal.tex` — a 2.5-page LaTeX proposal covering all 6
professor requirements: background, significance, problem statement, data acquisition,
validation metrics, and DL framework.

### Structure
1. Introduction (significance + innovation claim)
2. Background and Related Work (DDPM, DDIM, AnoDDPM, DDAD, DiT)
3. Problem Statement (unsupervised detection + localization)
4. Dataset and Data Acquisition (MVTec AD, 15 categories, free)
5. Methodology and Evaluation (DiT-S pipeline, AUROC/PRO metrics, 4 ablations)
6. Framework and Reproducibility (PyTorch stack, zero budget)

8 references via `\thebibliography`. Preamble matches A2 report style.

### Next Step
P1-06: Compile on Overleaf, review, submit.

---

## Log Entry 001 — Project Selection
**Date:** 2026-03-05
**Status:** Project Selected — Pre-Planning Phase

### Decision
Chose **Project #11: Anomaly Detection via Diffusion Reconstruction** (Diffusion Track).

### Why This Project
- Non-obvious, research-grade use of diffusion architecture: instead of generating images,
  we exploit the model's reconstruction error as an anomaly score.
- Fully unsupervised during training — no labeled anomaly data needed.
- Highly differentiating — no other group in this class will likely approach diffusion this way.
- Maps directly to a real industrial problem (manufacturing defect detection).
- The "twist" is the entire project narrative: diffusion is typically a generator; here it becomes a detector.

### Core Concept (One Sentence)
A diffusion model trained only on normal images learns the distribution of normality;
anything it cannot reconstruct cleanly (high residual error) is flagged as anomalous.

---

## Log Entry 005 -- Documentation Restructured for AI Agents
**Date:** 2026-03-06
**Status:** All docs restructured, task files created, beginning implementation

### What Happened
Scored the updated plan at **90/100 (A-)**. Then restructured all documentation
for AI agent consumption per project lead's directive.

### Changes Made
1. **TEAM_ROLES.md** -- stripped task tables with owners/deadlines/deliverables.
   Now contains only role assignments. Tasks tracked in `tracking/TASKS.md`.
2. **TASKS.md** -- complete rewrite. Each task now has: ID, title, type
   (AGENT/HUMAN), status, and dependency chain. 41 tasks across 5 phases.
3. **CHECKLIST.md** -- fixed section 3 inconsistency (still referenced old
   UNet/DDPM pipeline). Updated to reflect DiT/DDIM/dual-scoring architecture.
4. **PLAN.md** -- removed human-oriented owner columns from all phase tables.
   Tables now show ID, task, type (AGENT/HUMAN), and "Done When" criteria.
5. **docs/tasks/** -- created 8 detailed task specification files for all
   AGENT-executable tasks through Phase 2:
   - P0-02 (requirements.txt), P0-03 (dataset.py)
   - P2-01/02 (diffusion.py), P2-03 (dit.py), P2-04 (DDIM)
   - P2-06 (train.py), P2-08/09/10/11 (scoring pipeline), P2-12 (visualize.py)
   Each file has: Context, Inputs, Outputs, Specification, Success Criteria.

### Architecture for AI Agents
```
docs/tracking/TASKS.md     -- Index: what needs doing (ID, type, status, deps)
docs/tasks/{ID}.md         -- Detail: exactly how to do it (spec, success criteria)
docs/plan/PLAN.md          -- Big picture: architecture, phases, risk register
docs/KNOWLEDGE_BASE.md     -- Technical reference: math, papers, settled decisions
```

An agent reads TASKS.md to find the next TODO task with all dependencies DONE,
then reads the corresponding task file in docs/tasks/ for the detailed spec.

### Next
Begin implementation: P0-02 (requirements.txt) and P0-03 (dataset.py).

---

## Log Entry 004 — Master Plan & Documentation Architecture
**Date:** 2026-03-05
**Status:** Plan created, documentation structure established

### What Happened
Incorporated all 18 items from Remarks.MD into a formal execution plan (PLAN.md).
Created full project documentation architecture with structured tracking.

### Decisions Settled (from Remarks.MD open questions)
1. **DiT backbone** — confirmed as primary (UNet as ablation baseline)
2. **Cosine noise schedule** — confirmed (linear as ablation only)
3. **SSIM** — confirmed for pixel-level anomaly maps (L2 as ablation)
4. **DDIM 50-step sampling** — confirmed (DDPM 1000-step as ablation)
5. **Per-category training** — confirmed (standard MVTec protocol)
6. **128x128 primary resolution** — confirmed (free Colab compatible)
7. **Dual-level scoring** — pixel (SSIM) + feature (ResNet-18 layers 1-3)
8. **SOTA baselines via anomalib** — PatchCore + Reverse Distillation
9. **Target AUROC revised** — 85% -> 95%+ (matching modern methods)
10. **Zero budget** — all tools, data, compute confirmed free

### Cost Verification
Verified every component in the tech stack is free and open-source.
Compute: Google Colab Free (T4) + Kaggle Free (P100) = ~240 GPU hrs/week
across 4 accounts. Training all 15 categories estimated at ~45-50 total GPU hours.

### Documents Created
- `docs/plan/PLAN.md` — Master execution plan (10 sections)
- `docs/tracking/TASKS.md` — 52 tasks across 4 phases with owners
- `docs/tracking/STATUS.md` — Weekly status report (rolling)
- Updated PROJECT_LOG.md (this entry)

### Documentation Architecture
```
docs/
  PROJECT_LOG.md        <- Decision journal (append-only, newest first)
  CHECKLIST.md          <- Pre-planning reference
  KNOWLEDGE_BASE.md     <- Running technical reference
  TEAM_ROLES.md         <- Role assignments
  plan/
    PLAN.md             <- Master execution plan
  tracking/
    TASKS.md            <- Task tracker (52 tasks, 4 phases)
    STATUS.md           <- Weekly status snapshots
```

---

## Log Entry 003 -- Architecture Critique & Remarks (AI Audit)
**Date:** 2026-03-05
**Status:** Remarks generated -- plan revision needed before implementation

### What Happened
Conducted a full brutal critique of the project plan against current state-of-the-art
(2024-2025 papers). Generated `Remarks.MD` with 18 actionable items.

### Key Findings
1. **Architecture is outdated:** Vanilla DDPM (Ho 2020) + AnoDDPM (2022) achieves ~88% AUROC.
   Modern methods (DDAD 2023, DiffusionAD 2023) achieve 99.8% on the same dataset.
   Our 85% AUROC target is below what non-diffusion methods (PatchCore) achieve.
2. **Missing critical techniques:** No DDIM sampling (10-20x faster inference), no feature-level
   anomaly comparison (10%+ AUROC boost), no SSIM/LPIPS for anomaly maps, no multi-scale fusion.
3. **Uniqueness opportunity identified:** Using a Diffusion Transformer (DiT) backbone instead of
   UNet would combine BOTH project tracks (Transformer + Diffusion) and be genuinely novel --
   no published DiT-based anomaly detection on MVTec exists.
4. **Baselines are too weak:** PCA and vanilla AE are 2019-era. Need PatchCore and Reverse
   Distillation via `anomalib` library for credible comparison.
5. **PRO metric missed:** We already have the official MVTec evaluation scripts but the plan
   doesn't mention using PRO (Per-Region Overlap) -- the official localization metric.
6. **Zero code written after a full day of planning.** Need to start smoke-testing.

### Decisions Needed
- [ ] DiT vs UNet backbone (recommendation: DiT for uniqueness)
- [ ] DDAD-style conditioned denoising vs DiffusionAD noise-to-norm
- [ ] Target AUROC revision: 85% -> 95%+
- [ ] Which ablation studies to include (minimum 3 recommended)
- [ ] Concrete deadlines (all currently TBD)

### Reference
Remarks.MD was in project root — all 18 items have been incorporated into
`docs/plan/PLAN.md`. File removed after full integration (Session 3).

---

## Log Entry 002 — Pre-Planning Checklist Session
**Date:** 2026-03-05
**Status:** Checklist created, questions answered before proposal writing
**Reference:** See `CHECKLIST.md` for full breakdown

### Session Summary
Before writing the proposal, the team conducted a pre-planning session to answer:
1. How do we get the data — difficulty and time cost?
2. How do we preprocess it — pipeline complexity?
3. What exactly are we coding — diffusion architecture depth?
4. How automatable is this workflow with Claude?
5. What will change between proposal and implementation?

All findings documented in `CHECKLIST.md`. Key conclusion: this project is feasible for
4 students over a typical semester project timeline. The largest risk is diffusion training
time on CPU; mitigation is using Google Colab Pro or a pre-trained backbone.

---
