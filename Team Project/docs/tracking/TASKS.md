# TASK TRACKER
## DiT-Based Anomaly Detection -- Group 6

Legend: `AGENT` = AI can execute | `HUMAN` = requires human | `DONE` / `TODO` / `WIP`

---

## PHASE 0 -- Environment & Data

| ID | Task | Type | Status | Depends On |
|----|------|------|--------|------------|
| P0-01 | Download MVTec AD dataset | HUMAN | DONE | -- |
| P0-02 | Create requirements.txt | AGENT | TODO | -- |
| P0-03 | Write MVTec Dataset class (dataset.py) | AGENT | TODO | P0-02 |
| P0-04 | Set up shared Google Drive | HUMAN | TODO | -- |
| P0-05 | Set up GitHub repo | HUMAN | TODO | -- |
| P0-06 | Verify Colab Free T4 + AMP | HUMAN | TODO | P0-02 |

## PHASE 1 -- Research & Proposal

| ID | Task | Type | Status | Depends On |
|----|------|------|--------|------------|
| P1-01 | Read DDPM paper (Ho 2020) sec 1-3 | HUMAN | TODO | -- |
| P1-02 | Read DiT paper (Peebles 2023) sec 1-3 | HUMAN | TODO | -- |
| P1-03 | Read DDAD paper (Mousakhan 2023) sec 1-3 | HUMAN | TODO | -- |
| P1-04 | Read DDIM paper (Song 2021) sec 4 | HUMAN | TODO | -- |
| P1-05 | Write proposal draft (LaTeX) | AGENT | DONE | -- |
| P1-06 | Compile proposal + submit | HUMAN | TODO | P1-05 |

## PHASE 2 -- Core Implementation

| ID | Task | Type | Status | Depends On |
|----|------|------|--------|------------|
| P2-01 | Implement cosine noise schedule | AGENT | TODO | P0-02 |
| P2-02 | Implement forward process q(x_t|x_0) | AGENT | TODO | P2-01 |
| P2-03 | Implement DiT backbone | AGENT | TODO | P0-02 |
| P2-04 | Implement DDIM reverse sampling | AGENT | TODO | P2-01 |
| P2-05 | Add data augmentations to Dataset | AGENT | TODO | P0-03 |
| P2-06 | Implement training loop with AMP | AGENT | TODO | P2-02, P2-03 |
| P2-07 | Smoke test: 5 epochs on hazelnut | HUMAN | TODO | P2-06, P0-01 |
| P2-08 | Implement reconstruction inference | AGENT | TODO | P2-04 |
| P2-09 | Implement SSIM pixel anomaly map | AGENT | TODO | P2-08 |
| P2-10 | Implement ResNet-18 feature anomaly map | AGENT | TODO | P2-08 |
| P2-11 | Implement combined scoring + AUROC eval | AGENT | TODO | P2-09, P2-10 |
| P2-12 | Implement 6-panel visualization | AGENT | TODO | P2-11 |
| P2-13 | Full training on hazelnut (100 epochs) | HUMAN | TODO | P2-07 |
| P2-14 | Evaluate hazelnut: AUROC + visual maps | HUMAN | TODO | P2-11, P2-13 |

## PHASE 3 -- Full Evaluation & Baselines

| ID | Task | Type | Status | Depends On |
|----|------|------|--------|------------|
| P3-01 | Train all 15 categories | HUMAN | TODO | P2-13 |
| P3-02 | Run PatchCore baseline (anomalib) | AGENT | TODO | P0-03 |
| P3-03 | Run Reverse Distillation baseline (anomalib) | AGENT | TODO | P0-03 |
| P3-04 | Run PCA + AE baselines | AGENT | TODO | P0-03 |
| P3-05 | Ablation: DiT vs UNet backbone | AGENT | TODO | P2-13 |
| P3-06 | Ablation: T_partial sweep (100-500) | HUMAN | TODO | P2-13 |
| P3-07 | Ablation: SSIM vs L2 vs LPIPS | AGENT | TODO | P2-13 |
| P3-08 | Ablation: With/without feature scoring | AGENT | TODO | P2-13 |
| P3-09 | Compute pixel-level AUROC all categories | HUMAN | TODO | P3-01 |
| P3-10 | Export .tiff maps + run MVTec eval (PRO) | AGENT | TODO | P3-01 |

## PHASE 4 -- Report & Presentation

| ID | Task | Type | Status | Depends On |
|----|------|------|--------|------------|
| P4-01 | Write final report (LaTeX) | AGENT | TODO | P3-01 |
| P4-02 | Finalize all figures | AGENT | TODO | P3-01 |
| P4-03 | Compile report PDF | HUMAN | TODO | P4-01, P4-02 |
| P4-04 | Create presentation slides | AGENT | TODO | P4-01 |
| P4-05 | Presentation rehearsal | HUMAN | TODO | P4-04 |
| P4-06 | Code cleanup + README | AGENT | TODO | P3-01 |

---

For AGENT tasks: detailed specs are in `docs/tasks/{ID}.md`.
For HUMAN tasks: team members execute these manually (Colab training, paper reading, etc.).