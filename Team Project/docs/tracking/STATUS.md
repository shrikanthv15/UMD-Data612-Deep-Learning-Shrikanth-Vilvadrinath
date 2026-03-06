# STATUS LOG
## DiT-Based Anomaly Detection -- Group 6

**Convention:** Append new entries at the top after each session.

---

## 2026-03-06 -- Session 5: Full Core Implementation Complete
- Implemented ALL Phase 2 AGENT tasks (P2-01 through P2-12)
- **code/src/dataset.py** -- MVTec AD PyTorch Dataset with augmentations (P0-03, P2-05)
- **code/src/diffusion.py** -- Cosine schedule + forward process + DDIM sampling (P2-01, P2-02, P2-04, P2-08)
- **code/src/dit.py** -- DiT-S (33M params) and DiT-Tiny (4.4M params) backbones (P2-03)
- **code/src/train.py** -- Training loop with AMP, warmup+cosine LR, gradient clipping (P2-06)
- **code/src/scoring.py** -- Dual-level scoring: SSIM pixel + ResNet-18 feature maps (P2-09, P2-10, P2-11)
- **code/src/evaluate.py** -- Full evaluation pipeline: per-category AUROC computation (P2-11)
- **code/src/visualize.py** -- 6-panel comparison visualization (P2-12)
- **code/requirements.txt** -- All dependencies (P0-02)
- All 7 modules pass import + smoke tests on CPU
- 14/41 tasks DONE (all Phase 0-2 AGENT tasks complete)
- Next: P2-07 (smoke test on GPU), P3 baselines, P4 report

## 2026-03-06 -- Session 4: Docs Restructured + Implementation Started
- Scored updated plan: 90/100 (A-)
- Rewrote TEAM_ROLES.md (stripped task tables)
- Rewrote TASKS.md (agent-executable format with types and deps)
- Fixed CHECKLIST.md section 3 (was still referencing UNet/DDPM)
- Cleaned PLAN.md phase tables (removed owner columns)
- Created 8 agent task files in docs/tasks/
- Created code/ directory structure
- Created P0-02 (requirements.txt) and P0-03 (dataset.py)
- Next: P2-01/02 (diffusion.py), P2-03 (dit.py)

## 2026-03-05 -- Session 3: Plan Finalized + Docs Restructured
- Simplified documentation structure (removed over-engineering)
- Rewrote TASKS.md: clean ID/Task/Status format, no owners or deadlines
- Created docs/tasks/ folder for detailed next-task instructions
- Updated KNOWLEDGE_BASE.md and CHECKLIST.md to reflect revised plan
- Removed Remarks.MD (fully incorporated into PLAN.md)
- Marked P0-01 (MVTec download) as DONE
- Next: Proposal draft (P1-05)

## 2026-03-05 -- Session 2: Plan Created
- Created master execution plan (PLAN.md) incorporating all 18 critique items
- Created task tracker (TASKS.md) and status log (STATUS.md)
- Verified all tools/data/compute are free -- zero budget
- Settled all open architectural questions (DiT, DDIM, cosine, SSIM, etc.)
- Established documentation folder structure

## 2026-03-05 -- Session 1: Project Selected + Pre-Planning
- Selected project: Anomaly Detection via Diffusion Reconstruction
- Created pre-planning checklist (CHECKLIST.md)
- Created knowledge base (KNOWLEDGE_BASE.md)
- Created team roles draft (TEAM_ROLES.md)
- Assessed feasibility: data, compute, coding complexity, automation