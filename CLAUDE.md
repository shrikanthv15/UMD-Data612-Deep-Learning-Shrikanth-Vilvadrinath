# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## DATA-MSML 612 — Deep Learning (University of Maryland)

Course repo containing assignments, lecture slides, and transcripts. Student: Shrikanth Vilvadrinath.

## Environment

- Conda environment: `data612` (Python 3.10, TensorFlow 2.10, scikeras 0.11)
- Environment spec: `Assignments/Assignment <N>/code/environment.yml`

```bash
# One-time setup (from the assignment's code/ directory)
conda env create -f environment.yml

# Every session
conda activate data612
```

## Running Assignments

**Assignment 1 — Boston Housing MLP regression:**
```bash
cd "Assignments/Assignment 1/code"
conda activate data612
python assignment1.py        # ~10-15 min on CPU
python versions.py           # verify environment
python dl_versions.py
```

**Assignment 2 — IMDB Sentiment CNN:**
```bash
cd "Assignments/Assignment 2/code"
conda activate data612
python assignment2.py        # ~10-12 min on CPU; downloads IMDB ~80MB on first run
```

**Compile report (LaTeX):**
```bash
cd "Assignments/Assignment <N>/report"
pdflatex report.tex
pdflatex report.tex   # run twice for ToC/refs
```

## Assignment Architecture

### Assignment 1 — Boston Housing (`assignment1.py`)
- 7 sequential steps: load data → split X/Y → define model → pipeline → K-fold CV → hidden unit sweep → depth sweep
- Dataset: `code/data/housing dataset.csv` — whitespace-delimited, no header, 506×14 (cols 0–12 features, col 13 = MEDV target)
  - Load with: `pd.read_csv('housing dataset.csv', sep=r'\s+', header=None)`
- Outputs PNG plots to `output/`: `step6_hidden_units.png`, `step7_depth.png`, `step7_convergence.png`

### Assignment 2 — IMDB Sentiment (`assignment2.py`)
- 7 tasks: load/preprocess → train-test split → baselines (LogReg + FFNN) → CNN architecture → train/eval CNN → stopword effect → comparison
- Dataset: `keras.datasets.imdb`, top 5000 words, padded to length 500, cached in `~/.keras/datasets/`
- CNN architecture: `Embedding(5000, 32) → Conv1D(32, 7, relu) → MaxPooling1D(5) → Flatten → Dense(250, relu) → Dense(1, sigmoid)`
- Outputs PNG plots to `output/`: `task5_cnn_accuracy.png`, `task5_cnn_loss.png`, `task7_model_comparison.png`
- NLTK stopwords used in Task 6; script has a fallback list if NLTK download fails

## Key Patterns

- Use `Pipeline([('standardize', StandardScaler()), ('mlp', KerasRegressor(...))])` for cross-validation to prevent data leakage
- Use `scikeras.wrappers.KerasRegressor` with `model=` parameter (not deprecated `keras.wrappers` or `build_fn=`)
- Pass model hyperparameters via `model__<param>` prefix in `KerasRegressor()`
- Negate sklearn's `neg_mean_squared_error` scores: `mse = -cross_val_score(...)`
- Seeds: `np.random.seed(42)`, `tf.random.set_seed(42)`, `random.seed(42)`
- Use `matplotlib.use('Agg')` and `plt.savefig()` (no interactive display); suppress TF logs with `TF_CPP_MIN_LOG_LEVEL=2`

## Folder Convention per Assignment

```
Assignment N/
  AssignmentN.md       # Assignment spec
  howtorun.md          # Setup + run instructions
  code/
    assignmentN.py     # Main deliverable script
    environment.yml    # Conda spec
    versions.py        # Check non-DL package versions
    dl_versions.py     # Check TF/Keras versions
    data/              # (Assignment 1 only) dataset files
  output/              # Generated plots (git-tracked)
  report/              # LaTeX source (.tex references ../output/ for plots)
  Submission/          # Final files submitted to instructor
```
