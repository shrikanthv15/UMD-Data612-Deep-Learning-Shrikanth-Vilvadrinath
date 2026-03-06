# DATA-MSML 612 — Deep Learning (University of Maryland)

## Repository Structure
- `SLIDES/` — Lecture slides
- `Transcripts/` — Lecture transcripts
- `Assignments/` — Homework assignments

## Framework
- **Keras / TensorFlow** with **scikit-learn** for model evaluation
- **scikeras** for Pipeline integration (`KerasRegressor`)
- Conda environment: `data612` (created from `Assignments/Assignment 1/environment.yml`)

## Commands
```bash
conda activate data612
cd "Assignments/Assignment 1"
python assignment1.py
```

## Dataset Convention
- Boston Housing CSV (`housing dataset.csv`) is whitespace-delimited, no header row
- Load with: `pd.read_csv('housing dataset.csv', sep=r'\s+', header=None)`
- 506 rows, 14 columns (13 features + 1 target MEDV)

## Key Patterns
- Use `Pipeline(StandardScaler + KerasRegressor)` for proper cross-validation (prevents data leakage)
- Use `scikeras.wrappers.KerasRegressor` (not the deprecated `keras.wrappers`)
- Use `model=` parameter (not `build_fn=`) for scikeras
- Negate sklearn's `neg_mean_squared_error` when reporting MSE
- Set random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) for reproducibility
