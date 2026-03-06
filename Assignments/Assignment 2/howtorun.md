# Assignment 2 -- How to Run

## Prerequisites

- Anaconda or Miniconda installed
- Internet connection (first run downloads IMDB dataset ~80MB and NLTK stopwords)
- LaTeX distribution (for compiling the report PDF; e.g., TeX Live, MiKTeX)

## Setup

```bash
# Start from the repository root (C:\Data612 or wherever you cloned it)
cd "Assignments/Assignment 2/code"

# Create or update the conda environment
conda env update -f environment.yml
conda activate data612
```

## Run

```bash
python assignment2.py
```

## Expected Output

- Terminal output with all 7 task sections and accuracy results
- 3 PNG plots saved to `output/`:
  - `task5_cnn_accuracy.png` --CNN training vs validation accuracy curves
  - `task5_cnn_loss.png` --CNN training vs validation loss curves
  - `task7_model_comparison.png` --Grouped bar chart comparing all models

## Estimated Runtime

| Task | Description | Time (CPU) |
|------|-------------|------------|
| 1-2 | Data loading & verification | ~10s |
| 3a | Logistic Regression | ~30s |
| 3b | Feed-Forward NN (10 epochs) | ~3 min |
| 4 | CNN architecture (no training) | <1s |
| 5 | CNN training (10 epochs) | ~3 min |
| 6 | Stopword CNN (10 epochs) | ~3 min |
| 7 | Comparison & plots | <5s |
| **Total** | | **~10-12 min** |

## Troubleshooting

- **NLTK stopwords error:** The script includes a fallback stopword list and will work even if NLTK download fails.
- **Memory issues:** The bag-of-words matrix requires ~500MB RAM. Ensure at least 8GB total RAM.
- **Low accuracy:** Ensure you are using `NUM_WORDS=5000` and `MAX_LEN=500`. If CNN accuracy is below 85%, try increasing `EPOCHS` to 15.
- **TensorFlow warnings:** Suppressed via `TF_CPP_MIN_LOG_LEVEL=2`. If errors persist, verify `tensorflow==2.10` is installed.
- **IMDB download:** Dataset is cached in `~/.keras/datasets/` after the first download.

## Compile Report (PDF)

```bash
# From the report/ directory
cd "Assignments/Assignment 2/report"
pdflatex report.tex
pdflatex report.tex  # Run twice for table of contents + references

# Copy to Submission with correct naming
cp report.pdf "../Submission/Data612_Assignment_2_Shrikanth_Vilvadrinath_Report.pdf"
```

Note: The `.tex` file references images via `../output/` relative paths. Compile from the `report/` directory so LaTeX can find the PNG plots.

## Verify Versions

```bash
python versions.py
python dl_versions.py
```
