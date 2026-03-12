# Assignment 3 -- How to Run

## Prerequisites

- Anaconda or Miniconda installed
- LaTeX distribution (for compiling the report PDF; e.g., TeX Live, MiKTeX)

## Setup

```bash
# Start from the repository root (C:\Data612 or wherever you cloned it)
cd "Assignments/Assignment 3/code"

# Create or update the conda environment
conda env update -f environment.yml
conda activate data612
```

## Run

```bash
python assignment3.py
```

## Expected Output

- Terminal output with all 11 step sections and accuracy results
- 5 PNG plots saved to `output/`:
  - `step10_depth_accuracy.png` -- Accuracy curves for 1-4 LSTM layers over 500 epochs
  - `step10_depth_loss.png` -- Loss curves for 1-4 LSTM layers over 500 epochs
  - `step11_learning_rate.png` -- Bar chart of accuracy for 6 learning rate values
  - `step11_hidden_size.png` -- Bar chart of accuracy for 6 hidden unit values
  - `step11_num_layers.png` -- Bar chart of accuracy for 5 depth values

## Estimated Runtime

| Step | Description | Time (CPU) |
|------|-------------|------------|
| 1-6 | Data preparation | <1s |
| 7 | Baseline training (500 epochs) | ~15s |
| 8-9 | Accuracy + predictions | <5s |
| 10 | Depth experiments (4 configs) | ~1 min |
| 11 | Hyperparameter sweeps (17 configs) | ~4 min |
| **Total** | | **~5-7 min** |

## Troubleshooting

- **TensorFlow warnings:** Suppressed via `TF_CPP_MIN_LOG_LEVEL=2`. If errors persist, verify `tensorflow==2.10` is installed.
- **Low accuracy:** The baseline should exceed 80%. If not, ensure seeds are set correctly and `learning_rate=0.01`.
- **Memory:** This assignment uses only 25 samples; memory should not be an issue.
- **Slow training:** Each model trains for 500 epochs with batch_size=1 on 25 samples. Total runtime is ~5-7 minutes on CPU.

## Compile Report (PDF)

```bash
# From the report/ directory
cd "Assignments/Assignment 3/report"
pdflatex Data612_Assignment_3_Shrikanth_Vilvadrinath_Report.tex
pdflatex Data612_Assignment_3_Shrikanth_Vilvadrinath_Report.tex  # Run twice for ToC/refs
```

Note: The `.tex` file references images via `output/` relative paths. For Overleaf, upload the `output/` folder alongside the `.tex` file.

## Verify Versions

```bash
python versions.py
python dl_versions.py
```
