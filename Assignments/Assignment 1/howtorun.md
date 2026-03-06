# How to Run — Assignment 1

## Prerequisites

- **Anaconda** installed on your machine
- A terminal (Anaconda Prompt, Git Bash, or similar)

---

## 1. Set Up the Conda Environment (one-time)

Open a terminal and navigate to the `code/` folder:

```bash
cd "C:\Data612\Assignments\Assignment 1\code"
```

Create the environment from the provided spec:

```bash
conda env create -f environment.yml
```

This installs Python 3.10, TensorFlow 2.10, scikit-learn, matplotlib, and all other
dependencies into a conda environment called **data612**.

> This takes a few minutes the first time. You only need to do it once.

---

## 2. Activate the Environment

Every time you open a new terminal, activate the environment first:

```bash
conda activate data612
```

Your prompt should now show `(data612)` at the beginning.

---

## 3. (Optional) Verify the Environment

Run the version-check scripts to make sure everything installed correctly:

```bash
python versions.py
python dl_versions.py
```

You should see output like:

```
scipy:        1.15.3
numpy:        1.26.4
matplotlib:   3.10.8
pandas:       2.3.3
statsmodels:  0.14.6
scikit-learn: 1.7.2

tensorflow:   2.10.0
keras:        2.10.0
```

---

## 4. Run the Assignment Script

Make sure you're still in the `code/` folder with the environment active, then run:

```bash
python assignment1.py
```

### What happens when you run it:

| Step | What it does                                               | Approx. time |
| ---- | ---------------------------------------------------------- | ------------ |
| 1    | Loads the Boston Housing dataset (506 rows, 14 cols)       | instant      |
| 2    | Splits into X (13 features) and Y (target)                 | instant      |
| 3    | Prints the baseline model architecture (3,009 params)      | instant      |
| 4    | Explains the Pipeline standardization approach             | instant      |
| 5    | Runs 10-fold CV on the baseline model (100 epochs each)    | ~2 min       |
| 6    | Tests 7 different hidden-layer widths with 10-fold CV each | ~5 min       |
| 7    | Tests 6 different network depths + convergence curves      | ~5 min       |

**Total runtime: ~10-15 minutes on CPU.**

### Output:

- **Terminal** — MSE results for every experiment, plus a final summary table
- **Plots** saved to `../output/`:
  - `step6_hidden_units.png` — Hidden layer width vs MSE
  - `step7_depth.png` — Network depth vs MSE
  - `step7_convergence.png` — Training/validation loss curves by depth

---

## Folder Structure

```
Assignment 1/
  Assignment1.md        <-- Assignment description
  howtorun.md           <-- This file
  code/
    assignment1.py      <-- Main script (the deliverable)
    environment.yml     <-- Conda environment spec
    versions.py         <-- Library version check
    dl_versions.py      <-- DL library version check
  data/
    housing dataset.csv             <-- Dataset (506 samples, 13 features)
    Setting Up Your DL Environment.pdf  <-- Course setup reference
  output/
    step6_hidden_units.png  <-- Generated plot
    step7_depth.png         <-- Generated plot
    step7_convergence.png   <-- Generated plot
```

---

## Troubleshooting

**"conda is not recognized"**

> Use the Anaconda Prompt instead of regular Command Prompt, or add conda to your PATH.

**"ModuleNotFoundError: No module named 'tensorflow'"**

> Make sure you ran `conda activate data612` before running the script.

**Script is very slow**

> It's CPU-only by design (TF 2.10 on Windows). ~10-15 min is normal. Don't close the terminal.

**MSE values look way off (>100 for baseline)**

> Something went wrong with standardization. Make sure you're running the unmodified script and haven't changed the Pipeline setup.
