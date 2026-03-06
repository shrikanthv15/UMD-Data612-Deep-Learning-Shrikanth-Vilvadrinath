# Task P0-02: Create requirements.txt

## Context
All Python dependencies for the project. Must work with `pip install -r requirements.txt`
on Python 3.10+ inside Google Colab Free or Kaggle Notebooks.

## Output
`Team Project/code/requirements.txt`

## Specification
Include these packages (no version pins unless critical -- Colab/Kaggle handle versions):
```
torch>=2.0
torchvision>=0.15
numpy
Pillow
scikit-learn
scikit-image
matplotlib
seaborn
tqdm
pytorch-msssim
lpips
tifffile
scipy
tabulate
anomalib
```

Do NOT include tensorflow, keras, or conda-specific packages.
Do NOT pin exact versions -- Colab updates frequently.

## Success Criteria
- `pip install -r requirements.txt` completes without errors on Python 3.10
- All imports work: `import torch, torchvision, anomalib, pytorch_msssim, lpips`
