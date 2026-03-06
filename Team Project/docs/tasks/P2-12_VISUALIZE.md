# Task P2-12: 6-Panel Visualization

## Context
Create publication-quality visualization showing the anomaly detection pipeline
output for individual test images.

## Output
`Team Project/code/src/visualize.py`

## Specification

```python
def plot_anomaly_grid(
    original: Tensor,        # (3, H, W) in [-1, 1]
    noised: Tensor,          # (3, H, W) in [-1, 1]
    reconstruction: Tensor,  # (3, H, W) in [-1, 1]
    pixel_map: Tensor,       # (1, H, W) in [0, 1]
    feature_map: Tensor,     # (1, H, W) in [0, 1]
    gt_mask: Tensor,         # (1, H, W) binary
    save_path: str = None,
    title: str = None,
):
    """
    6-panel visualization grid:
    | Original | Noised | Reconstruction | Pixel Map | Feature Map | GT Mask |
    
    - Images: denormalize from [-1,1] to [0,1] for display
    - Anomaly maps: use 'hot' colormap
    - GT mask: use 'gray' colormap
    - Figure size: (18, 3) inches
    - DPI: 150
    - Tight layout, no extra whitespace
    """

def plot_auroc_bar_chart(
    category_results: dict[str, float],  # {category: auroc}
    method_name: str = "DiT",
    save_path: str = None,
):
    """Horizontal bar chart of per-category AUROC. Include 95% target line."""

def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],  # {method: (fpr, tpr)}
    save_path: str = None,
):
    """Overlay ROC curves for DiT vs baselines. Include diagonal reference."""

def plot_training_loss(
    loss_csv_path: str,
    save_path: str = None,
):
    """Plot training loss curve from CSV (epoch, loss columns)."""

def plot_tpartial_sensitivity(
    t_values: list[int],
    auroc_values: list[float],
    save_path: str = None,
):
    """Line plot: AUROC vs T_partial. Mark the optimal point."""
```

## Success Criteria
- `plot_anomaly_grid(...)` saves a clear 6-panel PNG
- Anomaly maps use 'hot' colormap (red = anomalous)
- All plots use consistent style (matplotlib default + seaborn context)
- All functions work with both file save and inline display
- Figures suitable for LaTeX report inclusion (150+ DPI, vector-friendly)
